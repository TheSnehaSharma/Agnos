import logging
import os
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
from keras_facenet import FaceNet

from config import (
    DETECTION_MIN_NEIGHBORS,
    DETECTION_SCALE_FACTOR,
    FACE_INPUT_SIZE,
    MATCH_THRESHOLD,
    MIN_FACE_SIZE,
    UNKNOWN_NAME,
)

logger = logging.getLogger(__name__)

FaceBox = Tuple[int, int, int, int]  # x, y, w, h


# ---------------------------------------------------------------- detection


class FaceDetector:
    """Haar-cascade frontal face detector."""

    def __init__(self) -> None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(path)
        if self._cascade.empty():
            raise RuntimeError(f"Unable to load face cascade from {path}")

    def detect(self, bgr_image: np.ndarray) -> List[FaceBox]:
        """Return detected faces, largest first."""
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=DETECTION_SCALE_FACTOR,
            minNeighbors=DETECTION_MIN_NEIGHBORS,
            minSize=MIN_FACE_SIZE,
        )
        boxes = [tuple(int(v) for v in face) for face in faces]
        return sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)


def crop_face_rgb(bgr_image: np.ndarray, box: FaceBox) -> Optional[np.ndarray]:
    """Return an independent RGB copy of the face region, or None if empty."""
    x, y, w, h = box
    region = bgr_image[y : y + h, x : x + w]
    if region.size == 0:
        return None
    return cv2.cvtColor(region, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------- embedding


class FaceEmbedder:
    """Wraps a FaceNet model; inference is serialized with a lock."""

    def __init__(self, model) -> None:
        self._model = model
        self._lock = threading.Lock()

    @classmethod
    def load(cls) -> "FaceEmbedder":
        return cls(FaceNet())

    def embed(self, face_rgb: np.ndarray) -> np.ndarray:
        resized = cv2.resize(face_rgb, FACE_INPUT_SIZE, interpolation=cv2.INTER_AREA)
        batch = np.expand_dims(resized.astype(np.float32), axis=0)
        with self._lock:
            return self._model.embeddings(batch)[0]


# ----------------------------------------------------------------- matching


@dataclass(frozen=True, eq=False)
class FaceGallery:
    """Immutable snapshot of registered faces, safe to share across threads."""

    names: Tuple[str, ...] = ()
    encodings: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float32)
    )

    def __len__(self) -> int:
        return len(self.names)

    @classmethod
    def build(
        cls, names: Sequence[str], encodings: Sequence[np.ndarray]
    ) -> "FaceGallery":
        if not names:
            return cls()

        dimension = np.asarray(encodings[0]).size
        kept_names, kept_encodings = [], []
        for name, encoding in zip(names, encodings):
            array = np.asarray(encoding, dtype=np.float32).ravel()
            if array.size != dimension:
                logger.warning("Skipping '%s': embedding size mismatch", name)
                continue
            kept_names.append(name)
            kept_encodings.append(array)

        return cls(names=tuple(kept_names), encodings=np.stack(kept_encodings))


def cosine_distances(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine distance between query and every row of matrix.

    Zero-norm vectors have distance 1.0.
    """
    query = np.asarray(query, dtype=np.float32).ravel()
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
    safe_norms = np.where(norms == 0, 1.0, norms)
    similarities = (matrix @ query) / safe_norms
    return np.where(norms == 0, 1.0, 1.0 - similarities)


def find_match(
    encoding: np.ndarray,
    gallery: FaceGallery,
    threshold: float = MATCH_THRESHOLD,
) -> str:
    """Return the closest registered name below threshold, else UNKNOWN_NAME."""
    if len(gallery) == 0:
        return UNKNOWN_NAME

    query = np.asarray(encoding, dtype=np.float32).ravel()
    if query.size != gallery.encodings.shape[1]:
        logger.warning("Embedding size does not match the registered faces")
        return UNKNOWN_NAME

    distances = cosine_distances(query, gallery.encodings)
    best = int(np.argmin(distances))
    if distances[best] < threshold:
        return gallery.names[best]
    return UNKNOWN_NAME


# --------------------------------------------------------------- extraction


class FaceExtractionError(Exception):
    """Base class for problems extracting a face from an uploaded image."""


class InvalidImageError(FaceExtractionError):
    def __init__(self) -> None:
        super().__init__("Invalid image file.")


class NoFaceFoundError(FaceExtractionError):
    def __init__(self) -> None:
        super().__init__("No face found! Please try again with better lighting.")


class MultipleFacesError(FaceExtractionError):
    def __init__(self) -> None:
        super().__init__(
            "Multiple faces found! Please ensure only one person is in the frame."
        )


class FaceCropError(FaceExtractionError):
    def __init__(self) -> None:
        super().__init__("Unable to extract the face.")


def extract_single_face(image_bytes: bytes, detector: FaceDetector) -> np.ndarray:
    """Decode an image and return the RGB crop of the only face in it."""
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise InvalidImageError()

    boxes = detector.detect(image)
    if not boxes:
        raise NoFaceFoundError()
    if len(boxes) > 1:
        raise MultipleFacesError()

    crop = crop_face_rgb(image, boxes[0])
    if crop is None:
        raise FaceCropError()
    return crop
