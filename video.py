import concurrent.futures
import logging
from typing import Optional, Tuple

import av
import cv2
import numpy as np
from streamlit_webrtc import VideoProcessorBase

from config import DETECTION_INTERVAL_FRAMES, SCANNING_LABEL, UNKNOWN_NAME
from database import mark_attendance
from recognition import (
    FaceBox,
    FaceDetector,
    FaceEmbedder,
    FaceGallery,
    crop_face_rgb,
    find_match,
)

logger = logging.getLogger(__name__)

# BGR colors
_YELLOW = (0, 255, 255)
_GREEN = (0, 255, 0)
_RED = (0, 0, 255)


class FaceRecognitionProcessor(VideoProcessorBase):
    """Detects faces on every Nth frame and recognizes them off-thread.

    Recognition runs on a single worker thread so the video stream never blocks.
    """

    def __init__(self, detector: FaceDetector, embedder: FaceEmbedder) -> None:
        super().__init__()
        self._detector = detector
        self._embedder = embedder
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        self._task: Optional[concurrent.futures.Future] = None
        self._frame_count = 0
        self._last_face: Optional[FaceBox] = None
        self._label = SCANNING_LABEL
        self._color = _YELLOW

        # Replaced atomically as a single tuple by update_context().
        self._context: Tuple[FaceGallery, Optional[int]] = (FaceGallery(), None)

    def update_context(self, gallery: FaceGallery, org_id: Optional[int]) -> None:
        """Called from the Streamlit thread to supply the registered faces."""
        self._context = (gallery, org_id)

    def on_ended(self) -> None:
        """Release the worker thread when the stream stops."""
        self._executor.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------ recognition

    def _recognize(
        self,
        face_rgb: np.ndarray,
        gallery: FaceGallery,
        org_id: Optional[int],
    ) -> str:
        try:
            encoding = self._embedder.embed(face_rgb)
            name = find_match(encoding, gallery)
        except Exception:
            logger.exception("Face recognition failed")
            return UNKNOWN_NAME

        if name != UNKNOWN_NAME and org_id is not None:
            try:
                mark_attendance(org_id, name)
            except Exception:
                logger.exception("Failed to record attendance")
        return name

    # ------------------------------------------------------------ frame steps

    def _collect_result(self) -> None:
        if self._task is None or not self._task.done():
            return

        try:
            name = self._task.result()
        except Exception:
            logger.exception("Recognition task failed")
            name = UNKNOWN_NAME
        self._task = None

        if self._last_face is None:
            return

        self._label = name
        self._color = _GREEN if name != UNKNOWN_NAME else _RED

    def _update_detection(self, img: np.ndarray) -> None:
        faces = self._detector.detect(img)
        if faces:
            self._last_face = faces[0]
        else:
            self._last_face = None
            self._label = SCANNING_LABEL
            self._color = _YELLOW

    def _start_recognition(self, img: np.ndarray, box: FaceBox) -> None:
        if self._task is not None:
            return
        face_rgb = crop_face_rgb(img, box)
        if face_rgb is None:
            return
        gallery, org_id = self._context
        self._task = self._executor.submit(self._recognize, face_rgb, gallery, org_id)

    def _draw(self, img: np.ndarray, box: FaceBox) -> None:
        x, y, w, h = box
        cv2.rectangle(img, (x, y), (x + w, y + h), self._color, 3)
        cv2.putText(
            img,
            self._label,
            (x, max(y - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            self._color,
            2,
        )

    # ------------------------------------------------------------- WebRTC hook

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self._frame_count += 1

        self._collect_result()

        if self._frame_count % DETECTION_INTERVAL_FRAMES == 0:
            self._update_detection(img)

        if self._last_face is not None:
            self._start_recognition(img, self._last_face)
            self._draw(img, self._last_face)

        return av.VideoFrame.from_ndarray(img, format="bgr24")
