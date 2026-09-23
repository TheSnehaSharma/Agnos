import streamlit as st
import pandas as pd
import cv2
import numpy as np
import hashlib
import os
import concurrent.futures
import sqlite3
import secrets
from datetime import datetime
import pytz
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode
import av
import io

# ============================================================
# PREVENT TENSORFLOW MEMORY CRASHES
# ============================================================

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from keras_facenet import FaceNet


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "agnos_data"

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

DB_FILE = os.path.join(DATA_DIR, "agnos.db")

RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {
            "urls": ["stun:stun.l.google.com:19302"]
        }
    ]
})

st.set_page_config(
    page_title="Agnos",
    page_icon="👁️",
    layout="wide"
)


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def get_embedder():
    return FaceNet()


embedder = get_embedder()


# ============================================================
# DATABASE
# ============================================================

def get_db_connection():
    """
    Creates a new SQLite connection for each operation/thread.

    This is important because WebRTC recognition runs in a
    background thread.
    """
    conn = sqlite3.connect(
        DB_FILE,
        timeout=30,
        check_same_thread=False
    )

    conn.row_factory = sqlite3.Row

    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")

    return conn


def init_database():
    """
    Creates the complete application database.

    Existing databases are preserved.
    """

    conn = get_db_connection()

    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS organizations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                org_key TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                session_token TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                organization_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                encoding BLOB NOT NULL,
                embedding_dimension INTEGER NOT NULL,
                created_at TEXT NOT NULL,

                FOREIGN KEY (organization_id)
                    REFERENCES organizations(id)
                    ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                organization_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                attendance_date TEXT NOT NULL,
                attendance_time TEXT NOT NULL,
                created_at TEXT NOT NULL,

                FOREIGN KEY (organization_id)
                    REFERENCES organizations(id)
                    ON DELETE CASCADE,

                UNIQUE (
                    organization_id,
                    name,
                    attendance_date
                )
            );

            CREATE INDEX IF NOT EXISTS idx_faces_org
            ON faces(organization_id);

            CREATE INDEX IF NOT EXISTS idx_attendance_org
            ON attendance(organization_id);

            CREATE INDEX IF NOT EXISTS idx_attendance_date
            ON attendance(organization_id, attendance_date);
            """
        )

        conn.commit()

    finally:
        conn.close()


init_database()


# ============================================================
# PASSWORD SECURITY
# ============================================================

PBKDF2_ITERATIONS = 310_000


def hash_password(password):
    """
    PBKDF2-HMAC-SHA256 with a random salt.

    Much safer than storing raw SHA-256(password).
    """

    salt = secrets.token_bytes(16)

    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS
    )

    return (
        salt.hex()
        + ":"
        + password_hash.hex()
    )


def verify_password(password, stored_hash):
    """
    Verifies a PBKDF2 password hash.
    """

    try:
        salt_hex, hash_hex = stored_hash.split(":")

        salt = bytes.fromhex(salt_hex)
        expected_hash = bytes.fromhex(hash_hex)

        actual_hash = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            PBKDF2_ITERATIONS
        )

        return secrets.compare_digest(
            actual_hash,
            expected_hash
        )

    except (ValueError, TypeError):
        return False


# ============================================================
# ORGANIZATION / AUTH HELPERS
# ============================================================

def get_organization(org_key):
    conn = get_db_connection()

    try:
        return conn.execute(
            """
            SELECT *
            FROM organizations
            WHERE org_key = ?
            """,
            (org_key,)
        ).fetchone()

    finally:
        conn.close()


def create_organization(org_key, password):
    """
    Creates a new organization.

    The session token is intentionally different from the
    password hash.
    """

    password_hash = hash_password(password)
    session_token = secrets.token_urlsafe(32)

    now = datetime.utcnow().isoformat()

    conn = get_db_connection()

    try:
        cursor = conn.execute(
            """
            INSERT INTO organizations (
                org_key,
                password_hash,
                session_token,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                org_key,
                password_hash,
                session_token,
                now,
                now
            )
        )

        conn.commit()

        return cursor.lastrowid, session_token

    finally:
        conn.close()


def authenticate_with_password(org_key, password):
    organization = get_organization(org_key)

    if organization is None:
        return None

    if not verify_password(
        password,
        organization["password_hash"]
    ):
        return None

    return organization


def authenticate_with_token(org_key, token):
    """
    Authenticates using a random session token.

    The password hash is never exposed as the URL token.
    """

    conn = get_db_connection()

    try:
        return conn.execute(
            """
            SELECT *
            FROM organizations
            WHERE org_key = ?
              AND session_token = ?
            """,
            (
                org_key,
                token
            )
        ).fetchone()

    finally:
        conn.close()


def reset_password(org_key, new_password):
    """
    Changes password and rotates the session token.
    """

    new_password_hash = hash_password(new_password)
    new_token = secrets.token_urlsafe(32)

    now = datetime.utcnow().isoformat()

    conn = get_db_connection()

    try:
        conn.execute(
            """
            UPDATE organizations
            SET password_hash = ?,
                session_token = ?,
                updated_at = ?
            WHERE org_key = ?
            """,
            (
                new_password_hash,
                new_token,
                now,
                org_key
            )
        )

        conn.commit()

        return new_token

    finally:
        conn.close()


# ============================================================
# FACE DATABASE HELPERS
# ============================================================

def encode_embedding(encoding):
    """
    Converts a numpy embedding into a compact binary representation.
    """

    array = np.asarray(
        encoding,
        dtype=np.float32
    )

    return array.tobytes()


def decode_embedding(blob, dimension):
    """
    Converts the database BLOB back into a numpy embedding.
    """

    return np.frombuffer(
        blob,
        dtype=np.float32,
        count=dimension
    ).copy()


def load_org_data(org_key):
    """
    Loads all registered faces for an organization.
    """

    organization = get_organization(org_key)

    if organization is None:
        return [], []

    conn = get_db_connection()

    try:
        rows = conn.execute(
            """
            SELECT
                name,
                encoding,
                embedding_dimension
            FROM faces
            WHERE organization_id = ?
            ORDER BY id ASC
            """,
            (organization["id"],)
        ).fetchall()

    finally:
        conn.close()

    names = []
    encodings = []

    for row in rows:
        try:
            encoding = decode_embedding(
                row["encoding"],
                row["embedding_dimension"]
            )

            names.append(row["name"])
            encodings.append(encoding)

        except Exception:
            # Skip corrupted embeddings rather than crashing
            continue

    return names, encodings


def save_face(
    org_key,
    name,
    encoding
):
    """
    Inserts a face into the database.
    """

    organization = get_organization(org_key)

    if organization is None:
        raise ValueError("Organization does not exist.")

    encoding_array = np.asarray(
        encoding,
        dtype=np.float32
    )

    blob = encode_embedding(encoding_array)

    now = datetime.utcnow().isoformat()

    conn = get_db_connection()

    try:
        conn.execute(
            """
            INSERT INTO faces (
                organization_id,
                name,
                encoding,
                embedding_dimension,
                created_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                organization["id"],
                name,
                sqlite3.Binary(blob),
                int(encoding_array.size),
                now
            )
        )

        conn.commit()

    finally:
        conn.close()


def delete_face(org_key, face_index):
    """
    Deletes a face based on the current ordered database list.
    """

    organization = get_organization(org_key)

    if organization is None:
        return

    conn = get_db_connection()

    try:
        row = conn.execute(
            """
            SELECT id
            FROM faces
            WHERE organization_id = ?
            ORDER BY id ASC
            LIMIT 1 OFFSET ?
            """,
            (
                organization["id"],
                face_index
            )
        ).fetchone()

        if row is not None:
            conn.execute(
                """
                DELETE FROM faces
                WHERE id = ?
                """,
                (row["id"],)
            )

            conn.commit()

    finally:
        conn.close()


# ============================================================
# ATTENDANCE
# ============================================================

def log_attendance(name, org_key):
    """
    Records attendance exactly once per person per day.

    The UNIQUE constraint in SQLite prevents race-condition
    duplicates.
    """

    if not name or name == "Unknown":
        return False

    organization = get_organization(org_key)

    if organization is None:
        return False

    local_tz = pytz.timezone("Asia/Kolkata")

    now = datetime.now(local_tz)

    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    conn = get_db_connection()

    try:
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO attendance (
                organization_id,
                name,
                attendance_date,
                attendance_time,
                created_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                organization["id"],
                name,
                date_str,
                time_str,
                now.isoformat()
            )
        )

        conn.commit()

        return cursor.rowcount > 0

    finally:
        conn.close()


def load_attendance(org_key):
    """
    Loads attendance records for the organization.
    """

    organization = get_organization(org_key)

    if organization is None:
        return pd.DataFrame(
            columns=[
                "Name",
                "Time",
                "Date"
            ]
        )

    conn = get_db_connection()

    try:
        rows = conn.execute(
            """
            SELECT
                name AS Name,
                attendance_time AS Time,
                attendance_date AS Date
            FROM attendance
            WHERE organization_id = ?
            ORDER BY attendance_date DESC,
                     attendance_time DESC
            """,
            (organization["id"],)
        ).fetchall()

    finally:
        conn.close()

    return pd.DataFrame(
        [dict(row) for row in rows],
        columns=[
            "Name",
            "Time",
            "Date"
        ]
    )


def clear_attendance(org_key):
    organization = get_organization(org_key)

    if organization is None:
        return

    conn = get_db_connection()

    try:
        conn.execute(
            """
            DELETE FROM attendance
            WHERE organization_id = ?
            """,
            (organization["id"],)
        )

        conn.commit()

    finally:
        conn.close()


# ============================================================
# FACE RECOGNITION
# ============================================================

def cosine_distance(a, b):
    """
    Calculates cosine distance.

    Lower value = more similar.
    """

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 1.0

    similarity = np.dot(a, b) / (
        norm_a * norm_b
    )

    return 1.0 - similarity


# ============================================================
# SESSION STATE
# ============================================================

if "auth_status" not in st.session_state:
    st.session_state.auth_status = False

if "org_key" not in st.session_state:
    st.session_state.org_key = None

if "known_names" not in st.session_state:
    st.session_state.known_names = []

if "known_encodings" not in st.session_state:
    st.session_state.known_encodings = []


# ============================================================
# AUTO LOGIN
# ============================================================

if (
    not st.session_state.auth_status
    and "org" in st.query_params
    and "token" in st.query_params
):

    q_org = st.query_params["org"]
    q_token = st.query_params["token"]

    organization = authenticate_with_token(
        q_org,
        q_token
    )

    if organization is not None:
        st.session_state.auth_status = True
        st.session_state.org_key = q_org

        (
            st.session_state.known_names,
            st.session_state.known_encodings
        ) = load_org_data(q_org)


# ============================================================
# WEBRTC THREAD PROCESSOR
# ============================================================

class AsyncFaceProcessor:

    def __init__(self):

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades
            + "haarcascade_frontalface_default.xml"
        )

        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1
        )

        self.ai_task = None

        self.frame_count = 0

        self.last_face = None

        self.current_name = "Finding Match..."

        self.box_color = (
            0,
            255,
            255
        )

        self.known_names = []

        self.known_encodings = []

        self.org_key = None

    def recognize_face(self, face_crop):

        try:

            # ------------------------------------------------
            # Resize
            # ------------------------------------------------

            face_crop = cv2.resize(
                face_crop,
                (160, 160),
                interpolation=cv2.INTER_AREA
            )

            # ------------------------------------------------
            # FaceNet expects RGB images
            # ------------------------------------------------

            face_crop = np.asarray(
                face_crop,
                dtype=np.float32
            )

            face_crop = np.expand_dims(
                face_crop,
                axis=0
            )

            # ------------------------------------------------
            # Generate embedding
            # ------------------------------------------------

            encoding = embedder.embeddings(
                face_crop
            )[0]

            best_name = "Unknown"

            # Existing threshold retained so recognition
            # behavior is not unexpectedly changed.
            best_dist = 0.40

            names = self.known_names
            encodings = self.known_encodings

            # ------------------------------------------------
            # Compare against registered users
            # ------------------------------------------------

            for known_name, known_enc in zip(
                names,
                encodings
            ):

                dist = cosine_distance(
                    encoding,
                    known_enc
                )

                if dist < best_dist:

                    best_dist = dist
                    best_name = known_name

            # ------------------------------------------------
            # Attendance
            # ------------------------------------------------

            if best_name != "Unknown":

                try:
                    log_attendance(
                        best_name,
                        self.org_key
                    )

                except Exception:
                    # Recognition should not crash if logging
                    # encounters a transient database problem.
                    pass

            return best_name

        except Exception:

            # Do not expose internal processing errors to the
            # WebRTC thread.
            return "Unknown"

    def recv(self, frame: av.VideoFrame):

        img = frame.to_ndarray(
            format="bgr24"
        )

        self.frame_count += 1

        # ----------------------------------------------------
        # Pick up completed AI task
        # ----------------------------------------------------

        if (
            self.ai_task is not None
            and self.ai_task.done()
        ):

            try:

                result = self.ai_task.result()

            except Exception:

                result = "Unknown"

            self.current_name = result

            if result != "Unknown":

                self.box_color = (
                    0,
                    255,
                    0
                )

            else:

                self.box_color = (
                    0,
                    0,
                    255
                )

            self.ai_task = None

        # ----------------------------------------------------
        # Face detection every third frame
        # ----------------------------------------------------

        if self.frame_count % 3 == 0:

            gray = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2GRAY
            )

            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60)
            )

            if len(faces) > 0:

                # Largest face
                faces = sorted(
                    faces,
                    key=lambda f: f[2] * f[3],
                    reverse=True
                )

                self.last_face = faces[0]

            else:

                self.last_face = None

                self.current_name = (
                    "Finding Match..."
                )

                self.box_color = (
                    0,
                    255,
                    255
                )

        # ----------------------------------------------------
        # Draw result
        # ----------------------------------------------------

        if self.last_face is not None:

            x, y, w, h = self.last_face

            cv2.rectangle(
                img,
                (x, y),
                (x + w, y + h),
                self.box_color,
                3
            )

            cv2.putText(
                img,
                self.current_name,
                (x, max(y - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                self.box_color,
                2
            )

            # ------------------------------------------------
            # Start asynchronous recognition
            # ------------------------------------------------

            if self.ai_task is None:

                rgb_img = cv2.cvtColor(
                    img,
                    cv2.COLOR_BGR2RGB
                )

                face_crop = rgb_img[
                    y:y + h,
                    x:x + w
                ]

                if face_crop.size > 0:

                    self.ai_task = (
                        self.executor.submit(
                            self.recognize_face,
                            face_crop.copy()
                        )
                    )

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )


# ============================================================
# UI
# ============================================================

if not st.session_state.auth_status:

    col1, col2, col3 = st.columns(
        [1, 2, 1]
    )

    with col2:

        st.title("👁️ AGNOS LOGIN")

        key_in = st.text_input(
            "Org Key",
            max_chars=5
        ).upper()

        organization = (
            get_organization(key_in)
            if len(key_in) == 5
            else None
        )

        is_known = organization is not None

        with st.form("auth"):

            btn = "Sign In / Sign Up"

            pw = st.text_input(
                "Password",
                type="password"
            )

            if st.form_submit_button(
                btn,
                type="primary"
            ):

                # --------------------------------------------
                # Basic input validation
                # --------------------------------------------

                if len(key_in) != 5:

                    st.error(
                        "Org Key must contain exactly 5 characters."
                    )

                elif not pw:

                    st.error(
                        "Password cannot be empty."
                    )

                else:

                    if is_known:

                        organization = (
                            authenticate_with_password(
                                key_in,
                                pw
                            )
                        )

                        if organization is None:

                            st.error(
                                "Wrong Password"
                            )

                        else:

                            session_token = (
                                organization["session_token"]
                            )

                            st.session_state.auth_status = True

                            st.session_state.org_key = (
                                key_in
                            )

                            (
                                st.session_state.known_names,
                                st.session_state.known_encodings
                            ) = load_org_data(key_in)

                            # --------------------------------
                            # Password hash is NOT put in URL.
                            # --------------------------------

                            st.query_params[
                                "org"
                            ] = key_in

                            st.query_params[
                                "token"
                            ] = session_token

                            st.rerun()

                    else:

                        try:

                            (
                                organization_id,
                                session_token
                            ) = create_organization(
                                key_in,
                                pw
                            )

                            st.session_state.auth_status = True

                            st.session_state.org_key = (
                                key_in
                            )

                            st.session_state.known_names = []

                            st.session_state.known_encodings = []

                            st.query_params[
                                "org"
                            ] = key_in

                            st.query_params[
                                "token"
                            ] = session_token

                            st.rerun()

                        except sqlite3.IntegrityError:

                            st.error(
                                "Organization already exists."
                            )

else:

    # ========================================================
    # SIDEBAR
    # ========================================================

    with st.sidebar:

        st.title("👁️ AGNOS")

        st.caption(
            f"ORG: {st.session_state.org_key}"
        )

        st.metric(
            "Registered Users",
            len(st.session_state.known_names)
        )

        st.markdown("---")

        with st.expander("Forgot Password"):

            with st.form(
                "reset_pwd_form"
            ):

                new_pwd = st.text_input(
                    "New Password",
                    type="password"
                )

                confirm_pwd = st.text_input(
                    "Confirm New Password",
                    type="password"
                )

                if st.form_submit_button(
                    "Reset Password",
                    use_container_width=True
                ):

                    if not new_pwd:

                        st.error(
                            "Password cannot be empty."
                        )

                    elif new_pwd != confirm_pwd:

                        st.error(
                            "Passwords do not match. Try again."
                        )

                    else:

                        try:

                            new_token = reset_password(
                                st.session_state.org_key,
                                new_pwd
                            )

                            # Rotate the authentication token.
                            st.query_params[
                                "token"
                            ] = new_token

                            st.success(
                                "Password successfully reset!"
                            )

                        except Exception:

                            st.error(
                                "Unable to reset password."
                            )

        st.markdown("---")

        if st.button("Log Out"):

            st.query_params.clear()

            for k in list(
                st.session_state.keys()
            ):

                del st.session_state[k]

            st.rerun()

    # ========================================================
    # TABS
    # ========================================================

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🎥 Live Scanner",
            "👤 Register Face",
            "📊 Access Logs",
            "🗄️ Manage Database"
        ]
    )

    # ========================================================
    # TAB 1 - LIVE SCANNER
    # ========================================================

    with tab1:

        st.markdown(
            "### Facial Recognition System"
        )

        st.caption(
            "Yellow = Scanning | Green = Verified | Red = Unknown"
        )

        ctx = webrtc_streamer(
            key="ai-scanner",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=AsyncFaceProcessor,
            media_stream_constraints={
                "video": True,
                "audio": False
            },
            async_processing=True
        )

        if ctx.video_processor:

            ctx.video_processor.known_names = (
                st.session_state.known_names
            )

            ctx.video_processor.known_encodings = (
                st.session_state.known_encodings
            )

            ctx.video_processor.org_key = (
                st.session_state.org_key
            )

    # ========================================================
    # TAB 2 - REGISTER FACE
    # ========================================================

    with tab2:

        st.markdown(
            "### Add New User"
        )

        new_name = st.text_input(
            "Name"
        ).upper()

        st.markdown("---")

        c_cam, c_or, c_up = st.columns(
            [4, 1, 4]
        )

        with c_cam:

            cam_photo = st.camera_input(
                "Take a photo"
            )

        with c_or:

            st.markdown(
                """
                <h3 style='text-align: center;
                margin-top: 50px;
                color: gray;'>
                OR
                </h3>
                """,
                unsafe_allow_html=True
            )

        with c_up:

            up_photo = st.file_uploader(
                "Upload a photo",
                type=[
                    "jpg",
                    "png",
                    "jpeg"
                ]
            )

        photo_file = (
            cam_photo
            if cam_photo is not None
            else up_photo
        )

        if photo_file and new_name:

            if st.button(
                "Register User",
                type="primary",
                use_container_width=True
            ):

                with st.spinner(
                    "Extracting Facial Features..."
                ):

                    try:

                        bytes_data = (
                            photo_file.getvalue()
                        )

                        cv2_img = cv2.imdecode(
                            np.frombuffer(
                                bytes_data,
                                np.uint8
                            ),
                            cv2.IMREAD_COLOR
                        )

                        if cv2_img is None:

                            st.error(
                                "Invalid image file."
                            )

                        else:

                            gray = cv2.cvtColor(
                                cv2_img,
                                cv2.COLOR_BGR2GRAY
                            )

                            face_cascade = (
                                cv2.CascadeClassifier(
                                    cv2.data.haarcascades
                                    + "haarcascade_frontalface_default.xml"
                                )
                            )

                            faces = (
                                face_cascade.detectMultiScale(
                                    gray,
                                    1.1,
                                    5,
                                    minSize=(60, 60)
                                )
                            )

                            if len(faces) == 0:

                                st.error(
                                    "No face found! Please try again with better lighting."
                                )

                            elif len(faces) > 1:

                                st.error(
                                    "Multiple faces found! Please ensure only one person is in the frame."
                                )

                            else:

                                x, y, w, h = faces[0]

                                rgb_img = cv2.cvtColor(
                                    cv2_img,
                                    cv2.COLOR_BGR2RGB
                                )

                                face_crop = rgb_img[
                                    y:y + h,
                                    x:x + w
                                ]

                                if face_crop.size == 0:

                                    st.error(
                                        "Unable to extract the face."
                                    )

                                else:

                                    face_crop = cv2.resize(
                                        face_crop,
                                        (160, 160),
                                        interpolation=cv2.INTER_AREA
                                    )

                                    face_crop = np.asarray(
                                        face_crop,
                                        dtype=np.float32
                                    )

                                    face_crop = (
                                        np.expand_dims(
                                            face_crop,
                                            axis=0
                                        )
                                    )

                                    encoding = (
                                        embedder.embeddings(
                                            face_crop
                                        )[0]
                                    )

                                    # --------------------------------
                                    # Save to SQLite
                                    # --------------------------------

                                    save_face(
                                        st.session_state.org_key,
                                        new_name,
                                        encoding
                                    )

                                    # --------------------------------
                                    # Refresh session data
                                    # --------------------------------

                                    (
                                        st.session_state.known_names,
                                        st.session_state.known_encodings
                                    ) = load_org_data(
                                        st.session_state.org_key
                                    )

                                    st.success(
                                        f"Successfully registered {new_name}!"
                                    )

                                    st.rerun()

                    except Exception as e:

                        st.error(
                            f"Unable to register user: {str(e)}"
                        )

    # ========================================================
    # TAB 3 - ACCESS LOGS
    # ========================================================

    with tab3:

        c1, c2 = st.columns(
            [4, 1]
        )

        with c1:

            if st.button(
                "Refresh Logs"
            ):

                st.rerun()

        with c2:

            if st.button(
                "🗑️ Clear All Logs",
                type="secondary"
            ):

                clear_attendance(
                    st.session_state.org_key
                )

                st.rerun()

        df = load_attendance(
            st.session_state.org_key
        )

        if not df.empty:

            display_df = df.copy()

            display_df.index = (
                display_df.index + 1
            )

            st.dataframe(
                display_df,
                use_container_width=True
            )

            st.download_button(
                "Download CSV",
                df.to_csv(
                    index=False
                ).encode("utf-8"),
                "logs.csv",
                "text/csv"
            )

        else:

            st.info(
                "No logs found."
            )

    # ========================================================
    # TAB 4 - MANAGE DATABASE
    # ========================================================

    with tab4:

        if st.session_state.known_names:

            for idx, name in enumerate(
                st.session_state.known_names
            ):

                c1, c2 = st.columns(
                    [4, 1]
                )

                with c1:

                    st.markdown(
                        f"**{idx + 1}. {name}**"
                    )

                with c2:

                    if st.button(
                        "Delete",
                        key=f"del_{idx}_{name}"
                    ):

                        delete_face(
                            st.session_state.org_key,
                            idx
                        )

                        (
                            st.session_state.known_names,
                            st.session_state.known_encodings
                        ) = load_org_data(
                            st.session_state.org_key
                        )

                        st.rerun()

        else:

            st.info(
                "Database empty."
            )
