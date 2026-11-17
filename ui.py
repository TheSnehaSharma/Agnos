import logging
from typing import List, Optional

import streamlit as st
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer

from config import ORG_KEY_LENGTH, STUN_SERVER_URLS
from database import (
    FaceRecord,
    Organization,
    OrganizationExistsError,
    add_face,
    authenticate_with_token,
    clear_attendance,
    create_organization,
    delete_face,
    get_organization,
    init_schema,
    list_faces,
    load_attendance,
    reset_password,
)
from recognition import (
    FaceDetector,
    FaceEmbedder,
    FaceExtractionError,
    FaceGallery,
    extract_single_face,
)
from security import verify_password
from video import FaceRecognitionProcessor

logger = logging.getLogger(__name__)

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": STUN_SERVER_URLS}]})


# ======================================================== cached resources


@st.cache_resource
def bootstrap() -> bool:
    """One-time application setup (database schema)."""
    init_schema()
    return True


@st.cache_resource
def get_embedder() -> FaceEmbedder:
    return FaceEmbedder.load()


# ============================================================ session state

_AUTH = "auth_status"
_ORG_KEY = "org_key"
_ORG_ID = "org_id"
_FACES = "faces"
_GALLERY = "gallery"
_FLASH = "flash_message"


def init_session() -> None:
    defaults = {
        _AUTH: False,
        _ORG_KEY: None,
        _ORG_ID: None,
        _FACES: [],
        _GALLERY: FaceGallery(),
        _FLASH: None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def is_authenticated() -> bool:
    return bool(st.session_state[_AUTH])


def get_org_key() -> Optional[str]:
    return st.session_state[_ORG_KEY]


def get_org_id() -> Optional[int]:
    return st.session_state[_ORG_ID]


def get_faces() -> List[FaceRecord]:
    return st.session_state[_FACES]


def get_gallery() -> FaceGallery:
    return st.session_state[_GALLERY]


def refresh_faces() -> None:
    records = list_faces(st.session_state[_ORG_ID])
    st.session_state[_FACES] = records
    st.session_state[_GALLERY] = FaceGallery.build(
        [r.name for r in records],
        [r.encoding for r in records],
    )


def sign_in(organization: Organization) -> None:
    st.session_state[_AUTH] = True
    st.session_state[_ORG_KEY] = organization.org_key
    st.session_state[_ORG_ID] = organization.id
    refresh_faces()


def sign_out() -> None:
    st.query_params.clear()
    for key in list(st.session_state.keys()):
        del st.session_state[key]


def flash(message: str) -> None:
    """Store a message to display once after the next rerun."""
    st.session_state[_FLASH] = message


def pop_flash() -> Optional[str]:
    message = st.session_state[_FLASH]
    st.session_state[_FLASH] = None
    return message


# ================================================================ auth view


def try_token_login() -> None:
    """Restore a session from the ``org`` and ``token`` query parameters."""
    if is_authenticated():
        return
    if "org" not in st.query_params or "token" not in st.query_params:
        return

    organization = authenticate_with_token(
        st.query_params["org"],
        st.query_params["token"],
    )
    if organization is not None:
        sign_in(organization)


def _handle_login_submit(org_key: str, password: str) -> None:
    if len(org_key) != ORG_KEY_LENGTH:
        st.error(f"Org Key must contain exactly {ORG_KEY_LENGTH} characters.")
        return
    if not password:
        st.error("Password cannot be empty.")
        return

    organization = get_organization(org_key)

    if organization is not None:
        if not verify_password(password, organization.password_hash):
            st.error("Wrong Password")
            return
    else:
        try:
            organization = create_organization(org_key, password)
        except OrganizationExistsError:
            st.error("Organization already exists.")
            return

    sign_in(organization)
    st.query_params["org"] = organization.org_key
    st.query_params["token"] = organization.session_token
    st.rerun()


def render_login() -> None:
    _, center, _ = st.columns([1, 2, 1])

    with center:
        st.title("👁️ AGNOS LOGIN")

        org_key = st.text_input("Org Key", max_chars=ORG_KEY_LENGTH).upper()

        with st.form("auth"):
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign In / Sign Up", type="primary")

            if submitted:
                _handle_login_submit(org_key, password)


# ================================================================== sidebar


def _render_reset_form() -> None:
    with st.form("reset_pwd_form"):
        new_password = st.text_input("New Password", type="password")
        confirm_password = st.text_input("Confirm New Password", type="password")

        if not st.form_submit_button("Reset Password", use_container_width=True):
            return

        if not new_password:
            st.error("Password cannot be empty.")
        elif new_password != confirm_password:
            st.error("Passwords do not match. Try again.")
        else:
            try:
                new_token = reset_password(get_org_key(), new_password)
            except Exception:
                logger.exception("Password reset failed")
                st.error("Unable to reset password.")
            else:
                st.query_params["token"] = new_token
                st.success("Password successfully reset!")


def render_sidebar() -> None:
    with st.sidebar:
        st.title("👁️ AGNOS")
        st.caption(f"ORG: {get_org_key()}")
        st.metric("Registered Users", len(get_faces()))

        st.markdown("---")

        with st.expander("Forgot Password"):
            _render_reset_form()

        st.markdown("---")

        if st.button("Log Out"):
            sign_out()
            st.rerun()


# ============================================================ scanner tab


def render_scanner_tab() -> None:
    st.markdown("### Facial Recognition System")
    st.caption("Yellow = Scanning | Green = Verified | Red = Unknown")

    # Resolved on the Streamlit thread; the factory runs on a WebRTC thread.
    embedder = get_embedder()

    ctx = webrtc_streamer(
        key="ai-scanner",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=lambda: FaceRecognitionProcessor(
            FaceDetector(), embedder
        ),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if ctx.video_processor:
        ctx.video_processor.update_context(get_gallery(), get_org_id())


# ============================================================ register tab


def _register_user(name: str, photo_file) -> None:
    with st.spinner("Extracting Facial Features..."):
        try:
            face_rgb = extract_single_face(photo_file.getvalue(), FaceDetector())
            encoding = get_embedder().embed(face_rgb)
            add_face(get_org_id(), name, encoding)
        except FaceExtractionError as exc:
            st.error(str(exc))
            return
        except Exception as exc:
            logger.exception("User registration failed")
            st.error(f"Unable to register user: {exc}")
            return

    refresh_faces()
    flash(f"Successfully registered {name}!")
    st.rerun()


def render_register_tab() -> None:
    st.markdown("### Add New User")

    message = pop_flash()
    if message:
        st.success(message)

    new_name = st.text_input("Name").strip().upper()

    st.markdown("---")

    col_cam, col_or, col_up = st.columns([4, 1, 4])

    with col_cam:
        cam_photo = st.camera_input("Take a photo")

    with col_or:
        st.markdown(
            """
            <h3 style='text-align: center; margin-top: 50px; color: gray;'>
            OR
            </h3>
            """,
            unsafe_allow_html=True,
        )

    with col_up:
        up_photo = st.file_uploader("Upload a photo", type=["jpg", "png", "jpeg"])

    photo_file = cam_photo if cam_photo is not None else up_photo

    if photo_file and new_name:
        if st.button("Register User", type="primary", use_container_width=True):
            _register_user(new_name, photo_file)


# ================================================================ logs tab


def render_logs_tab() -> None:
    col_refresh, col_clear = st.columns([4, 1])

    with col_refresh:
        if st.button("Refresh Logs"):
            st.rerun()

    with col_clear:
        if st.button("🗑️ Clear All Logs", type="secondary"):
            clear_attendance(get_org_id())
            st.rerun()

    df = load_attendance(get_org_id())

    if df.empty:
        st.info("No logs found.")
        return

    display_df = df.copy()
    display_df.index = display_df.index + 1

    st.dataframe(display_df, use_container_width=True)

    st.download_button(
        "Download CSV",
        df.to_csv(index=False).encode("utf-8"),
        "logs.csv",
        "text/csv",
    )


# ============================================================ database tab


def render_database_tab() -> None:
    faces = get_faces()

    if not faces:
        st.info("Database empty.")
        return

    for position, face in enumerate(faces, start=1):
        col_name, col_action = st.columns([4, 1])

        with col_name:
            st.markdown(f"**{position}. {face.name}**")

        with col_action:
            if st.button("Delete", key=f"del_{face.id}"):
                delete_face(get_org_id(), face.id)
                refresh_faces()
                st.rerun()


# ============================================================ entry point


def _render_main() -> None:
    render_sidebar()

    scanner, register, logs, database = st.tabs(
        [
            "🎥 Live Scanner",
            "👤 Register Face",
            "📊 Access Logs",
            "🗄️ Manage Database",
        ]
    )
    with scanner:
        render_scanner_tab()
    with register:
        render_register_tab()
    with logs:
        render_logs_tab()
    with database:
        render_database_tab()


def run() -> None:
    bootstrap()
    init_session()
    try_token_login()

    if is_authenticated():
        _render_main()
    else:
        render_login()
