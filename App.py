import os
# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import streamlit as st
import pandas as pd
import cv2
import numpy as np
import pickle
import hashlib
from datetime import datetime
from deepface import DeepFace

# WebRTC Imports
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

# --- 1. CONFIGURATION ---
DATA_DIR = "agnos_data"
if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
AUTH_FILE = os.path.join(DATA_DIR, "auth_registry.json")

# Required for Cloud Deployment (STUN servers help WebRTC connect over the internet)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.set_page_config(page_title="Agnos Enterprise", layout="wide", initial_sidebar_state="expanded")

# --- Math Helper for DeepFace ---
def cosine_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# --- 2. BACKEND HELPERS ---
def load_auth():
    if os.path.exists(AUTH_FILE):
        import json
        with open(AUTH_FILE, "r") as f: return json.load(f)
    return {}

def save_auth(data):
    import json
    with open(AUTH_FILE, "w") as f: json.dump(data, f)

def get_file_paths(org_key):
    return {
        "db": os.path.join(DATA_DIR, f"faces_{org_key}.pkl"),
        "logs": os.path.join(DATA_DIR, f"logs_{org_key}.csv")
    }

# --- 3. SESSION STATE & LOGIN PERSISTENCE ---
if "auth_status" not in st.session_state: st.session_state.auth_status = False
if "org_key" not in st.session_state: st.session_state.org_key = None
if "known_names" not in st.session_state: st.session_state.known_names = []
if "known_encodings" not in st.session_state: st.session_state.known_encodings = []
if "logged_set" not in st.session_state: st.session_state.logged_set = set()

def load_org_data(org_key):
    paths = get_file_paths(org_key)
    st.session_state.known_names = []
    st.session_state.known_encodings = []
    st.session_state.logged_set = set()
    
    if os.path.exists(paths["db"]):
        with open(paths["db"], "rb") as f:
            data = pickle.load(f)
            st.session_state.known_names = data.get("names", [])
            st.session_state.known_encodings = data.get("encodings", [])
    
    if os.path.exists(paths["logs"]) and os.path.getsize(paths["logs"]) > 0:
        try:
            df = pd.read_csv(paths["logs"])
            today = datetime.now().strftime("%Y-%m-%d")
            todays_names = df[df['Date'] == today]['Name'].unique()
            st.session_state.logged_set = set(todays_names)
        except Exception as e:
            pass

# AUTO-LOGIN CHECK (Survives F5 Reloads)
if not st.session_state.auth_status:
    if "org" in st.query_params and "token" in st.query_params:
        q_org = st.query_params["org"]
        q_token = st.query_params["token"]
        auth_db = load_auth()
        if q_org in auth_db and auth_db[q_org] == q_token:
            st.session_state.auth_status = True
            st.session_state.org_key = q_org
            load_org_data(q_org)

def save_face_data():
    paths = get_file_paths(st.session_state.org_key)
    data = {"names": st.session_state.known_names, "encodings": st.session_state.known_encodings}
    with open(paths["db"], "wb") as f: pickle.dump(data, f)

# --- 4. WEBRTC THREAD PROCESSOR ---
# Note: This runs in a background thread. It cannot easily access st.session_state directly.
class FaceRecognitionProcessor(VideoProcessorBase):
    def __init__(self):
        self.known_names = []
        self.known_encodings = []
        self.org_key = None
        self.frame_count = 0
        self.face_locations = []
        self.face_names = []

    def log_attendance_thread_safe(self, name):
        if not self.org_key or name == "Unknown": return
        
        paths = get_file_paths(self.org_key)
        csv_path = paths["logs"]
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        
        # Quick check if already logged today (thread-safe-ish read)
        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
            try:
                with open(csv_path, "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        if name in line and date_str in line: return # Already logged
            except: pass

        # Write to log
        write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
        time_str = now.strftime("%H:%M:%S")
        with open(csv_path, "a") as f:
            if write_header: f.write("Name,Time,Date\n")
            f.write(f"{name},{time_str},{date_str}\n")

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        # Process every 5th frame to prevent massive lag
        if self.frame_count % 5 == 0:
            small_frame = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            try:
                objs = DeepFace.represent(img_path=rgb_small_frame, model_name="Facenet", enforce_detection=True)
                self.face_locations = []
                self.face_names = []

                for obj in objs:
                    encoding = obj["embedding"]
                    area = obj["facial_area"]
                    left, top, w, h = area['x']*2, area['y']*2, area['w']*2, area['h']*2
                    right, bottom = left + w, top + h
                    self.face_locations.append((top, right, bottom, left))

                    name = "Unknown"
                    best_dist = 0.40 # Facenet Threshold

                    for known_name, known_enc in zip(self.known_names, self.known_encodings):
                        dist = cosine_distance(encoding, known_enc)
                        if dist < best_dist:
                            best_dist = dist
                            name = known_name

                    self.face_names.append(name)
                    if name != "Unknown":
                        self.log_attendance_thread_safe(name)

            except ValueError:
                self.face_locations = []
                self.face_names = []

        # Draw boxes (always runs on every frame to keep video smooth)
        for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(img, (left, top), (right, bottom), color, 2)
            cv2.rectangle(img, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(img, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# --- 5. UI ---
if not st.session_state.auth_status:
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.title("🔐 Agnos Login")
        key_in = st.text_input("Org Key (5 Char)", max_chars=5).upper()
        auth_db = load_auth()
        is_known = (len(key_in) == 5 and key_in in auth_db)
        
        if len(key_in) == 5:
            if is_known: st.info(f"✅ Found: {key_in}")
            else: st.warning(f"🆕 Creating: {key_in}")

        with st.form("auth"):
            btn = "Sign In" if is_known else "Create Account"
            pw = st.text_input("Password", type="password")
            if st.form_submit_button(btn, type="primary"):
                h_pw = hashlib.sha256(pw.encode()).hexdigest()
                if is_known and auth_db[key_in] != h_pw:
                    st.error("Wrong Password")
                else:
                    if not is_known:
                        auth_db[key_in] = h_pw
                        save_auth(auth_db)
                    
                    st.session_state.auth_status = True
                    st.session_state.org_key = key_in
                    
                    # Set URL Params for persistence
                    st.query_params["org"] = key_in
                    st.query_params["token"] = h_pw
                    
                    load_org_data(key_in)
                    st.rerun()
else:
    with st.sidebar:
        st.title("🛡️ Agnos")
        st.caption(f"ORG: {st.session_state.org_key}")
        st.metric("Users", len(st.session_state.known_names))
        st.markdown("---")
        if st.button("Log Out"):
            st.query_params.clear() # Clear cookies
            for k in list(st.session_state.keys()): del st.session_state[k]
            st.rerun()

    st.title("Agnos Enterprise (WebRTC + DeepFace)")
    tab1, tab2, tab3, tab4 = st.tabs(["🎥 Live Scanner", "👤 Register", "📊 Logs", "🗄️ Database"])

    with tab1:
        st.markdown("### WebRTC Scanner")
        st.caption("Video processing runs directly in your browser. Start the stream below.")
        
        ctx = webrtc_streamer(
            key="face-recognition",
            mode=1, # SENDRECV
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=FaceRecognitionProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True
        )

        # Inject current state into the WebRTC thread if it's running
        if ctx.video_processor:
            ctx.video_processor.known_names = st.session_state.known_names
            ctx.video_processor.known_encodings = st.session_state.known_encodings
            ctx.video_processor.org_key = st.session_state.org_key

    with tab2:
        col1, col2 = st.columns([1, 1])
        with col1:
            new_name = st.text_input("Employee Name").upper()
            photo_file = st.camera_input("Take a photo") or st.file_uploader("Upload a photo", type=["jpg", "png", "jpeg"])
            
            if photo_file and new_name:
                if st.button("Register Face", type="primary"):
                    with st.spinner("Analyzing face..."):
                        bytes_data = photo_file.getvalue()
                        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                        rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
                        
                        try:
                            objs = DeepFace.represent(img_path=rgb_img, model_name="Facenet", enforce_detection=True)
                            if len(objs) > 1:
                                st.error("Multiple faces found! Please ensure only one person is in the frame.")
                            else:
                                st.session_state.known_names.append(new_name)
                                st.session_state.known_encodings.append(objs[0]["embedding"])
                                save_face_data()
                                st.success(f"Successfully registered {new_name}!")
                                st.rerun()
                        except ValueError:
                            st.error("No face found! Please try again with better lighting.")

    with tab3:
        paths = get_file_paths(st.session_state.org_key)
        if st.button("Refresh Logs"): load_org_data(st.session_state.org_key) # Force reload CSV
        
        if os.path.exists(paths["logs"]) and os.path.getsize(paths["logs"]) > 0:
            df = pd.read_csv(paths["logs"])
            # Reverse order to show newest first
            st.dataframe(df.iloc[::-1], use_container_width=True)
            st.download_button("Download CSV", df.to_csv(index=False).encode('utf-8'), "logs.csv", "text/csv")
        else:
            st.info("No logs found.")

    with tab4:
        if st.session_state.known_names:
            for idx, name in enumerate(st.session_state.known_names):
                c1, c2 = st.columns([3, 1])
                c1.text(f"👤 {name}")
                if c2.button("Delete", key=f"del_{idx}_{name}"):
                    st.session_state.known_names.pop(idx)
                    st.session_state.known_encodings.pop(idx)
                    save_face_data()
                    st.rerun()
        else:
            st.info("Database empty.")
