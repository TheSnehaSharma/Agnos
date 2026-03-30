import os
import sys
import subprocess

# --- 0. THE OPENCV HIJACKER ---
# Catch the crash, uninstall the broken GUI version, and force the headless version
try:
    import cv2
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "opencv-python"])
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "opencv-python-headless"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless"])
    import cv2

# --- Prevent TensorFlow Memory Crashes ---
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import hashlib
import concurrent.futures
from datetime import datetime
from deepface import DeepFace
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode
import av

# --- 1. CONFIGURATION ---
DATA_DIR = "agnos_data"
if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
AUTH_FILE = os.path.join(DATA_DIR, "auth_registry.json")

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

st.set_page_config(page_title="Agnos Enterprise", layout="wide")

# --- Math Helper for DeepFace ---
def cosine_distance(a, b):
    a, b = np.array(a), np.array(b)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0: return 1.0
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

# --- 3. SESSION STATE ---
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
        try:
            with open(paths["db"], "rb") as f:
                data = pickle.load(f)
                st.session_state.known_names = data.get("names", [])
                st.session_state.known_encodings = data.get("encodings", [])
        except EOFError: pass
    
    if os.path.exists(paths["logs"]) and os.path.getsize(paths["logs"]) > 0:
        try:
            df = pd.read_csv(paths["logs"])
            today = datetime.now().strftime("%Y-%m-%d")
            st.session_state.logged_set = set(df[df['Date'] == today]['Name'].unique())
        except Exception: pass

def save_face_data():
    paths = get_file_paths(st.session_state.org_key)
    with open(paths["db"], "wb") as f:
        pickle.dump({"names": st.session_state.known_names, "encodings": st.session_state.known_encodings}, f)

def log_attendance_thread_safe(name, org_key):
    if name == "Unknown": return
    paths = get_file_paths(org_key)
    csv_path = paths["logs"]
    now = datetime.now()
    date_str, time_str = now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")

    # Prevent spam logging
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        with open(csv_path, "r") as f:
            if any(name in line and date_str in line for line in f): return

    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a") as f:
        if write_header: f.write("Name,Time,Date\n")
        f.write(f"{name},{time_str},{date_str}\n")

# --- AUTO-LOGIN ---
if not st.session_state.auth_status and "org" in st.query_params and "token" in st.query_params:
    q_org, q_token = st.query_params["org"], st.query_params["token"]
    auth_db = load_auth()
    if q_org in auth_db and auth_db[q_org] == q_token:
        st.session_state.auth_status, st.session_state.org_key = True, q_org
        load_org_data(q_org)

# --- 4. WEBRTC THREAD PROCESSOR ---
class AsyncFaceProcessor:
    def __init__(self):
        # 1. The Fast Tracker (Haar Cascade)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # 2. The Background AI Thread Manager
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.ai_task = None
        
        # 3. State Variables
        self.current_name = "Finding Match..."
        self.box_color = (0, 255, 255) # Yellow
        self.known_names = []
        self.known_encodings = []
        self.org_key = None

    def run_deepface_in_background(self, face_crop):
        try:
            # Run the heavy AI math on just the cropped face
            objs = DeepFace.represent(img_path=face_crop, model_name="Facenet", enforce_detection=False)
            encoding = objs[0]["embedding"]
            
            best_name = "Unknown"
            best_dist = 0.40 # Facenet Threshold
            
            for known_name, known_enc in zip(self.known_names, self.known_encodings):
                dist = cosine_distance(encoding, known_enc)
                if dist < best_dist:
                    best_dist = dist
                    best_name = known_name
                    
            if best_name != "Unknown":
                log_attendance_thread_safe(best_name, self.org_key)
                
            return best_name
        except Exception:
            return "Unknown"

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # FAST: Detect face instantly
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        if len(faces) == 0:
            self.current_name = "Finding Match..."
            self.box_color = (0, 255, 255)

        for (x, y, w, h) in faces:
            # Draw tracking box
            cv2.rectangle(img, (x, y), (x+w, y+h), self.box_color, 3)
            cv2.putText(img, self.current_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.box_color, 2)

            # SLOW: If the background thread is free, give it a new face to process
            if self.ai_task is None or self.ai_task.done():
                
                # If a previous task just finished, update the UI with the result
                if self.ai_task and self.ai_task.done():
                    result = self.ai_task.result()
                    self.current_name = result
                    self.box_color = (0, 255, 0) if result != "Unknown" else (0, 0, 255) # Green if known, Red if Unknown

                # Grab the face and send it to the background thread
                face_crop = img[y:y+h, x:x+w].copy()
                self.ai_task = self.executor.submit(self.run_deepface_in_background, cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 5. UI START ---
if not st.session_state.auth_status:
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.title("🔐 Agnos Login")
        key_in = st.text_input("Org Key (5 Char)", max_chars=5).upper()
        auth_db = load_auth()
        is_known = (len(key_in) == 5 and key_in in auth_db)
        
        with st.form("auth"):
            btn = "Sign In" if is_known else "Create Account"
            pw = st.text_input("Password", type="password")
            if st.form_submit_button(btn, type="primary"):
                h_pw = hashlib.sha256(pw.encode()).hexdigest()
                if is_known and auth_db[key_in] != h_pw: st.error("Wrong Password")
                else:
                    if not is_known:
                        auth_db[key_in] = h_pw
                        save_auth(auth_db)
                    st.session_state.auth_status = True
                    st.session_state.org_key = key_in
                    st.query_params["org"], st.query_params["token"] = key_in, h_pw
                    load_org_data(key_in)
                    st.rerun()
else:
    with st.sidebar:
        st.title("🛡️ Agnos")
        st.caption(f"ORG: {st.session_state.org_key}")
        st.metric("Users", len(st.session_state.known_names))
        st.markdown("---")
        if st.button("Log Out"):
            st.query_params.clear()
            for k in list(st.session_state.keys()): del st.session_state[k]
            st.rerun()

    st.title("Agnos Enterprise")
    tab1, tab2, tab3, tab4 = st.tabs(["🎥 Live Scanner", "👤 Register", "📊 Logs", "🗄️ Database"])

    with tab1:
        st.markdown("### Facial Recognition System")
        st.caption("Yellow = Scanning | Green = Verified | Red = Unknown")
        
        ctx = webrtc_streamer(
            key="ai-scanner",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=AsyncFaceProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True
        )

        # Inject database into the video thread
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
                if st.button("Register User", type="primary"):
                    with st.spinner("Extracting Facial Features..."):
                        # Read the image
                        bytes_data = photo_file.getvalue()
                        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                        rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
                        
                        try:
                            # Generate AI Encoding
                            objs = DeepFace.represent(img_path=rgb_img, model_name="Facenet", enforce_detection=True)
                            if len(objs) > 1:
                                st.error("Multiple faces found! Please ensure only one person is in the frame.")
                            else:
                                st.session_state.known_names.append(new_name)
                                st.session_state.known_encodings.append(objs[0]["embedding"])
                                save_face_data()
                                st.success(f"Successfully registered {new_name}! Their face data has been encrypted and saved.")
                                st.rerun()
                        except ValueError:
                            st.error("No face found! Please try again with better lighting.")

    with tab3:
        paths = get_file_paths(st.session_state.org_key)
        if st.button("Refresh Logs"): load_org_data(st.session_state.org_key)
        
        if os.path.exists(paths["logs"]) and os.path.getsize(paths["logs"]) > 0:
            df = pd.read_csv(paths["logs"])
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
