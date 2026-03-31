import streamlit as st
import pandas as pd
import cv2
import numpy as np
import pickle
import hashlib
import os
import concurrent.futures
from datetime import datetime
import pytz
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode
import av
import json

# --- Prevent TensorFlow Memory Crashes ---
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras_facenet import FaceNet

# --- CONFIGURATION ---
DATA_DIR = "agnos_data"
if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
AUTH_FILE = os.path.join(DATA_DIR, "auth_registry.json")

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

st.set_page_config(page_title="Agnos", page_icon="👁️", layout="wide")

@st.cache_resource
def get_embedder():
    return FaceNet()

embedder = get_embedder()

def cosine_distance(a, b):
    a, b = np.array(a), np.array(b)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0: 
        return 1.0
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# --- BACKEND HELPERS ---
def load_auth():
    if os.path.exists(AUTH_FILE):
        with open(AUTH_FILE, "r") as f: 
            return json.load(f)
    return {}

def save_auth(data):
    with open(AUTH_FILE, "w") as f: 
        json.dump(data, f)

def get_file_paths(org_key):
    return {
        "db": os.path.join(DATA_DIR, f"faces_{org_key}.pkl"),
        "logs": os.path.join(DATA_DIR, f"logs_{org_key}.csv")
    }

# --- SESSION STATE ---
if "auth_status" not in st.session_state: st.session_state.auth_status = False
if "org_key" not in st.session_state: st.session_state.org_key = None
if "known_names" not in st.session_state: st.session_state.known_names = []
if "known_encodings" not in st.session_state: st.session_state.known_encodings = []

def load_org_data(org_key):
    paths = get_file_paths(org_key)
    st.session_state.known_names = []
    st.session_state.known_encodings = []
    
    if os.path.exists(paths["db"]):
        try:
            with open(paths["db"], "rb") as f:
                data = pickle.load(f)
                st.session_state.known_names = data.get("names", [])
                st.session_state.known_encodings = data.get("encodings", [])
        except EOFError: pass

def save_face_data():
    paths = get_file_paths(st.session_state.org_key)
    with open(paths["db"], "wb") as f:
        pickle.dump({"names": st.session_state.known_names, "encodings": st.session_state.known_encodings}, f)

def log_attendance(name, org_key):
    if name == "Unknown": return
    paths = get_file_paths(org_key)
    csv_path = paths["logs"]
    
    local_tz = pytz.timezone('Asia/Kolkata') 
    now = datetime.now(local_tz)
    date_str, time_str = now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")

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

# --- WEBRTC THREAD PROCESSOR ---
class AsyncFaceProcessor:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.ai_task = None
        
        self.frame_count = 0
        self.last_face = None 
        self.current_name = "Finding Match..."
        self.box_color = (0, 255, 255) 
        
        self.known_names = []
        self.known_encodings = []
        self.org_key = None

    def recognize_face(self, face_crop):
        try:
            face_crop = cv2.resize(face_crop, (160, 160))
            face_crop = np.expand_dims(face_crop, axis=0) 
            
            encoding = embedder.embeddings(face_crop)[0]
            
            best_name = "Unknown"
            best_dist = 0.40 
            
            for known_name, known_enc in zip(self.known_names, self.known_encodings):
                dist = cosine_distance(encoding, known_enc)
                if dist < best_dist:
                    best_dist = dist
                    best_name = known_name
                    
            if best_name != "Unknown":
                log_attendance(best_name, self.org_key)
                
            return best_name
            
        except Exception:
            return "Unknown"

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        if self.ai_task and self.ai_task.done():
            result = self.ai_task.result()
            self.current_name = result
            self.box_color = (0, 255, 0) if result != "Unknown" else (0, 0, 255)
            self.ai_task = None 

        if self.frame_count % 3 == 0:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

            if len(faces) > 0:
                faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
                self.last_face = faces[0]
            else:
                self.last_face = None
                self.current_name = "Finding Match..."
                self.box_color = (0, 255, 255)

        if self.last_face is not None:
            x, y, w, h = self.last_face
            cv2.rectangle(img, (x, y), (x+w, y+h), self.box_color, 3)
            cv2.putText(img, self.current_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.box_color, 2)

            if self.ai_task is None:
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                face_crop = rgb_img[y:y+h, x:x+w]
                if face_crop.size > 0:
                    self.ai_task = self.executor.submit(self.recognize_face, face_crop)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- UI START ---
if not st.session_state.auth_status:
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.title("👁️ AGNOS LOGIN")
        key_in = st.text_input("Org Key", max_chars=5).upper()
        auth_db = load_auth()
        is_known = (len(key_in) == 5 and key_in in auth_db)
        
        with st.form("auth"):
            btn = "Sign In / Sign Up"
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
        st.title("👁️ AGNOS")
        st.caption(f"ORG: {st.session_state.org_key}")
        st.metric("Registered Users", len(st.session_state.known_names))

        st.markdown("---")
        with st.expander("Forgot Password"):
            with st.form("reset_pwd_form"):
                new_pwd = st.text_input("New Password", type="password")
                confirm_pwd = st.text_input("Confirm New Password", type="password")
                
                if st.form_submit_button("Reset Password", use_container_width=True):
                    if new_pwd != confirm_pwd:
                        st.error("Passwords do not match. Try again.")
                    else:
                        auth_db = load_auth()
                        new_hash = hashlib.sha256(new_pwd.encode()).hexdigest()
                        auth_db[st.session_state.org_key] = new_hash
                        save_auth(auth_db)
                        st.query_params["token"] = new_hash
                        st.success("Password successfully reset!")

        st.markdown("---")
        if st.button("Log Out"):
            st.query_params.clear()
            for k in list(st.session_state.keys()): del st.session_state[k]
            st.rerun()

    tab1, tab2, tab3, tab4 = st.tabs(["🎥 Live Scanner", "👤 Register Face", "📊 Access Logs", "🗄️ Manage Database"])

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

        if ctx.video_processor:
            ctx.video_processor.known_names = st.session_state.known_names
            ctx.video_processor.known_encodings = st.session_state.known_encodings
            ctx.video_processor.org_key = st.session_state.org_key

    with tab2:
        st.markdown("### Add New User")
        new_name = st.text_input("Name").upper()
        st.markdown("---")
        
        c_cam, c_or, c_up = st.columns([4, 1, 4])
        with c_cam:
            cam_photo = st.camera_input("Take a photo")
        with c_or:
            st.markdown("<h3 style='text-align: center; margin-top: 50px; color: gray;'>OR</h3>", unsafe_allow_html=True)
        with c_up:
            up_photo = st.file_uploader("Upload a photo", type=["jpg", "png", "jpeg"])
        
        photo_file = cam_photo or up_photo
        
        if photo_file and new_name:
            if st.button("Register User", type="primary", use_container_width=True):
                with st.spinner("Extracting Facial Features..."):
                    bytes_data = photo_file.getvalue()
                    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
                    
                    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
                    
                    if len(faces) == 0:
                        st.error("No face found! Please try again with better lighting.")
                    elif len(faces) > 1:
                        st.error("Multiple faces found! Please ensure only one person is in the frame.")
                    else:
                        x, y, w, h = faces[0]
                        rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
                        face_crop = rgb_img[y:y+h, x:x+w]
                        face_crop = cv2.resize(face_crop, (160, 160))
                        face_crop = np.expand_dims(face_crop, axis=0)
                        
                        encoding = embedder.embeddings(face_crop)[0]
                        st.session_state.known_names.append(new_name)
                        st.session_state.known_encodings.append(encoding)
                        save_face_data()
                        st.success(f"Successfully registered {new_name}!")
                        st.rerun()

    with tab3:
        paths = get_file_paths(st.session_state.org_key)
        
        c1, c2 = st.columns([4, 1])
        with c1:
            if st.button("Refresh Logs"): load_org_data(st.session_state.org_key)
        with c2:
            if st.button("🗑️ Clear All Logs", type="secondary"):
                if os.path.exists(paths["logs"]):
                    os.remove(paths["logs"])
                st.rerun()
        
        if os.path.exists(paths["logs"]) and os.path.getsize(paths["logs"]) > 0:
            df = pd.read_csv(paths["logs"])
            df.index = df.index + 1
            st.dataframe(df.iloc[::-1], use_container_width=True)
            st.download_button("Download CSV", df.to_csv(index=False).encode('utf-8'), "logs.csv", "text/csv")
        else:
            st.info("No logs found.")

    with tab4:
        if st.session_state.known_names:
            for idx, name in enumerate(st.session_state.known_names):
                c1, c2 = st.columns([4, 1])
                
                with c1:
                    st.markdown(f"**{idx + 1}. {name}**")
                    
                with c2:
                    if st.button("Delete", key=f"del_{idx}_{name}"):
                        st.session_state.known_names.pop(idx)
                        st.session_state.known_encodings.pop(idx)
                        save_face_data()
                        st.rerun()
        else:
            st.info("Database empty.")
