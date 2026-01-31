import streamlit as st
import pandas as pd
import cv2
import os
import pickle
import av
from deepface import DeepFace
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# --- CONFIG & DIRS ---
DB_FOLDER = "registered_faces"
PKL_LOG = "attendance_data.pkl"
MODEL_NAME = "Facenet512"

if not os.path.exists(DB_FOLDER):
    os.makedirs(DB_FOLDER)

st.set_page_config(page_title="Live Bio-Auth Pro", layout="wide")

# --- MODEL CACHE ---
@st.cache_resource
def load_ai_models():
    DeepFace.build_model(MODEL_NAME)
    return True

load_ai_models()

# --- THE REAL-TIME ENGINE ---
class FaceRecognitionTransformer(VideoTransformerBase):
    def __init__(self):
        self.logged_today = set()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        try:
            # Match current frame against DB_FOLDER
            results = DeepFace.find(
                img_path=img, 
                db_path=DB_FOLDER, 
                model_name=MODEL_NAME, 
                enforce_detection=False, 
                detector_backend="opencv", 
                silent=True
            )
            
            if len(results) > 0 and not results[0].empty:
                candidate = results[0].iloc[0]
                dist = candidate['distance']
                
                # If match is strong enough
                if dist < 0.35:
                    name = os.path.basename(candidate['identity']).split('.')[0]
                    
                    # Drawing the overlay in the live stream
                    cv2.putText(img, f"Verified: {name}", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # LOGGING LOGIC (This runs inside the video thread)
                    self.log_attendance(name)
        except:
            pass

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def log_attendance(self, name):
        # We use a session-independent check to prevent double-logging
        if name not in self.logged_today:
            logs = []
            if os.path.exists(PKL_LOG):
                with open(PKL_LOG, "rb") as f:
                    try: logs = pickle.load(f)
                    except: logs = []
            
            logs.append({"Name": name, "Time": datetime.now().strftime("%H:%M:%S")})
            with open(PKL_LOG, "wb") as f:
                pickle.dump(logs, f)
            self.logged_today.add(name)

# --- UI NAVIGATION ---
page = st.sidebar.radio("Navigate", ["Register", "Live Feed", "Log History"])

if page == "Register":
    st.header("👤 Face Registration")
    name = st.text_input("Name").upper()
    file = st.file_uploader("Upload Profile Image", type=['jpg', 'png', 'jpeg'])
    if st.button("Save User") and name and file:
        with open(os.path.join(DB_FOLDER, f"{name}.jpg"), "wb") as f:
            f.write(file.getbuffer())
        for p in [f for f in os.listdir(DB_FOLDER) if f.endswith('.pkl')]:
            os.remove(os.path.join(DB_FOLDER, p))
        st.success(f"Registered {name}")

elif page == "Live Feed":
    st.header("📹 Live Biometric Feed")
    
    # Check if database exists
    if not any(f.endswith(('.jpg', '.png')) for f in os.listdir(DB_FOLDER)):
        st.warning("⚠️ No faces registered yet.")
        st.stop()

    # WebRTC configuration for Cloud deployment
    RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

    webrtc_streamer(
        key="face-recognition",
        video_transformer_factory=FaceRecognitionTransformer,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
    )

elif page == "Log History":
    st.header("📊 Attendance Log")
    if os.path.exists(PKL_LOG):
        with open(PKL_LOG, "rb") as f:
            data = pickle.load(f)
        st.table(pd.DataFrame(data))
        if st.button("🔥 WIPE SESSION"):
            os.remove(PKL_LOG)
            st.rerun()
    else:
        st.info("No logs found. Start the live feed to record data.")
