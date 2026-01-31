import streamlit as st
import pandas as pd
import cv2
import numpy as np
import base64
import os
import pickle
from deepface import DeepFace
from datetime import datetime
from streamlit_javascript import st_javascript

# --- CONFIG ---
DB_FOLDER = "registered_faces"
PKL_LOG = "attendance_data.pkl"
MODEL_NAME = "Facenet512"

if not os.path.exists(DB_FOLDER):
    os.makedirs(DB_FOLDER)

st.set_page_config(page_title="Biometric Bridge Pro", layout="wide")

# --- MODEL LOADING ---
@st.cache_resource
def load_ai_models():
    DeepFace.build_model(MODEL_NAME)
    return True

load_ai_models()

# --- THE JS HANDSHAKE CODE ---
# This script captures the frame and stores it in a global JS variable 
# that Python can "scrape" using st_javascript.
JS_CAMERA_CODE = """
async () => {
    const video = document.createElement('video');
    const canvas = document.createElement('canvas');
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 240, height: 180 } });
    video.srcObject = stream;
    await video.play();
    
    canvas.width = 240;
    canvas.height = 180;
    canvas.getContext('2d').drawImage(video, 0, 0, 240, 180);
    
    const dataURL = canvas.toDataURL('image/webp', 0.3);
    // Stop tracks to save battery/resources
    stream.getTracks().forEach(t => t.stop());
    return dataURL;
}
"""

# --- UI NAVIGATION ---
page = st.sidebar.radio("Navigate", ["Register", "Live Feed", "Log History"])

if page == "Register":
    st.header("👤 Face Registration")
    name = st.text_input("Full Name").upper()
    img_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
    
    if st.button("Save User") and name and img_file:
        with open(os.path.join(DB_FOLDER, f"{name}.jpg"), "wb") as f:
            f.write(img_file.getbuffer())
        # Clear DeepFace cache
        for p in [f for f in os.listdir(DB_FOLDER) if f.endswith('.pkl')]:
            os.remove(os.path.join(DB_FOLDER, p))
        st.success(f"Registered {name}!")

elif page == "Live Feed":
    st.header("📹 Biometric Terminal")
    col_v, col_s = st.columns([2, 1])
    
    with col_v:
        st.write("Click 'Scan' to capture a frame and verify.")
        if st.button("📸 Scan Face"):
            # This is the bi-directional bridge! 
            # It executes JS and returns the value to the Python variable 'return_data'
            return_data = st_javascript(JS_CAMERA_CODE)
            
            if return_data and len(str(return_data)) > 100:
                st.session_state.current_frame = return_data
            else:
                st.warning("Could not access camera. Ensure permissions are granted.")

    with col_s:
        if "current_frame" in st.session_state:
            try:
                # 1. Clean and Decode
                raw_b64 = st.session_state.current_frame.split(",")[1]
                missing_padding = len(raw_b64) % 4
                if missing_padding: raw_b64 += "=" * (4 - missing_padding)
                
                nparr = np.frombuffer(base64.b64decode(raw_b64), np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # 2. Match
                res = DeepFace.find(img_path=frame, db_path=DB_FOLDER, 
                                   model_name=MODEL_NAME, enforce_detection=False, 
                                   detector_backend="opencv", silent=True)
                
                if len(res) > 0 and not res[0].empty:
                    m_name = os.path.basename(res[0].iloc[0]['identity']).split('.')[0]
                    dist = res[0].iloc[0]['distance']
                    acc = max(0, int((1 - dist/0.35) * 100))
                    
                    st.metric("Detected", m_name, f"{acc}% Match")
                    
                    # 3. Log to Pickle
                    if "logged_today" not in st.session_state: st.session_state.logged_today = set()
                    if m_name not in st.session_state.logged_set:
                        # Append to pkl
                        logs = []
                        if os.path.exists(PKL_LOG):
                            with open(PKL_LOG, "rb") as f: logs = pickle.load(f)
                        logs.append({"Name": m_name, "Time": datetime.now().strftime("%H:%M:%S")})
                        with open(PKL_LOG, "wb") as f: pickle.dump(logs, f)
                        
                        st.session_state.logged_set.add(m_name)
                        st.toast(f"✅ Logged: {m_name}")
                else:
                    st.warning("Identity: Unknown")
            except Exception as e:
                st.error(f"Sync Error: {e}")

elif page == "Log History":
    st.header("📊 Attendance Log")
    if os.path.exists(PKL_LOG):
        with open(PKL_LOG, "rb") as f: logs = pickle.load(f)
        st.table(pd.DataFrame(logs))
        if st.button("🔥 WIPE SESSION"):
            os.remove(PKL_LOG)
            st.session_state.logged_set = set()
            st.rerun()
