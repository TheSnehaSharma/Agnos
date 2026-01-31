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

# --- 1. SETUP & CONFIG ---
DB_FOLDER = "registered_faces"
PKL_LOG = "attendance_data.pkl"
MODEL_NAME = "Facenet512"

for folder in [DB_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

st.set_page_config(page_title="Iron-Vision Biometric", layout="wide")

# --- 2. AI MODEL CACHE ---
@st.cache_resource
def load_ai_models():
    try:
        DeepFace.build_model(MODEL_NAME)
        return True
    except Exception as e:
        return str(e)

load_ai_models()

# --- 3. JS BRIDGE DEFINITION ---
# This is the "Pull" script that returns a frame to Python
JS_CAMERA_BRIDGE = """
async () => {
    if (!window.videoStream) {
        window.videoStream = await navigator.mediaDevices.getUserMedia({ video: { width: 320, height: 240 } });
        window.vEl = document.createElement('video');
        window.vEl.srcObject = window.videoStream;
        await window.vEl.play();
        window.cEl = document.createElement('canvas');
        window.cEl.width = 320;
        window.cEl.height = 240;
    }
    const ctx = window.cEl.getContext('2d');
    ctx.drawImage(window.vEl, 0, 0, 320, 240);
    return window.cEl.toDataURL('image/jpeg', 0.5);
}
"""

# --- 4. NAVIGATION ---
page = st.sidebar.radio("Navigate", ["Register Face", "Live Attendance", "Log History"])

# --- PAGE 1: REGISTER ---
if page == "Register Face":
    st.header("👤 Face Registration")
    
    with st.form("reg_form", clear_on_submit=True):
        name = st.text_input("FULL NAME").upper()
        file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
        if st.form_submit_button("Save to Database"):
            if name and file:
                path = os.path.join(DB_FOLDER, f"{name}.jpg")
                with open(path, "wb") as f:
                    f.write(file.getbuffer())
                # Clean AI cache so it picks up the new face
                for p in [f for f in os.listdir(DB_FOLDER) if f.endswith('.pkl')]:
                    os.remove(os.path.join(DB_FOLDER, p))
                st.success(f"Registered {name}")

    st.markdown("---")
    st.subheader("🗂️ Database Manager")
    db_files = [f for f in os.listdir(DB_FOLDER) if f.endswith(('.jpg', '.png', '.jpeg'))]
    if db_files:
        for f in db_files:
            c1, c2 = st.columns([4, 1])
            c1.write(f"✅ {f.split('.')[0]}")
            if c2.button("Delete", key=f"del_{f}"):
                os.remove(os.path.join(DB_FOLDER, f))
                st.rerun()
    else:
        st.info("Database is empty.")

# --- PAGE 2: LIVE FEED ---
elif page == "Live Attendance":
    st.header("📹 Live Identification")
    
    col_v, col_s = st.columns([2, 1])
    
    with col_v:
        # THE BRIDGE: Python pulls the frame from JS
        st.write("Initializing secure camera link...")
        raw_frame = st_javascript(JS_CAMERA_BRIDGE)
        
        if raw_frame and len(str(raw_frame)) > 100:
            st.image(raw_frame, use_container_width=True, caption="Live Feed Active")
        else:
            st.info("Awaiting Camera Permissions...")

    with col_s:
        st.subheader("System Status")
        if raw_frame and len(str(raw_frame)) > 100:
            try:
                # Decode
                header, encoded = raw_frame.split(",", 1)
                nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Identify
                res = DeepFace.find(img_path=frame, db_path=DB_FOLDER, 
                                   model_name=MODEL_NAME, enforce_detection=False, 
                                   detector_backend="opencv", silent=True)
                
                if len(res) > 0 and not res[0].empty:
                    m_name = os.path.basename(res[0].iloc[0]['identity']).split('.')[0]
                    dist = res[0].iloc[0]['distance']
                    acc = max(0, int((1 - dist/0.38) * 100))
                    
                    if acc > 30:
                        st.metric("Identity", m_name, f"{acc}% Confidence")
                        # Persistence Log
                        logs = []
                        if os.path.exists(PKL_LOG):
                            with open(PKL_LOG, "rb") as f: logs = pickle.load(f)
                        
                        today = datetime.now().strftime("%Y-%m-%d")
                        if not any(e['Name'] == m_name and e['Date'] == today for e in logs):
                            logs.append({"Name": m_name, "Time": datetime.now().strftime("%H:%M:%S"), "Date": today})
                            with open(PKL_LOG, "wb") as f: pickle.dump(logs, f)
                            st.toast(f"✅ Logged {m_name}")
                    else:
                        st.warning("Identity Unknown")
                else:
                    st.info("Scanning...")
            except Exception as e:
                st.error(f"Syncing AI Engine...")
        
        if st.button("📸 Sync New Frame"):
            st.rerun()

# --- PAGE 3: HISTORY ---
elif page == "Log History":
    st.header("📊 Attendance Records")
    if os.path.exists(PKL_LOG):
        with open(PKL_LOG, "rb") as f:
            data = pickle.load(f)
        if data:
            st.table(pd.DataFrame(data))
            if st.button("🔥 Wipe All Logs"):
                os.remove(PKL_LOG)
                st.rerun()
        else:
            st.info("No records found.")
    else:
        st.info("Log file not yet created.")
