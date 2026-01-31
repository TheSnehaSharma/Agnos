import streamlit as st
import pandas as pd
import cv2
import numpy as np
import base64
import os
import pickle
from deepface import DeepFace
from datetime import datetime
from streamlit_js_eval import streamlit_js_eval

# --- CONFIG ---
DB_FOLDER = "registered_faces"
PKL_LOG = "attendance_data.pkl"
MODEL_NAME = "Facenet512"

if not os.path.exists(DB_FOLDER):
    os.makedirs(DB_FOLDER)

st.set_page_config(page_title="Iron-Vision Portal", layout="wide")

# --- MODEL CACHE ---
@st.cache_resource
def load_ai():
    try:
        DeepFace.build_model(MODEL_NAME)
        return True
    except Exception as e:
        return str(e)

# --- THE JS BRIDGE SCRIPTS ---
# Script 1: Initializes the camera and MediaPipe once (persistent)
INIT_JS = """
(async () => {
    if (!window.vStream) {
        window.vStream = await navigator.mediaDevices.getUserMedia({ video: { width: 320, height: 240 } });
        window.vEl = document.createElement('video');
        window.vEl.srcObject = window.vStream;
        await window.vEl.play();
        window.cEl = document.createElement('canvas');
        window.cEl.width = 320;
        window.cEl.height = 240;
    }
    return "Camera Initialized";
})()
"""

# Script 2: Grabs a single frame from the live video
CAPTURE_JS = """
(() => {
    if (!window.vEl) return null;
    const ctx = window.cEl.getContext('2d');
    ctx.drawImage(window.vEl, 0, 0, 320, 240);
    return window.cEl.toDataURL('image/jpeg', 0.4);
})()
"""

# --- NAVIGATION ---
page = st.sidebar.radio("Navigate", ["Register", "Live Feed", "Logs"])

if page == "Register":
    st.header("👤 Face Registration")
    with st.form("reg"):
        name = st.text_input("Name").upper()
        file = st.file_uploader("Photo", type=['jpg', 'png'])
        if st.form_submit_button("Save User"):
            if name and file:
                with open(os.path.join(DB_FOLDER, f"{name}.jpg"), "wb") as f:
                    f.write(file.getbuffer())
                # Clear AI metadata
                for p in [f for f in os.listdir(DB_FOLDER) if f.endswith('.pkl')]:
                    os.remove(os.path.join(DB_FOLDER, p))
                st.success(f"Registered {name}")

    st.subheader("🗂️ Database Manager")
    db_list = [f for f in os.listdir(DB_FOLDER) if f.endswith(('.jpg', '.png'))]
    for f in db_list:
        c1, c2 = st.columns([4, 1])
        c1.write(f"✅ {f.split('.')[0]}")
        if c2.button("Delete", key=f"del_{f}"):
            os.remove(os.path.join(DB_FOLDER, f))
            st.rerun()

elif page == "Live Feed":
    st.header("📹 Live Biometric Scanner")
    
    # Check Model Health
    if load_ai() is not True:
        st.error("AI Engine failed to load.")
        st.stop()

    # 1. Initialize Camera Portal
    streamlit_js_eval(js_expressions=INIT_JS, key="init_cam")
    
    col_v, col_s = st.columns([2, 1])
    
    with col_v:
        # 2. Grab the frame from the browser (THE BRIDGE)
        img_data = streamlit_js_eval(js_expressions=CAPTURE_JS, key="capture_frame")
        
        if img_data:
            st.image(img_data, use_container_width=True, caption="Active Portal")
        else:
            st.info("Searching for Camera Portal...")

    with col_s:
        st.subheader("System Status")
        if img_data and "base64," in img_data:
            try:
                # 3. Decode
                encoded = img_data.split(",")[1]
                nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # 4. DeepFace Recognition
                res = DeepFace.find(img_path=frame, db_path=DB_FOLDER, 
                                   model_name=MODEL_NAME, enforce_detection=False, 
                                   detector_backend="opencv", silent=True)
                
                if len(res) > 0 and not res[0].empty:
                    m_name = os.path.basename(res[0].iloc[0]['identity']).split('.')[0]
                    dist = res[0].iloc[0]['distance']
                    acc = max(0, int((1 - dist/0.4) * 100))
                    
                    if acc > 30:
                        st.metric("Detected", m_name, f"{acc}% Match")
                        # Logging logic...
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
                    st.info("Scanning database...")
            except Exception as e:
                st.error("Engine Resyncing...")
        
        # Manual Trigger for Rerun
        if st.button("🔄 Refresh Analysis"):
            st.rerun()

elif page == "Logs":
    st.header("📊 Attendance Log")
    if os.path.exists(PKL_LOG):
        with open(PKL_LOG, "rb") as f: data = pickle.load(f)
        st.table(pd.DataFrame(data))
    else:
        st.info("No records.")
