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

# --- CONFIG & DIRS ---
DB_FOLDER = "registered_faces"
PKL_LOG = "attendance_data.pkl"
MODEL_NAME = "Facenet512"

if not os.path.exists(DB_FOLDER):
    os.makedirs(DB_FOLDER)

st.set_page_config(page_title="Iron-Vision: Continuous Bridge", layout="wide")

# --- MODEL CACHE ---
@st.cache_resource
def load_ai():
    try:
        DeepFace.build_model(MODEL_NAME)
        return True
    except Exception as e:
        return str(e)

load_ai()

# --- JAVASCRIPT: THE DUAL-BUFFER BRIDGE ---
# This script creates a persistent video and returns the data to Python.
JS_BRIDGE_CODE = """
async () => {
    // 1. Initialize persistent camera stream
    if (!window.vStream) {
        window.vStream = await navigator.mediaDevices.getUserMedia({ video: { width: 320, height: 240 } });
        window.vEl = document.createElement('video');
        window.vEl.srcObject = window.vStream;
        await window.vEl.play();
        window.cEl = document.createElement('canvas');
        window.cEl.width = 320;
        window.cEl.height = 240;
    }
    
    const ctx = window.cEl.getContext('2d');
    ctx.drawImage(window.vEl, 0, 0, 320, 240);
    
    // Return frame to Python
    return window.cEl.toDataURL('image/jpeg', 0.5);
}
"""

# --- UI NAVIGATION ---
page = st.sidebar.radio("Navigate", ["Register", "Live Feed", "Log History"])

if page == "Register":
    st.header("👤 Face Registration")
    with st.form("registration_form"):
        name = st.text_input("FULL NAME").upper()
        file = st.file_uploader("Upload Profile Image", type=['jpg', 'png'])
        if st.form_submit_button("Register User"):
            if name and file:
                path = os.path.join(DB_FOLDER, f"{name}.jpg")
                with open(path, "wb") as f:
                    f.write(file.getbuffer())
                for p in [f for f in os.listdir(DB_FOLDER) if f.endswith('.pkl')]:
                    os.remove(os.path.join(DB_FOLDER, p))
                st.success(f"Registered {name}")

    st.markdown("---")
    st.subheader("🗂️ Database Manager")
    db_files = [f for f in os.listdir(DB_FOLDER) if f.endswith(('.jpg', '.png'))]
    for f in db_files:
        c1, c2 = st.columns([4, 1])
        c1.write(f"✅ {f.split('.')[0]}")
        if c2.button("Delete", key=f"del_{f}"):
            os.remove(os.path.join(DB_FOLDER, f))
            st.rerun()

elif page == "Live Feed":
    st.header("📹 Biometric Scanner")
    col_v, col_s = st.columns([2, 1])
    
    with col_v:
        # THE BRIDGE: Python calls JS and waits for the return string
        img_data = st_javascript(JS_BRIDGE_CODE)
        
        if img_data and len(str(img_data)) > 100:
            # Displaying the frame in Python to verify it reached the server
            st.image(img_data, use_container_width=True, caption="Verified Bridge Link")
        else:
            st.warning("Awaiting Camera Portal... (Check browser permissions)")

    with col_s:
        st.subheader("System Status")
        if img_data and "base64," in img_data:
            try:
                # Decode
                encoded_data = img_data.split(",")[1]
                nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
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
                        st.metric("Detected", m_name, f"{acc}% Match")
                        # Log logic
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
        
        # This keeps the "Real-time" loop moving
        if st.button("🔄 Sync New Frame"):
            st.rerun()

elif page == "Log History":
    st.header("📊 Attendance Log")
    if os.path.exists(PKL_LOG):
        with open(PKL_LOG, "rb") as f: data = pickle.load(f)
        if data:
            st.table(pd.DataFrame(data))
            if st.button("Clear Logs"):
                os.remove(PKL_LOG)
                st.rerun()
        else: st.info("No records.")
    else: st.info("No logs found.")
