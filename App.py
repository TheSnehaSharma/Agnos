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

if not os.path.exists(DB_FOLDER):
    os.makedirs(DB_FOLDER)

st.set_page_config(page_title="Iron-Vision: Full Bridge", layout="wide")

# --- 2. AI ENGINE CACHE ---
@st.cache_resource
def load_ai():
    try:
        DeepFace.build_model(MODEL_NAME)
        return True
    except Exception as e:
        return str(e)

load_ai()

# --- 3. THE "HARDENED" JS BRIDGE ---
JS_CAMERA_PORTAL = """
async () => {
    // 1. Diagnostic: Check for API access
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        return "ERROR: Camera API blocked by browser security.";
    }

    try {
        // 2. Persistent Stream setup
        if (!window.activeStream) {
            window.activeStream = await navigator.mediaDevices.getUserMedia({ 
                video: { width: 320, height: 240, facingMode: "user" } 
            });
            window.videoElement = document.createElement('video');
            window.videoElement.srcObject = window.activeStream;
            window.videoElement.setAttribute("playsinline", true);
            await window.videoElement.play();
            window.canvasElement = document.createElement('canvas');
        }

        // 3. Frame Capture
        const ctx = window.canvasElement.getContext('2d');
        window.canvasElement.width = 320;
        window.canvasElement.height = 240;
        ctx.drawImage(window.videoElement, 0, 0, 320, 240);
        
        // 4. Return Base64 String
        return window.canvasElement.toDataURL('image/jpeg', 0.5);
    } catch (err) {
        return "ERROR: " + err.message;
    }
}
"""

# --- 4. NAVIGATION ---
page = st.sidebar.radio("Navigate", ["Register Face", "Live Attendance", "Manage Logs"])

if page == "Register Face":
    st.header("👤 Face Registration")
    
    with st.form("registration", clear_on_submit=True):
        name = st.text_input("FULL NAME").upper()
        file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
        if st.form_submit_button("Register User"):
            if name and file:
                with open(os.path.join(DB_FOLDER, f"{name}.jpg"), "wb") as f:
                    f.write(file.getbuffer())
                # Clean DeepFace indices
                for p in [f for f in os.listdir(DB_FOLDER) if f.endswith('.pkl')]:
                    os.remove(os.path.join(DB_FOLDER, p))
                st.success(f"Registered {name}")

    st.markdown("---")
    st.subheader("🗂️ Database Management")
    db_files = [f for f in os.listdir(DB_FOLDER) if f.endswith(('.jpg', '.png'))]
    for f in db_files:
        c1, c2 = st.columns([4, 1])
        c1.write(f"✅ {f.split('.')[0]}")
        if c2.button("Delete", key=f"del_{f}"):
            os.remove(os.path.join(DB_FOLDER, f))
            st.rerun()

elif page == "Live Attendance":
    st.header("📹 Biometric Scanner")
    col_v, col_s = st.columns([2, 1])
    
    with col_v:
        # PULLING THE PORTAL
        raw_frame = st_javascript(JS_CAMERA_PORTAL)
        
        if raw_frame:
            if str(raw_frame).startswith("ERROR"):
                st.error(raw_frame)
                st.info("💡 Tip: Ensure you are using HTTPS and have granted camera permissions to the site.")
            elif len(str(raw_frame)) > 100:
                st.image(raw_frame, use_container_width=True, caption="Active Neural Link")
            else:
                st.warning("Handshake Active: Warming up sensor...")
        else:
            st.info("Awaiting Camera Portal...")

    with col_s:
        st.subheader("System Status")
        if raw_frame and "base64," in str(raw_frame):
            try:
                # Decoding
                header, encoded = str(raw_frame).split(",", 1)
                nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Identification
                res = DeepFace.find(img_path=frame, db_path=DB_FOLDER, 
                                   model_name=MODEL_NAME, enforce_detection=False, 
                                   detector_backend="opencv", silent=True)
                
                if len(res) > 0 and not res[0].empty:
                    m_name = os.path.basename(res[0].iloc[0]['identity']).split('.')[0]
                    dist = res[0].iloc[0]['distance']
                    acc = max(0, int((1 - dist/0.4) * 100))
                    
                    if acc > 30:
                        st.metric("Detected", m_name, f"{acc}% Match")
                        # Add to Logs
                        logs = []
                        if os.path.exists(PKL_LOG):
                            with open(PKL_LOG, "rb") as f: logs = pickle.load(f)
                        
                        today = datetime.now().strftime("%Y-%m-%d")
                        if not any(e['Name'] == m_name and e['Date'] == today for e in logs):
                            logs.append({"Name": m_name, "Time": datetime.now().strftime("%H:%M:%S"), "Date": today})
                            with open(PKL_LOG, "wb") as f: pickle.dump(logs, f)
                            st.toast(f"✅ Logged {m_name}")
                    else:
                        st.warning("Access Denied: Unknown")
                else:
                    st.info("Scanning...")
            except Exception as e:
                st.error("Engine Resyncing...")
        
        if st.button("📸 Sync Feed"):
            st.rerun()

elif page == "Manage Logs":
    st.header("📊 Attendance Log")
    if os.path.exists(PKL_LOG):
        with open(PKL_LOG, "rb") as f: data = pickle.load(f)
        if data:
            st.table(pd.DataFrame(data))
            if st.button("🔥 Clear All Records"):
                os.remove(PKL_LOG)
                st.rerun()
        else: st.info("No records found.")
    else: st.info("Log file not found.")
