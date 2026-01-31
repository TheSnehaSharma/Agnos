import streamlit as st
import pandas as pd
import cv2
import numpy as np
import base64
import os
import pickle
from deepface import DeepFace
from datetime import datetime

# --- CONFIG & DIRS ---
DB_FOLDER = "registered_faces"
PKL_LOG = "attendance_data.pkl"
MODEL_NAME = "Facenet512" # High-accuracy model

if not os.path.exists(DB_FOLDER):
    os.makedirs(DB_FOLDER)

st.set_page_config(page_title="DeepFace Auth Pro", layout="wide")

# --- FORCE MODEL DOWNLOAD ON BOOT ---
@st.cache_resource
def load_ai_models():
    # This dummy call forces Streamlit Cloud to download the model files on startup
    # You will see the download progress in the 'Manage App' terminal now.
    try:
        DeepFace.build_model(MODEL_NAME)
        return True
    except Exception as e:
        return str(e)

with st.spinner("Downloading AI Models (FaceNet512)... Check 'Manage App' for progress."):
    model_status = load_ai_models()
    if model_status is not True:
        st.error(f"Model Download Failed: {model_status}")

# --- STORAGE ---
if "logged_set" not in st.session_state:
    st.session_state.logged_set = set()

# --- PICKLE ENGINE ---
def save_attendance_pkl(name):
    logs = []
    if os.path.exists(PKL_LOG):
        with open(PKL_LOG, "rb") as f:
            try: logs = pickle.load(f)
            except: logs = []
    entry = {"Name": name, "Time": datetime.now().strftime("%H:%M:%S"), "Date": datetime.now().strftime("%Y-%m-%d")}
    logs.append(entry)
    with open(PKL_LOG, "wb") as f:
        pickle.dump(logs, f)

# --- JAVASCRIPT CAMERA BRIDGE ---
JS_CODE = """
<div style="background:#000; padding:10px; border-radius:15px; border:2px solid #333;">
    <video id="webcam" autoplay playsinline style="width:100%; border-radius:10px;"></video>
</div>
<script>
    const video = document.getElementById('webcam');
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    async function init() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 400, height: 300 } });
            video.srcObject = stream;
        } catch (e) { alert("Camera access denied: " + e); }
    }

    function capture() {
        if (video.readyState === video.HAVE_ENOUGH_DATA) {
            canvas.width = 400; canvas.height = 300;
            ctx.drawImage(video, 0, 0, 400, 300);
            const data = canvas.toDataURL('image/jpeg', 0.6);
            window.parent.postMessage({type: "streamlit:setComponentValue", value: data}, "*");
        }
    }

    init();
    setInterval(capture, 3000); // Send image to Python every 3 seconds
</script>
"""

# --- UI NAVIGATION ---
page = st.sidebar.radio("Navigate", ["Register", "Live Feed", "Log History"])

if page == "Register":
    st.header("👤 Identity Registration")
    name = st.text_input("Name").upper()
    file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
    if st.button("Save User") and name and file:
        with open(os.path.join(DB_FOLDER, f"{name}.jpg"), "wb") as f:
            f.write(file.getbuffer())
        # Clear DeepFace cache pkl if it exists
        for p in [f for f in os.listdir(DB_FOLDER) if f.endswith('.pkl')]:
            os.remove(os.path.join(DB_FOLDER, p))
        st.success(f"Registered {name}")

elif page == "Live Feed":
    st.header("📹 Biometric Scanner")
    col_v, col_s = st.columns([2, 1])
    
    with col_v:
        # This will be None until the first interval (3 seconds)
        img_data = st.components.v1.html(JS_CODE, height=350)

    with col_s:
        if isinstance(img_data, str) and len(img_data) > 1000:
            try:
                # Decode
                b64 = img_data.split(',')[1]
                nparr = np.frombuffer(base64.b64decode(b64), np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Match
                results = DeepFace.find(img_path=frame, db_path=DB_FOLDER, 
                                       model_name=MODEL_NAME, enforce_detection=False, 
                                       silent=True, detector_backend="opencv")
                
                if len(results) > 0 and not results[0].empty:
                    match_name = os.path.basename(results[0].iloc[0]['identity']).split('.')[0]
                    dist = results[0].iloc[0]['distance']
                    # Facenet512 threshold is roughly 0.3
                    acc = max(0, int((1 - dist/0.35) * 100))
                    
                    st.metric("Identity", match_name, f"{acc}% Match")
                    if match_name not in st.session_state.logged_set:
                        save_attendance_pkl(match_name)
                        st.session_state.logged_set.add(match_name)
                        st.toast(f"✅ Logged: {match_name}")
                else:
                    st.warning("Unknown User")
            except Exception as e:
                st.error("Engine busy...")
        else:
            st.info("Awaiting Camera Feed (3s loop)...")

elif page == "Log History":
    st.header("📊 Attendance Log")
    if os.path.exists(PKL_LOG):
        with open(PKL_LOG, "rb") as f: logs = pickle.load(f)
        st.table(pd.DataFrame(logs))
        if st.button("🔥 WIPE DATA"):
            os.remove(PKL_LOG)
            st.session_state.logged_set = set()
            st.rerun()
    else: st.info("No logs found.")
