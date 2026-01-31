import streamlit as st
import pandas as pd
import cv2
import numpy as np
import base64
import os
import pickle
from deepface import DeepFace
from datetime import datetime

# --- CONFIG ---
DB_FOLDER = "registered_faces"
PKL_LOG = "attendance_data.pkl"

if not os.path.exists(DB_FOLDER):
    os.makedirs(DB_FOLDER)

st.set_page_config(page_title="DeepFace Auth System", layout="wide")

# Persistent state to prevent "Aligning Camera" loops
if "logged_set" not in st.session_state:
    st.session_state.logged_set = set()
if "last_frame" not in st.session_state:
    st.session_state.last_frame = None

# --- HELPER: PICKLE ENGINE ---
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

# --- FRONTEND ASSETS ---
JS_CODE = """
<div id="video-container" style="background:#000; border-radius:12px; width:100%; height:300px; position:relative;">
    <video id="v" autoplay playsinline style="width:100%; border-radius:12px;"></video>
</div>
<script>
    const video = document.getElementById('v');
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    async function startCamera() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                video: { width: 400, height: 300, facingMode: "user" } 
            });
            video.srcObject = stream;
        } catch (err) { console.error(err); }
    }

    function sendFrame() {
        if (video.readyState === video.HAVE_ENOUGH_DATA) {
            canvas.width = 400; 
            canvas.height = 300;
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            const dataURL = canvas.toDataURL('image/jpeg', 0.5);
            window.parent.postMessage({
                type: "streamlit:setComponentValue",
                value: dataURL
            }, "*");
        }
    }

    startCamera();
    setInterval(sendFrame, 2000); // 2 second intervals to allow Python to breathe
</script>
"""

# --- UI NAVIGATION ---
page = st.sidebar.radio("Navigate", ["Register", "Live Feed", "Log History"])

if page == "Register":
    st.header("👤 Face Registration")
    name = st.text_input("Full Name").upper()
    img_file = st.file_uploader("Upload Profile Image", type=['jpg', 'jpeg', 'png'])
    
    if st.button("Register User") and name and img_file:
        img_path = os.path.join(DB_FOLDER, f"{name}.jpg")
        with open(img_path, "wb") as f:
            f.write(img_file.getbuffer())
        # Clear DeepFace cache
        for cache in [f for f in os.listdir(DB_FOLDER) if f.endswith('.pkl')]:
            os.remove(os.path.join(DB_FOLDER, cache))
        st.success(f"Registered {name}!")

    st.markdown("---")
    all_users = [f for f in os.listdir(DB_FOLDER) if f.endswith(('.jpg', '.png'))]
    for f in all_users:
        c1, c2 = st.columns([4, 1])
        c1.write(f"✅ {f.split('.')[0]}")
        if c2.button("Delete", key=f"del_{f}"):
            os.remove(os.path.join(DB_FOLDER, f))
            st.rerun()

elif page == "Live Feed":
    st.header("📹 DeepFace Scanner")
    col_v, col_s = st.columns([2, 1])
    
    with col_v:
        # Capture frame via JS
        incoming_frame = st.components.v1.html(JS_CODE, height=350)
        # Update session state if a new frame arrives
        if incoming_frame:
            st.session_state.last_frame = incoming_frame

    with col_s:
        st.subheader("System Status")
        if st.session_state.last_frame:
            try:
                # Decode Frame
                data = st.session_state.last_frame.split(',')[1]
                nparr = np.frombuffer(base64.b64decode(data), np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Perform Search
                # We use 'Facenet512' for higher accuracy than the standard Facenet
                dfs = DeepFace.find(img_path=frame, db_path=DB_FOLDER, 
                                   model_name="Facenet512", enforce_detection=False, 
                                   silent=True, detector_backend="opencv")
                
                if len(dfs) > 0 and not dfs[0].empty:
                    id_path = dfs[0].iloc[0]['identity']
                    id_name = os.path.basename(id_path).split('.')[0]
                    dist = dfs[0].iloc[0]['distance']
                    
                    # Facenet512 threshold is usually around 0.3
                    accuracy = max(0, int((1 - dist/0.3) * 100))
                    st.metric("Identity", id_name, f"{accuracy}% Match")
                    
                    if id_name not in st.session_state.logged_set:
                        save_attendance_pkl(id_name)
                        st.session_state.logged_set.add(id_name)
                        st.toast(f"✅ Logged: {id_name}")
                else:
                    st.warning("Identity: Unknown")
            except Exception as e:
                st.info("System initializing models...")
        else:
            st.info("Waiting for first camera handshake...")

elif page == "Log History":
    st.header("📊 Attendance Log")
    if os.path.exists(PKL_LOG):
        with open(PKL_LOG, "rb") as f: data = pickle.load(f)
        df = pd.DataFrame(data)
        st.table(df)
        if st.button("🔥 WIPE ALL ATTENDANCE DATA"):
            if os.path.exists(PKL_LOG): os.remove(PKL_LOG)
            st.session_state.logged_set = set()
            st.rerun()
    else: st.info("No logs found.")
