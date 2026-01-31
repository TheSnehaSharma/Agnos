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
MODEL_NAME = "Facenet512"

if not os.path.exists(DB_FOLDER):
    os.makedirs(DB_FOLDER)

st.set_page_config(page_title="DeepFace Auth Pro", layout="wide")

# --- MODEL CACHE ---
@st.cache_resource
def load_ai_models():
    try:
        DeepFace.build_model(MODEL_NAME)
        return True
    except Exception as e:
        return str(e)

load_ai_models()

# --- STORAGE ---
if "logged_set" not in st.session_state:
    st.session_state.logged_set = set()

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

# --- JAVASCRIPT CAMERA BRIDGE (OPTIMIZED) ---
JS_CODE = """
<div style="background:#000; border-radius:12px; overflow:hidden; position:relative; width:100%; height:300px;">
    <video id="webcam" autoplay playsinline style="width:100%; height:100%; object-fit:contain;"></video>
    <canvas id="canvas" style="display:none;"></canvas>
    <button onclick="capture()" style="position:absolute; bottom:10px; right:10px; background:#00FF00; color:#000; border:none; padding:5px 10px; border-radius:5px; font-weight:bold; cursor:pointer; z-index:1000;">SYNC NOW</button>
</div>
<script>
    const video = document.getElementById('webcam');
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');

    async function init() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                video: { width: 320, height: 240 } // Lower res for stability
            });
            video.srcObject = stream;
        } catch (e) { console.error("Camera Error:", e); }
    }

    function capture() {
        if (video.readyState === video.HAVE_ENOUGH_DATA) {
            canvas.width = 320; 
            canvas.height = 240;
            ctx.drawImage(video, 0, 0, 320, 240);
            // Using webp and 0.4 quality to keep the Base64 string very short
            const data = canvas.toDataURL('image/webp', 0.4);
            window.parent.postMessage({
                type: "streamlit:setComponentValue",
                value: data
            }, "*");
        }
    }

    init();
    setInterval(capture, 3000); 
</script>
"""

# --- UI NAVIGATION ---
page = st.sidebar.radio("Navigate", ["Register", "Live Feed", "Log History"])

if page == "Register":
    st.header("👤 Identity Registration")
    name = st.text_input("Full Name").upper()
    file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
    if st.button("Save User") and name and file:
        with open(os.path.join(DB_FOLDER, f"{name}.jpg"), "wb") as f:
            f.write(file.getbuffer())
        for p in [f for f in os.listdir(DB_FOLDER) if f.endswith('.pkl')]:
            os.remove(os.path.join(DB_FOLDER, p))
        st.success(f"Registered {name}")

    st.markdown("---")
    st.subheader("🗂️ Manage Database")
    all_users = [f for f in os.listdir(DB_FOLDER) if f.endswith(('.jpg', '.png'))]
    for f in all_users:
        col_n, col_b = st.columns([4, 1])
        col_n.write(f"✅ {f.split('.')[0]}")
        if col_b.button("Delete", key=f"del_{f}"):
            os.remove(os.path.join(DB_FOLDER, f))
            st.rerun()

elif page == "Live Feed":
    st.header("📹 Biometric Scanner")
    col_v, col_s = st.columns([2, 1])
    
    with col_v:
        img_data = st.components.v1.html(JS_CODE, height=350)

    with col_s:
        st.subheader("System Status")
        if img_data:
            try:
                # Decode image
                b64 = str(img_data).split(',')[1]
                nparr = np.frombuffer(base64.b64decode(b64), np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # DeepFace Search
                results = DeepFace.find(img_path=frame, db_path=DB_FOLDER, 
                                       model_name=MODEL_NAME, enforce_detection=False, 
                                       silent=True, detector_backend="opencv")
                
                if len(results) > 0 and not results[0].empty:
                    match_name = os.path.basename(results[0].iloc[0]['identity']).split('.')[0]
                    dist = results[0].iloc[0]['distance']
                    acc = max(0, int((1 - dist/0.38) * 100))
                    
                    st.metric("Detected Identity", match_name, f"{acc}% Match")
                    
                    if match_name not in st.session_state.logged_set:
                        save_attendance_pkl(match_name)
                        st.session_state.logged_set.add(match_name)
                        st.toast(f"✅ Attendance Logged: {match_name}")
                else:
                    st.warning("Identity: Unknown")
            except Exception as e:
                st.error("Engine processing frame...")
        else:
            st.info("⌛ Awaiting Frame... If stuck, click 'SYNC NOW' on the video feed.")

elif page == "Log History":
    st.header("📊 Attendance Log")
    if os.path.exists(PKL_LOG):
        with open(PKL_LOG, "rb") as f: logs = pickle.load(f)
        df = pd.DataFrame(logs)
        st.table(df)
        if st.button("🔥 WIPE ALL SESSION DATA"):
            if os.path.exists(PKL_LOG): os.remove(PKL_LOG)
            st.session_state.logged_set = set()
            st.rerun()
    else: st.info("No logs found.")
