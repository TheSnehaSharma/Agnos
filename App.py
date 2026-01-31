import streamlit as st
import pandas as pd
import cv2
import numpy as np
import base64
import os
import pickle
import urllib.parse
from deepface import DeepFace
from datetime import datetime

# --- CONFIG ---
DB_FOLDER = "registered_faces"
PKL_LOG = "attendance_data.pkl"
MODEL_NAME = "Facenet512"

if not os.path.exists(DB_FOLDER):
    os.makedirs(DB_FOLDER)

st.set_page_config(page_title="Iron-Clad Face Auth", layout="wide")

# --- MODEL CACHE ---
@st.cache_resource
def load_ai_models():
    try:
        DeepFace.build_model(MODEL_NAME)
        return True
    except Exception as e:
        return str(e)

load_ai_models()

if "logged_set" not in st.session_state:
    st.session_state.logged_set = set()

# --- HELPERS ---
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

# --- JS BRIDGE ---
JS_CODE = """
<div style="background:#000; border-radius:12px; overflow:hidden; width:100%; height:250px;">
    <video id="v" autoplay playsinline style="width:100%; height:100%; object-fit:contain;"></video>
    <canvas id="c" style="display:none;"></canvas>
</div>
<script>
    const v = document.getElementById('v');
    const c = document.getElementById('c');
    const ctx = c.getContext('2d');

    async function start() {
        try {
            const s = await navigator.mediaDevices.getUserMedia({ video: { width: 160, height: 120 } });
            v.srcObject = s;
        } catch (e) { console.error(e); }
    }

    function sync() {
        if (v.readyState === v.HAVE_ENOUGH_DATA) {
            c.width = 160; c.height = 120;
            ctx.drawImage(v, 0, 0, 160, 120);
            const dataURL = c.toDataURL('image/webp', 0.1); 
            const wrapped = encodeURIComponent(dataURL);
            window.parent.postMessage({type: "streamlit:setComponentValue", value: wrapped}, "*");
        }
    }

    start();
    setInterval(sync, 3000); 
</script>
"""

# --- UI ---
page = st.sidebar.radio("Navigate", ["Register", "Live Feed", "Log History"])

if page == "Register":
    st.header("👤 Face Registration")
    name = st.text_input("Name").upper()
    file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
    if st.button("Save User") and name and file:
        with open(os.path.join(DB_FOLDER, f"{name}.jpg"), "wb") as f:
            f.write(file.getbuffer())
        # Clear index cache
        for p in [f for f in os.listdir(DB_FOLDER) if f.endswith('.pkl')]:
            os.remove(os.path.join(DB_FOLDER, p))
        st.success(f"Registered {name}")

elif page == "Live Feed":
    st.header("📹 Biometric Terminal")
    
    # Check if database is empty first
    registered_count = len([f for f in os.listdir(DB_FOLDER) if f.endswith(('.jpg', '.png'))])
    if registered_count == 0:
        st.warning("⚠️ No faces registered. Go to the 'Register' page first.")
        st.stop()

    col_v, col_s = st.columns([2, 1])
    with col_v:
        img_data = st.components.v1.html(JS_CODE, height=280)

    with col_s:
        if img_data:
            try:
                # 1. Decode
                unquoted = urllib.parse.unquote(str(img_data))
                raw_b64 = unquoted.split(',')[1]
                missing_padding = len(raw_b64) % 4
                if missing_padding: raw_b64 += "=" * (4 - missing_padding)
                
                nparr = np.frombuffer(base64.b64decode(raw_b64), np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    st.error("Frame Decode Error")
                else:
                    # 2. Identify
                    res = DeepFace.find(
                        img_path=frame, 
                        db_path=DB_FOLDER, 
                        model_name=MODEL_NAME, 
                        enforce_detection=False, 
                        detector_backend="opencv", 
                        silent=True
                    )
                    
                    if len(res) > 0 and not res[0].empty:
                        m_name = os.path.basename(res[0].iloc[0]['identity']).split('.')[0]
                        dist = res[0].iloc[0]['distance']
                        acc = max(0, int((1 - dist/0.4) * 100))
                        
                        if acc > 25:
                            st.metric("Detected", m_name, f"{acc}% Match")
                            if m_name not in st.session_state.logged_set:
                                save_attendance_pkl(m_name)
                                st.session_state.logged_set.add(m_name)
                                st.toast(f"✅ Logged: {m_name}")
                        else:
                            st.warning("Identity: Unknown")
                    else:
                        st.info("Scanning for match...")
            except Exception as e:
                # This will tell us the EXACT error
                st.error(f"Engine Debug: {str(e)}")
        else:
            st.info("Awaiting Handshake...")

elif page == "Log History":
    st.header("📊 Attendance Log")
    if os.path.exists(PKL_LOG):
        with open(PKL_LOG, "rb") as f: data = pickle.load(f)
        df = pd.DataFrame(data)
        st.table(df)
        if st.button("🔥 WIPE SESSION"):
            if os.path.exists(PKL_LOG): os.remove(PKL_LOG)
            st.session_state.logged_set = set()
            st.rerun()
    else: st.info("No logs found.")
