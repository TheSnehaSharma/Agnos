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
MODEL_NAME = "Facenet512"

if not os.path.exists(DB_FOLDER):
    os.makedirs(DB_FOLDER)

st.set_page_config(page_title="Ultra-Slim Auth", layout="wide")

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

# --- HELPER: LOGGING ---
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

# --- JAVASCRIPT CAMERA (GRAYSCALE & COMPRESSED) ---
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
        const s = await navigator.mediaDevices.getUserMedia({ video: { width: 240, height: 180 } });
        v.srcObject = s;
    }

    function sync() {
        if (v.readyState === v.HAVE_ENOUGH_DATA) {
            c.width = 240; c.height = 180;
            ctx.drawImage(v, 0, 0, 240, 180);
            
            // --- GRAYSCALE CONVERSION (Reduces data significantly) ---
            let imgData = ctx.getImageData(0, 0, c.width, c.height);
            let data = imgData.data;
            for (let i = 0; i < data.length; i += 4) {
                let avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
                data[i] = avg; data[i + 1] = avg; data[i + 2] = avg;
            }
            ctx.putImageData(imgData, 0, 0);

            // WebP is the most efficient string-based format
            const dataURL = c.toDataURL('image/webp', 0.2); 
            window.parent.postMessage({type: "streamlit:setComponentValue", value: dataURL}, "*");
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
        img_data = st.components.v1.html(JS_CODE, height=280)

    with col_s:
        if img_data:
            try:
                # REPAIR PADDING
                b64_str = str(img_data).split(',')[1]
                missing_padding = len(b64_str) % 4
                if missing_padding:
                    b64_str += '=' * (4 - missing_padding)
                
                # Decode
                nparr = np.frombuffer(base64.b64decode(b64_str), np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Search
                res = DeepFace.find(img_path=frame, db_path=DB_FOLDER, 
                                   model_name=MODEL_NAME, enforce_detection=False, 
                                   detector_backend="opencv", silent=True)
                
                if len(res) > 0 and not res[0].empty:
                    match_name = os.path.basename(res[0].iloc[0]['identity']).split('.')[0]
                    dist = res[0].iloc[0]['distance']
                    acc = max(0, int((1 - dist/0.38) * 100))
                    
                    if acc > 30:
                        st.metric("Detected", match_name, f"{acc}% Match")
                        if match_name not in st.session_state.logged_set:
                            save_attendance_pkl(match_name)
                            st.session_state.logged_set.add(match_name)
                            st.toast(f"✅ Logged: {match_name}")
                    else:
                        st.warning("Identity: Unknown")
                else:
                    st.warning("Searching for match...")
            except Exception as e:
                st.error("Processing sync...")
        else:
            st.info("Awaiting Handshake...")

elif page == "Log History":
    st.header("📊 Attendance Log")
    if os.path.exists(PKL_LOG):
        with open(PKL_LOG, "rb") as f: data = pickle.load(f)
        df = pd.DataFrame(data)
        st.table(df)
        if st.button("🔥 WIPE SESSION DATA"):
            if os.path.exists(PKL_LOG): os.remove(PKL_LOG)
            st.session_state.logged_set = set()
            st.rerun()
    else: st.info("No logs found.")
