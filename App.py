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

# --- HELPER: LOGGING & PADDING ---
def repair_padding(b64_string):
    """Adds missing = padding to a Base64 string."""
    return b64_string + "=" * ((4 - len(b64_string) % 4) % 4)

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

# --- JAVASCRIPT CAMERA (ULTRA COMPACT) ---
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
        const s = await navigator.mediaDevices.getUserMedia({ video: { width: 160, height: 120 } });
        v.srcObject = s;
    }

    function sync() {
        if (v.readyState === v.HAVE_ENOUGH_DATA) {
            c.width = 160; c.height = 120;
            ctx.drawImage(v, 0, 0, 160, 120);
            
            // Aggressive Grayscale
            let imgData = ctx.getImageData(0, 0, c.width, c.height);
            let d = imgData.data;
            for (let i = 0; i < d.length; i += 4) {
                let avg = (d[i] + d[i+1] + d[i+2]) / 3;
                d[i] = avg; d[i+1] = avg; d[i+2] = avg;
            }
            ctx.putImageData(imgData, 0, 0);

            // Tiny WebP packet
            const dataURL = c.toDataURL('image/webp', 0.1); 
            window.parent.postMessage({type: "streamlit:setComponentValue", value: dataURL}, "*");
        }
    }

    start();
    setInterval(sync, 2500); 
</script>
"""

# --- UI NAVIGATION ---
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

elif page == "Live Feed":
    st.header("📹 Turbo Biometric Scanner")
    col_v, col_s = st.columns([2, 1])
    
    with col_v:
        img_data = st.components.v1.html(JS_CODE, height=280)

    with col_s:
        if img_data:
            try:
                # 1. Extract and Repair String
                raw_b64 = str(img_data).split(',')[1]
                clean_b64 = repair_padding(raw_b64)
                
                # 2. Decode
                nparr = np.frombuffer(base64.b64decode(clean_b64), np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # 3. Match (Skip detector for speed)
                res = DeepFace.find(
                    img_path=frame, 
                    db_path=DB_FOLDER, 
                    model_name=MODEL_NAME, 
                    enforce_detection=False, 
                    detector_backend="skip", 
                    silent=True
                )
                
                if len(res) > 0 and not res[0].empty:
                    match_name = os.path.basename(res[0].iloc[0]['identity']).split('.')[0]
                    dist = res[0].iloc[0]['distance']
                    acc = max(0, int((1 - dist/0.4) * 100))
                    
                    if acc > 25:
                        st.metric("Detected", match_name, f"{acc}% Match")
                        if match_name not in st.session_state.logged_set:
                            save_attendance_pkl(match_name)
                            st.session_state.logged_set.add(match_name)
                            st.toast(f"✅ Logged: {match_name}")
                    else:
                        st.warning("Identity: Unknown")
                else:
                    st.info("No face in database.")
            except Exception as e:
                st.error(f"Sync Fix in Progress: {str(e)}")
        else:
            st.info("Awaiting Handshake...")

elif page == "Log History":
    st.header("📊 Attendance Log")
    if os.path.exists(PKL_LOG):
        with open(PKL_LOG, "rb") as f: data = pickle.load(f)
        st.table(pd.DataFrame(data))
        if st.button("🔥 WIPE SESSION DATA"):
            if os.path.exists(PKL_LOG): os.remove(PKL_LOG)
            st.session_state.logged_set = set()
            st.rerun()
    else: st.info("No logs found.")
