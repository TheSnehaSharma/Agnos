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

st.set_page_config(page_title="Iron-Bridge Auth", layout="wide")

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

# --- HELPER: ROBUST LOGGING ---
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

# --- JAVASCRIPT: THE STABLE BRIDGE ---
# We use a custom component that returns the value directly.
def camera_bridge():
    js_code = """
    <div style="background:#000; border-radius:15px; overflow:hidden;">
        <video id="v" autoplay playsinline style="width:100%; height:auto; object-fit:contain;"></video>
        <canvas id="c" style="display:none;"></canvas>
    </div>
    <script>
        const v = document.getElementById('v');
        const c = document.getElementById('c');
        const ctx = c.getContext('2d');

        async function start() {
            const s = await navigator.mediaDevices.getUserMedia({ video: { width: 200, height: 150 } });
            v.srcObject = s;
        }

        function sync() {
            if (v.readyState === v.HAVE_ENOUGH_DATA) {
                c.width = 200; c.height = 150;
                ctx.drawImage(v, 0, 0, 200, 150);
                // URL-Safe Base64 via WebP
                const data = c.toDataURL('image/webp', 0.2); 
                window.parent.postMessage({
                    type: "streamlit:setComponentValue",
                    value: encodeURIComponent(data)
                }, "*");
            }
        }

        start();
        setInterval(sync, 3000); // 3-second heartbeat
    </script>
    """
    return st.components.v1.html(js_code, height=300)

# --- UI NAVIGATION ---
page = st.sidebar.radio("Navigate", ["Register", "Live Feed", "Log History"])

if page == "Register":
    st.header("👤 Face Registration")
    name = st.text_input("Name").upper()
    file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
    if st.button("Save User") and name and file:
        with open(os.path.join(DB_FOLDER, f"{name}.jpg"), "wb") as f:
            f.write(file.getbuffer())
        # Clear DeepFace cache
        for p in [f for f in os.listdir(DB_FOLDER) if f.endswith('.pkl')]:
            os.remove(os.path.join(DB_FOLDER, p))
        st.success(f"Registered {name}")

    st.markdown("---")
    st.subheader("🗂️ Database")
    for f in [f for f in os.listdir(DB_FOLDER) if f.endswith(('.jpg', '.png'))]:
        c1, c2 = st.columns([4, 1])
        c1.write(f"✅ {f.split('.')[0]}")
        if c2.button("Delete", key=f"del_{f}"):
            os.remove(os.path.join(DB_FOLDER, f))
            st.rerun()

elif page == "Live Feed":
    st.header("📹 Biometric Scanner")
    col_v, col_s = st.columns([2, 1])
    
    with col_v:
        # Capture the return value of the component
        img_data = camera_bridge()

    with col_s:
        if img_data:
            try:
                # 1. Unwrap the URL encoding
                unquoted = urllib.parse.unquote(str(img_data))
                raw_b64 = unquoted.split(',')[1]
                
                # 2. Safety Padding
                raw_b64 += "=" * ((4 - len(raw_b64) % 4) % 4)
                
                # 3. Decode
                nparr = np.frombuffer(base64.b64decode(raw_b64), np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # 4. DeepFace (Skip detection for speed)
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
                        st.metric("Identity", match_name, f"{acc}% Match")
                        if match_name not in st.session_state.logged_set:
                            save_attendance_pkl(match_name)
                            st.session_state.logged_set.add(match_name)
                            st.toast(f"✅ Logged: {match_name}")
                    else:
                        st.warning("Unknown User")
                else:
                    st.info("Scanning...")
            except Exception as e:
                st.error("Engine Resyncing...")
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
