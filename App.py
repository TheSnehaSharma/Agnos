import streamlit as st
import pandas as pd
import cv2
import numpy as np
import base64
import os
import pickle
import gzip
import urllib.parse
from deepface import DeepFace
from datetime import datetime

# --- CONFIG ---
DB_FOLDER = "registered_faces"
PKL_LOG = "attendance_data.pkl"
MODEL_NAME = "Facenet512"

if not os.path.exists(DB_FOLDER):
    os.makedirs(DB_FOLDER)

st.set_page_config(page_title="Iron-Vision Pro", layout="wide")

# --- MODEL CACHE ---
@st.cache_resource
def load_ai_models():
    try:
        DeepFace.build_model(MODEL_NAME)
        return True
    except Exception as e:
        return str(e)

load_ai_models()

# --- STATE ---
if "detected_name" not in st.session_state:
    st.session_state.detected_name = "SEARCHING..."

# --- HELPERS ---
def robust_decompress(compressed_data):
    try:
        unquoted = urllib.parse.unquote(str(compressed_data))
        raw_bytes = base64.b64decode(unquoted)
        decompressed = gzip.decompress(raw_bytes)
        nparr = np.frombuffer(decompressed, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except:
        return None

def save_log(name):
    logs = []
    if os.path.exists(PKL_LOG):
        with open(PKL_LOG, "rb") as f:
            try: logs = pickle.load(f)
            except: logs = []
    
    today = datetime.now().strftime("%Y-%m-%d")
    if not any(e['Name'] == name and e['Date'] == today for e in logs):
        logs.append({"Name": name, "Time": datetime.now().strftime("%H:%M:%S"), "Date": today})
        with open(PKL_LOG, "wb") as f:
            pickle.dump(logs, f)
        return True
    return False

# --- JAVASCRIPT TEMPLATE (No f-string) ---
RAW_JS_TEMPLATE = """
<div style="position:relative; width:100%; max-width:400px; margin:auto; background:#000; border-radius:15px; overflow:hidden;">
    <video id="v" autoplay playsinline style="display:none;"></video>
    <canvas id="display_canvas" style="width:100%; height:auto; display:block;"></canvas>
    <div id="name_label" style="position:absolute; bottom:20px; left:0; width:100%; text-align:center; color:#0F0; font-family:monospace; font-weight:bold; font-size:18px; text-shadow:2px 2px #000;">ST_NAME_HERE</div>
</div>

<script src="https://cdn.jsdelivr.net/npm/@mediapipe/face_detection"></script>
<script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/pako/2.1.0/pako.min.js"></script>

<script>
    const v = document.getElementById('v');
    const d_can = document.getElementById('display_canvas');
    const d_ctx = d_can.getContext('2d');

    const faceDetection = new FaceDetection({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_detection/${file}`
    });

    faceDetection.setOptions({ model: 'short', minDetectionConfidence: 0.6 });

    faceDetection.onResults((results) => {
        d_can.width = v.videoWidth; d_can.height = v.videoHeight;
        d_ctx.drawImage(v, 0, 0, d_can.width, d_can.height);
        
        if (results.detections.length > 0) {
            const bbox = results.detections[0].boundingBox;
            
            d_ctx.strokeStyle = "#00FF00";
            d_ctx.lineWidth = 3;
            d_ctx.strokeRect(
                bbox.xCenter * d_can.width - (bbox.width * d_can.width / 2),
                bbox.yCenter * d_can.height - (bbox.height * d_can.height / 2),
                bbox.width * d_can.width, bbox.height * d_can.height
            );

            const temp_can = document.createElement('canvas');
            temp_can.width = 160; temp_can.height = 160;
            temp_can.getContext('2d').drawImage(v, 
                bbox.xCenter * v.videoWidth - (bbox.width * v.videoWidth / 2), 
                bbox.yCenter * v.videoHeight - (bbox.height * v.videoHeight / 2), 
                bbox.width * v.videoWidth, bbox.height * v.videoHeight,
                0, 0, 160, 160
            );
            
            const dataURL = temp_can.toDataURL('image/jpeg', 0.4);
            const bytes = Uint8Array.from(atob(dataURL.split(',')[1]), c => c.charCodeAt(0));
            const compressed = btoa(String.fromCharCode.apply(null, pako.gzip(bytes)));
            
            window.parent.postMessage({
                type: "streamlit:setComponentValue", 
                value: encodeURIComponent(compressed)
            }, "*");
        }
    });

    const camera = new Camera(v, {
        onFrame: async () => { await faceDetection.send({image: v}); },
        width: 400, height: 300
    });
    camera.start();
</script>
"""

# --- UI NAVIGATION ---
page = st.sidebar.radio("Navigate", ["Register", "Live Feed", "Log History"])

if page == "Register":
    st.header("👤 Face Registration")
    
    with st.form("registration"):
        name = st.text_input("Name").upper()
        file = st.file_uploader("Image", type=['jpg', 'jpeg', 'png'])
        if st.form_submit_button("Save User"):
            if name and file:
                with open(os.path.join(DB_FOLDER, f"{name}.jpg"), "wb") as f:
                    f.write(file.getbuffer())
                for p in [f for f in os.listdir(DB_FOLDER) if f.endswith('.pkl')]:
                    os.remove(os.path.join(DB_FOLDER, p))
                st.success(f"Registered {name}")

    st.markdown("---")
    st.subheader("🗂️ Database Management")
    reg_files = [f for f in os.listdir(DB_FOLDER) if f.endswith(('.jpg', '.png'))]
    if reg_files:
        for f in reg_files:
            c1, c2 = st.columns([4, 1])
            c1.write(f"✅ {f.split('.')[0]}")
            if c2.button("Delete", key=f"del_{f}"):
                os.remove(os.path.join(DB_FOLDER, f))
                st.rerun()
    else:
        st.info("No users registered.")

elif page == "Live Feed":
    st.header("📹 Live Identification Feed")
    
    # 1. Prepare JS with the current session state name
    current_js = RAW_JS_TEMPLATE.replace("ST_NAME_HERE", st.session_state.detected_name)

    col_v, col_s = st.columns([2, 1])
    
    with col_v:
        img_data = st.components.v1.html(current_js, height=350)

    with col_s:
        st.subheader("System Status")
        if img_data:
            frame = robust_decompress(img_data)
            if frame is not None:
                try:
                    res = DeepFace.find(img_path=frame, db_path=DB_FOLDER, 
                                       model_name=MODEL_NAME, enforce_detection=False, 
                                       detector_backend="skip", silent=True)
                    if len(res) > 0 and not res[0].empty:
                        m_name = os.path.basename(res[0].iloc[0]['identity']).split('.')[0]
                        dist = res[0].iloc[0]['distance']
                        acc = max(0, int((1 - dist/0.38) * 100))
                        
                        if acc > 30:
                            st.session_state.detected_name = f"VERIFIED: {m_name}"
                            st.metric("Detected", m_name, f"{acc}% Match")
                            if save_log(m_name): st.toast(f"✅ Logged: {m_name}")
                        else:
                            st.session_state.detected_name = "UNKNOWN USER"
                    else:
                        st.session_state.detected_name = "SEARCHING..."
                except:
                    st.session_state.detected_name = "ENGINE SYNCING..."
        else:
            st.info("Awaiting Camera...")

elif page == "Log History":
    st.header("📊 Attendance Log")
    if os.path.exists(PKL_LOG):
        with open(PKL_LOG, "rb") as f: data = pickle.load(f)
        if data:
            st.table(pd.DataFrame(data))
            if st.button("Wipe Logs"):
                os.remove(PKL_LOG)
                st.rerun()
        else: st.info("No records.")
