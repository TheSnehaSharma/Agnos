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

for folder in [DB_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

st.set_page_config(page_title="Hybrid Iron-Vision", layout="wide")

# --- MODEL CACHE ---
@st.cache_resource
def load_ai_models():
    DeepFace.build_model(MODEL_NAME)
    return True

load_ai_models()

# --- HELPERS ---
def decompress_frame(compressed_data):
    try:
        raw_bytes = base64.b64decode(urllib.parse.unquote(compressed_data))
        decompressed = gzip.decompress(raw_bytes)
        nparr = np.frombuffer(decompressed, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except: return None

def save_log(name):
    logs = []
    if os.path.exists(PKL_LOG):
        with open(PKL_LOG, "rb") as f:
            try: logs = pickle.load(f)
            except: logs = []
    if not any(entry['Name'] == name and entry['Date'] == datetime.now().strftime("%Y-%m-%d") for entry in logs):
        logs.append({
            "Name": name, 
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Date": datetime.now().strftime("%Y-%m-%d")
        })
        with open(PKL_LOG, "wb") as f:
            pickle.dump(logs, f)
        return True
    return False

# --- JS: VISUAL OVERLAY + GZIP CROP ---
JS_CODE = """
<div style="position:relative; width:100%; max-width:400px; margin:auto; background:#000; border-radius:15px; overflow:hidden;">
    <video id="v" autoplay playsinline style="display:none;"></video>
    <canvas id="display_canvas" style="width:100%; height:auto; display:block;"></canvas>
    <canvas id="crop_canvas" style="display:none;"></canvas>
    <div id="status" style="position:absolute; top:10px; left:10px; color:#0F0; font-family:monospace; font-size:12px; text-shadow:1px 1px #000;">AI: INITIALIZING...</div>
</div>

<script src="https://cdn.jsdelivr.net/npm/@mediapipe/face_detection"></script>
<script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/pako/2.1.0/pako.min.js"></script>

<script>
    const v = document.getElementById('v');
    const d_can = document.getElementById('display_canvas');
    const d_ctx = d_can.getContext('2d');
    const c_can = document.getElementById('crop_canvas');
    const c_ctx = c_can.getContext('2d');
    const status = document.getElementById('status');

    const faceDetection = new FaceDetection({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_detection/${file}`
    });

    faceDetection.setOptions({ model: 'short', minDetectionConfidence: 0.6 });

    faceDetection.onResults((results) => {
        d_can.width = v.videoWidth; d_can.height = v.videoHeight;
        d_ctx.drawImage(v, 0, 0, d_can.width, d_can.height);
        
        if (results.detections.length > 0) {
            status.innerText = "AI: TRACKING FACE";
            const bbox = results.detections[0].boundingBox;
            
            // DRAW RECTANGLE (Visual Only)
            d_ctx.strokeStyle = "#00FF00";
            d_ctx.lineWidth = 4;
            d_ctx.strokeRect(
                bbox.xCenter * d_can.width - (bbox.width * d_can.width / 2),
                bbox.yCenter * d_can.height - (bbox.height * d_can.height / 2),
                bbox.width * d_can.width, bbox.height * d_can.height
            );

            // CROP & COMPRESS
            c_can.width = 160; c_can.height = 160;
            c_ctx.drawImage(v, 
                bbox.xCenter * v.videoWidth - (bbox.width * v.videoWidth / 2), 
                bbox.yCenter * v.videoHeight - (bbox.height * v.videoHeight / 2), 
                bbox.width * v.videoWidth, bbox.height * v.videoHeight,
                0, 0, 160, 160
            );
            
            const dataURL = c_can.toDataURL('image/jpeg', 0.4);
            const bytes = Uint8Array.from(atob(dataURL.split(',')[1]), c => c.charCodeAt(0));
            const compressed = btoa(String.fromCharCode.apply(null, pako.gzip(bytes)));
            
            window.parent.postMessage({type: "streamlit:setComponentValue", value: encodeURIComponent(compressed)}, "*");
        } else {
            status.innerText = "AI: SEARCHING...";
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
    name = st.text_input("Full Name").upper()
    file = st.file_uploader("Upload Profile Image", type=['jpg', 'jpeg', 'png'])
    if st.button("Register User") and name and file:
        with open(os.path.join(DB_FOLDER, f"{name}.jpg"), "wb") as f:
            f.write(file.getbuffer())
        for p in [f for f in os.listdir(DB_FOLDER) if f.endswith('.pkl')]:
            os.remove(os.path.join(DB_FOLDER, p))
        st.success(f"Registered {name}!")

elif page == "Live Feed":
    st.header("📹 Live Biometric Scanner")
    col_v, col_s = st.columns([2, 1])
    
    with col_v:
        img_data = st.components.v1.html(JS_CODE, height=350)

    with col_s:
        if img_data:
            frame = decompress_frame(img_data)
            if frame is not None:
                try:
                    res = DeepFace.find(img_path=frame, db_path=DB_FOLDER, 
                                       model_name=MODEL_NAME, enforce_detection=False, 
                                       detector_backend="skip", silent=True)
                    if len(res) > 0 and not res[0].empty:
                        m_name = os.path.basename(res[0].iloc[0]['identity']).split('.')[0]
                        dist = res[0].iloc[0]['distance']
                        acc = max(0, int((1 - dist/0.35) * 100))
                        st.metric("Identity", m_name, f"{acc}% Match")
                        if save_log(m_name): st.toast(f"✅ Logged: {m_name}")
                    else: st.info("Comparing identity...")
                except: st.error("Engine Resyncing...")
        else: st.info("Waiting for face detection...")

elif page == "Log History":
    st.header("📊 Attendance Log")
    if os.path.exists(PKL_LOG):
        with open(PKL_LOG, "rb") as f: data = pickle.load(f)
        if data:
            st.table(pd.DataFrame(data))
            if st.button("🔥 WIPE DATA"):
                os.remove(PKL_LOG)
                st.rerun()
        else: st.info("No entries for today.")
    else: st.info("Log file not found.")
