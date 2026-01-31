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

# --- CONFIG & DIRS ---
DB_FOLDER = "registered_faces"
PKL_LOG = "attendance_data.pkl"
MODEL_NAME = "Facenet512"

for folder in [DB_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

st.set_page_config(page_title="Iron-Vision Live", layout="wide")

# --- AI CACHE ---
@st.cache_resource
def load_ai():
    DeepFace.build_model(MODEL_NAME)
    return True

load_ai()

# --- HELPERS ---
def decompress_frame(data):
    try:
        raw = base64.b64decode(urllib.parse.unquote(data))
        decompressed = gzip.decompress(raw)
        nparr = np.frombuffer(decompressed, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except: return None

# --- JS: CONTINUOUS LIVE FEED + RECTANGLE ---
# This code runs independently in the browser. It doesn't wait for Python.
JS_FEED_CODE = """
<div style="position:relative; width:100%; max-width:500px; margin:auto; background:#000; border-radius:20px; overflow:hidden; border:3px solid #333;">
    <video id="v" autoplay playsinline style="display:none;"></video>
    <canvas id="d" style="width:100%; height:auto; display:block;"></canvas>
    <div id="stat" style="position:absolute; top:15px; left:15px; color:#0F0; font-family:monospace; font-size:14px; text-shadow:2px 2px #000;">SYSTEM: LIVE</div>
</div>

<script src="https://cdn.jsdelivr.net/npm/@mediapipe/face_detection"></script>
<script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/pako/2.1.0/pako.min.js"></script>

<script>
    const v = document.getElementById('v');
    const d = document.getElementById('d');
    const ctx = d.getContext('2d');
    const stat = document.getElementById('stat');

    const faceDetection = new FaceDetection({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_detection/${file}`
    });

    faceDetection.setOptions({ model: 'short', minDetectionConfidence: 0.6 });

    faceDetection.onResults((results) => {
        d.width = v.videoWidth; d.height = v.videoHeight;
        ctx.drawImage(v, 0, 0, d.width, d.height);
        
        if (results.detections.length > 0) {
            const bbox = results.detections[0].boundingBox;
            
            // 1. DRAW LIVE RECTANGLE (NO LAG)
            ctx.strokeStyle = "#00FF00";
            ctx.lineWidth = 4;
            ctx.strokeRect(
                bbox.xCenter * d.width - (bbox.width * d.width / 2),
                bbox.yCenter * d.height - (bbox.height * d.height / 2),
                bbox.width * d.width, bbox.height * d.height
            );

            // 2. SEND DATA TO PYTHON (IN BACKGROUND)
            const c = document.createElement('canvas');
            c.width = 160; c.height = 160;
            c.getContext('2d').drawImage(v, 
                bbox.xCenter * v.videoWidth - (bbox.width * v.videoWidth / 2), 
                bbox.yCenter * v.videoHeight - (bbox.height * v.videoHeight / 2), 
                bbox.width * v.videoWidth, bbox.height * v.videoHeight,
                0, 0, 160, 160
            );
            
            const data = c.toDataURL('image/jpeg', 0.3);
            const bytes = Uint8Array.from(atob(data.split(',')[1]), c => c.charCodeAt(0));
            const comp = btoa(String.fromCharCode.apply(null, pako.gzip(bytes)));
            
            // This is the bridge back to Streamlit variable
            window.parent.postMessage({type: "streamlit:setComponentValue", value: encodeURIComponent(comp)}, "*");
        }
    });

    const camera = new Camera(v, {
        onFrame: async () => { await faceDetection.send({image: v}); },
        width: 640, height: 480
    });
    camera.start();
</script>
"""

# --- NAVIGATION ---
page = st.sidebar.radio("Navigate", ["Register", "Live Feed", "Logs"])

if page == "Register":
    st.header("👤 Face Registration")
    with st.form("reg"):
        name = st.text_input("Name").upper()
        file = st.file_uploader("Photo", type=['jpg','png'])
        if st.form_submit_button("Save"):
            if name and file:
                with open(os.path.join(DB_FOLDER, f"{name}.jpg"), "wb") as f:
                    f.write(file.getbuffer())
                for p in [f for f in os.listdir(DB_FOLDER) if f.endswith('.pkl')]:
                    os.remove(os.path.join(DB_FOLDER, p))
                st.success(f"Registered {name}")

    st.subheader("🗂️ Database Management")
    for f in [f for f in os.listdir(DB_FOLDER) if f.endswith(('.jpg', '.png'))]:
        c1, c2 = st.columns([4, 1])
        c1.write(f"✅ {f.split('.')[0]}")
        if c2.button("Delete", key=f"del_{f}"):
            os.remove(os.path.join(DB_FOLDER, f))
            st.rerun()

elif page == "Live Feed":
    st.header("📹 Live Biometric Scanner")
    col_v, col_s = st.columns([2, 1])
    
    with col_v:
        # THE BRIDGE: This is the ONLY component that can return a value
        # We must use a key to maintain state
        img_data = st.components.v1.html(JS_FEED_CODE, height=450, key="cam_bridge")

    with col_s:
        st.subheader("AI Analysis")
        if img_data:
            frame = decompress_frame(img_data)
            if frame is not None:
                try:
                    res = DeepFace.find(img_path=frame, db_path=DB_FOLDER, 
                                       model_name=MODEL_NAME, enforce_detection=False, 
                                       detector_backend="skip", silent=True)
                    if len(res) > 0 and not res[0].empty:
                        m_name = os.path.basename(res[0].iloc[0]['identity']).split('.')[0]
                        acc = max(0, int((1 - res[0].iloc[0]['distance']/0.38) * 100))
                        if acc > 30:
                            st.metric("Identity", m_name, f"{acc}% Match")
                            # Add logging logic here...
                        else: st.warning("Identity Unknown")
                    else: st.info("Scanning...")
                except: st.error("Engine Busy...")
        else:
            st.info("Awaiting Camera Permissions...")

elif page == "Logs":
    st.header("📊 Attendance Log")
    if os.path.exists(PKL_LOG):
        with open(PKL_LOG, "rb") as f: data = pickle.load(f)
        st.table(pd.DataFrame(data))
