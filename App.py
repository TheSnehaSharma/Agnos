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

st.set_page_config(page_title="Iron-Vision: URL Bridge", layout="wide")

# --- MODEL CACHE ---
@st.cache_resource
def load_ai_models():
    try:
        DeepFace.build_model(MODEL_NAME)
        return True
    except Exception as e:
        return str(e)

load_ai_models()

# --- JAVASCRIPT: THE URL INJECTOR ---
# Instead of postMessage, we write to window.parent.location.hash
JS_URL_BRIDGE = """
<div style="position:relative; width:100%; max-width:400px; margin:auto; background:#000; border-radius:15px; overflow:hidden;">
    <video id="v" autoplay playsinline style="display:none;"></video>
    <canvas id="display_canvas" style="width:100%; height:auto; display:block;"></canvas>
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
            temp_can.width = 120; temp_can.height = 120; // Tiny for URL limits
            temp_can.getContext('2d').drawImage(v, 
                bbox.xCenter * v.videoWidth - (bbox.width * v.videoWidth / 2), 
                bbox.yCenter * v.videoHeight - (bbox.height * v.videoHeight / 2), 
                bbox.width * v.videoWidth, bbox.height * v.videoHeight,
                0, 0, 120, 120
            );
            
            const dataURL = temp_can.toDataURL('image/jpeg', 0.2);
            const bytes = Uint8Array.from(atob(dataURL.split(',')[1]), c => c.charCodeAt(0));
            const compressed = btoa(String.fromCharCode.apply(null, pako.gzip(bytes)));
            
            // THE URL BRIDGE: We push the data into the URL hash
            window.parent.location.hash = "frame=" + compressed;
        }
    });

    const camera = new Camera(v, {
        onFrame: async () => { await faceDetection.send({image: v}); },
        width: 320, height: 240
    });
    camera.start();
</script>
"""

# --- UI NAVIGATION ---
page = st.sidebar.radio("Navigate", ["Register", "Live Feed", "Log History"])

if page == "Register":
    st.header("👤 Face Registration")
    with st.form("reg"):
        name = st.text_input("Name").upper()
        file = st.file_uploader("Image", type=['jpg', 'jpeg', 'png'])
        if st.form_submit_button("Save"):
            if name and file:
                with open(os.path.join(DB_FOLDER, f"{name}.jpg"), "wb") as f:
                    f.write(file.getbuffer())
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
    st.header("📹 Live Identification")
    
    # 1. Render the Video
    st.components.v1.html(JS_URL_BRIDGE, height=300)

    # 2. Read the URL Bridge
    # Streamlit can't read the 'hash' directly easily, so we use a Manual Refresh button 
    # OR we use the experimental query params
    params = st.query_params
    img_data = params.get("frame")

    st.subheader("System Status")
    if img_data:
        try:
            raw_bytes = base64.b64decode(img_data)
            decompressed = gzip.decompress(raw_bytes)
            nparr = np.frombuffer(decompressed, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            res = DeepFace.find(img_path=frame, db_path=DB_FOLDER, 
                               model_name=MODEL_NAME, enforce_detection=False, 
                               detector_backend="skip", silent=True)
            
            if len(res) > 0 and not res[0].empty:
                m_name = os.path.basename(res[0].iloc[0]['identity']).split('.')[0]
                st.metric("Identity", m_name)
            else:
                st.info("Scanning...")
        except:
            st.error("Decoding frame...")
    else:
        st.info("Waiting for first face lock...")
        if st.button("🔄 Sync Feed"):
            st.rerun()

elif page == "Log History":
    st.header("📊 Logs")
    if os.path.exists(PKL_LOG):
        with open(PKL_LOG, "rb") as f: data = pickle.load(f)
        st.table(pd.DataFrame(data))
