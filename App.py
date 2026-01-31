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

st.set_page_config(page_title="Hybrid Gzip Auth", layout="wide")

# --- MODEL CACHE ---
@st.cache_resource
def load_ai_models():
    DeepFace.build_model(MODEL_NAME)
    return True

load_ai_models()

# --- HELPERS ---
def decompress_and_decode(compressed_data):
    """Decompress Gzip data and convert back to CV2 image."""
    try:
        # 1. URL Unquote & Base64 Decode
        raw_bytes = base64.b64decode(urllib.parse.unquote(compressed_data))
        # 2. Gzip Decompress
        decompressed = gzip.decompress(raw_bytes)
        # 3. Decode Image
        nparr = np.frombuffer(decompressed, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        return None

# --- JAVASCRIPT: MEDIAPIPE + GZIP ---
# We use Pako (Gzip) via CDN to compress the cropped face
JS_CODE = """
<div style="background:#000; border-radius:12px; overflow:hidden; width:100%; height:300px; position:relative;">
    <video id="v" autoplay playsinline style="width:100%; height:100%; object-fit:contain;"></video>
    <canvas id="c" style="display:none;"></canvas>
</div>
<script src="https://cdn.jsdelivr.net/npm/@mediapipe/face_detection"></script>
<script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/pako/2.1.0/pako.min.js"></script>

<script>
    const video = document.getElementById('v');
    const canvas = document.getElementById('c');
    const ctx = canvas.getContext('2d');
    
    const faceDetection = new FaceDetection({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_detection/${file}`
    });

    faceDetection.setOptions({ model: 'short', minDetectionConfidence: 0.5 });

    faceDetection.onResults((results) => {
        if (results.detections.length > 0) {
            const bbox = results.detections[0].boundingBox;
            // Crop face from video
            canvas.width = 160; canvas.height = 160;
            ctx.drawImage(video, 
                bbox.xCenter * video.videoWidth - (bbox.width * video.videoWidth / 2), 
                bbox.yCenter * video.videoHeight - (bbox.height * video.videoHeight / 2), 
                bbox.width * video.videoWidth, bbox.height * video.videoHeight,
                0, 0, 160, 160
            );
            
            // Get Image Data and GZIP it
            const dataURL = canvas.toDataURL('image/jpeg', 0.4);
            const binaryString = atob(dataURL.split(',')[1]);
            const bytes = new Uint8Array(binaryString.length);
            for (let i = 0; i < binaryString.length; i++) bytes[i] = binaryString.charCodeAt(i);
            
            const compressed = pako.gzip(bytes);
            const b64compressed = btoa(String.fromCharCode.apply(null, compressed));
            
            window.parent.postMessage({
                type: "streamlit:setComponentValue",
                value: encodeURIComponent(b64compressed)
            }, "*");
        }
    });

    const camera = new Camera(video, {
        onFrame: async () => { await faceDetection.send({image: video}); },
        width: 320, height: 240
    });
    camera.start();
</script>
"""

# --- UI LOGIC ---
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
    st.header("📹 Hybrid Biometric Feed")
    col_v, col_s = st.columns([2, 1])
    
    with col_v:
        img_data = st.components.v1.html(JS_CODE, height=350)

    with col_s:
        if img_data:
            frame = decompress_and_decode(img_data)
            if frame is not None:
                try:
                    res = DeepFace.find(img_path=frame, db_path=DB_FOLDER, 
                                       model_name=MODEL_NAME, enforce_detection=False, 
                                       detector_backend="skip", silent=True)
                    
                    if len(res) > 0 and not res[0].empty:
                        m_name = os.path.basename(res[0].iloc[0]['identity']).split('.')[0]
                        dist = res[0].iloc[0]['distance']
                        acc = max(0, int((1 - dist/0.4) * 100))
                        
                        st.metric("Detected", m_name, f"{acc}% Match")
                        if m_name not in st.session_state.get('logged_set', set()):
                            # Save attendance...
                            st.toast(f"✅ Logged: {m_name}")
                    else: st.info("Scanning...")
                except Exception as e: st.error("Engine Resyncing...")
        else: st.info("Awaiting Face...")

elif page == "Log History":
    # ... standard table logic ...
    pass
