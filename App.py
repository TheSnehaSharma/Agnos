import streamlit as st
import pandas as pd
import cv2
import numpy as np
import base64
import os
import pickle
from insightface.app import FaceAnalysis
from datetime import datetime

# --- CONFIG ---
DB_FOLDER = "registered_faces"
PKL_LOG = "attendance_data.pkl"

if not os.path.exists(DB_FOLDER):
    os.makedirs(DB_FOLDER)

st.set_page_config(page_title="Agnos: InsightFace Edition", layout="wide")

# --- AI ENGINE (Pre-compiled ONNX) ---
@st.cache_resource
def load_insightface():
    # 'buffalo_l' is their high-accuracy model pack
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

# --- DATABASE HANDLER ---
@st.cache_resource
def load_face_db():
    engine = load_insightface()
    db = {}
    for file in os.listdir(DB_FOLDER):
        if file.endswith(('.jpg', '.png', '.jpeg')):
            img = cv2.imread(os.path.join(DB_FOLDER, file))
            faces = engine.get(img)
            if faces:
                # We store the 512-dimensional embedding
                db[file.split('.')[0]] = faces[0].normed_embedding
    return db

# --- JS BRIDGE (MediaPipe Rectangles) ---
JS_CODE = """
<div style="position:relative; width:100%; max-width:500px; margin:auto; background:#000; border-radius:15px; overflow:hidden;">
    <video id="v" autoplay playsinline style="display:none;"></video>
    <canvas id="d" style="width:100%; height:auto; display:block;"></canvas>
    <div id="status" style="position:absolute; bottom:15px; left:0; width:100%; text-align:center; color:#0F0; font-family:monospace; font-weight:bold; font-size:18px; text-shadow:2px 2px #000;">ST_NAME_HERE</div>
</div>
<script src="https://cdn.jsdelivr.net/npm/@mediapipe/face_detection"></script>
<script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils"></script>
<script>
    const v = document.getElementById('v');
    const d = document.getElementById('d');
    const ctx = d.getContext('2d');
    const faceDetection = new FaceDetection({locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_detection/${file}`});
    faceDetection.setOptions({ model: 'short', minDetectionConfidence: 0.6 });
    faceDetection.onResults((results) => {
        d.width = v.videoWidth; d.height = v.videoHeight;
        ctx.drawImage(v, 0, 0, d.width, d.height);
        if (results.detections.length > 0) {
            const bbox = results.detections[0].boundingBox;
            ctx.strokeStyle = "#00FF00"; ctx.lineWidth = 4;
            ctx.strokeRect(bbox.xCenter * d.width - (bbox.width * d.width / 2),
                           bbox.yCenter * d.height - (bbox.height * d.height / 2),
                           bbox.width * d.width, bbox.height * d.height);
            const c = document.createElement('canvas');
            c.width = 160; c.height = 160;
            c.getContext('2d').drawImage(v, 
                bbox.xCenter * v.videoWidth - (bbox.width * v.videoWidth / 2), 
                bbox.yCenter * v.videoHeight - (bbox.height * v.videoHeight / 2), 
                bbox.width * v.videoWidth, bbox.height * v.videoHeight,
                0, 0, 160, 160
            );
            window.parent.postMessage({type: "streamlit:setComponentValue", value: c.toDataURL('image/jpeg', 0.5)}, "*");
        }
    });
    const camera = new Camera(v, {onFrame: async () => { await faceDetection.send({image: v}); }, width: 640, height: 480});
    camera.start();
</script>
"""

# --- PAGE LOGIC ---
nav = st.sidebar.radio("Navigation", ["Register", "Live Feed", "Logs"])

if nav == "Register":
    st.header("👤 Face Registration")
    with st.form("reg"):
        name = st.text_input("NAME").upper()
        file = st.file_uploader("Upload", type=['jpg', 'png'])
        if st.form_submit_button("Save"):
            if name and file:
                with open(os.path.join(DB_FOLDER, f"{name}.jpg"), "wb") as f:
                    f.write(file.getbuffer())
                st.cache_resource.clear(); st.success(f"Registered {name}")

elif nav == "Live Feed":
    st.header("📹 InsightFace Scanner")
    app = load_insightface()
    db = load_face_db()
    
    current_status = st.session_state.get("last_id", "SEARCHING...")
    
    col_v, col_s = st.columns([2, 1])
    with col_v:
        img_data = st.components.v1.html(JS_CODE.replace("ST_NAME_HERE", current_status), height=450)

    with col_s:
        if img_data:
            try:
                encoded = str(img_data).split(",")[1]
                nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                faces = app.get(frame)
                if faces:
                    feat = faces[0].normed_embedding
                    best_name = "UNKNOWN"
                    max_sim = 0
                    
                    for name, saved_feat in db.items():
                        # Cosine similarity
                        sim = np.dot(feat, saved_feat)
                        if sim > max_sim:
                            max_sim = sim
                            best_name = name
                    
                    if max_sim > 0.4: # Accuracy threshold
                        st.session_state.last_id = f"HI, {best_name}"
                        st.metric("Identity", best_name, f"{int(max_sim*100)}% Match")
                else:
                    st.info("No face in frame...")
            except: st.error("Processing...")
