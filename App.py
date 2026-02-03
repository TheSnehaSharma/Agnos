import streamlit as st
import pandas as pd
import cv2
import numpy as np
import base64
import os
import pickle
from insightface.app import FaceAnalysis
from datetime import datetime

# --- 1. SETUP & DIRECTORIES ---
DB_FOLDER = "registered_faces"
PKL_LOG = "attendance_data.pkl"

if not os.path.exists(DB_FOLDER):
    os.makedirs(DB_FOLDER)

st.set_page_config(page_title="Iron-Vision Edge", layout="wide")

# --- 2. AI ENGINE (Optimized for Embeddings) ---
@st.cache_resource
def load_ai():
    # buffalo_s is fastest; we set det_size small because we are sending cropped faces
    app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(160, 160)) 
    return app

@st.cache_resource
def load_face_db():
    engine = load_ai()
    db = {}
    if not os.listdir(DB_FOLDER):
        return db
    for file in os.listdir(DB_FOLDER):
        if file.endswith(('.jpg', '.png', '.jpeg')):
            img = cv2.imread(os.path.join(DB_FOLDER, file))
            faces = engine.get(img)
            if faces:
                db[file.split('.')[0]] = faces[0].normed_embedding
    return db

# --- 3. THE HYBRID BRIDGE (MediaPipe + JS) ---
# This script detects the face locally and only sends a 128x128 crop to Python
JS_BRIDGE = """
<div style="position: relative; width: 320px; margin: auto;">
    <video id="v" autoplay playsinline style="width: 320px; height: 240px; border-radius: 10px; background: #000;"></video>
    <canvas id="overlay" width="320" height="240" style="position: absolute; top: 0; left: 0;"></canvas>
    <div id="status" style="color: #0F0; font-family: monospace; font-size: 10px; margin-top: 5px;">LOADED: EDGE_AI_ACTIVE</div>
</div>

<script src="https://cdn.jsdelivr.net/npm/@mediapipe/face_detection"></script>
<script>
    const video = document.getElementById('v');
    const overlay = document.getElementById('overlay');
    const ctx = overlay.getContext('2d');
    const status = document.getElementById('status');
    
    // Hidden canvas for cropping
    const cropCanvas = document.createElement('canvas');
    const cropCtx = cropCanvas.getContext('2d');

    const faceDetection = new FaceDetection({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_detection/${file}`
    });

    faceDetection.setOptions({ model: 'short', minDetectionConfidence: 0.7 });

    faceDetection.onResults(results => {
        ctx.clearRect(0, 0, overlay.width, overlay.height);
        
        if (results.detections.length > 0) {
            const detection = results.detections[0].boundingBox;
            
            // Draw Box on Overlay (Local UI)
            ctx.strokeStyle = "#0F0";
            ctx.lineWidth = 2;
            ctx.strokeRect(detection.xCenter * 320 - (detection.width * 320 / 2), 
                           detection.yCenter * 240 - (detection.height * 240 / 2), 
                           detection.width * 320, detection.height * 240);

            // CROP & SEND
            const x = Math.max(0, (detection.xCenter - detection.width/2) * video.videoWidth);
            const y = Math.max(0, (detection.yCenter - detection.height/2) * video.videoHeight);
            const w = detection.width * video.videoWidth;
            const h = detection.height * video.videoHeight;

            cropCanvas.width = 128;
            cropCanvas.height = 128;
            cropCtx.drawImage(video, x, y, w, h, 0, 0, 128, 128);
            
            const data = cropCanvas.toDataURL('image/jpeg', 0.6);
            const inputs = window.parent.document.querySelectorAll('input');
            for (let i of inputs) {
                if (i.ariaLabel === "image_bridge") {
                    i.value = data;
                    i.dispatchEvent(new Event('input', { bubbles: true }));
                    break;
                }
            }
        }
    });

    navigator.mediaDevices.getUserMedia({ video: { width: 320, height: 240 } })
        .then(stream => { 
            video.srcObject = stream;
            async function predict() {
                await faceDetection.send({image: video});
                setTimeout(predict, 1000); // Send 1 face per second to Python
            }
            predict();
        });
</script>
"""

st.markdown("<style>div[data-testid='stTextInput'] { display: none !important; }</style>", unsafe_allow_html=True)

# --- 4. NAVIGATION ---
page = st.sidebar.radio("Navigation", ["Live Scanner", "Register Face", "Attendance Log"])

if page == "Live Scanner":
    st.header("⚡ Real-time Edge Biometrics")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.components.v1.html(JS_BRIDGE, height=300)
        img_data = st.text_input("bridge", label_visibility="collapsed", key="image_bridge", help="image_bridge")

    with col2:
        if img_data and len(img_data) > 1000:
            try:
                # Decode the crop
                encoded = img_data.split(",")[1]
                nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
                face_chip = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                engine = load_ai()
                db = load_face_db()
                faces = engine.get(face_chip)

                if faces:
                    emb = faces[0].normed_embedding
                    best_name = "UNKNOWN"
                    score = 0
                    
                    for name, saved_emb in db.items():
                        sim = np.dot(emb, saved_emb)
                        if sim > score:
                            score = sim
                            best_name = name
                    
                    if score > 0.40:
                        st.metric("Access Granted", best_name, f"{int(score*100)}% Match")
                        st.image(face_chip, width=100, caption="Verified Chip")
                        
                        # Logging
                        logs = []
                        if os.path.exists(PKL_LOG):
                            with open(PKL_LOG, "rb") as f: logs = pickle.load(f)
                        today = datetime.now().strftime("%Y-%m-%d")
                        if not any(e['Name'] == best_name and e['Date'] == today for e in logs):
                            logs.append({"Name": best_name, "Time": datetime.now().strftime("%H:%M:%S"), "Date": today})
                            with open(PKL_LOG, "wb") as f: pickle.dump(logs, f)
                            st.toast(f"Logged: {best_name}")
                    else:
                        st.error("Identity: UNKNOWN")
                else:
                    st.info("Aligning Face...")
            except:
                st.error("Bridge Sync Error")
        else:
            st.info("Awaiting Face Detection...")

elif page == "Register Face":
    st.header("👤 User Registration")
    with st.form("reg"):
        name = st.text_input("NAME").upper().strip()
        file = st.file_uploader("Upload Image", type=['jpg', 'png'])
        if st.form_submit_button("Register"):
            if name and file:
                path = os.path.join(DB_FOLDER, f"{name}.jpg")
                with open(path, "wb") as f: f.write(file.getbuffer())
                load_face_db.clear()
                st.success(f"{name} added to database.")

elif page == "Attendance Log":
    st.header("📊 Records")
    if os.path.exists(PKL_LOG):
        with open(PKL_LOG, "rb") as f: data = pickle.load(f)
        st.dataframe(pd.DataFrame(data), use_container_width=True)
        if st.button("Clear Records"):
            os.remove(PKL_LOG); st.rerun()
