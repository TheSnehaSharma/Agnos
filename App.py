import streamlit as st
import pandas as pd
import cv2
import numpy as np
import base64
import os
import pickle
from insightface.app import FaceAnalysis
from datetime import datetime
import streamlit.components.v1 as components

# --- 1. SETUP & DIRECTORIES ---
DB_FOLDER = "registered_faces"
PKL_LOG = "attendance_data.pkl"

if not os.path.exists(DB_FOLDER):
    os.makedirs(DB_FOLDER)

st.set_page_config(page_title="Iron-Vision Edge", layout="wide")

# --- 2. AI ENGINE (Cached) ---
@st.cache_resource
def load_ai():
    app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(160, 160)) 
    return app

@st.cache_resource
def load_face_db():
    engine = load_ai()
    db = {}
    for file in os.listdir(DB_FOLDER):
        if file.lower().endswith(('.jpg', '.png', '.jpeg')):
            img = cv2.imread(os.path.join(DB_FOLDER, file))
            if img is not None:
                faces = engine.get(img)
                if faces:
                    db[file.split('.')[0]] = faces[0].normed_embedding
    return db

# --- 3. THE "IRON-BRIDGE" COMPONENT ---
# This uses Streamlit's messaging API to bypass Cross-Origin blocks
def camera_bridge():
    bridge_html = """
    <div style="position: relative; width: 320px; margin: auto;">
        <video id="v" autoplay playsinline style="width: 320px; height: 240px; border-radius: 10px; background: #000; transform: scaleX(-1);"></video>
        <canvas id="overlay" width="320" height="240" style="position: absolute; top: 0; left: 0; transform: scaleX(-1);"></canvas>
        <div id="status" style="color: #0F0; font-family: monospace; font-size: 10px; margin-top: 5px; text-align: center;">SYSTEM ACTIVE</div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/face_detection"></script>
    <script>
        const video = document.getElementById('v');
        const overlay = document.getElementById('overlay');
        const ctx = overlay.getContext('2d');
        const cropCanvas = document.createElement('canvas');
        const cropCtx = cropCanvas.getContext('2d');

        const faceDetection = new FaceDetection({
          locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_detection/${file}`
        });

        faceDetection.setOptions({ model: 'short', minDetectionConfidence: 0.7 });

        faceDetection.onResults(results => {
            ctx.clearRect(0, 0, overlay.width, overlay.height);
            if (results.detections.length > 0) {
                const face = results.detections[0].boundingBox;
                ctx.strokeStyle = "#00FF00";
                ctx.lineWidth = 2;
                ctx.strokeRect(face.xCenter * 320 - (face.width * 320 / 2), 
                               face.yCenter * 240 - (face.height * 240 / 2), 
                               face.width * 320, face.height * 240);

                // Crop
                const x = (face.xCenter - face.width/2) * video.videoWidth;
                const y = (face.yCenter - face.height/2) * video.videoHeight;
                cropCanvas.width = 128; cropCanvas.height = 128;
                cropCtx.drawImage(video, x, y, face.width * video.videoWidth, face.height * video.videoHeight, 0, 0, 128, 128);
                
                const data = cropCanvas.toDataURL('image/jpeg', 0.5);
                
                // STREAMLIT CLOUD COMPATIBLE HANDSHAKE
                window.parent.postMessage({
                    type: 'streamlit:setComponentValue',
                    value: data
                }, '*');
            }
        });

        navigator.mediaDevices.getUserMedia({ video: { width: 320, height: 240 } })
            .then(stream => { 
                video.srcObject = stream;
                async function predict() {
                    await faceDetection.send({image: video});
                    setTimeout(predict, 1000); 
                }
                predict();
            });
    </script>
    """
    return components.html(bridge_html, height=300)

# --- 4. APP LOGIC ---
page = st.sidebar.radio("Navigation", ["Live Scanner", "Register Face", "Attendance Log"])

if page == "Live Scanner":
    st.header("📹 Biometric Scanner (Cloud Optimized)")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # We catch the data returned by the component
        img_data = camera_bridge()
        # Since components.html doesn't return value directly like custom components, 
        # we still use the text_input trick but with a "Global" selector fix.
        st.markdown('<input type="text" id="bridge_input" style="display:none;">', unsafe_allow_html=True)
        img_data = st.text_input("bridge", key="image_bridge", label_visibility="hidden")

    with col2:
        if img_data and len(img_data) > 1000:
            try:
                encoded = img_data.split(",")[1]
                nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
                face_chip = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                engine = load_ai()
                db = load_face_db()
                faces = engine.get(face_chip)

                if faces:
                    emb = faces[0].normed_embedding
                    best_name, score = "UNKNOWN", 0
                    for name, saved_emb in db.items():
                        sim = np.dot(emb, saved_emb)
                        if sim > score:
                            score, best_name = sim, name
                    
                    if score > 0.45:
                        st.metric("Identity", best_name, f"{int(score*100)}% Match")
                        st.image(face_chip, width=150)
                        
                        logs = []
                        if os.path.exists(PKL_LOG):
                            with open(PKL_LOG, "rb") as f: logs = pickle.load(f)
                        today = datetime.now().strftime("%Y-%m-%d")
                        if not any(e['Name'] == best_name and e['Date'] == today for e in logs):
                            logs.append({"Name": best_name, "Time": datetime.now().strftime("%H:%M:%S"), "Date": today})
                            with open(PKL_LOG, "wb") as f: pickle.dump(logs, f)
                            st.toast(f"✅ Logged: {best_name}")
                    else:
                        st.warning("Face Not Recognized")
            except:
                st.error("Processing...")
        else:
            st.info("Awaiting Face Detection...")

# --- RESTORED REGISTER PAGE ---
elif page == "Register Face":
    st.header("👤 Register New User")
    with st.form("reg"):
        name = st.text_input("FULL NAME").upper().strip()
        file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
        if st.form_submit_button("Save to Database"):
            if name and file:
                with open(os.path.join(DB_FOLDER, f"{name}.jpg"), "wb") as f:
                    f.write(file.getbuffer())
                load_face_db.clear()
                st.success(f"Successfully Registered {name}")

    st.markdown("---")
    st.subheader("🗂️ Database Manager")
    for f in [x for x in os.listdir(DB_FOLDER) if x.lower().endswith(('.jpg', '.png'))]:
        c1, c2 = st.columns([4, 1])
        c1.write(f"✅ {f.split('.')[0]}")
        if c2.button("Delete", key=f):
            os.remove(os.path.join(DB_FOLDER, f))
            load_face_db.clear()
            st.rerun()

elif page == "Attendance Log":
    st.header("📊 Attendance Records")
    if os.path.exists(PKL_LOG):
        with open(PKL_LOG, "rb") as f: data = pickle.load(f)
        st.table(pd.DataFrame(data))
        if st.button("Clear Logs"):
            os.remove(PKL_LOG); st.rerun()
