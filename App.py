import streamlit as st
import pandas as pd
import cv2
import numpy as np
import base64
import os
import pickle
import face_recognition
from datetime import datetime

# --- DIRECTORY SETUP ---
DB_FOLDER = "registered_faces"
PKL_LOG = "attendance_data.pkl"

for folder in [DB_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

st.set_page_config(page_title="Agnos Cloud Biometric", layout="wide")

# --- DLIB DATABASE ENGINE ---
@st.cache_resource
def load_face_encodings():
    """Generates 128-d face vectors for all registered images once."""
    known_encodings = []
    known_names = []
    
    if not os.listdir(DB_FOLDER):
        return [], []

    for file in os.listdir(DB_FOLDER):
        if file.endswith(('.jpg', '.png', '.jpeg')):
            path = os.path.join(DB_FOLDER, file)
            # Load and encode
            img = face_recognition.load_image_file(path)
            encs = face_recognition.face_encodings(img)
            if encs:
                known_encodings.append(encs[0])
                known_names.append(file.split('.')[0])
    return known_encodings, known_names

# --- JAVASCRIPT: THE HYBRID BRIDGE ---
# Uses MediaPipe for browser-side rectangles
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

    const faceDetection = new FaceDetection({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_detection/${file}`
    });

    faceDetection.setOptions({ model: 'short', minDetectionConfidence: 0.6 });

    faceDetection.onResults((results) => {
        d.width = v.videoWidth; d.height = v.videoHeight;
        ctx.drawImage(v, 0, 0, d.width, d.height);
        
        if (results.detections.length > 0) {
            const bbox = results.detections[0].boundingBox;
            
            // Draw Box
            ctx.strokeStyle = "#00FF00"; ctx.lineWidth = 4;
            ctx.strokeRect(bbox.xCenter * d.width - (bbox.width * d.width / 2),
                           bbox.yCenter * d.height - (bbox.height * d.height / 2),
                           bbox.width * d.width, bbox.height * d.height);

            // Send 160x160 Crop to Python
            const c = document.createElement('canvas');
            c.width = 160; c.height = 160;
            c.getContext('2d').drawImage(v, 
                bbox.xCenter * v.videoWidth - (bbox.width * v.videoWidth / 2), 
                bbox.yCenter * v.videoHeight - (bbox.height * v.videoHeight / 2), 
                bbox.width * v.videoWidth, bbox.height * v.videoHeight,
                0, 0, 160, 160
            );
            
            window.parent.postMessage({
                type: "streamlit:setComponentValue",
                value: c.toDataURL('image/jpeg', 0.5)
            }, "*");
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
nav = st.sidebar.radio("Go to", ["Register User", "Live Attendance", "Attendance Logs"])

if nav == "Register User":
    st.header("👤 Face Registration")
    with st.form("reg"):
        name = st.text_input("FULL NAME").upper()
        file = st.file_uploader("Upload Profile Image", type=['jpg', 'png', 'jpeg'])
        if st.form_submit_button("Save to Database"):
            if name and file:
                with open(os.path.join(DB_FOLDER, f"{name}.jpg"), "wb") as f:
                    f.write(file.getbuffer())
                st.cache_resource.clear() # Force re-encoding
                st.success(f"Successfully Registered {name}")

    st.markdown("---")
    st.subheader("🗂️ Database Manager")
    db_files = [f for f in os.listdir(DB_FOLDER) if f.endswith(('.jpg', '.png'))]
    for f in db_files:
        c1, c2 = st.columns([4, 1])
        c1.write(f"✅ {f.split('.')[0]}")
        if c2.button("Delete", key=f):
            os.remove(os.path.join(DB_FOLDER, f))
            st.cache_resource.clear()
            st.rerun()

elif nav == "Live Attendance":
    st.header("📹 Live Scanner")
    known_enc, known_names = load_face_encodings()
    
    current_status = st.session_state.get("last_id", "SEARCHING...")
    final_js = JS_CODE.replace("ST_NAME_HERE", current_status)
    
    col_v, col_s = st.columns([2, 1])
    with col_v:
        img_data = st.components.v1.html(final_js, height=450)

    with col_s:
        st.subheader("System Status")
        if not known_enc:
            st.warning("No users in database. Please register first.")
        elif img_data:
            try:
                # 1. Decode & Fix Padding
                encoded = str(img_data).split(",")[1]
                encoded += "=" * ((4 - len(encoded) % 4) % 4)
                nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 2. Identify
                curr_enc = face_recognition.face_encodings(rgb)
                if curr_enc:
                    matches = face_recognition.compare_faces(known_enc, curr_enc[0], tolerance=0.5)
                    if True in matches:
                        name = known_names[matches.index(True)]
                        st.session_state.last_id = f"VERIFIED: {name}"
                        st.metric("Detected", name)
                        
                        # Save Log
                        logs = []
                        if os.path.exists(PKL_LOG):
                            with open(PKL_LOG, "rb") as f: logs = pickle.load(f)
                        today = datetime.now().strftime("%Y-%m-%d")
                        if not any(e['Name'] == name and e['Date'] == today for e in logs):
                            logs.append({"Name": name, "Time": datetime.now().strftime("%H:%M:%S"), "Date": today})
                            with open(PKL_LOG, "wb") as f: pickle.dump(logs, f)
                            st.toast(f"✅ Attendance Logged: {name}")
                    else:
                        st.session_state.last_id = "UNKNOWN USER"
            except Exception as e:
                st.error("Processing Frame...")

elif nav == "Attendance Logs":
    st.header("📊 Logs")
    if os.path.exists(PKL_LOG):
        with open(PKL_LOG, "rb") as f: data = pickle.load(f)
        if data:
            st.table(pd.DataFrame(data))
            if st.button("Clear Records"):
                os.remove(PKL_LOG); st.rerun()
        else: st.info("No records today.")
