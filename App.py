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

# --- 1. SETUP & CONFIG ---
DB_FOLDER = "registered_faces"
PKL_LOG = "attendance_data.pkl"
MODEL_NAME = "Facenet512"

if not os.path.exists(DB_FOLDER):
    os.makedirs(DB_FOLDER)

st.set_page_config(page_title="Agnos Biometric Pro", layout="wide")

# --- 2. AI ENGINE ---
@st.cache_resource
def load_ai():
    try:
        DeepFace.build_model(MODEL_NAME)
        return True
    except Exception as e:
        return str(e)

load_ai()

# --- 3. THE JS BRIDGE (MEDIA PIPE + URL BRIDGE) ---
# This script draws rectangles in the browser AND sends data to Python
JS_CODE = f"""
<div style="position:relative; width:100%; max-width:500px; margin:auto; background:#000; border-radius:20px; overflow:hidden;">
    <video id="v" autoplay playsinline style="display:none;"></video>
    <canvas id="d" style="width:100%; height:auto; display:block;"></canvas>
    <div id="name_label" style="position:absolute; bottom:15px; left:0; width:100%; text-align:center; color:#0F0; font-family:monospace; font-weight:bold; font-size:22px; text-shadow:2px 2px #000;">ST_NAME_HERE</div>
</div>

<script src="https://cdn.jsdelivr.net/npm/@mediapipe/face_detection"></script>
<script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils"></script>

<script>
    const v = document.getElementById('v');
    const d = document.getElementById('d');
    const ctx = d.getContext('2d');

    const faceDetection = new FaceDetection({{
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_detection/${{file}}`
    }});

    faceDetection.setOptions({{ model: 'short', minDetectionConfidence: 0.6 }});

    faceDetection.onResults((results) => {{
        d.width = v.videoWidth; d.height = v.videoHeight;
        ctx.drawImage(v, 0, 0, d.width, d.height);
        
        if (results.detections.length > 0) {{
            const bbox = results.detections[0].boundingBox;
            
            // 1. LIVE RECTANGLE
            ctx.strokeStyle = "#00FF00"; ctx.lineWidth = 4;
            const x = bbox.xCenter * d.width - (bbox.width * d.width / 2);
            const y = bbox.yCenter * d.height - (bbox.height * d.height / 2);
            ctx.strokeRect(x, y, bbox.width * d.width, bbox.height * d.height);

            // 2. SEND TO PYTHON (URL BRIDGE)
            const c = document.createElement('canvas');
            c.width = 160; c.height = 160;
            c.getContext('2d').drawImage(v, 
                bbox.xCenter * v.videoWidth - (bbox.width * v.videoWidth / 2), 
                bbox.yCenter * v.videoHeight - (bbox.height * v.videoHeight / 2), 
                bbox.width * v.videoWidth, bbox.height * v.videoHeight,
                0, 0, 160, 160
            );
            
            // Using setComponentValue as the primary bridge
            window.parent.postMessage({{
                type: "streamlit:setComponentValue",
                value: c.toDataURL('image/jpeg', 0.4)
            }}, "*");
        }}
    }});

    const camera = new Camera(v, {{
        onFrame: async () => {{ await faceDetection.send({{image: v}}); }},
        width: 640, height: 480
    }});
    camera.start();
</script>
"""

# --- 4. NAVIGATION ---
page = st.sidebar.radio("Navigate", ["Register Face", "Live Attendance", "Log History"])

if page == "Register Face":
    st.header("👤 Face Registration")
    with st.form("reg"):
        name = st.text_input("NAME").upper()
        file = st.file_uploader("Upload Image", type=['jpg','png'])
        if st.form_submit_button("Save User"):
            if name and file:
                with open(os.path.join(DB_FOLDER, f"{name}.jpg"), "wb") as f:
                    f.write(file.getbuffer())
                st.success(f"Registered {name}")

    st.markdown("---")
    st.subheader("🗂️ Manage Database")
    for f in [f for f in os.listdir(DB_FOLDER) if f.endswith(('.jpg', '.png'))]:
        c1, c2 = st.columns([4, 1])
        c1.write(f"✅ {f.split('.')[0]}")
        if c2.button("Delete", key=f"del_{f}"):
            os.remove(os.path.join(DB_FOLDER, f)); st.rerun()

elif page == "Live Attendance":
    st.header("📹 Biometric Feed")
    
    # Injects the current detected name into the JS
    current_name = st.session_state.get("last_name", "SEARCHING...")
    final_js = JS_CODE.replace("ST_NAME_HERE", current_name)

    col_v, col_s = st.columns([2, 1])
    
    with col_v:
        # This is the bridge capture
        img_data = st.components.v1.html(final_js, height=450)

    with col_s:
        st.subheader("System Status")
        if img_data:
            try:
                encoded = str(img_data).split(",")[1]
                nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                res = DeepFace.find(img_path=frame, db_path=DB_FOLDER, 
                                   model_name=MODEL_NAME, enforce_detection=False, 
                                   detector_backend="skip", silent=True)
                
                if len(res) > 0 and not res[0].empty:
                    m_name = os.path.basename(res[0].iloc[0]['identity']).split('.')[0]
                    st.session_state.last_name = f"VERIFIED: {m_name}"
                    st.metric("Identity", m_name)
                    
                    # Log Attendance
                    logs = []
                    if os.path.exists(PKL_LOG):
                        with open(PKL_LOG, "rb") as f: logs = pickle.load(f)
                    today = datetime.now().strftime("%Y-%m-%d")
                    if not any(e['Name'] == m_name and e['Date'] == today for e in logs):
                        logs.append({"Name": m_name, "Time": datetime.now().strftime("%H:%M:%S"), "Date": today})
                        with open(PKL_LOG, "wb") as f: pickle.dump(logs, f)
                        st.toast(f"✅ Logged {m_name}")
                else:
                    st.session_state.last_name = "UNKNOWN USER"
            except:
                st.error("Engine Resyncing...")
        else:
            st.info("Awaiting Handshake...")

elif page == "Log History":
    st.header("📊 Attendance Records")
    if os.path.exists(PKL_LOG):
        with open(PKL_LOG, "rb") as f: data = pickle.load(f)
        st.table(pd.DataFrame(data))
    else: st.info("No records.")
