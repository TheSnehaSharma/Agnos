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
    DeepFace.build_model(MODEL_NAME)
    return True

load_ai_models()

# --- STATE ---
if "detected_name" not in st.session_state:
    st.session_state.detected_name = "SEARCHING..."

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
    # Log once per day per person
    today = datetime.now().strftime("%Y-%m-%d")
    if not any(e['Name'] == name and e['Date'] == today for e in logs):
        logs.append({"Name": name, "Time": datetime.now().strftime("%H:%M:%S"), "Date": today})
        with open(PKL_LOG, "wb") as f:
            pickle.dump(logs, f)
        return True
    return False

# --- JS: VIDEO + OVERLAY + NAME SYNC ---
JS_CODE = f"""
<div style="position:relative; width:100%; max-width:400px; margin:auto; background:#000; border-radius:15px; overflow:hidden;">
    <video id="v" autoplay playsinline style="display:none;"></video>
    <canvas id="display_canvas" style="width:100%; height:auto; display:block;"></canvas>
    <div id="name_label" style="position:absolute; bottom:20px; left:0; width:100%; text-align:center; color:#0F0; font-family:monospace; font-weight:bold; font-size:18px; text-shadow:2px 2px #000;">{{name_placeholder}}</div>
</div>

<script src="https://cdn.jsdelivr.net/npm/@mediapipe/face_detection"></script>
<script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/pako/2.1.0/pako.min.js"></script>

<script>
    const v = document.getElementById('v');
    const d_can = document.getElementById('display_canvas');
    const d_ctx = d_can.getContext('2d');
    const label = document.getElementById('name_label');

    const faceDetection = new FaceDetection({{
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_detection/${{file}}`
    }});

    faceDetection.setOptions({{ model: 'short', minDetectionConfidence: 0.6 }});

    faceDetection.onResults((results) => {{
        d_can.width = v.videoWidth; d_can.height = v.videoHeight;
        d_ctx.drawImage(v, 0, 0, d_can.width, d_can.height);
        
        if (results.detections.length > 0) {{
            const bbox = results.detections[0].boundingBox;
            
            // Draw Box
            d_ctx.strokeStyle = "#00FF00";
            d_ctx.lineWidth = 3;
            const x = bbox.xCenter * d_can.width - (bbox.width * d_can.width / 2);
            const y = bbox.yCenter * d_can.height - (bbox.height * d_can.height / 2);
            const w = bbox.width * d_can.width;
            const h = bbox.height * d_can.height;
            d_ctx.strokeRect(x, y, w, h);

            // Gzip & Send Crop
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
            
            window.parent.postMessage({{
                type: "streamlit:setComponentValue", 
                value: encodeURIComponent(compressed)
            }}, "*");
        }}
    }});

    const camera = new Camera(v, {{
        onFrame: async () => {{ await faceDetection.send({{image: v}}); }},
        width: 400, height: 300
    }});
    camera.start();
</script>
""".replace("{name_placeholder}", st.session_state.detected_name)

# --- UI NAVIGATION ---
page = st.sidebar.radio("Navigate", ["Register", "Live Feed", "Log History"])

if page == "Register":
    st.header("👤 Face Registration")
    
    # 1. Registration Form
    with st.form("reg_form", clear_on_submit=True):
        name = st.text_input("Full Name").upper()
        file = st.file_uploader("Upload Profile Image", type=['jpg', 'jpeg', 'png'])
        if st.form_submit_button("Save to Database"):
            if name and file:
                with open(os.path.join(DB_FOLDER, f"{name}.jpg"), "wb") as f:
                    f.write(file.getbuffer())
                # Wipe DeepFace metadata
                for p in [f for f in os.listdir(DB_FOLDER) if f.endswith('.pkl')]:
                    os.remove(os.path.join(DB_FOLDER, p))
                st.success(f"Successfully Registered {name}")
            else:
                st.error("Please provide both name and image.")

    st.markdown("---")
    
    # 2. Database Management Section
    st.subheader("🗂️ Manage Registered Users")
    reg_files = [f for f in os.listdir(DB_FOLDER) if f.endswith(('.jpg', '.png', '.jpeg'))]
    if not reg_files:
        st.info("The database is currently empty.")
    else:
        for f in reg_files:
            c1, c2 = st.columns([4, 1])
            c1.write(f"✅ {f.split('.')[0]}")
            if c2.button("Delete User", key=f"del_{f}"):
                os.remove(os.path.join(DB_FOLDER, f))
                st.rerun()

elif page == "Live Feed":
    st.header("📹 Live Identification Feed")
    col_v, col_s = st.columns([2, 1])
    
    with col_v:
        img_data = st.components.v1.html(JS_CODE, height=350)

    with col_s:
        st.subheader("System Status")
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
                    st.session_state.detected_name = "RESYNCING..."
        else:
            st.info("Looking for face...")

elif page == "Log History":
    st.header("📊 Attendance Log")
    if os.path.exists(PKL_LOG):
        with open(PKL_LOG, "rb") as f: 
            data = pickle.load(f)
        if data:
            df = pd.DataFrame(data)
            st.table(df)
            if st.button("🔥 CLEAR ALL LOGS"):
                os.remove(PKL_LOG)
                st.rerun()
        else: st.info("No attendance recorded yet.")
    else: st.info("No log file found.")
