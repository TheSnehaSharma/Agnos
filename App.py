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
    try:
        DeepFace.build_model(MODEL_NAME)
        return True
    except Exception as e:
        return str(e)

with st.sidebar:
    st.title("System Health")
    status_ai = load_ai_models()
    if status_ai is True:
        st.success("AI Engine: Ready")
    else:
        st.error(f"AI Engine Error: {status_ai}")

# --- PERSISTENT STATE ---
if "detected_name" not in st.session_state:
    st.session_state.detected_name = "INITIALIZING..."
if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0

# --- HELPERS ---
def robust_decompress(compressed_data):
    try:
        # 1. Decode URL and Base64
        unquoted = urllib.parse.unquote(str(compressed_data))
        raw_bytes = base64.b64decode(unquoted)
        # 2. Gzip Decompress
        decompressed = gzip.decompress(raw_bytes)
        # 3. Decode to CV2
        nparr = np.frombuffer(decompressed, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        return None

def save_log(name):
    logs = []
    if os.path.exists(PKL_LOG):
        with open(PKL_LOG, "rb") as f:
            try: logs = pickle.load(f)
            except: logs = []
    
    today = datetime.now().strftime("%Y-%m-%d")
    # Only log once per minute to prevent file bloat
    now_min = datetime.now().strftime("%H:%M")
    if not any(e['Name'] == name and e['Date'] == today and e['Time'].startswith(now_min) for e in logs):
        logs.append({"Name": name, "Time": datetime.now().strftime("%H:%M:%S"), "Date": today})
        with open(PKL_LOG, "wb") as f:
            pickle.dump(logs, f)
        return True
    return False

# --- JS: VIDEO + OVERLAY ---
JS_CODE = f"""
<div style="position:relative; width:100%; max-width:400px; margin:auto; background:#000; border-radius:15px; overflow:hidden; border:2px solid #333;">
    <video id="v" autoplay playsinline style="display:none;"></video>
    <canvas id="display_canvas" style="width:100%; height:auto; display:block;"></canvas>
    <div id="name_label" style="position:absolute; bottom:15px; left:0; width:100%; text-align:center; color:#0F0; font-family:monospace; font-weight:bold; font-size:20px; text-shadow:2px 2px #000;">{{name_placeholder}}</div>
</div>

<script src="https://cdn.jsdelivr.net/npm/@mediapipe/face_detection"></script>
<script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/pako/2.1.0/pako.min.js"></script>

<script>
    const v = document.getElementById('v');
    const d_can = document.getElementById('display_canvas');
    const d_ctx = d_can.getContext('2d');

    const faceDetection = new FaceDetection({{
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_detection/${{file}}`
    }});

    faceDetection.setOptions({{ model: 'short', minDetectionConfidence: 0.6 }});

    faceDetection.onResults((results) => {{
        d_can.width = v.videoWidth; d_can.height = v.videoHeight;
        d_ctx.drawImage(v, 0, 0, d_can.width, d_can.height);
        
        if (results.detections.length > 0) {{
            const bbox = results.detections[0].boundingBox;
            
            // Draw Interactive Box
            d_ctx.strokeStyle = "#00FF00";
            d_ctx.lineWidth = 3;
            const x = bbox.xCenter * d_can.width - (bbox.width * d_can.width / 2);
            const y = bbox.yCenter * d_can.height - (bbox.height * d_can.height / 2);
            d_ctx.strokeRect(x, y, bbox.width * d_can.width, bbox.height * d_can.height);

            // Export Face for Python
            const c_can = document.createElement('canvas');
            c_can.width = 160; c_can.height = 160;
            c_can.getContext('2d').drawImage(v, 
                bbox.xCenter * v.videoWidth - (bbox.width * v.videoWidth / 2), 
                bbox.yCenter * v.videoHeight - (bbox.height * v.videoHeight / 2), 
                bbox.width * v.videoWidth, bbox.height * v.videoHeight,
                0, 0, 160, 160
            );
            
            const dataURL = c_can.toDataURL('image/jpeg', 0.4);
            const b64 = dataURL.split(',')[1];
            const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
            const compressed = btoa(String.fromCharCode.apply(null, pako.gzip(bytes)));
            
            window.parent.postMessage({{
                type: "streamlit:setComponentValue", 
                value: encodeURIComponent(compressed)
            }}, "*");
        }
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
    
    with st.form("registration_form", clear_on_submit=True):
        name = st.text_input("Name (e.g., JOHN DOE)").upper()
        file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
        submit = st.form_submit_button("Register User")
        
        if submit and name and file:
            path = os.path.join(DB_FOLDER, f"{name}.jpg")
            with open(path, "wb") as f:
                f.write(file.getbuffer())
            # Clean AI cache
            for p in [f for f in os.listdir(DB_FOLDER) if f.endswith('.pkl')]:
                os.remove(os.path.join(DB_FOLDER, p))
            st.success(f"Registered {name}")

    st.markdown("---")
    st.subheader("🗂️ Database Management")
    db_list = [f for f in os.listdir(DB_FOLDER) if f.endswith(('.jpg', '.png'))]
    if db_list:
        for f in db_list:
            c1, c2 = st.columns([4, 1])
            c1.write(f"✅ {f.split('.')[0]}")
            if c2.button("Delete", key=f"del_{f}"):
                os.remove(os.path.join(DB_FOLDER, f))
                st.rerun()
    else:
        st.info("Database is empty.")

elif page == "Live Feed":
    st.header("📹 Biometric Scanner")
    
    # Pre-check database
    if not any(f.endswith(('.jpg', '.png')) for f in os.listdir(DB_FOLDER)):
        st.warning("⚠️ Database is empty. Please register a face first.")
        st.stop()

    col_v, col_s = st.columns([2, 1])
    
    with col_v:
        # Capture Handshake
        img_data = st.components.v1.html(JS_CODE, height=360)

    with col_s:
        st.subheader("System Status")
        if img_data:
            st.session_state.frame_count += 1
            frame = robust_decompress(img_data)
            
            if frame is not None:
                try:
                    # Recognition
                    res = DeepFace.find(img_path=frame, db_path=DB_FOLDER, 
                                       model_name=MODEL_NAME, enforce_detection=False, 
                                       detector_backend="skip", silent=True)
                    
                    if len(res) > 0 and not res[0].empty:
                        m_name = os.path.basename(res[0].iloc[0]['identity']).split('.')[0]
                        dist = res[0].iloc[0]['distance']
                        acc = max(0, int((1 - dist/0.38) * 100))
                        
                        if acc > 30:
                            st.session_state.detected_name = f"VERIFIED: {m_name}"
                            st.metric("Identity", m_name, f"{acc}% Match")
                            if save_log(m_name): st.toast(f"✅ Attendance Logged: {m_name}")
                        else:
                            st.session_state.detected_name = "UNKNOWN USER"
                            st.warning("Access Denied: Unknown")
                    else:
                        st.session_state.detected_name = "SEARCHING..."
                        st.info("Searching database...")
                except Exception as e:
                    st.session_state.detected_name = "ENGINE ERROR"
                    st.error(f"Logic Error: {e}")
            else:
                st.error("Gzip Packet Corrupted")
        else:
            st.info("Waiting for MediaPipe Handshake...")
            st.session_state.detected_name = "AWAITING CAMERA..."

elif page == "Log History":
    st.header("📊 Attendance Log")
    if os.path.exists(PKL_LOG):
        with open(PKL_LOG, "rb") as f:
            data = pickle.load(f)
        if data:
            st.table(pd.DataFrame(data))
            if st.button("🔥 WIPE ALL LOGS"):
                os.remove(PKL_LOG)
                st.rerun()
        else:
            st.info("No records found.")
    else:
        st.info("Log file not initialized.")
