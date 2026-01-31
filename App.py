import streamlit as st
import pandas as pd
import cv2
import numpy as np
import base64
import os
import pickle
from insightface.app import FaceAnalysis
from datetime import datetime

# --- 1. SETUP & CONFIG ---
DB_FOLDER = "registered_faces"
PKL_LOG = "attendance_data.pkl"

for folder in [DB_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

st.set_page_config(page_title="Iron-Vision Biometric", layout="wide")

# --- 2. AI ENGINE ---
@st.cache_resource
def load_ai():
    # 'buffalo_s' is pre-compiled and works without C++ build tools
    app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(320, 320))
    return app

# --- 3. JAVASCRIPT BRIDGE (THE HIDDEN INPUT HACK) ---
def camera_bridge():
    JS_CODE = """
    <div style="background:#000; border-radius:15px; padding:10px; text-align:center;">
        <video id="v" autoplay playsinline style="width:100%; max-width:320px; border-radius:10px;"></video>
        <canvas id="c" style="display:none;"></canvas>
        <div id="msg" style="color:#0F0; font-family:monospace; margin-top:10px;">SYSTEM: INITIALIZING...</div>
    </div>

    <script>
        const v = document.getElementById('v');
        const c = document.getElementById('c');
        const ctx = c.getContext('2d');
        const msg = document.getElementById('msg');

        // Request Camera
        navigator.mediaDevices.getUserMedia({ video: { width: 320, height: 240 } })
            .then(s => { v.srcObject = s; msg.innerText = "SYSTEM: ONLINE"; })
            .catch(e => { msg.innerText = "ERROR: " + e.name; });

        function sendToPython() {
            if (v.videoWidth > 0) {
                c.width = 160; c.height = 120; // Low res for speed
                ctx.drawImage(v, 0, 0, 160, 120);
                const data = c.toDataURL('image/jpeg', 0.4);
                
                // This is the magic: finding the Streamlit text input and "typing" into it
                const inputs = window.parent.document.querySelectorAll('input');
                for (let i of inputs) {
                    if (i.ariaLabel === "hidden_bridge") {
                        i.value = data;
                        i.dispatchEvent(new Event('input', { bubbles: true }));
                        msg.innerText = "SYNCING: " + data.length + " bytes";
                        break;
                    }
                }
            }
        }
        setInterval(sendToPython, 2000); // Sync every 2 seconds
    </script>
    """
    return st.components.v1.html(JS_CODE, height=320)

# --- 4. NAVIGATION ---
page = st.sidebar.radio("Navigate", ["Register Face", "Live Scanner", "Attendance Logs"])

# --- PAGE 1: REGISTER ---
if page == "Register Face":
    st.header("👤 Face Registration")
    with st.form("reg_form", clear_on_submit=True):
        name = st.text_input("FULL NAME").upper()
        file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
        if st.form_submit_button("Save to Database"):
            if name and file:
                path = os.path.join(DB_FOLDER, f"{name}.jpg")
                with open(path, "wb") as f:
                    f.write(file.getbuffer())
                st.success(f"Registered {name} successfully.")

    st.markdown("---")
    st.subheader("🗂️ Database Manager")
    db_files = [f for f in os.listdir(DB_FOLDER) if f.endswith(('.jpg', '.png'))]
    for f in db_files:
        c1, c2 = st.columns([4, 1])
        c1.write(f"✅ {f.split('.')[0]}")
        if c2.button("Delete", key=f):
            os.remove(os.path.join(DB_FOLDER, f))
            st.rerun()

# --- PAGE 2: LIVE SCANNER ---
elif page == "Live Scanner":
    st.header("📹 Live Identification")
    
    col_v, col_s = st.columns([1, 1])
    
    with col_v:
        # 1. The visible camera UI
        camera_bridge()
        
        # 2. The HIDDEN bridge widget
        # The aria-label MUST match the JS selector above
        raw_data = st.text_input("bridge", label_visibility="collapsed", key="hidden_bridge", help="hidden_bridge")

    with col_s:
        st.subheader("AI System Status")
        if raw_data and len(raw_data) > 2000:
            try:
                # Decode
                encoded = raw_data.split(",")[1]
                nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    # Run AI
                    engine = load_ai()
                    faces = engine.get(frame)
                    
                    if faces:
                        st.success(f"🎯 FACE DETECTED!")
                        st.image(frame, width=150)
                        # Here you would add the vector comparison logic
                    else:
                        st.warning("Scanning... (Align your face)")
                else:
                    st.error("Frame Corrupted.")
            except Exception as e:
                st.error(f"Syncing: {e}")
        else:
            st.info("Awaiting Handshake... (Check Camera Permissions)")

# --- PAGE 3: LOGS ---
elif page == "Attendance Logs":
    st.header("📊 History")
    if os.path.exists(PKL_LOG):
        with open(PKL_LOG, "rb") as f: data = pickle.load(f)
        st.table(pd.DataFrame(data))
    else:
        st.info("No records found.")
