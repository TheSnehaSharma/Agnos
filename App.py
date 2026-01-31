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

for folder in [DB_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

st.set_page_config(page_title="Iron-Vision Biometric", layout="wide")

# --- 2. AI ENGINE (Pre-compiled ONNX) ---
@st.cache_resource
def load_ai():
    # 'buffalo_s' is the fastest pre-compiled model
    app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(320, 320))
    return app

@st.cache_resource
def load_face_db():
    engine = load_ai()
    db = {}
    for file in os.listdir(DB_FOLDER):
        if file.endswith(('.jpg', '.png', '.jpeg')):
            img = cv2.imread(os.path.join(DB_FOLDER, file))
            faces = engine.get(img)
            if faces:
                db[file.split('.')[0]] = faces[0].normed_embedding
    return db

# --- 3. THE HIDDEN BRIDGE (CSS + JS) ---
# This hides the text input so it looks professional
st.markdown("""
    <style>
    div[data-testid="stTextInput"] {
        display: none !important;
    }
    </style>
""", unsafe_allow_html=True)

JS_BRIDGE = """
<div style="background:#000; border-radius:15px; padding:10px; text-align:center;">
    <video id="v" autoplay playsinline style="width:100%; max-width:320px; border-radius:10px;"></video>
    <canvas id="c" style="display:none;"></canvas>
    <div id="msg" style="color:#0F0; font-family:monospace; margin-top:10px; font-size:12px;">SYSTEM: ACTIVE</div>
</div>

<script>
    const v = document.getElementById('v');
    const c = document.getElementById('c');
    const ctx = c.getContext('2d');
    const msg = document.getElementById('msg');

    navigator.mediaDevices.getUserMedia({ video: { width: 320, height: 240 } })
        .then(s => { v.srcObject = s; })
        .catch(e => { msg.innerText = "ERROR: " + e.name; });

    function sendToPython() {
        if (v.videoWidth > 0) {
            c.width = 160; c.height = 120; // Lower res for high speed
            ctx.drawImage(v, 0, 0, 160, 120);
            const data = c.toDataURL('image/jpeg', 0.4);
            
            // Find the hidden Streamlit input and inject data
            const inputs = window.parent.document.querySelectorAll('input');
            for (let i of inputs) {
                if (i.ariaLabel === "image_bridge") {
                    i.value = data;
                    i.dispatchEvent(new Event('input', { bubbles: true }));
                    msg.innerText = "SYNCING: " + data.length + " bytes";
                    break;
                }
            }
        }
    }
    setInterval(sendToPython, 1500); // Process every 1.5 seconds
</script>
"""

# --- 4. NAVIGATION ---
page = st.sidebar.radio("Navigation", ["Live Scanner", "Register Face", "Attendance Log"])

# --- PAGE: LIVE SCANNER ---
if page == "Live Scanner":
    st.header("📹 Real-time Biometric Scanner")
    
    col_v, col_s = st.columns([1, 1])
    
    with col_v:
        # Render the Camera
        st.components.v1.html(JS_BRIDGE, height=320)
        # The Hidden Input (JS targets this via the aria-label/help)
        img_data = st.text_input("bridge", label_visibility="collapsed", key="image_bridge", help="image_bridge")

    with col_s:
        st.subheader("Identification Status")
        if img_data and len(img_data) > 1000:
            try:
                # Decode
                encoded = img_data.split(",")[1]
                nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # AI Recognition
                engine = load_ai()
                db = load_face_db()
                faces = engine.get(frame)
                
                if faces:
                    feat = faces[0].normed_embedding
                    best_name = "UNKNOWN"
                    max_sim = 0
                    
                    for name, saved_feat in db.items():
                        sim = np.dot(feat, saved_feat) # Cosine similarity
                        if sim > max_sim:
                            max_sim = sim
                            best_name = name
                    
                    if max_sim > 0.45: # Confidence threshold
                        st.metric("Identity", best_name, f"{int(max_sim*100)}% Match")
                        st.image(frame, width=150, caption="Captured Face")
                        
                        # Save Log
                        logs = []
                        if os.path.exists(PKL_LOG):
                            with open(PKL_LOG, "rb") as f: logs = pickle.load(f)
                        today = datetime.now().strftime("%Y-%m-%d")
                        if not any(e['Name'] == best_name and e['Date'] == today for e in logs):
                            logs.append({"Name": best_name, "Time": datetime.now().strftime("%H:%M:%S"), "Date": today})
                            with open(PKL_LOG, "wb") as f: pickle.dump(logs, f)
                            st.toast(f"✅ Logged: {best_name}")
                    else:
                        st.warning("Face Unknown")
                else:
                    st.info("Searching for face...")
            except Exception as e:
                st.error("Stabilizing AI...")
        else:
            st.info("Awaiting Sensor Handshake...")

# --- PAGE: REGISTER ---
elif page == "Register Face":
    st.header("👤 Register New User")
    with st.form("reg"):
        name = st.text_input("FULL NAME").upper()
        file = st.file_uploader("Upload Image", type=['jpg', 'png'])
        if st.form_submit_button("Save to Database"):
            if name and file:
                with open(os.path.join(DB_FOLDER, f"{name}.jpg"), "wb") as f:
                    f.write(file.getbuffer())
                st.cache_resource.clear() # Force re-encoding of database
                st.success(f"Successfully Registered {name}")

    st.markdown("---")
    st.subheader("🗂️ Database Manager")
    for f in [x for x in os.listdir(DB_FOLDER) if x.endswith(('.jpg', '.png'))]:
        c1, c2 = st.columns([4, 1])
        c1.write(f"✅ {f.split('.')[0]}")
        if c2.button("Delete", key=f):
            os.remove(os.path.join(DB_FOLDER, f))
            st.cache_resource.clear()
            st.rerun()

# --- PAGE: LOGS ---
elif page == "Attendance Log":
    st.header("📊 Attendance Records")
    if os.path.exists(PKL_LOG):
        with open(PKL_LOG, "rb") as f: data = pickle.load(f)
        st.table(pd.DataFrame(data))
        if st.button("Clear Logs"):
            os.remove(PKL_LOG); st.rerun()
    else:
        st.info("No records found.")
