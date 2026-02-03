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

for folder in [DB_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

st.set_page_config(page_title="Iron-Vision Biometric", layout="wide")

# --- 2. AI ENGINE (Cached) ---
@st.cache_resource
def load_ai():
    app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(320, 320))
    return app

@st.cache_resource
def load_face_db():
    engine = load_ai()
    db = {}
    valid_extensions = ('.jpg', '.png', '.jpeg')
    if os.path.exists(DB_FOLDER):
        for file in os.listdir(DB_FOLDER):
            if file.lower().endswith(valid_extensions):
                img = cv2.imread(os.path.join(DB_FOLDER, file))
                if img is not None:
                    faces = engine.get(img)
                    if faces:
                        db[file.split('.')[0]] = faces[0].normed_embedding
    return db

# --- 3. THE 2X CROP BRIDGE (Cloud Optimized) ---
JS_BRIDGE = """
<div style="position: relative; width: 100%; max-width: 400px; margin: auto; background: #000; border-radius: 15px; overflow: hidden; aspect-ratio: 4/3;">
    <video id="v" autoplay playsinline style="width: 100%; height: 100%; transform: scaleX(-1); object-fit: contain;"></video>
    <canvas id="overlay" width="400" height="300" style="position: absolute; top: 0; left: 0; transform: scaleX(-1); width: 100%; height: 100%;"></canvas>
    <div id="msg" style="position: absolute; bottom: 10px; left: 10px; color:#0F0; font-family:monospace; font-size:12px; background:rgba(0,0,0,0.5); padding: 5px;">SYSTEM: ACTIVE</div>
</div>

<script src="https://cdn.jsdelivr.net/npm/@mediapipe/face_detection"></script>
<script>
    const v = document.getElementById('v');
    const overlay = document.getElementById('overlay');
    const ctx = overlay.getContext('2d');
    const msg = document.getElementById('msg');
    const c = document.createElement('canvas');
    const c_ctx = c.getContext('2d');

    const faceDetection = new FaceDetection({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_detection/${file}`
    });
    faceDetection.setOptions({ model: 'short', minDetectionConfidence: 0.7 });

    faceDetection.onResults(results => {
        ctx.clearRect(0, 0, overlay.width, overlay.height);
        if (results.detections.length > 0) {
            const face = results.detections[0].boundingBox;
            
            // Draw Feedback Box
            ctx.strokeStyle = "#0F0"; ctx.lineWidth = 3;
            ctx.strokeRect(face.xCenter * 400 - (face.width * 400 / 2), 
                           face.yCenter * 300 - (face.height * 300 / 2), 
                           face.width * 400, face.height * 300);

            // --- 2X CROP LOGIC ---
            const w_v = v.videoWidth; const h_v = v.videoHeight;
            const targetW = face.width * 2 * w_v;
            const targetH = face.height * 2 * h_v;
            const targetX = (face.xCenter * w_v) - (targetW / 2);
            const targetY = (face.yCenter * h_v) - (targetH / 2);

            c.width = 160; c.height = 160;
            c_ctx.drawImage(v, targetX, targetY, targetW, targetH, 0, 0, 160, 160);
            const data = c.toDataURL('image/jpeg', 0.6);

            // EXTREME HANDSHAKE: Try every possible way to find the Streamlit input
            const findAndSet = (win) => {
                const inputs = win.document.querySelectorAll('input');
                for (let i of inputs) {
                    if (i.ariaLabel === "image_bridge") {
                        i.value = data;
                        i.dispatchEvent(new Event('input', { bubbles: true }));
                        return true;
                    }
                }
                return false;
            };

            let success = findAndSet(window.parent);
            if (!success) success = findAndSet(window.top);
            if (!success) success = findAndSet(window);
            
            msg.innerText = success ? "SYNCING..." : "SEARCHING FOR BRIDGE...";
        } else {
            msg.innerText = "NO FACE DETECTED";
        }
    });

    navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } })
        .then(s => { 
            v.srcObject = s;
            async function loop() { await faceDetection.send({image: v}); setTimeout(loop, 1000); }
            loop();
        });
</script>
"""

st.markdown("<style>div[data-testid='stTextInput'] { display: none !important; }</style>", unsafe_allow_html=True)

# --- 4. NAVIGATION ---
page = st.sidebar.radio("Navigation", ["Live Scanner", "Register Face", "Attendance Log"])

# --- PAGE: LIVE SCANNER ---
if page == "Live Scanner":
    st.header("📹 Biometric Scanner")
    col_v, col_s = st.columns([1, 1])
    
    with col_v:
        st.components.v1.html(JS_BRIDGE, height=350)
        # JS finds this by the aria-label matching the label string "image_bridge"
        img_data = st.text_input("image_bridge", key="image_bridge", label_visibility="collapsed")

    with col_s:
        st.subheader("Identification Status")
        if img_data and len(img_data) > 1000:
            try:
                encoded = img_data.split(",")[1]
                nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                engine = load_ai()
                db = load_face_db()
                faces = engine.get(frame)
                
                if faces:
                    feat = faces[0].normed_embedding
                    best_name, max_sim = "UNKNOWN", 0
                    
                    for name, saved_feat in db.items():
                        sim = np.dot(feat, saved_feat)
                        if sim > max_sim:
                            max_sim, best_name = sim, name
                    
                    if max_sim > 0.45:
                        st.metric("Identity", best_name, f"{int(max_sim*100)}% Match")
                        st.image(frame, width=150, caption="2x Expanded Capture")
                        
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
                    st.info("Aligning features...")
            except:
                st.error("Stabilizing AI...")
        else:
            st.info("Awaiting Sensor Handshake...")

# --- PAGE: REGISTER (ORIGINAL RESTORED) ---
elif page == "Register Face":
    st.header("👤 Register New User")
    # This is exactly your original form code
    with st.form("reg"):
        name = st.text_input("FULL NAME").upper().strip()
        file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
        if st.form_submit_button("Save to Database"):
            if name and file:
                with open(os.path.join(DB_FOLDER, f"{name}.jpg"), "wb") as f:
                    f.write(file.getbuffer())
                st.cache_resource.clear() # Force re-encoding of database
                st.success(f"Successfully Registered {name}")
            else:
                st.error("Please provide both name and image.")

    st.markdown("---")
    st.subheader("🗂️ Database Manager")
    if os.path.exists(DB_FOLDER):
        for f in [x for x in os.listdir(DB_FOLDER) if x.lower().endswith(('.jpg', '.png'))]:
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
