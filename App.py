import streamlit as st
import pd
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

st.set_page_config(page_title="Iron-Vision Edge Pro", layout="wide")

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
    valid_ext = ('.jpg', '.png', '.jpeg')
    if os.path.exists(DB_FOLDER):
        for file in os.listdir(DB_FOLDER):
            if file.lower().endswith(valid_ext):
                img = cv2.imread(os.path.join(DB_FOLDER, file))
                if img is not None:
                    faces = engine.get(img)
                    if faces:
                        db[file.split('.')[0]] = faces[0].normed_embedding
    return db

# --- 3. THE 2X CROP BRIDGE (Updated Handshake) ---
JS_BRIDGE = """
<div style="position: relative; width: 320px; height: 240px; margin: auto; background: #000; border-radius: 10px; overflow: hidden;">
    <video id="v" autoplay playsinline style="width: 320px; height: 240px; transform: scaleX(-1); object-fit: contain;"></video>
    <canvas id="overlay" width="320" height="240" style="position: absolute; top: 0; left: 0; transform: scaleX(-1);"></canvas>
    <div id="status" style="position: absolute; bottom: 5px; width: 100%; color: #0F0; font-family: monospace; font-size: 10px; text-align: center; background: rgba(0,0,0,0.4);">INITIATING...</div>
</div>

<script src="https://cdn.jsdelivr.net/npm/@mediapipe/face_detection"></script>
<script>
    const video = document.getElementById('v');
    const overlay = document.getElementById('overlay');
    const ctx = overlay.getContext('2d');
    const cropCanvas = document.createElement('canvas');
    const cropCtx = cropCanvas.getContext('2d');
    const statusText = document.getElementById('status');

    const faceDetection = new FaceDetection({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_detection/${file}`
    });

    faceDetection.setOptions({ model: 'short', minDetectionConfidence: 0.7 });

    faceDetection.onResults(results => {
        ctx.clearRect(0, 0, overlay.width, overlay.height);
        if (results.detections.length > 0) {
            const face = results.detections[0].boundingBox;
            
            // Draw UI
            ctx.strokeStyle = "#00FF00";
            ctx.lineWidth = 2;
            ctx.strokeRect(face.xCenter * 320 - (face.width * 320 / 2), 
                           face.yCenter * 240 - (face.height * 240 / 2), 
                           face.width * 320, face.height * 240);

            // 2X Expanded Crop Logic
            const w_vid = video.videoWidth;
            const h_vid = video.videoHeight;
            const new_w = face.width * 2 * w_vid;
            const new_h = face.height * 2 * h_vid;
            const x = (face.xCenter * w_vid) - (new_w / 2);
            const y = (face.yCenter * h_vid) - (new_h / 2);

            cropCanvas.width = 160; 
            cropCanvas.height = 160;
            cropCtx.drawImage(video, x, y, new_w, new_h, 0, 0, 160, 160);
            
            const data = cropCanvas.toDataURL('image/jpeg', 0.6);
            
            // --- STABLE STREAMLIT CLOUD HANDSHAKE ---
            // On Cloud, we must search the window.parent for the input precisely.
            const inputs = window.parent.document.querySelectorAll('input[data-testid="stTextInputEnterChat"]');
            let found = false;
            for (let i of window.parent.document.querySelectorAll('input')) {
                if (i.parentElement && i.parentElement.innerHTML.includes('bridge')) {
                    i.value = data;
                    i.dispatchEvent(new Event('input', { bubbles: true }));
                    i.dispatchEvent(new Event('change', { bubbles: true }));
                    found = true;
                    break;
                }
            }
            statusText.innerText = found ? "SYNCING FACE DATA" : "LOOKING FOR BRIDGE...";
        } else {
            statusText.innerText = "NO FACE DETECTED";
        }
    });

    navigator.mediaDevices.getUserMedia({ video: { width: 320, height: 240 } })
        .then(stream => { 
            video.srcObject = stream;
            statusText.innerText = "CAMERA READY";
            async function predict() {
                await faceDetection.send({image: video});
                setTimeout(predict, 1000); 
            }
            predict();
        })
        .catch(err => {
            statusText.innerText = "CAMERA ERROR: " + err.name;
        });
</script>
"""

# CSS to hide the bridge input
st.markdown("<style>div[data-testid='stTextInput'] { display: none !important; }</style>", unsafe_allow_html=True)

# --- 4. NAVIGATION ---
page = st.sidebar.radio("Navigation", ["Live Scanner", "Register Face", "Attendance Log"])

if page == "Live Scanner":
    st.header("⚡ Biometric Scanner")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.components.v1.html(JS_BRIDGE, height=280)
        # Using a very simple label so JS can find it by searching innerHTML
        img_data = st.text_input("bridge", key="image_bridge", label_visibility="collapsed")

    with col2:
        st.subheader("Results")
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
                    
                    if score > 0.42:
                        st.metric("Subject", best_name, f"{int(score*100)}% Confidence")
                        st.image(face_chip, width=150, caption="Detection Crop")
                        
                        # Logging
                        logs = []
                        if os.path.exists(PKL_LOG):
                            with open(PKL_LOG, "rb") as f: logs = pickle.load(f)
                        today = datetime.now().strftime("%Y-%m-%d")
                        if not any(e['Name'] == best_name and e['Date'] == today for e in logs):
                            logs.append({"Name": best_name, "Time": datetime.now().strftime("%H:%M:%S"), "Date": today})
                            with open(PKL_LOG, "wb") as f: pickle.dump(logs, f)
                            st.toast(f"✅ Logged: {best_name}")
                    else:
                        st.warning("Identity Unknown")
                else:
                    st.info("Face detected, but features unclear.")
            except Exception as e:
                st.error("Error decoding face.")
        else:
            st.info("Awaiting Handshake...")

# --- 5. REGISTER PAGE ---
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
    if os.path.exists(DB_FOLDER):
        files = [x for x in os.listdir(DB_FOLDER) if x.lower().endswith(('.jpg', '.png'))]
        for f in files:
            c1, c2 = st.columns([4, 1])
            c1.write(f"✅ {f.split('.')[0]}")
            if c2.button("Delete", key=f):
                os.remove(os.path.join(DB_FOLDER, f))
                load_face_db.clear()
                st.rerun()

# --- 6. LOGS ---
elif page == "Attendance Log":
    st.header("📊 Records")
    if os.path.exists(PKL_LOG):
        with open(PKL_LOG, "rb") as f: data = pickle.load(f)
        st.table(pd.DataFrame(data))
        if st.button("Clear Logs"):
            os.remove(PKL_LOG); st.rerun()
