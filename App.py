import streamlit as st
import cv2
import numpy as np
import os
import pickle
import base64
from datetime import datetime
from insightface.app import FaceAnalysis
import pandas as pd

# ---------------- CONFIG ----------------
DB_FOLDER = "registered_faces"
LOG_FILE = "attendance.pkl"
SIM_THRESHOLD = 0.5

os.makedirs(DB_FOLDER, exist_ok=True)

st.set_page_config("Iron-Vision Biometric", layout="wide")

# ---------------- AI ENGINE ----------------
@st.cache_resource
def load_engine():
    app = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(320, 320))
    return app

@st.cache_resource
def load_face_db():
    engine = load_engine()
    db = {}

    for f in os.listdir(DB_FOLDER):
        if f.lower().endswith((".jpg", ".png", ".jpeg")):
            img = cv2.imread(os.path.join(DB_FOLDER, f))
            if img is None:
                continue
            faces = engine.get(img)
            if faces:
                db[f.split(".")[0]] = faces[0].normed_embedding
    return db

# ---------------- MEDIAPIPE COMPONENT ----------------
JS_BRIDGE = """
<div style="position:relative;width:100%;max-width:420px;margin:auto;
background:black;border-radius:14px;overflow:hidden;aspect-ratio:4/3;">
  <video id="v" autoplay playsinline
    style="width:100%;height:100%;object-fit:contain;transform:scaleX(-1)"></video>
  <canvas id="o" width="420" height="315"
    style="position:absolute;top:0;left:0;transform:scaleX(-1);"></canvas>
  <div id="t" style="position:absolute;bottom:8px;left:8px;
    font-family:monospace;font-size:12px;color:#0f0;
    background:rgba(0,0,0,.5);padding:4px">ACTIVE</div>
</div>

<script src="https://cdn.jsdelivr.net/npm/@mediapipe/face_detection"></script>
<script>
const v = document.getElementById("v");
const o = document.getElementById("o");
const ctx = o.getContext("2d");
const txt = document.getElementById("t");
const c = document.createElement("canvas");
const cctx = c.getContext("2d");

const fd = new FaceDetection({
  locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/face_detection/${f}`
});
fd.setOptions({ model: "short", minDetectionConfidence: 0.7 });

fd.onResults(r => {
  ctx.clearRect(0,0,o.width,o.height);
  if (!r.detections.length) {
    txt.innerText = "NO FACE";
    return;
  }

  const b = r.detections[0].boundingBox;
  ctx.strokeStyle = "#0f0";
  ctx.lineWidth = 3;
  ctx.strokeRect(
    b.xCenter*420 - b.width*420/2,
    b.yCenter*315 - b.height*315/2,
    b.width*420,
    b.height*315
  );

  const vw = v.videoWidth, vh = v.videoHeight;
  const w = b.width * 2 * vw;
  const h = b.height * 2 * vh;
  const x = b.xCenter*vw - w/2;
  const y = b.yCenter*vh - h/2;

  c.width = 160; c.height = 160;
  cctx.drawImage(v, x,y,w,h,0,0,160,160);
  const data = c.toDataURL("image/jpeg",0.6);

  if (window.Streamlit) {
    window.Streamlit.setComponentValue(data);
    txt.innerText = "SYNCED";
  }
});

navigator.mediaDevices.getUserMedia({video:{width:640,height:480}})
.then(s=>{
  v.srcObject=s;
  async function loop(){
    await fd.send({image:v});
    setTimeout(loop, 900);
  }
  loop();
});
</script>
"""

# ---------------- NAV ----------------
page = st.sidebar.radio("Navigation", ["Live Scanner", "Register Face", "Attendance"])

# ---------------- LIVE SCANNER ----------------
if page == "Live Scanner":
    st.header("📹 Live Biometric Scanner")
    col1, col2 = st.columns([1,1])

    with col1:
        img_data = st.components.v1.html(JS_BRIDGE, height=360)

    with col2:
        st.subheader("Recognition")
        if img_data and len(img_data) > 1000:
            try:
                encoded = img_data.split(",")[1]
                img = cv2.imdecode(
                    np.frombuffer(base64.b64decode(encoded), np.uint8),
                    cv2.IMREAD_COLOR
                )

                engine = load_engine()
                db = load_face_db()
                faces = engine.get(img)

                if faces:
                    emb = faces[0].normed_embedding
                    best, score = "UNKNOWN", 0

                    for name, ref in db.items():
                        s = float(np.dot(emb, ref))
                        if s > score:
                            best, score = name, s

                    if score >= SIM_THRESHOLD:
                        st.success(f"{best} — {int(score*100)}%")
                        st.image(img, width=160)

                        logs=[]
                        if os.path.exists(LOG_FILE):
                            with open(LOG_FILE,"rb") as f:
                                logs=pickle.load(f)

                        today = datetime.now().strftime("%Y-%m-%d")
                        if not any(l["Name"]==best and l["Date"]==today for l in logs):
                            logs.append({
                                "Name": best,
                                "Time": datetime.now().strftime("%H:%M:%S"),
                                "Date": today
                            })
                            with open(LOG_FILE,"wb") as f:
                                pickle.dump(logs,f)
                            st.toast(f"Logged {best}")

                    else:
                        st.warning("Unknown Face")
                else:
                    st.info("Aligning...")
            except Exception as e:
                st.error("Processing Error")
        else:
            st.info("Waiting for camera")

# ---------------- REGISTER ----------------
elif page == "Register Face":
    st.header("👤 Register User")
    with st.form("reg"):
        name = st.text_input("Full Name").upper().strip()
        img = st.file_uploader("Image", ["jpg","png","jpeg"])
        if st.form_submit_button("Save"):
            if name and img:
                with open(os.path.join(DB_FOLDER,f"{name}.jpg"),"wb") as f:
                    f.write(img.getbuffer())
                st.cache_resource.clear()
                st.success("Registered")
            else:
                st.error("Missing data")

# ---------------- LOGS ----------------
elif page == "Attendance":
    st.header("📊 Attendance")
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE,"rb") as f:
            st.table(pd.DataFrame(pickle.load(f)))
        if st.button("Clear"):
            os.remove(LOG_FILE)
            st.rerun()
    else:
        st.info("No logs")
