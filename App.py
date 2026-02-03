import streamlit as st
import cv2
import numpy as np
import os
import base64
import pickle
from insightface.app import FaceAnalysis

# ---------------- CONFIG ----------------
DB_FOLDER = "registered_faces"
SIM_THRESHOLD = 0.5
os.makedirs(DB_FOLDER, exist_ok=True)

st.set_page_config(page_title="Iron-Vision", layout="wide")

# ---------------- STATE ----------------
if "identity" not in st.session_state:
    st.session_state.identity = "SCANNING"
    st.session_state.score = 0

# ---------------- AI ENGINE ----------------
@st.cache_resource
def load_engine():
    app = FaceAnalysis(
        name="buffalo_s",
        providers=["CPUExecutionProvider"]
    )
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

# ---------------- FULLSCREEN JS COMPONENT ----------------
JS_BRIDGE = r"""
<style>
html, body {
  margin: 0;
  padding: 0;
  background: black;
}
</style>

<div style="position:relative;width:100vw;height:100vh;overflow:hidden;">
  <video id="v" autoplay playsinline
    style="width:100%;height:100%;object-fit:cover;transform:scaleX(-1)"></video>
  <canvas id="o"
    style="position:absolute;top:0;left:0;width:100%;height:100%;
           transform:scaleX(-1);"></canvas>
</div>

<script src="https://cdn.jsdelivr.net/npm/@mediapipe/face_detection"></script>
<script>
const NAME = "{{NAME}}";
const SCORE = "{{SCORE}}";

const v = document.getElementById("v");
const o = document.getElementById("o");
const ctx = o.getContext("2d");
const c = document.createElement("canvas");
const cctx = c.getContext("2d");

const fd = new FaceDetection({
  locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/face_detection/${f}`
});
fd.setOptions({ model:"short", minDetectionConfidence:0.7 });

fd.onResults(r => {
  o.width = o.clientWidth;
  o.height = o.clientHeight;
  ctx.clearRect(0,0,o.width,o.height);

  if (!r.detections.length) return;

  const b = r.detections[0].boundingBox;
  const x = b.xCenter*o.width - b.width*o.width/2;
  const y = b.yCenter*o.height - b.height*o.height/2;
  const w = b.width*o.width;
  const h = b.height*o.height;

  ctx.strokeStyle = "#00ff00";
  ctx.lineWidth = 3;
  ctx.strokeRect(x,y,w,h);

  ctx.fillStyle = "#00ff00";
  ctx.font = "18px monospace";
  ctx.fillText(`${NAME} (${SCORE}%)`, x, y-10);

  const vw = v.videoWidth, vh = v.videoHeight;
  const cw = b.width*2*vw;
  const ch = b.height*2*vh;
  const cx = b.xCenter*vw - cw/2;
  const cy = b.yCenter*vh - ch/2;

  c.width = 160; c.height = 160;
  cctx.drawImage(v, cx,cy,cw,ch, 0,0,160,160);

  if (window.Streamlit) {
    Streamlit.setComponentValue(
      c.toDataURL("image/jpeg",0.6)
    );
  }
});

navigator.mediaDevices.getUserMedia({video:true})
.then(s=>{
  v.srcObject=s;
  async function loop(){
    await fd.send({image:v});
    setTimeout(loop, 800);
  }
  loop();
});
</script>
"""

# ---------------- RENDER CAMERA ----------------
html = JS_BRIDGE \
    .replace("{{NAME}}", st.session_state.identity) \
    .replace("{{SCORE}}", str(st.session_state.score))

st.components.v1.html(html, height=800, key="camera")
img_data = st.session_state.get("camera")

# ---------------- FACE RECOGNITION ----------------
if isinstance(img_data, str) and img_data.startswith("data:image"):
    frame = cv2.imdecode(
        np.frombuffer(
            base64.b64decode(img_data.split(",")[1]),
            np.uint8
        ),
        cv2.IMREAD_COLOR
    )

    faces = load_engine().get(frame)
    if faces:
        emb = faces[0].normed_embedding
        best, score = "UNKNOWN", 0.0

        for name, ref in load_face_db().items():
            s = float(np.dot(emb, ref))
            if s > score:
                best, score = name, s

        if score >= SIM_THRESHOLD:
            st.session_state.identity = best
            st.session_state.score = int(score * 100)
        else:
            st.session_state.identity = "UNKNOWN"
            st.session_state.score = int(score * 100)
