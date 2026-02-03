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
  const h = b.height
