import streamlit as st
import pandas as pd
import json
import numpy as np
import os
import base64
import pickle
from datetime import datetime

# --- CONFIG & STORAGE ---
DB_FILE = "registered_faces.json"
PKL_LOG = "attendance_data.pkl"

st.set_page_config(page_title="Pickle-Sync Auth", layout="wide")

# 1. Database Initialization
if "db" not in st.session_state:
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f: st.session_state.db = json.load(f)
    else: st.session_state.db = {}

# 2. Gatekeeper (Ensures one log per person per day)
if "logged_set" not in st.session_state:
    st.session_state.logged_set = set()

# --- ASSETS ---
CSS_CODE = """
<style>
    body { margin:0; background: #0e1117; color: #00FF00; font-family: monospace; overflow: hidden; }
    #view { position: relative; width: 100%; height: 400px; border-radius: 12px; overflow: hidden; background: #000; border: 1px solid #333; }
    video, canvas, img { position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover; }
    #status-bar { position: absolute; top: 0; left: 0; right: 0; background: rgba(0,0,0,0.8); padding: 8px; font-size: 11px; z-index: 100; }
</style>
"""

JS_CODE = """
<script type="module">
    import { FaceLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

    const video = document.getElementById("webcam");
    const staticImg = document.getElementById("static-img");
    const canvas = document.getElementById("overlay");
    const ctx = canvas.getContext("2d");
    const log = document.getElementById("status-bar");
    
    const staticImgSrc = "STATIC_IMG_PLACEHOLDER";
    const runMode = "RUN_MODE_PLACEHOLDER";
    const registry = JSON.parse('DB_JSON_PLACEHOLDER');
    
    let faceLandmarker;

    async function init() {
        try {
            const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm");
            faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
                baseOptions: {
                    modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
                    delegate: "GPU"
                },
                runningMode: runMode,
                numFaces: 1
            });

            if (staticImgSrc !== "null") {
                staticImg.src = staticImgSrc;
                staticImg.onload = async () => {
                    const results = await faceLandmarker.detect(staticImg);
                    if (results.faceLandmarks && results.faceLandmarks.length > 0) {
                        const dataString = btoa(JSON.stringify(results.faceLandmarks[0]));
                        const url = new URL(window.parent.location.href);
                        url.searchParams.set("face_data", dataString);
                        window.parent.history.replaceState({}, "", url);
                        window.parent.postMessage({type: 'streamlit:setComponentValue', value: 'READY'}, "*");
                    }
                };
            } else {
                const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                video.srcObject = stream;
                video.onloadeddata = () => predictVideo();
            }
        } catch (err) { log.innerText = "ERROR: " + err.message; }
    }

    function findMatch(current) {
        let match = { name: "Unknown", conf: 0 };
        for (const [name, saved] of Object.entries(registry)) {
            let dist = 0;
            for (let i = 0; i < 30; i++) {
                const dx = current[i].x - saved[i].x;
                const dy = current[i].y - saved[i].y;
                dist += Math.sqrt(dx*dx + dy*dy);
            }
            dist /= 30;
            const conf = Math.max(0, Math.floor((1 - (dist / 0.05)) * 100));
            if (dist < 0.05 && conf > match.conf) match = { name, conf };
        }
        return match;
    }

    async function predictVideo() {
        const results = faceLandmarker.detectForVideo(video, performance.now());
        canvas.width = video.videoWidth; canvas.height = video.videoHeight;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (results.faceLandmarks && results.faceLandmarks.length > 0) {
            const landmarks = results.faceLandmarks[0];
            const match = findMatch(landmarks);
            
            const xs = landmarks.map(p => p.x * canvas.width);
            const ys = landmarks.map(p => p.y * canvas.height);
            const x = Math.min(...xs), y = Math.min(...ys), w = Math.max(...xs)-x, h = Math.max(...ys)-y;
            
            const color = match.name === "Unknown" ? "#FF4B4B" : "#00FF00";
            ctx.strokeStyle = color; ctx.lineWidth = 4;
            ctx.strokeRect(x, y, w, h);
            ctx.fillStyle = color;
            ctx.fillRect(x, y-25, w, 25);
            ctx.fillStyle = "white";
            ctx.font = "bold 14px monospace";
            ctx.fillText(`${match.name} ${match.conf}%`, x+5, y-8);

            // BRIDGE: If matched, update the parent URL
            if (match.name !== "Unknown") {
                const url = new URL(window.parent.location.href);
                if (url.searchParams.get("detected") !== match.name) {
                    url.searchParams.set("detected", match.name);
                    window.parent.history.replaceState({}, "", url);
                }
            }
        }
        window.requestAnimationFrame(predictVideo);
    }
    init();
</script>
"""

def get_component_html(img_b64=None):
    db_json = json.dumps(st.session_state.db)
    img_val = f"data:image/jpeg;base64,{img_b64}" if img_b64 else "null"
    html = f"<!DOCTYPE html><html><head>{CSS_CODE}</head><body>"
    html += f'<div id="view"><div id="status-bar">BRIDGE-SYNC ACTIVE</div>'
    html += f'<video id="webcam" autoplay muted playsinline style="display: {"none" if img_b64 else "block"}"></video>'
    html += f'<img id="static-img" style="display: {"block" if img_b64 else "none"}">'
    html += f'<canvas id="overlay"></canvas></div>{JS_CODE}</body></html>'
    return html.replace("STATIC_IMG_PLACEHOLDER", img_val).replace("RUN_MODE_PLACEHOLDER", "IMAGE" if img_b64 else "VIDEO").replace("DB_JSON_PLACEHOLDER", db_json)

# --- HELPER: PICKLE ENGINE ---
def save_attendance_pkl(name):
    logs = []
    if os.path.exists(PKL_LOG):
        with open(PKL_LOG, "rb") as f: logs = pickle.load(f)
    entry = {"Name": name, "Time": datetime.now().strftime("%H:%M:%S"), "Date": datetime.now().strftime("%Y-%m-%d")}
    logs.append(entry)
    with open(PKL_LOG, "wb") as f: pickle.dump(logs, f)

# --- UI NAVIGATION ---
page = st.sidebar.radio("Navigate", ["Register", "Live Feed", "Log"])

if page == "Register":
    st.header("👤 Face Registration")
    name = st.text_input("Full Name").upper()
    uploaded = st.file_uploader("Upload Profile Image", type=['jpg', 'jpeg', 'png'])
    if uploaded:
        b64 = base64.b64encode(uploaded.getvalue()).decode()
        st.components.v1.html(get_component_html(b64), height=420)
    
    url_data = st.query_params.get("face_data")
    if url_data and name:
        if st.button("Confirm Registration"):
            st.session_state.db[name] = json.loads(base64.b64decode(url_data).decode())
            with open(DB_FILE, "w") as f: json.dump(st.session_state.db, f)
            st.query_params.clear()
            st.success(f"Registered {name}!")
            st.rerun()

    st.markdown("---")
    st.subheader("🗂️ Manage Database")
    for reg_name in list(st.session_state.db.keys()):
        col_n, col_b = st.columns([4, 1])
        col_n.write(f"✅ {reg_name}")
        if col_b.button("Delete", key=f"del_{reg_name}"):
            del st.session_state.db[reg_name]
            with open(DB_FILE, "w") as f: json.dump(st.session_state.db, f)
            st.rerun()

elif page == "Live Feed":
    st.header("📹 Live Scanner")
    col_v, col_m = st.columns([3, 1])
    
    # 1. Capture via URL Bridge
    detected_name = st.query_params.get("detected")
