import streamlit as st
import pandas as pd
import json
import numpy as np
import os
import base64
from datetime import datetime

# --- CONFIG & STORAGE ---
DB_FILE = "registered_faces.json"
LOG_FILE = "attendance_log.csv"

st.set_page_config(page_title="Biometric Scanner", layout="wide")

# Initialize Storage
if "db" not in st.session_state:
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f: st.session_state.db = json.load(f)
    else: st.session_state.db = {}

if "logs" not in st.session_state:
    if os.path.exists(LOG_FILE):
        try: st.session_state.logs = pd.read_csv(LOG_FILE)
        except: st.session_state.logs = pd.DataFrame(columns=["Name", "Time"])
    else: st.session_state.logs = pd.DataFrame(columns=["Name", "Time"])

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
    
    // Values injected from Python
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
            for (let i = 0; i < 30; i++) { // Using first 30 anchor points for speed
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
            
            // DRAW OVERLAY
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

            // SEND TO PYTHON FOR LOGGING
            window.parent.postMessage({
                type: "streamlit:setComponentValue",
                value: JSON.stringify({ name: match.name, ts: Date.now() })
            }, "*");
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
    html += f'<div id="view"><div id="status-bar">SECURE FEED</div>'
    html += f'<video id="webcam" autoplay muted playsinline style="display: {"none" if img_b64 else "block"}"></video>'
    html += f'<img id="static-img" style="display: {"block" if img_b64 else "none"}">'
    html += f'<canvas id="overlay"></canvas></div>{JS_CODE}</body></html>'
    return html.replace("STATIC_IMG_PLACEHOLDER", img_val).replace("RUN_MODE_PLACEHOLDER", "IMAGE" if img_b64 else "VIDEO").replace("DB_JSON_PLACEHOLDER", db_json)

# --- UI NAVIGATION ---
page = st.sidebar.radio("Navigate", ["Register", "Live Feed", "Log"])

if page == "Register":
    st.header("👤 Face Registration")
    col1, col2 = st.columns([1, 1])
    with col1:
        name = st.text_input("Full Name").upper()
        uploaded = st.file_uploader("Upload Profile Image", type=['jpg', 'jpeg', 'png'])
        if uploaded:
            b64 = base64.b64encode(uploaded.getvalue()).decode()
            st.components.v1.html(get_component_html(b64), height=420)
        
        # URL Bridge check
        url_data = st.query_params.get("face_data")
        if url_data and name:
            if st.button("Confirm Registration"):
                st.session_state.db[name] = json.loads(base64.b64decode(url_data).decode())
                with open(DB_FILE, "w") as f: json.dump(st.session_state.db, f)
                st.query_params.clear()
                st.success(f"Registered {name}!")
                st.rerun()

    with col2:
        st.subheader("Manage Database")
        for reg_name in list(st.session_state.db.keys()):
            c_n, c_b = st.columns([3, 1])
            c_n.write(reg_name)
            if c_b.button("Delete", key=f"del_{reg_name}"):
                del st.session_state.db[reg_name]
                with open(DB_FILE, "w") as f: json.dump(st.session_state.db, f)
                st.rerun()

elif page == "Live Feed":
    st.header("📹 Live Scanner")
    col_v, col_m = st.columns([3, 1])
    with col_v:
        # Standard component capture
        capture = st.components.v1.html(get_component_html(), height=420)
    
    with col_m:
        if isinstance(capture, str) and capture != "READY":
            res = json.loads(capture)
            identified = res.get("name")
            st.metric("Detected", identified)
            
            # --- LOGGING ENGINE ---
            if identified != "Unknown":
                if "logged_names" not in st.session_state: st.session_state.logged_names = set()
                if identified not in st.session_state.logged_names:
                    now = datetime.now().strftime("%H:%M:%S")
                    new_log = pd.DataFrame({"Name": [identified], "Time": [now]})
                    st.session_state.logs = pd.concat([st.session_state.logs, new_log], ignore_index=True)
                    st.session_state.logs.to_csv(LOG_FILE, index=False)
                    st.session_state.logged_names.add(identified)
                    st.toast(f"✅ Logged {identified}")

elif page == "Log":
    st.header("📊 Attendance Log")
    st.dataframe(st.session_state.logs, use_container_width=True)
    if st.button("Download CSV"):
        st.download_button("Export", st.session_state.logs.to_csv(index=False), "attendance.csv")
