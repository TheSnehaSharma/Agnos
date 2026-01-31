import streamlit as st
import pandas as pd
import json
import numpy as np
import os
import base64
from datetime import datetime

# --- CONFIGURATION ---
DB_FILE = "registered_faces.json"
LOG_FILE = "attendance_log.csv"

st.set_page_config(page_title="Privacy Face Auth", layout="wide")

# Persistent Storage
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
    body { margin:0; background: #0e1117; color: #00FF00; font-family: monospace; }
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
                log.innerText = "STATUS: Extracting Face Landmarks...";
                staticImg.src = staticImgSrc;
                staticImg.onload = async () => {
                    const results = await faceLandmarker.detect(staticImg);
                    if (results.faceLandmarks && results.faceLandmarks.length > 0) {
                        processResults(results);
                    } else {
                        log.innerText = "❌ ERROR: No face found in image.";
                    }
                };
            } else {
                const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                video.srcObject = stream;
                video.onloadeddata = () => { predictVideo(); };
            }
        } catch (err) { log.innerText = "ERROR: " + err.message; }
    }

    function processResults(results) {
        canvas.width = (staticImgSrc !== "null") ? staticImg.naturalWidth : video.videoWidth;
        canvas.height = (staticImgSrc !== "null") ? staticImg.naturalHeight : video.videoHeight;
        const landmarks = results.faceLandmarks[0];
        
        // Visual feedback
        const xs = landmarks.map(p => p.x * canvas.width);
        const ys = landmarks.map(p => p.y * canvas.height);
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = "#00FF00"; ctx.lineWidth = 5;
        ctx.strokeRect(Math.min(...xs), Math.min(...ys), Math.max(...xs)-Math.min(...xs), Math.max(...ys)-Math.min(...ys));
        
        log.innerText = "✅ ENCODING COMPLETE";

        // Dispatch back to Streamlit
        window.parent.postMessage({
            type: "streamlit:setComponentValue",
            value: JSON.stringify(landmarks)
        }, "*");
    }

    async function predictVideo() {
        const results = faceLandmarker.detectForVideo(video, performance.now());
        canvas.width = video.videoWidth; canvas.height = video.videoHeight;
        if (results.faceLandmarks && results.faceLandmarks.length > 0) {
            const landmarks = results.faceLandmarks[0];
            const xs = landmarks.map(p => p.x * canvas.width);
            const ys = landmarks.map(p => p.y * canvas.height);
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.strokeStyle = "#00FF00"; ctx.lineWidth = 5;
            ctx.strokeRect(Math.min(...xs), Math.min(...ys), Math.max(...xs)-Math.min(...xs), Math.max(...ys)-Math.min(...ys));
            
            window.parent.postMessage({
                type: "streamlit:setComponentValue",
                value: JSON.stringify(landmarks)
            }, "*");
        }
        window.requestAnimationFrame(predictVideo);
    }

    init();
</script>
"""

def get_component_html(img_b64=None):
    html_template = f"<!DOCTYPE html><html><head>{CSS_CODE}</head><body>"
    html_template += f'<div id="view"><div id="status-bar">...</div>'
    html_template += f'<video id="webcam" autoplay muted playsinline style="display: {"none" if img_b64 else "block"}"></video>'
    html_template += f'<img id="static-img" style="display: {"block" if img_b64 else "none"}">'
    html_template += f'<canvas id="overlay"></canvas></div>{JS_CODE}</body></html>'
    
    img_val = f"data:image/jpeg;base64,{img_b64}" if img_b64 else "null"
    return html_template.replace("STATIC_IMG_PLACEHOLDER", img_val).replace("RUN_MODE_PLACEHOLDER", "IMAGE" if img_b64 else "VIDEO")

# --- UI ---

page = st.sidebar.radio("Navigate", ["Register", "Live Feed", "Log"])

if page == "Register":
    st.header("👤 Identity Registration")
    name = st.text_input("Name").upper()
    uploaded_file = st.file_uploader("Upload Profile Photo", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        b64_img = base64.b64encode(uploaded_file.getvalue()).decode()
        # The return value of the component is assigned to 'captured_json'
        captured_json = st.components.v1.html(get_component_html(b64_img), height=420)
        
        # Buffer the data immediately if it's a string
        if isinstance(captured_json, str):
            st.session_state.buffered_reg = captured_json

    # UI Flow
    if "buffered_reg" in st.session_state and name:
        st.success(f"Face encoded for {name}!")
        if st.button("Confirm & Save to Database"):
            try:
                # Retrieve from session state, not the component direct
                data = st.session_state.buffered_reg
                st.session_state.db[name] = json.loads(data)
                with open(DB_FILE, "w") as f:
                    json.dump(st.session_state.db, f)
                st.success(f"✅ {name} added to vault.")
                # Clean up and reset
                del st.session_state.buffered_reg
                st.rerun()
            except Exception as e:
                st.error(f"Save failed: {e}")
    elif uploaded_file:
        st.info("Reading image... please wait for 'Encoding Complete' message above.")

elif page == "Live Feed":
    st.header("📹 Attendance Feed")
    col1, col2 = st.columns([3, 1])
    with col1:
        feed_val = st.components.v1.html(get_component_html(), height=420)
    with col2:
        if isinstance(feed_val, str):
            current_face = json.loads(feed_val)
            identified = "Unknown"
            
            # Efficient landmark comparison
            for db_name, saved_face in st.session_state.db.items():
                curr_arr = np.array([[p['x'], p['y']] for p in current_face[:30]])
                save_arr = np.array([[p['x'], p['y']] for p in saved_face[:30]])
                dist = np.mean(np.linalg.norm(curr_arr - save_arr, axis=1))
                if dist < 0.05:
                    identified = db_name
                    break
            
            st.subheader(f"Status: {identified}")
            if identified != "Unknown":
                if identified not in st.session_state.logs["Name"].values:
                    now = datetime.now().strftime("%H:%M:%S")
                    new_entry = pd.DataFrame({"Name": [identified], "Time": [now]})
                    st.session_state.logs = pd.concat([st.session_state.logs, new_entry], ignore_index=True)
                    st.session_state.logs.to_csv(LOG_FILE, index=False)
                    st.toast(f"Logged {identified}")

elif page == "Log":
    st.header("📊 History")
    st.dataframe(st.session_state.logs, use_container_width=True)
    st.download_button("Export CSV", st.session_state.logs.to_csv(index=False), "attendance.csv")
