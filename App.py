import streamlit as st
import pandas as pd
import json
import numpy as np
import os
from datetime import datetime

# --- PERSISTENCE ---
DB_FILE = "registered_faces.json"
LOG_FILE = "attendance_log.csv"

st.set_page_config(page_title="Privacy Face Auth", layout="wide")

if "db" not in st.session_state:
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f: st.session_state.db = json.load(f)
    else: st.session_state.db = {}

if "logs" not in st.session_state:
    if os.path.exists(LOG_FILE):
        try: st.session_state.logs = pd.read_csv(LOG_FILE)
        except: st.session_state.logs = pd.DataFrame(columns=["Name", "Time"])
    else: st.session_state.logs = pd.DataFrame(columns=["Name", "Time"])

# --- FRONTEND CODE ---
# Corrected CDN URLs for MediaPipe 0.10.3
INTERFACE_CODE = """
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/vision_bundle.mjs" type="module"></script>
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/vision_bundle.js"></script>
    <style>
        body { margin:0; background: #0e1117; color: #00FF00; font-family: monospace; }
        #view { position: relative; width: 100%; height: 400px; border-radius: 12px; overflow: hidden; background: #000; border: 1px solid #333; }
        video, canvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover; }
        #status-bar { position: absolute; top: 0; left: 0; right: 0; background: rgba(0,0,0,0.8); padding: 8px; font-size: 11px; z-index: 100; }
    </style>
</head>
<body>
    <div id="view">
        <div id="status-bar">STATUS: Checking Google AI Engine...</div>
        <video id="webcam" autoplay playsinline muted></video>
        <canvas id="overlay"></canvas>
    </div>
    <script>
        const video = document.getElementById("webcam");
        const canvas = document.getElementById("overlay");
        const ctx = canvas.getContext("2d");
        const log = document.getElementById("status-bar");
        let faceLandmarker;

        async function init() {
            // Check if library loaded via global namespace
            const visionLib = window.tasksVision;
            
            if (!visionLib) {
                log.innerText = "ERROR: CDN blocked or file not found. Try disabling Ad-Block.";
                log.style.color = "#FF4B4B";
                return;
            }

            try {
                log.innerText = "STATUS: Connecting to WASM... (Wait 5-10s)";
                const vision = await visionLib.FilesetResolver.forVisionTasks(
                    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
                );
                
                log.innerText = "STATUS: Fetching Face Model...";
                faceLandmarker = await visionLib.FaceLandmarker.createFromOptions(vision, {
                    baseOptions: {
                        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
                        delegate: "GPU"
                    },
                    runningMode: "VIDEO",
                    numFaces: 1
                });

                log.innerText = "STATUS: Requesting Webcam...";
                const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                video.srcObject = stream;
                
                video.onloadeddata = () => {
                    log.innerText = "ONLINE: Privacy-Focused Scanner Active";
                    predict();
                };
            } catch (err) {
                log.innerText = "CRITICAL: " + err.message;
                log.style.color = "#FF4B4B";
            }
        }

        async function predict() {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            
            const startTimeMs = performance.now();
            const results = faceLandmarker.detectForVideo(video, startTimeMs);

            ctx.clearRect(0, 0, canvas.width, canvas.height);
            if (results.faceLandmarks && results.faceLandmarks.length > 0) {
                const landmarks = results.faceLandmarks[0];
                const xs = landmarks.map(p => p.x * canvas.width);
                const ys = landmarks.map(p => p.y * canvas.height);
                const minX = Math.min(...xs), maxX = Math.max(...xs);
                const minY = Math.min(...ys), maxY = Math.max(...ys);

                ctx.strokeStyle = "#00FF00";
                ctx.lineWidth = 3;
                ctx.strokeRect(minX, minY, maxX - minX, maxY - minY);
                
                // Privacy: Encoded landmarks only
                window.parent.postMessage({
                    type: "streamlit:setComponentValue",
                    value: JSON.stringify(landmarks)
                }, "*");
            }
            window.requestAnimationFrame(predict);
        }

        window.onload = init;
    </script>
</body>
</html>
"""

# --- UI LOGIC ---
page = st.sidebar.radio("Navigate", ["Register", "Live Feed", "Log"])

if page == "Register":
    st.header("👤 Identity Registration")
    name = st.text_input("Name").upper()
    val = st.components.v1.html(INTERFACE_CODE, height=420)
    
    if val and isinstance(val, str):
        st.session_state.buffered_encoding = val
        st.success(f"✅ Recognition Ready for {name}")
    
    if st.button("Confirm Registration"):
        if name and st.session_state.get('buffered_encoding'):
            st.session_state.db[name] = json.loads(st.session_state.buffered_encoding)
            with open(DB_FILE, "w") as f:
                json.dump(st.session_state.db, f)
            st.success(f"Saved {name}!")
            st.rerun()
        else:
            st.error("Ensure name is entered and face is detected.")

elif page == "Live Feed":
    st.header("📹 Attendance Feed")
    col1, col2 = st.columns([3, 1])
    with col1:
        feed_val = st.components.v1.html(INTERFACE_CODE, height=420)
    with col2:
        if feed_val and isinstance(feed_val, str):
            current_face = json.loads(feed_val)
            identified = "Unknown"
            for db_name, saved_face in st.session_state.db.items():
                curr_arr = np.array([[p['x'], p['y']] for p in current_face[:30]])
                save_arr = np.array([[p['x'], p['y']] for p in saved_face[:30]])
                dist = np.mean(np.linalg.norm(curr_arr - save_arr, axis=1))
                if dist < 0.05:
                    identified = db_name
                    break
            st.subheader(f"Status: {identified}")
            if identified != "Unknown" and identified not in st.session_state.logs["Name"].values:
                now = datetime.now().strftime("%H:%M:%S")
                new_entry = pd.DataFrame({"Name": [identified], "Time": [now]})
                st.session_state.logs = pd.concat([st.session_state.logs, new_entry], ignore_index=True)
                st.session_state.logs.to_csv(LOG_FILE, index=False)
                st.toast(f"Logged {identified}")

elif page == "Log":
    st.header("📊 History")
    st.dataframe(st.session_state.logs, use_container_width=True)
    st.download_button("Export CSV", st.session_state.logs.to_csv(index=False), "attendance.csv")
