import streamlit as st
import pandas as pd
import json
import numpy as np
import os
from datetime import datetime

# --- SETTINGS & PERSISTENCE ---
DB_FILE = "registered_faces.json"
LOG_FILE = "attendance_log.csv"

st.set_page_config(page_title="Privacy Face Auth", layout="wide")

if "db" not in st.session_state:
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f: st.session_state.db = json.load(f)
    else: st.session_state.db = {}

if "logs" not in st.session_state:
    if os.path.exists(LOG_FILE): st.session_state.logs = pd.read_csv(LOG_FILE)
    else: st.session_state.logs = pd.DataFrame(columns=["Name", "Time"])

# --- FRONTEND COMPONENT (RECTANGLE ONLY) ---
INTERFACE_CODE = """
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3"></script>
    <style>
        body { margin:0; background: #0e1117; }
        #view { position: relative; width: 100%; height: 400px; border-radius: 12px; overflow: hidden; background: #000; }
        video, canvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover; }
    </style>
</head>
<body>
    <div id="view">
        <video id="webcam" autoplay playsinline muted></video>
        <canvas id="overlay"></canvas>
    </div>
    <script type="module">
        import { FaceLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

        const video = document.getElementById("webcam");
        const canvas = document.getElementById("overlay");
        const ctx = canvas.getContext("2d");
        let faceLandmarker;

        async function setup() {
            const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm");
            faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
                baseOptions: {
                    modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
                    delegate: "GPU"
                },
                runningMode: "VIDEO",
                numFaces: 1
            });
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = stream;
            video.addEventListener("loadeddata", predict);
        }

        async function predict() {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const results = await faceLandmarker.detectForVideo(video, performance.now());

            ctx.clearRect(0, 0, canvas.width, canvas.height);
            if (results.faceLandmarks.length > 0) {
                const landmarks = results.faceLandmarks[0];
                
                // Calculate Bounding Box from landmarks
                const xs = landmarks.map(p => p.x * canvas.width);
                const ys = landmarks.map(p => p.y * canvas.height);
                const minX = Math.min(...xs);
                const maxX = Math.max(...xs);
                const minY = Math.min(...ys);
                const maxY = Math.max(...ys);

                // Draw Rectangle Only
                ctx.strokeStyle = "#00FF00";
                ctx.lineWidth = 3;
                ctx.strokeRect(minX, minY, maxX - minX, maxY - minY);
                
                window.parent.postMessage({
                    type: "streamlit:setComponentValue",
                    value: JSON.stringify(landmarks)
                }, "*");
            }
            window.requestAnimationFrame(predict);
        }
        setup();
    </script>
</body>
</html>
"""

# --- NAVIGATION ---
page = st.sidebar.radio("Navigate", ["Register", "Live Feed", "Log"])

if page == "Register":
    st.header("👤 Register Identity")
    name = st.text_input("Person Name").upper()
    raw_data = st.components.v1.html(INTERFACE_CODE, height=420)
    
    # Fix: Only attempt json.loads if raw_data is not None
    if st.button("Save Face Print") and name:
        if raw_data:
            st.session_state.db[name] = json.loads(raw_data)
            with open(DB_FILE, "w") as f: json.dump(st.session_state.db, f)
            st.success(f"Registered {name}!")
        else:
            st.error("Waiting for camera to detect face...")

elif page == "Live Feed":
    st.header("📹 Attendance Feed")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        current_data = st.components.v1.html(INTERFACE_CODE, height=420)
        
    with col2:
        if current_data:
            current_face = json.loads(current_data)
            identified = "Unknown"
            
            # Compare with database
            for name, saved_face in st.session_state.db.items():
                curr_arr = np.array([[p['x'], p['y']] for p in current_face[:30]])
                save_arr = np.array([[p['x'], p['y']] for p in saved_face[:30]])
                dist = np.mean(np.linalg.norm(curr_arr - save_arr, axis=1))
                
                if dist < 0.04:
                    identified = name
                    break
            
            st.subheader(f"Status: {identified}")
            
            # Fix: Only log if identified is NOT "Unknown"
            if identified != "Unknown":
                if identified not in st.session_state.logs["Name"].values:
                    now = datetime.now().strftime("%H:%M:%S")
                    new_entry = pd.DataFrame({"Name": [identified], "Time": [now]})
                    st.session_state.logs = pd.concat([st.session_state.logs, new_entry], ignore_index=True)
                    st.session_state.logs.to_csv(LOG_FILE, index=False)
                    st.toast(f"Logged {identified}")

elif page == "Log":
    st.header("📊 Attendance Log")
    st.dataframe(st.session_state.logs, use_container_width=True)
    st.download_button("Download CSV", st.session_state.logs.to_csv(index=False), "attendance.csv")
