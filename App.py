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

# Initialize Session State & Local Storage
if "db" not in st.session_state:
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f: st.session_state.db = json.load(f)
    else: st.session_state.db = {}

if "logs" not in st.session_state:
    if os.path.exists(LOG_FILE): st.session_state.logs = pd.read_csv(LOG_FILE)
    else: st.session_state.logs = pd.DataFrame(columns=["Name", "Time"])

# --- FRONTEND COMPONENT (HTML & JS BUNDLE) ---
# We use a raw string to ensure no external .html or .js files are needed
INTERFACE_CODE = """
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3"></script>
    <style>
        body { margin:0; background: #0e1117; color: white; font-family: sans-serif; }
        #view { position: relative; width: 100%; height: 400px; border-radius: 12px; overflow: hidden; background: #000; }
        video, canvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover; }
    </style>
</head>
<body>
    <div id="view">
        <video id="webcam" autoplay playsinline muted></video>
        <canvas id="overlay"></canvas>
    </div>
    <div id="status" style="padding: 10px; font-size: 12px; color: #00FF00;">Initializing Google AI...</div>

    <script type="module">
        import { FaceLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

        const video = document.getElementById("webcam");
        const canvas = document.getElementById("overlay");
        const ctx = canvas.getContext("2d");
        const status = document.getElementById("status");
        let faceLandmarker;

        async function setup() {
            try {
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
                status.innerText = "AI Active - Encoding locally";
            } catch (e) {
                status.innerText = "Error: " + e.message;
                status.style.color = "#FF4B4B";
            }
        }

        async function predict() {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const results = await faceLandmarker.detectForVideo(video, performance.now());

            ctx.clearRect(0, 0, canvas.width, canvas.height);
            if (results.faceLandmarks.length > 0) {
                const landmarks = results.faceLandmarks[0];
                
                // Draw mesh
                const drawingUtils = new DrawingUtils(ctx);
                drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_TESSELATION, {color: "#00FF0030", lineWidth: 1});
                
                // Send only mathematical coordinates to Python
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

# --- PAGE NAVIGATION ---
st.sidebar.title("Biometric Privacy v1")
page = st.sidebar.radio("Go to", ["1. Register", "2. Live Recognition", "3. Log"])

if page == "1. Register":
    st.header("👤 Register New Identity")
    name = st.text_input("Person Name (Uppercase)").upper()
    st.info("The browser will encode your face landmarks. No photo is stored.")
    
    # Render the JS component
    raw_data = st.components.v1.html(INTERFACE_CODE, height=480)
    
    if raw_data and name:
        if st.button(f"Save Face Print for {name}"):
            st.session_state.db[name] = json.loads(raw_data)
            with open(DB_FILE, "w") as f: json.dump(st.session_state.db, f)
            st.success(f"Registered {name}!")
            st.rerun()

elif page == "2. Live Recognition":
    st.header("📹 Secure Attendance Stream")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        current_face_json = st.components.v1.html(INTERFACE_CODE, height=480)
        
    with col2:
        if current_face_json:
            current_face = json.loads(current_face_json)
            identified = "Unknown"
            
            # Efficient Matching (Euclidean Distance on key landmarks)
            for name, saved_face in st.session_state.db.items():
                curr_arr = np.array([[p['x'], p['y']] for p in current_face[:30]])
                save_arr = np.array([[p['x'], p['y']] for p in saved_face[:30]])
                dist = np.mean(np.linalg.norm(curr_arr - save_arr, axis=1))
                
                if dist < 0.04: # Similarity threshold
                    identified = name
                    break
            
            st.metric("Detected Identity", identified)
            
            # Log attendance once per session
            if identified != "Unknown" and identified not in st.session_state.logs["Name"].values:
                now = datetime.now().strftime("%H:%M:%S")
                new_entry = pd.DataFrame({"Name": [identified], "Time": [now]})
                st.session_state.logs = pd.concat([st.session_state.logs, new_entry], ignore_index=True)
                st.session_state.logs.to_csv(LOG_FILE, index=False)
                st.toast(f"Attendance recorded for {identified}")

elif page == "3. Log":
    st.header("📊 Attendance Summary")
    st.dataframe(st.session_state.logs, use_container_width=True)
    
    csv = st.session_state.logs.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Records (.CSV)", data=csv, file_name="attendance.csv")
