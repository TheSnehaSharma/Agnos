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
    if os.path.exists(LOG_FILE):
        try: st.session_state.logs = pd.read_csv(LOG_FILE)
        except: st.session_state.logs = pd.DataFrame(columns=["Name", "Time"])
    else: st.session_state.logs = pd.DataFrame(columns=["Name", "Time"])

# --- COMPONENT CODE (UMD VERSION) ---
INTERFACE_CODE = """
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/vision_bundle.js"></script>
    <style>
        body { margin:0; background: #0e1117; color: #00FF00; font-family: sans-serif; }
        #view { position: relative; width: 100%; height: 400px; border-radius: 12px; overflow: hidden; background: #000; }
        video, canvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover; }
        #status { position: absolute; bottom: 10px; left: 10px; background: rgba(0,0,0,0.7); padding: 5px; font-size: 12px; }
    </style>
</head>
<body>
    <div id="view">
        <video id="webcam" autoplay playsinline muted></video>
        <canvas id="overlay"></canvas>
        <div id="status">Loading Google Models...</div>
    </div>
    <script>
        const video = document.getElementById("webcam");
        const canvas = document.getElementById("overlay");
        const ctx = canvas.getContext("2d");
        const status = document.getElementById("status");
        let faceLandmarker;

        async function init() {
            const vision = await tasksVision.FilesetResolver.forVisionTasks(
                "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
            );
            faceLandmarker = await tasksVision.FaceLandmarker.createFromOptions(vision, {
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
            status.innerText = "System Online - Scanning";
        }

        async function predict() {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const results = faceLandmarker.detectForVideo(video, performance.now());

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
                
                // Send JSON string to Python
                window.parent.postMessage({
                    type: "streamlit:setComponentValue",
                    value: JSON.stringify(landmarks)
                }, "*");
            }
            window.requestAnimationFrame(predict);
        }
        init();
    </script>
</body>
</html>
"""

# --- UI LOGIC ---
page = st.sidebar.radio("Navigate", ["Register", "Live Feed", "Log"])

if page == "Register":
    st.header("👤 Registration")
    name = st.text_input("Name").upper()
    
    val = st.components.v1.html(INTERFACE_CODE, height=420)
    
    if val and isinstance(val, str):
        st.session_state.buffered_encoding = val
        st.success(f"✅ Data Ready for {name if name else 'User'}")
    
    if st.button("Save Face to Database"):
        if name and st.session_state.get('buffered_encoding'):
            st.session_state.db[name] = json.loads(st.session_state.buffered_encoding)
            with open(DB_FILE, "w") as f:
                json.dump(st.session_state.db, f)
            st.success(f"Successfully saved {name}!")
            st.rerun()
        else:
            st.error("Cannot save: Ensure face is detected and name is entered.")

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
            
            if identified != "Unknown":
                if identified not in st.session_state.logs["Name"].values:
                    now = datetime.now().strftime("%H:%M:%S")
                    new_entry = pd.DataFrame({"Name": [identified], "Time": [now]})
                    st.session_state.logs = pd.concat([st.session_state.logs, new_entry], ignore_index=True)
                    st.session_state.logs.to_csv(LOG_FILE, index=False)
                    st.toast(f"Marked attendance for {identified}")

elif page == "Log":
    st.header("📊 Attendance Log")
    st.dataframe(st.session_state.logs, use_container_width=True)
    st.download_button("Download CSV", st.session_state.logs.to_csv(index=False), "attendance.csv")
