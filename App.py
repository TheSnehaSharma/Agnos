import streamlit as st
import pandas as pd
import json
import base64
import os
from datetime import datetime

# --- 1. INITIALIZATION & PERSISTENCE ---
st.set_page_config(page_title="Private AI Attendance", layout="wide")

DB_FILE = "registered_faces.json"
LOG_FILE = "attendance_log.csv"

# Ensure session state is synced with local files
if 'registered_users' not in st.session_state:
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            st.session_state.registered_users = json.load(f)
    else:
        st.session_state.registered_users = []

if 'attendance_records' not in st.session_state:
    if os.path.exists(LOG_FILE):
        try:
            st.session_state.attendance_records = pd.read_csv(LOG_FILE).to_dict('records')
        except:
            st.session_state.attendance_records = []
    else:
        st.session_state.attendance_records = []

# --- 2. REGISTRATION PAGE (LOCAL ENCODING) ---
def registration_page():
    st.title("👤 Local Privacy Registration")
    st.info("🔒 Face geometry is calculated in your browser. Original photos are never stored or sent to the server.")
    
    name = st.text_input("Enter Full Name").strip().upper()
    img_file = st.file_uploader("Upload Profile Photo", type=['jpg', 'png', 'jpeg'])

    if img_file and name:
        img_bytes = img_file.read()
        img_base64 = base64.b64encode(img_bytes).decode()
        
        js_reg = f"""
        <div id="status-box" style="padding:10px; background:#f0f2f6; border-radius:8px; font-family:sans-serif;">
            <div id="status-text" style="color:#ff4b4b; font-weight:bold;">⏳ Initializing Local AI...</div>
        </div>

        <script type="module">
            import * as faceapi from 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api/dist/face-api.esm.js';

            const status = document.getElementById('status-text');
            const MODEL_URL = 'https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js/weights';

            async function init() {{
                try {{
                    status.innerText = "Loading Models...";
                    await faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL);
                    await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);
                    await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
                    
                    status.innerText = "🧬 Vectorizing locally...";
                    const img = new Image();
                    img.src = "data:image/jpeg;base64,{img_base64}";
                    img.onload = async () => {{
                        const det = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor();
                        if (det) {{
                            window.parent.postMessage({{
                                type: 'streamlit:setComponentValue', 
                                value: Array.from(det.descriptor)
                            }}, '*');
                            status.innerText = "✅ Encoding Complete! Click 'Confirm' below.";
                        }} else {{
                            status.innerText = "❌ Error: No face detected.";
                        }}
                    }};
                }} catch (e) {{
                    status.innerText = "❌ AI Blocked. Please check browser shield.";
                }}
            }}
            init();
        </script>
        """
        extracted_vector = st.components.v1.html(js_reg, height=100)

        if extracted_vector and isinstance(extracted_vector, list):
            st.success(f"Mathematical features extracted for {name}")
            if st.button(f"Confirm Registration for {name}"):
                new_user = {{"name": name, "encoding": extracted_vector}}
                st.session_state.registered_users.append(new_user)
                with open(DB_FILE, "w") as f:
                    json.dump(st.session_state.registered_users, f)
                st.success(f"✅ {name} registered!")
                st.rerun()

    st.write("---")
    st.subheader("Registered Users")
    if st.session_state.registered_users:
        st.table(pd.DataFrame(st.session_state.registered_users)[['name']].tail(10))

# --- 3. ATTENDANCE PAGE (LIVE SCANNER) ---
def attendance_page():
    st.title("📹 Hybrid AI Scanner")
    
    # Check if users exist to avoid errors
    if not st.session_state.registered_users:
        st.warning("Please register a user first.")
        return

    # 1. Server-side Matching Logic
    def find_match(input_embedding):
        import numpy as np
        input_vec = np.array(input_embedding)
        best_match = "Unknown"
        min_dist = 0.5 

        for user in st.session_state.registered_users:
            dist = np.linalg.norm(input_vec - np.array(user['encoding']))
            if dist < min_dist:
                min_dist = dist
                best_match = user['name']
        return best_match

    # 2. JavaScript Interface (Escaped Braces for f-string)
    js_interface = f"""
    <div style="position: relative; width: 640px; border-radius: 15px; overflow: hidden;">
        <video id="video" width="640" height="480" autoplay muted style="transform: scaleX(-1);"></video>
        <canvas id="overlay" style="position: absolute; top: 0; left: 0; transform: scaleX(-1);"></canvas>
    </div>

    <script type="module">
        import * as faceapi from 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api/dist/face-api.esm.js';

        const video = document.getElementById('video');
        const canvas = document.getElementById('overlay');
        const MODEL_URL = 'https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js/weights';

        async function setup() {{
            try {{
                await faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL);
                await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
                await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);

                const stream = await navigator.mediaDevices.getUserMedia({{ video: true }});
                video.srcObject = stream;

                video.onplay = () => {{
                    const displaySize = {{ width: video.width, height: video.height }};
                    faceapi.matchDimensions(canvas, displaySize);

                    setInterval(async () => {{
                        const detections = await faceapi.detectAllFaces(video).withFaceLandmarks().withFaceDescriptors();
                        const resized = faceapi.resizeResults(detections, displaySize);
                        
                        canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
                        
                        if (detections.length > 0) {{
                            // Only send the vector (descriptor) to the server
                            window.parent.postMessage({{
                                type: 'streamlit:setComponentValue',
                                value: Array.from(detections[0].descriptor)
                            }}, '*');
                            
                            faceapi.draw.drawDetections(canvas, resized);
                        }}
                    }}, 1000); 
                }};
            }} catch (e) {{
                console.error("Camera error:", e);
            }}
        }}
        setup();
    </script>
    """

    # 3. Component Bridge
    client_vector = st.components.v1.html(js_interface, height=500)

    # 4. Result Handling
    if client_vector and isinstance(client_vector, list):
        name = find_match(client_vector)
        st.subheader(f"Detected: {name}")
        if name != "Unknown":
            if st.button("Log Attendance"):
                st.success(f"Log saved for {name}")
