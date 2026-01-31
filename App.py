import streamlit as st
import pandas as pd
import json
import base64
import os
import numpy as np
from datetime import datetime

# --- 1. INITIALIZATION ---
DB_FILE = "registered_faces.json"
if 'registered_users' not in st.session_state:
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            st.session_state.registered_users = json.load(f)
    else:
        st.session_state.registered_users = []

# --- 2. SERVER-SIDE CALCULATION (THE BRAIN) ---
def get_best_match(input_vector):
    """Calculates the Euclidean distance on the server."""
    if not st.session_state.registered_users:
        return "Unknown", 0
    
    input_vec = np.array(input_vector)
    best_name = "Unknown"
    min_dist = 0.55  # Threshold: smaller is stricter
    
    for user in st.session_state.registered_users:
        registered_vec = np.array(user['encoding'])
        # Euclidean Distance: sqrt(sum((a-b)^2))
        dist = np.linalg.norm(input_vec - registered_vec)
        if dist < min_dist:
            min_dist = dist
            best_name = user['name']
    
    confidence = round((1 - min_dist) * 100) if best_name != "Unknown" else 0
    return best_name, confidence

# --- 3. ATTENDANCE PAGE ---
def attendance_page():
    st.title("📹 Hybrid Privacy Scanner")
    st.info("The video stays in your browser. Only mathematical vectors are sent to the server for verification.")

    # JavaScript: Handles Camera + Detection, but sends VECTOR to Python
    raw_js = """
    <div style="position: relative; display: inline-block; width: 100%;">
        <video id="v" autoplay muted playsinline style="width: 100%; max-width: 600px; border-radius: 10px; background:#000; transform: scaleX(-1);"></video>
        <canvas id="c" style="position: absolute; top: 0; left: 0; transform: scaleX(-1);"></canvas>
    </div>
    <script type="module">
        import * as faceapi from 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api/dist/face-api.esm.js';
        const v = document.getElementById('v');
        const c = document.getElementById('c');
        
        async function start() {
            const URL = 'https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js/weights';
            await faceapi.nets.tinyFaceDetector.loadFromUri(URL);
            await faceapi.nets.faceLandmark68Net.loadFromUri(URL);
            await faceapi.nets.faceRecognitionNet.loadFromUri(URL);
            
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            v.srcObject = stream;
            
            v.onloadedmetadata = () => {
                c.width = v.videoWidth;
                c.height = v.videoHeight;
                run();
            };
        }

        async function run() {
            setInterval(async () => {
                const detections = await faceapi.detectAllFaces(v, new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks().withFaceDescriptors();
                const ctx = c.getContext('2d');
                ctx.clearRect(0, 0, c.width, c.height);

                if (detections.length > 0) {
                    // Draw the box locally for immediate feedback
                    const displaySize = { width: v.videoWidth, height: v.videoHeight };
                    const resized = faceapi.resizeResults(detections, displaySize);
                    faceapi.draw.drawDetections(c, resized);

                    // SEND THE VECTOR TO THE SERVER (PYTHON)
                    window.parent.postMessage({
                        type: 'streamlit:setComponentValue',
                        value: Array.from(detections[0].descriptor)
                    }, '*');
                }
            }, 600);
        }
        start();
    </script>
    """
    
    # This component returns the vector from the browser to Python
    face_vector = st.components.v1.html(raw_js, height=500)

    if face_vector:
        # Step 4: Python does the calculation
        name, conf = get_best_match(face_vector)
        
        # Display the server's conclusion
        if name != "Unknown":
            st.success(f"✅ Recognized: **{name}** ({conf}% match)")
            if st.button(f"Log Attendance for {name}"):
                st.balloons()
                # (Add your CSV logging logic here)
        else:
            st.warning("🔍 Scanning... (No match found in server database)")

# --- NAV ---
page = st.sidebar.radio("Navigation", ["Register Face", "Take Attendance"])
if page == "Take Attendance":
    attendance_page()
else:
    # (Your existing registration_page code)
    st.write("Registration page goes here.")
