import streamlit as st
import pandas as pd
import json
import base64
import os
import numpy as np
from datetime import datetime

# --- 1. INITIALIZATION & PERSISTENCE ---
st.set_page_config(page_title="Private AI Attendance", layout="wide")

DB_FILE = "registered_faces.json"
LOG_FILE = "attendance_log.csv"

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

# --- 2. SERVER-SIDE CALCULATION ENGINE ---
def calculate_match(input_vector):
    """Computes Euclidean distance between live vector and database on the server."""
    if not st.session_state.registered_users:
        return None, 0
    
    input_vec = np.array(input_vector)
    best_name = None
    min_dist = 0.5  # Strictness threshold (0.4 - 0.6 is typical)
    
    for user in st.session_state.registered_users:
        known_vec = np.array(user['encoding'])
        # Euclidean distance calculation
        dist = np.linalg.norm(input_vec - known_vec)
        if dist < min_dist:
            min_dist = dist
            best_name = user['name']
    
    confidence = round((1 - min_dist) * 100) if best_name else 0
    return best_name, confidence

# --- 3. REGISTRATION PAGE ---
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
                new_user = {"name": name, "encoding": extracted_vector}
                st.session_state.registered_users.append(new_user)
                with open(DB_FILE, "w") as f:
                    json.dump(st.session_state.registered_users, f)
                st.success(f"✅ {name} registered!")
                st.rerun()

    st.write("---")
    st.subheader("Registered Users")
    if st.session_state.registered_users:
        st.table(pd.DataFrame(st.session_state.registered_users)[['name']].tail(10))

# --- 4. ATTENDANCE PAGE (LIVE SCANNER) ---
def attendance_page():
    st.title("📹 Live Presence Scanner")
    
    if not st.session_state.registered_users:
        st.warning("⚠️ No users registered. Please go to Registration first.")
        return

    # JS code as a raw string to avoid f-string curly brace issues
    raw_js_attendance = """
    <div style="position: relative; display: inline-block; width: 100%;">
        <video id="v" autoplay muted playsinline style="width: 100%; max-width: 640px; border-radius: 10px; background:#000; transform: scaleX(-1);"></video>
        <canvas id="c" style="position: absolute; top: 0; left: 0; transform: scaleX(-1);"></canvas>
    </div>
    <p id="msg" style="font-family:sans-serif; color: #666; margin-top:10px;">Initializing Camera...</p>

    <script type="module">
        import * as faceapi from 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api/dist/face-api.esm.js';
        const v = document.getElementById('v');
        const c = document.getElementById('c');
        const m = document.getElementById('msg');
        
        async function start() {
            try {
                const URL = 'https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js/weights';
                await faceapi.nets.tinyFaceDetector.loadFromUri(URL);
                await faceapi.nets.faceLandmark68Net.loadFromUri(URL);
                await faceapi.nets.faceRecognitionNet.loadFromUri(URL);
                
                const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                v.srcObject = stream;
                v.onloadedmetadata = () => {
                    c.width = v.videoWidth;
                    c.height = v.videoHeight;
                    m.innerText = "🔒 Scanner Active (Privacy-Enforced)";
                    run();
                };
            } catch(e) { m.innerText = "❌ Error: " + e.message; }
        }

        async function run() {
            setInterval(async () => {
                const detections = await faceapi.detectAllFaces(v, new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks().withFaceDescriptors();
                const ctx = c.getContext('2d');
                ctx.clearRect(0, 0, c.width, c.height);

                if (detections.length > 0) {
                    // 1. Draw box locally for instant feedback
                    const displaySize = { width: v.videoWidth, height: v.videoHeight };
                    const resized = faceapi.resizeResults(detections, displaySize);
                    faceapi.draw.drawDetections(c, resized);

                    // 2. Send only the mathematical vector to Python for calculation
                    window.parent.postMessage({
                        type: 'streamlit:setComponentValue',
                        value: Array.from(detections[0].descriptor)
                    }, '*');
                }
            }, 800);
        }
        start();
    </script>
    """
    
    # Render the component and capture the face vector
    live_vector = st.components.v1.html(raw_js_attendance, height=520)

    if live_vector and isinstance(live_vector, list):
        # SERVER-SIDE CALCULATION
        matched_name, confidence = calculate_match(live_vector)
        
        if matched_name:
            st.success(f"👤 **{matched_name}** recognized ({confidence}% match)")
            if st.button(f"Log Presence for {matched_name}"):
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                record = {"name": matched_name, "timestamp": now}
                st.session_state.attendance_records.append(record)
                pd.DataFrame(st.session_state.attendance_records).to_csv(LOG_FILE, index=False)
                st.toast(f"Logged {matched_name} at {now}!")
        else:
            st.info("🔍 Searching database...")

# --- 5. NAVIGATION ---
choice = st.sidebar.radio("Navigation", ["Take Attendance", "Register Face"])

if choice == "Register Face":
    registration_page()
else:
    attendance_page()
