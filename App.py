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

# Initialize Session State
if 'registered_users' not in st.session_state:
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            st.session_state.registered_users = json.load(f)
    else:
        st.session_state.registered_users = []

if 'attendance_records' not in st.session_state:
    if os.path.exists(LOG_FILE):
        st.session_state.attendance_records = pd.read_csv(LOG_FILE).to_dict('records')
    else:
        st.session_state.attendance_records = []

# --- 2. REGISTRATION PAGE (ENCODE ON CLIENT) ---
def registration_page():
    st.title("👤 Local Privacy Registration")
    st.info("Your image never leaves your browser. Only a 128-digit mathematical map is saved.")
    
    name = st.text_input("Full Name").strip().upper()
    img_file = st.file_uploader("Upload Profile Photo", type=['jpg', 'png', 'jpeg'])

    if img_file and name:
        img_base64 = base64.b64encode(img_file.read()).decode()
        
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

# --- 3. ATTENDANCE PAGE (LIVE FEED) ---
def attendance_page():
    st.title("📹 Live Presence Scanner")
    
    if not st.session_state.registered_users:
        st.warning("⚠️ No users registered. Showing feed in 'Detection Only' mode.")
    
    known_json = json.dumps(st.session_state.registered_users)

    # Note the double {{ }} for JS logic
    js_attendance = f"""
    <div style="position: relative; display: inline-block; width: 100%;">
        <video id="v" autoplay muted playsinline style="width: 100%; max-width: 600px; border-radius: 10px; background:#000;"></video>
        <canvas id="c" style="position: absolute; top: 0; left: 0;"></canvas>
    </div>
    <p id="msg" style="font-family:sans-serif; color: #666; margin-top:10px;">Initializing Camera & AI...</p>

    <script type="module">
        import * as faceapi from 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api/dist/face-api.esm.js';

        const v = document.getElementById('v');
        const c = document.getElementById('c');
        const m = document.getElementById('msg');
        const known = {known_json};

        async function start() {{
            try {{
                const URL = 'https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js/weights';
                await faceapi.nets.tinyFaceDetector.loadFromUri(URL);
                await faceapi.nets.faceLandmark68Net.loadFromUri(URL);
                await faceapi.nets.faceRecognitionNet.loadFromUri(URL);

                const stream = await navigator.mediaDevices.getUserMedia({{ video: true }});
                v.srcObject = stream;

                v.onloadedmetadata = () => {{
                    c.width = v.videoWidth;
                    c.height = v.videoHeight;
                    m.innerText = "🔒 Scanner Active (0.5 Tolerance)";
                    run();
                }};
            }} catch(e) {{ m.innerText = "❌ Error: " + e.message; }}
        }}

        async function run() {{
            let faceMatcher = null;
            if (known.length > 0) {{
                const labels = known.map(u => new faceapi.Labeled
