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
    st.title("📹 Live Presence Scanner")
    
    if not st.session_state.registered_users:
        st.warning("⚠️ No users registered. Showing feed in 'Detection Only' mode.")
    
    known_json = json.dumps(st.session_state.registered_users)

    js_attendance = f"""
    <div style="position: relative; display: inline-block; width: 100
