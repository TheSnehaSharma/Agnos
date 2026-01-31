import streamlit as st
import pandas as pd
from datetime import datetime
import json
import base64
import os

# --- 1. INITIALIZATION ---
st.set_page_config(page_title="Privacy-First AI Attendance", layout="wide")

FACE_API_JS = "https://cdn.jsdelivr.net/npm/face-api.js@0.22.2/dist/face-api.min.js"
DB_FILE = "registered_faces.json"

# Load registered users from file or initialize empty list
if 'registered_users' not in st.session_state:
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            st.session_state.registered_users = json.load(f)
    else:
        st.session_state.registered_users = []

if 'attendance_records' not in st.session_state:
    st.session_state.attendance_records = []

# --- 2. REGISTRATION PAGE ---
def registration_page():
    st.title("👤 Private Registration")
    st.info("Photo is processed locally. Only a 'Face Map' (128 numbers) is stored.")
    
    name = st.text_input("Full Name").strip().upper()
    img_file = st.file_uploader("Upload Face Photo", type=['jpg', 'png', 'jpeg'])
    
    if img_file and name:
        img_bytes = img_file.read()
        img_base64 = base64.b64encode(img_bytes).decode()
        
        st.write("---")
        st.caption("AI Engine Status:")
        
        # Double {{ }} used for CSS/JS logic; Single { } for Python variables
        js_component = f"""
        <div id="status" style="font-family:sans-serif; color:#ff4b4b; font-size:14px; font-weight:bold;">🧬 Initializing Local AI...</div>
        <script src="{FACE_API_JS}"></script>
        <script>
            async function process() {{
                try {{
                    const MODEL_URL = 'https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js/weights';
                    await faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL);
                    await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);
                    await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
                    
                    document.getElementById('status').innerText = "Analyzing Image...";
                    const img = new Image();
                    img.src = "data:image/jpeg;base64,{img_base64}";
                    img.onload = async () => {{
                        const det = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor();
                        if (det) {{
                            document.getElementById('status').innerText = "✅ Extraction Complete!";
                            window.parent.postMessage({{
                                type: 'streamlit:setComponentValue', 
                                value: {{ "name": "{name}", "encoding": Array.from(det.descriptor) }}
                            }}, '*');
                        }} else {{
                            document.getElementById('status').innerText = "❌ No face detected!";
                            window.parent.postMessage({{type: 'streamlit:setComponentValue', value: "ERR_NO_FACE"}}, '*');
                        }}
                    }};
                }} catch (e) {{
                    document.getElementById('status').innerText = "❌ Model Load Error";
                    window.parent.postMessage({{type: 'streamlit:setComponentValue', value: "ERR_LOAD"}}, '*');
                }}
            }}
            process();
        </script>
        """
        returned_data = st.components.v1.html(js_component, height=70)

        if returned_data:
            if isinstance(returned_data, dict):
                if not any(u['name'] == returned_data['name'] for u in st.session_state.registered_users):
                    st.session_state.registered_users.append(returned_data)
                    # Save to file immediately
                    with open(DB_FILE, "w") as f:
                        json.dump(st.session_state.registeredimport streamlit as st
import pandas as pd
from datetime import datetime
import json
import base64

# --- 1. INITIALIZATION ---
st.set_page_config(page_title="Privacy-First AI Attendance", layout="wide")

FACE_API_JS = "https://cdn.jsdelivr.net/npm/face-api.js@0.22.2/dist/face-api.min.js"

if 'registered_users' not in st.session_state:
    st.session_state.registered_users = [] 
if 'attendance_records' not in st.session_state:
    st.session_state.attendance_records = []

# --- 2. REGISTRATION PAGE ---
def registration_page():
    st.title("👤 Private Registration")
    st.info("Photo is processed locally. Only a 'Face Map' (128 numbers) is stored.")
    
    name = st.text_input("Full Name").strip().upper()
    img_file = st.file_uploader("Upload Face Photo", type=['jpg', 'png', 'jpeg'])
    
    if img_file and name:
        img_bytes = img_file.read()
        img_base64 = base64.b64encode(img_bytes).decode()
        
        st.write("---")
        st.caption("AI Engine Status:")
        
        # Note the double {{ }} for JS logic and single { } for Python variables
        js_component = f"""
        <div id="status" style="font-family:sans-serif; color:#ff4b4b; font-size:14px; font-weight:bold;">🧬 Initializing Local AI...</div>
        <script src="{FACE_API_JS}"></script>
        <script>
            async function process() {{
                try {{
                    const MODEL_URL = 'https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js/weights';
                    await faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL);
                    await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);
                    await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
                    
                    document.getElementById('status').innerText = "Analyzing Image...";
                    const img = new Image();
                    img.src = "data:image/jpeg;base64,{img_base64}";
                    img.onload = async () => {{
                        const det = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor();
                        if (det) {{
                            document.getElementById('status').innerText = "✅ Extraction Complete!";
                            window.parent.postMessage({{
                                type: 'streamlit:setComponentValue', 
                                value: {{ "name": "{name}", "encoding": Array.from(det.descriptor) }}
                            }}, '*');
                        }} else {{
                            document.getElementById('status').innerText = "❌ No face detected!";
                            window.parent.postMessage({{type: 'streamlit:setComponentValue', value: "ERR_NO_FACE"}}, '*');
                        }}
                    }};
                }} catch (e) {{
                    document.getElementById('status').innerText = "❌ Model Load Error";
                    window.parent.postMessage({{type: 'streamlit:setComponentValue', value: "ERR_LOAD"}}, '*');
                }}
            }}
            process();
        </script>
        """
        returned_data = st.components.v1.html(js_component, height=70)

        if returned_data:
            if isinstance(returned_data, dict):
                if not any(u['name'] == returned_data['name'] for u in st.session_state.registered_users):
                    st.session_state.registered_users.append(returned_data)
                    st.success(f"✅ Facial map for {name} saved!")
                    st.balloons()
            elif returned_data == "ERR_NO_FACE":
                st.error("No face found in that image.")

# --- 3. ATTENDANCE PAGE ---
def attendance_page():
    st.title("📹 Live Private Attendance")
    
    if not st.session_state.registered_users:
        st.warning("⚠️ No facial maps found. Please register a user first.")
        return

    st.write(f"Users in Database: **{len(st.session_state.registered_users)}**")
    known_json = json.dumps(st.session_state.registered_users)

    js_attendance = f"""
    <div style="position: relative; display: inline-block;">
        <video id="v" autoplay muted playsinline style="width: 100%; max-width: 500px; border-radius: 10px; border: 2px solid #ff4b4b;"></video>
        <canvas id="c" style="position: absolute; top: 0; left: 0;"></canvas>
    </div>
    <p id="msg" style="font-family:sans-serif; color: #555; font-weight: bold;">Starting scanner...</p>

    <script src="{FACE_API_JS}"></script>
    <script>
        const v = document.getElementById('v');
        const c = document.getElementById('c');
        const m = document.getElementById('msg');
        const known = {known_json};

        async function start() {{
            try {{
                const URL = 'https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js/weights';
                await faceapi.nets.tinyFaceDetector.loadFromUri(URL);
                await faceapi.nets.faceLandmark68Net.loadFromUri(URL);
                await faceapi.nets.faceRecognitionNet.
