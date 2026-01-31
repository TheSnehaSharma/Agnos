import streamlit as st
import pandas as pd
from datetime import datetime
import streamlit.components.v1 as components
import json
import os
import base64

# --- 1. INITIALIZATION ---
if 'registered_users' not in st.session_state:
    st.session_state.registered_users = [] 
if 'attendance_records' not in st.session_state:
    st.session_state.attendance_records = []
if 'already_logged' not in st.session_state:
    st.session_state.already_logged = set()

LOG_FILE = "attendance_log.csv"
if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=["Name", "Date", "Time", "Status"]).to_csv(LOG_FILE, index=False)

FACE_API_JS = "https://cdn.jsdelivr.net/npm/face-api.js@0.22.2/dist/face-api.min.js"

# --- PAGE 1: REGISTRATION ---
def registration_page():
    st.title("👤 User Registration")
    
    with st.form("reg_form", clear_on_submit=False):
        name = st.text_input("Full Name").strip().upper()
        img_file = st.file_uploader("Upload Face Photo", type=['jpg', 'png', 'jpeg'])
        submit = st.form_submit_button("Register User")

    if submit:
        if not name or not img_file:
            st.warning("⚠️ Please provide both a name and a photo.")
        else:
            img_bytes = img_file.read()
            img_base64 = base64.b64encode(img_bytes).decode()
            
            # Using a visible component for registration so we can see errors
            st.info(f"🧬 AI is analyzing {name}'s face...")
            
            js_reg = f"""
            <div id="status" style="font-family: sans-serif; font-size: 12px; color: #666;">Initializing AI Models...</div>
            <script src="{FACE_API_JS}"></script>
            <script>
                const status = document.getElementById('status');
                async function encode() {{
                    try {{
                        const MODEL_URL = 'https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights';
                        status.innerText = "Loading weights...";
                        await faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL);
                        await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);
                        await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
                        
                        status.innerText = "Processing image...";
                        const img = new Image();
                        img.src = "data:image/jpeg;base64,{img_base64}";
                        img.onload = async () => {{
                            const detections = await faceapi.detectAllFaces(img).withFaceLandmarks().withFaceDescriptors();
                            if (detections.length === 0) {{
                                status.innerText = "Error: No face detected";
                                window.parent.postMessage({{type: 'streamlit:setComponentValue', value: 'ERR_NONE'}}, '*');
                            }} else if (detections.length > 1) {{
                                status.innerText = "Error: Multiple faces";
                                window.parent.postMessage({{type: 'streamlit:setComponentValue', value: 'ERR_MULTI'}}, '*');
                            }} else {{
                                status.innerText = "Success!";
                                window.parent.postMessage({{
                                    type: 'streamlit:setComponentValue', 
                                    value: Array.from(detections[0].descriptor)
                                }}, '*');
                            }}
                        }};
                    }} catch (e) {{
                        status.innerText = "Load Error: " + e.message;
                        window.parent.postMessage({{type: 'streamlit:setComponentValue', value: 'ERR_LOAD'}}, '*');
                    }}
                }}
                encode();
            </script>
            """
            # Height is 30 so you can see the small status text
            result = components.html(js_reg, height=50)
            
            if result == "ERR_NONE":
                st.error("❌ No face detected. Please use a clearer photo.")
            elif result == "ERR_MULTI":
                st.error("❌ Multiple people detected. Please upload a solo photo.")
            elif result == "ERR_LOAD":
                st.error("❌ AI Models failed to load. Please check your internet connection.")
            elif isinstance(result, list):
                if not any(u['name'] == name for u in st.session_state.registered_users):
                    st.session_state.registered_users.append({"name": name, "encoding": result})
                    st.success(f"✅ {name} registered successfully!")
                else:
                    st.info(f"{name} is already registered.")

# --- PAGE 2: LIVE ATTENDANCE ---
def attendance_page():
    st.title("📹 Live Attendance")
    
    if not st.session_state.registered_users:
        st.warning("⚠️ No users registered yet.")
        return

    known_data_json = json.dumps(st.session_state.registered_users)

    js_attendance = f"""
    <div style="position: relative; text-align: center;">
        <video id="video" autoplay muted style="width: 100%; max-width: 500px; border-radius: 10px;"></video>
        <canvas id="canvas" style="position: absolute; top: 0; left: 50%; transform: translateX(-50%);"></canvas>
        <audio id="chime" src="https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3"></audio>
        <div id="ui-label" style="background: rgba(0,0,0,0.6); color: white; padding: 10px; margin-top: 10px; border-radius: 5px; font-family: sans-serif;">
            Initializing AI...
        </div>
    </div>

    <script src="{FACE_API_JS}"></script>
    <script>
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const label = document.getElementById('ui-label');
        const chime = document.getElementById('chime');
        const knownData = {known_data_json};

        async function start() {{
            const MODEL_URL = 'https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights';
            try {{
                label.innerText = "Loading AI models...";
                await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
                await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
                await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);

                const labeledDescriptors = knownData.map(u => 
                    new faceapi.LabeledFaceDescriptors(u.name, [new Float32Array(u.encoding)])
                );
                const faceMatcher = new faceapi.FaceMatcher(labeledDescriptors, 0.55);

                const stream = await navigator.mediaDevices.getUserMedia({{ video: {{}} }});
                video.srcObject = stream;

                video.addEventListener('play', () => {{
                    const displaySize = {{ width: video.videoWidth, height: video.videoHeight }};
                    canvas.width = displaySize.width;
                    canvas.height = displaySize.height;
                    faceapi.matchDimensions(canvas, displaySize);
                    label.innerText = "AI Scanning Active";

                    setInterval(async () => {{
                        const detections = await faceapi.detectAllFaces(video, new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks().withFaceDescriptors();
                        const resizedDetections = faceapi.resizeResults(detections, displaySize);
                        const ctx = canvas.getContext('2d');
                        ctx.clearRect(0, 0, canvas.width, canvas.height);

                        if (detections.length > 1) {{
                            label.innerText = "⚠️ Multiple faces detected!";
                        }} else if (detections.length === 1) {{
                            const result = faceMatcher.findBestMatch(resizedDetections[0].descriptor);
