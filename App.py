import streamlit as st
import pandas as pd
from datetime import datetime
import json
import os
import base64

# --- 1. INITIALIZATION & CONFIG ---
st.set_page_config(page_title="AI Face Attendance", layout="wide")

LOG_FILE = "attendance_log.csv"
FACE_API_JS = "https://cdn.jsdelivr.net/npm/face-api.js@0.22.2/dist/face-api.min.js"

# Initialize Session States
if 'registered_users' not in st.session_state:
    st.session_state.registered_users = [] 
if 'already_logged' not in st.session_state:
    st.session_state.already_logged = set()
if 'last_detection' not in st.session_state:
    st.session_state.last_detection = None

# Ensure CSV exists
if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=["Name", "Date", "Time", "Status"]).to_csv(LOG_FILE, index=False)

# --- 2. REGISTRATION LOGIC ---
def registration_page():
    st.title("👤 User Registration")
    st.info("Upload a clear photo to extract your unique facial descriptor.")
    
    with st.form("reg_form", clear_on_submit=True):
        name = st.text_input("Full Name").strip().upper()
        img_file = st.file_uploader("Upload Face Photo", type=['jpg', 'png', 'jpeg'])
        submit = st.form_submit_button("Register User")

    if submit and name and img_file:
        img_base64 = base64.b64encode(img_file.read()).decode()
        
        # This hidden component extracts the encoding once and returns it to Python
        js_reg = f"""
        <script src="{FACE_API_JS}"></script>
        <script>
            async function encode() {{
                const MODEL_URL = 'https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js/weights';
                await faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL);
                await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);
                await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
                
                const img = new Image();
                img.src = "data:image/jpeg;base64,{img_base64}";
                img.onload = async () => {{
                    const det = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor();
                    if (det) {{
                        window.parent.postMessage({{
                            type: 'streamlit:setComponentValue', 
                            value: Array.from(det.descriptor)
                        }}, '*');
                    }} else {{
                        window.parent.postMessage({{type: 'streamlit:setComponentValue', value: 'ERR'}}, '*');
                    }}
                }};
            }}
            encode();
        </script>
        """
        result = st.components.v1.html(js_reg, height=0)
        
        if result == 'ERR':
            st.error("❌ No face detected. Please try again with a better photo.")
        elif isinstance(result, list):
            # Check if name already exists
            if not any(u['name'] == name for u in st.session_state.registered_users):
                st.session_state.registered_users.append({"name": name, "encoding": result})
                st.success(f"✅ {name} registered successfully!")
            else:
                st.warning(f"⚠️ {name} is already in the system.")

# --- 3. ATTENDANCE LOGIC ---
def attendance_page():
    st.title("📹 Live Smart Attendance")
    
    if not st.session_state.registered_users:
        st.warning("⚠️ No users registered. Please go to Registration first.")
        return

    # Data to pass to JavaScript for local matching
    known_faces_json = json.dumps(st.session_state.registered_users)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.write("### Camera Feed")
        # The JavaScript below handles the video feed and the drawing overlay
        js_attendance = f"""
        <div style="position: relative; width: 100%; max-width: 640px;">
            <video id="video" autoplay muted playsinline style="width: 100%; border-radius: 12px; background: #222;"></video>
            <canvas id="overlay" style="position: absolute; top: 0; left: 0;"></canvas>
        </div>
        <div id="status" style="margin-top:10px; font-family: sans-serif; color: #555;">Initializing AI...</div>

        <script src="{FACE_API_JS}"></script>
        <script>
            const video = document.getElementById('video');
            const canvas = document.getElementById('overlay');
            const status = document.getElementById('status');
            const knownData = {known_faces_json};
            let lastSent = "";

            async function start() {{
                const MODEL_URL = 'https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js/weights';
                try {{
                    await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
                    await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
                    await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);

                    const stream = await navigator.mediaDevices.getUserMedia({{ video: true }});
                    video.srcObject = stream;

                    video.onloadedmetadata = () => {{
                        canvas.width = video.videoWidth;
                        canvas.height = video.videoHeight;
                        status.innerText = "Scanning Active - Ready";
                        runRecognition();
                    }};
                }} catch(e) {{ status.innerText = "Error: " + e.message; }}
            }}

            async function runRecognition() {{
                const labeledDescriptors = knownData.map(u => 
                    new faceapi.LabeledFaceDescriptors(u.name, [new Float32Array(u.encoding)])
                );
                const faceMatcher = new faceapi.FaceMatcher(labeledDescriptors, 0.55);

                setInterval(async () => {{
                    const detections = await faceapi.detectAllFaces(video, new faceapi.TinyFaceDetectorOptions())
                        .withFaceLandmarks().withFaceDescriptors();
                    
                    const displaySize = {{ width: video.videoWidth, height: video.videoHeight }};
                    const resized = faceapi.resizeResults(detections, displaySize);
                    
                    const ctx = canvas.getContext('2d');
                    ctx.clearRect(0, 0, canvas.width, canvas.height);

                    resized.forEach(det => {{
                        const result = faceMatcher.findBestMatch(det.descriptor);
                        const box = det.detection.box;
                        
                        // Local Draw (No Lag)
                        new faceapi.draw.DrawBox(box, {{ label: result.toString() }}).draw(canvas);

                        if (result.label !== 'unknown' && result.label !== lastSent) {{
                            lastSent = result.label;
                            window.parent.postMessage({{
                                type: 'streamlit:setComponentValue', 
                                value: result.label
                            }}, '*');
                        }}
                    }});
                }}, 500); // Scan every 500ms
            }}
            start();
        </script>
        """
        detected_name = st.components.v1.html(js_attendance, height=520)

    with col2:
        st.write("### Recent Logs")
        
        # Process detection in Python
        if detected_name and isinstance(detected_name, str):
            today = datetime.now().strftime("%Y-%m-%d")
            log_key = f"{detected_name}_{today}"
            
            if log_key not in st.session_state.already_logged:
                st.session_state.already_logged.add(log_key)
                now = datetime.now()
                new_entry = {
                    "Name": detected_name,
                    "Date": today,
                    "Time": now.strftime("%H:%M:%S"),
                    "Status": "Present"
                }
                # Prepend to list for UI
                st.session_state.attendance_records.insert(0, new_entry)
                # Append to CSV
                pd.DataFrame([new_entry]).to_csv(LOG_FILE, mode='a', header=False, index=False)
                st.toast(f"Logged: {detected_name}!", icon="✅")

        if st.session_state.attendance_records:
            st.table(pd.DataFrame(st.session_state.attendance_records).head(10))
        else:
            st.info("No one has been logged yet today.")

# --- 4. NAVIGATION ---
page = st.sidebar.radio("Menu", ["User Registration", "Live Attendance"])

if page == "User Registration":
    registration_page()
else:
    attendance_page()

if st.sidebar.button("Reset Session Logs"):
    st.session_state.already_logged = set()
    st.session_state.attendance_records = []
    st.rerun()
