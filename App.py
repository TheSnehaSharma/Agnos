import streamlit as st
import pandas as pd
from datetime import datetime
import streamlit.components.v1 as components
import json
import os
import base64

# --- 1. INITIALIZATION ---
st.set_page_config(page_title="AI Attendance", layout="wide")

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
    
    with st.form("reg_form", clear_on_submit=True):
        name = st.text_input("Full Name").strip().upper()
        img_file = st.file_uploader("Upload Face Photo", type=['jpg', 'png', 'jpeg'])
        submit = st.form_submit_button("Register User")

    if submit:
        if not name or not img_file:
            st.warning("⚠️ Please provide both a name and a photo.")
        else:
            img_bytes = img_file.read()
            img_base64 = base64.b64encode(img_bytes).decode()
            st.info(f"🧬 AI is analyzing {name}'s face...")
            
            js_reg = f"""
            <div id="status" style="font-family: sans-serif; font-size: 14px; color: #333;">Initializing AI Models...</div>
            <script src="{FACE_API_JS}"></script>
            <script>
                const status = document.getElementById('status');
                async function encode() {{
                    try {{
                        const MODEL_URL = 'https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights';
                        await faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL);
                        await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);
                        await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
                        
                        const img = new Image();
                        img.src = "data:image/jpeg;base64,{img_base64}";
                        img.onload = async () => {{
                            const detections = await faceapi.detectAllFaces(img).withFaceLandmarks().withFaceDescriptors();
                            if (detections.length === 0) {{
                                window.parent.postMessage({{type: 'streamlit:setComponentValue', value: 'ERR_NONE'}}, '*');
                            }} else if (detections.length > 1) {{
                                window.parent.postMessage({{type: 'streamlit:setComponentValue', value: 'ERR_MULTI'}}, '*');
                            }} else {{
                                window.parent.postMessage({{
                                    type: 'streamlit:setComponentValue', 
                                    value: Array.from(detections[0].descriptor)
                                }}, '*');
                            }}
                        }};
                    }} catch (e) {{
                        window.parent.postMessage({{type: 'streamlit:setComponentValue', value: 'ERR_LOAD'}}, '*');
                    }}
                }}
                encode();
            </script>
            """
            result = components.html(js_reg, height=50)
            
            if result == "ERR_NONE":
                st.error("❌ No face detected.")
            elif result == "ERR_MULTI":
                st.error("❌ Multiple people detected.")
            elif isinstance(result, list):
                st.session_state.registered_users.append({{"name": name, "encoding": result}})
                st.success(f"✅ {name} registered!")

# --- PAGE 2: LIVE ATTENDANCE ---
def attendance_page():
    st.title("📹 Live Attendance")
    
    if not st.session_state.registered_users:
        st.warning("⚠️ No users registered yet.")
        return

    known_data_json = json.dumps(st.session_state.registered_users)

    # Use double curly braces for CSS and JS logic
    js_attendance = f"""
    <div style="position: relative; text-align: center;">
        <video id="video" autoplay muted playsinline style="width: 100%; max-width: 500px; border-radius: 10px;"></video>
        <canvas id="canvas" style="position: absolute; top: 0; left: 50%; transform: translateX(-50%);"></canvas>
        <div id="ui-label" style="background: rgba(0,0,0,0.6); color: white; padding: 10px; margin-top: 10px; border-radius: 5px;">
            Starting Camera...
        </div>
    </div>

    <script src="{FACE_API_JS}"></script>
    <script>
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const label = document.getElementById('ui-label');
        const knownData = {known_data_json};

        async function start() {{
            const MODEL_URL = 'https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights';
            try {{
                await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
                await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
                await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);

                const labeledDescriptors = knownData.map(u => 
                    new faceapi.LabeledFaceDescriptors(u.name, [new Float32Array(u.encoding)])
                );
                const faceMatcher = new faceapi.FaceMatcher(labeledDescriptors, 0.6);

                const stream = await navigator.mediaDevices.getUserMedia({{ video: {{}} }});
                video.srcObject = stream;

                video.addEventListener('play', () => {{
                    const displaySize = {{ width: video.videoWidth, height: video.videoHeight }};
                    faceapi.matchDimensions(canvas, displaySize);

                    setInterval(async () => {{
                        const detections = await faceapi.detectAllFaces(video, new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks().withFaceDescriptors();
                        const resizedDetections = faceapi.resizeResults(detections, displaySize);
                        
                        canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
                        faceapi.draw.drawDetections(canvas, resizedDetections);

                        if (detections.length === 1) {{
                            const result = faceMatcher.findBestMatch(resizedDetections[0].descriptor);
                            label.innerText = "Detected: " + result.toString();
                            if (result.label !== 'unknown') {{
                                window.parent.postMessage({{
                                    type: 'streamlit:setComponentValue', 
                                    value: result.label
                                }}, '*');
                            }}
                        }}
                    }}, 1000);
                }});
            }} catch (e) {{ label.innerText = "Error: " + e.message; }}
        }}
        start();
    </script>
    """
    
    detected_name = components.html(js_attendance, height=600)

    if detected_name and isinstance(detected_name, str):
        date_today = datetime.now().strftime("%Y-%m-%d")
        log_id = f"{detected_name}_{date_today}"
        
        if log_id not in st.session_state.already_logged:
            st.session_state.already_logged.add(log_id)
            new_entry = {
                "Name": detected_name, 
                "Date": date_today, 
                "Time": datetime.now().strftime("%H:%M:%S"), 
                "Status": "Present"
            }
            st.session_state.attendance_records.append(new_entry)
            pd.DataFrame([new_entry]).to_csv(LOG_FILE, mode='a', header=False, index=False)
            st.success(f"Verified: {detected_name}")

    st.subheader("Today's Attendance")
    st.table(pd.DataFrame(st.session_state.attendance_records))

# --- NAVIGATION ---
page = st.sidebar.selectbox("Go to", ["Registration", "Attendance"])
if page == "Registration":
    registration_page()
else:
    attendance_page()
