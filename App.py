import streamlit as st
import pandas as pd
from datetime import datetime
import streamlit.components.v1 as components
import json
import os

# --- 1. INITIALIZATION ---
if 'registered_users' not in st.session_state:
    st.session_state.registered_users = [] # Stores: {"name": "NAME", "encoding": [...]}
if 'attendance_records' not in st.session_state:
    st.session_state.attendance_records = []
if 'already_logged' not in st.session_state:
    st.session_state.already_logged = set()

LOG_FILE = "attendance_log.csv"
if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=["Name", "Date", "Time", "Status"]).to_csv(LOG_FILE, index=False)

# CDN for Face-API
FACE_API_JS = "https://cdn.jsdelivr.net/npm/face-api.js@0.22.2/dist/face-api.min.js"

# --- PAGE 1: REGISTRATION (with Local Encoding) ---
def registration_page():
    st.title("👤 Register with Face Encoding")
    name = st.text_input("Full Name").strip().upper()
    img_file = st.file_uploader("Upload Clear Photo", type=['jpg', 'png', 'jpeg'])

    if img_file and name:
        # Pass the image to JS to extract encoding locally
        img_bytes = img_file.read()
        img_base64 = pd.io.common.base64.b64encode(img_bytes).decode()
        
        js_reg = f"""
        <script src="{FACE_API_JS}"></script>
        <script>
            async function encode() {{
                await faceapi.nets.ssdMobilenetv1.loadFromUri('https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights');
                await faceapi.nets.faceRecognitionNet.loadFromUri('https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights');
                await faceapi.nets.faceLandmark68Net.loadFromUri('https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights');
                
                const img = new Image();
                img.src = "data:image/jpeg;base64,{img_base64}";
                img.onload = async () => {{
                    const detections = await faceapi.detectAllFaces(img).withFaceLandmarks().withFaceDescriptors();
                    if (detections.length === 0) {{
                        window.parent.postMessage({{type: 'streamlit:setComponentValue', value: 'ERROR_NO_FACE'}}, '*');
                    }} else if (detections.length > 1) {{
                        window.parent.postMessage({{type: 'streamlit:setComponentValue', value: 'ERROR_MULTIPLE'}}, '*');
                    }} else {{
                        window.parent.postMessage({{
                            type: 'streamlit:setComponentValue', 
                            value: Array.from(detections[0].descriptor)
                        }}, '*');
                    }}
                }};
            }}
            encode();
        </script>
        """
        result = components.html(js_reg, height=0)
        
        if result == "ERROR_NO_FACE":
            st.error("❌ No face detected. Use a clearer, higher-resolution photo.")
        elif result == "ERROR_MULTIPLE":
            st.error("❌ Multiple people detected. Please upload a photo with only one person.")
        elif isinstance(result, list):
            if not any(u['name'] == name for u in st.session_state.registered_users):
                st.session_state.registered_users.append({"name": name, "encoding": result})
                st.success(f"✅ {name} registered with high-speed local encoding!")

# --- PAGE 2: LIVE ATTENDANCE (with Local Comparison) ---
def attendance_page():
    st.title("📹 Live Attendance")
    
    if not st.session_state.registered_users:
        st.info("Database empty. Please register users first.")
        return

    # Prepare known faces for JS
    known_data_json = json.dumps(st.session_state.registered_users)

    js_attendance = f"""
    <div style="text-align: center;">
        <video id="video" autoplay muted style="width: 100%; max-width: 500px; border-radius: 10px;"></video>
        <audio id="chime" src="https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3"></audio>
        <p id="status" style="color: gray;">Initializing local AI...</p>
    </div>
    <script src="{FACE_API_JS}"></script>
    <script>
        const video = document.getElementById('video');
        const chime = document.getElementById('chime');
        const status = document.getElementById('status');
        const knownData = {known_data_json};

        async function start() {{
            const MODEL_URL = 'https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights';
            await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
            await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
            await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);

            const labeledDescriptors = knownData.map(u => 
                new faceapi.LabeledFaceDescriptors(u.name, [new Float32Array(u.encoding)])
            );
            const faceMatcher = new faceapi.FaceMatcher(labeledDescriptors, 0.6);

            const stream = await navigator.mediaDevices.getUserMedia({{ video: true }});
            video.srcObject = stream;
            status.innerText = "AI Active: Monitoring for faces...";

            setInterval(async () => {{
                const detections = await faceapi.detectAllFaces(video, new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks().withFaceDescriptors();
                
                if (detections.length > 1) {{
                    status.innerText = "⚠️ Multiple people detected! Please look one at a time.";
                }} else if (detections.length === 1) {{
                    const bestMatch = faceMatcher.findBestMatch(detections[0].descriptor);
                    if (bestMatch.label !== 'unknown') {{
                        chime.play();
                        window.parent.postMessage({{type: 'streamlit:setComponentValue', value: bestMatch.label}}, '*');
                    }}
                    status.innerText = "Scanning: " + bestMatch.label;
                }} else {{
                    status.innerText = "Monitoring...";
                }}
            }}, 1500);
        }}
        start();
    </script>
    """
    
    match_name = components.html(js_attendance, height=500)

    if match_name and match_name not in st.session_state.already_logged:
        timestamp = datetime.now()
        entry = {"Name": match_name, "Date": timestamp.strftime("%Y-%m-%d"), "Time": timestamp.strftime("%H:%M:%S"), "Status": "Present"}
        st.session_state.attendance_records.append(entry)
        st.session_state.already_logged.add(match_name)
        pd.DataFrame([entry]).to_csv(LOG_FILE, mode='a', header=False, index=False)
        st.success(f"🔔 {match_name} marked present!")
        st.balloons()

# --- PAGE 3: LOGS ---
def logs_page():
    st.title("📄 Attendance Logs")
    present_names = [r["Name"] for r in st.session_state.attendance_records]
    all_data = list(st.session_state.attendance_records)
    
    for user in st.session_state.registered_users:
        if user['name'] not in present_names:
            all_data.append({"Name": user['name'], "Date": datetime.now().strftime("%Y-%m-%d"), "Time": "-", "Status": "Absent"})
    
    df = pd.DataFrame(all_data)
    st.dataframe(df, use_container_width=True)
    st.download_button("📥 Download Report", df.to_csv(index=False).encode('utf-8'), "attendance.csv", "text/csv")

# --- NAV ---
page = st.sidebar.radio("Menu", ["Register", "Attendance", "View Logs"])
if page == "Register": registration_page()
elif page == "Attendance": attendance_page()
else: logs_page()
