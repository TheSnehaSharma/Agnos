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
            with st.spinner(f"🧬 AI is analyzing {name}'s face..."):
                img_bytes = img_file.read()
                img_base64 = base64.b64encode(img_bytes).decode()
                
                js_reg = f"""
                <script src="{FACE_API_JS}"></script>
                <script>
                    async function encode() {{
                        const MODEL_URL = 'https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights';
                        try {{
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
                result = components.html(js_reg, height=0)
                
                if result == "ERR_NONE":
                    st.error("❌ No face detected. Please use a clearer photo.")
                elif result == "ERR_MULTI":
                    st.error("❌ Multiple people detected. Please upload a solo photo.")
                elif result == "ERR_LOAD":
                    st.error("❌ AI Models failed to load. Check internet connection.")
                elif isinstance(result, list):
                    if not any(u['name'] == name for u in st.session_state.registered_users):
                        st.session_state.registered_users.append({"name": name, "encoding": result})
                        st.success(f"✅ {name} registered successfully!")

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
        <div id="ui-label" style="background: rgba(0,0,0,0.6); color: white; padding: 10px; margin-top: 10px; border-radius: 5px;">
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
                    label.innerText = "AI Active: Scanning...";

                    setInterval(async () => {{
                        const detections = await faceapi.detectAllFaces(video, new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks().withFaceDescriptors();
                        const resizedDetections = faceapi.resizeResults(detections, displaySize);
                        const ctx = canvas.getContext('2d');
                        ctx.clearRect(0, 0, canvas.width, canvas.height);

                        if (detections.length > 1) {{
                            label.innerText = "⚠️ Multiple faces detected!";
                        }} else if (detections.length === 1) {{
                            const result = faceMatcher.findBestMatch(resizedDetections[0].descriptor);
                            const box = resizedDetections[0].detection.box;
                            new faceapi.draw.DrawBox(box, {{ label: result.toString() }}).draw(canvas);

                            if (result.label !== 'unknown') {{
                                window.parent.postMessage({{type: 'streamlit:setComponentValue', value: result.label}}, '*');
                                chime.play();
                            }}
                        }}
                    }}, 1200);
                }});
            }} catch (e) {{ label.innerText = "Error: " + e.message; }}
        }}
        start();
    </script>
    """
    
    match_name = components.html(js_attendance, height=600)

    if match_name and match_name in [u['name'] for u in st.session_state.registered_users]:
        if match_name not in st.session_state.already_logged:
            ts = datetime.now()
            entry = {
                "Name": match_name, 
                "Date": ts.strftime("%Y-%m-%d"), 
                "Time": ts.strftime("%H:%M:%S"), 
                "Status": "Present"
            }
            st.session_state.attendance_records.append(entry)
            st.session_state.already_logged.add(match_name)
            pd.DataFrame([entry]).to_csv(LOG_FILE, mode='a', header=False, index=False)
            st.success(f"Attendance Recorded: {match_name}")
            st.balloons()

# --- PAGE 3: LOGS ---
def logs_page():
    st.title("📄 Attendance Logs")
    present_names = [r["Name"] for r in st.session_state.attendance_records]
    all_rows = list(st.session_state.attendance_records)
    
    for user in st.session_state.registered_users:
        if user['name'] not in present_names:
            all_rows.append({
                "Name": user['name'], 
                "Date": datetime.now().strftime("%Y-%m-%d"), 
                "Time": "-", 
                "Status": "Absent"
            })
    
    df = pd.DataFrame(all_rows)
    st.dataframe(df, use_container_width=True)
    st.download_button("📥 Download Report", df.to_csv(index=False).encode('utf-8'), "attendance.csv", "text/csv")

# --- NAVIGATION ---
page = st.sidebar.radio("Navigation", ["Register", "Attendance", "View Logs"])
if page == "Register":
    registration_page()
elif page == "Attendance":
    attendance_page()
else:
    logs_page()
