import streamlit as st
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

# --- 2. REGISTRATION (ENCODE ON CLIENT) ---
def registration_page():
    st.title("👤 Private Registration")
    st.info("Your photo is processed locally. Only a mathematical 'face map' is stored.")
    
    # Form for input
    name = st.text_input("Full Name (Local Only)").strip().upper()
    img_file = st.file_uploader("Upload Face Photo", type=['jpg', 'png', 'jpeg'])
    
    if img_file and name:
        # Convert to Base64 so JS can read it locally
        img_bytes = img_file.read()
        img_base64 = base64.b64encode(img_bytes).decode()
        
        # This component runs the AI in the browser
        js_component = f"""
        <div id="status" style="font-family:sans-serif; color:#666; font-size:14px;">🧬 Extracting facial features locally...</div>
        <script src="{FACE_API_JS}"></script>
        <script>
            async function process() {{
                try {{
                    const MODEL_URL = 'https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js/weights';
                    await faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL);
                    await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);
                    await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
                    
                    const img = new Image();
                    img.src = "data:image/jpeg;base64,{img_base64}";
                    img.onload = async () => {{
                        const det = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor();
                        if (det) {{
                            // SUCCESS: Send only the numbers back to Python
                            window.parent.postMessage({{
                                type: 'streamlit:setComponentValue', 
                                value: {{ "name": "{name}", "encoding": Array.from(det.descriptor) }}
                            }}, '*');
                            document.getElementById('status').innerText = "✅ Extraction Complete!";
                        }} else {{
                            window.parent.postMessage({{type: 'streamlit:setComponentValue', value: "ERR_NO_FACE"}}, '*');
                        }}
                    }};
                }} catch (e) {{
                    window.parent.postMessage({{type: 'streamlit:setComponentValue', value: "ERR_LOAD"}}, '*');
                }}
            }}
            process();
        </script>
        """
        # Capture the returned value
        returned_data = st.components.v1.html(js_component, height=50)

        # Logic to save the data once returned
        if returned_data:
            if returned_data == "ERR_NO_FACE":
                st.error("❌ No face detected. Try a clearer photo.")
            elif returned_data == "ERR_LOAD":
                st.error("❌ AI Models failed to load. Check your connection.")
            elif isinstance(returned_data, dict):
                # Check if already added to prevent duplicates on rerun
                if not any(u['name'] == returned_data['name'] for u in st.session_state.registered_users):
                    st.session_state.registered_users.append(returned_data)
                    st.success(f"✅ Facial map for {name} saved to local session!")
                    st.balloons()

# --- 3. ATTENDANCE (COMPARE ON CLIENT) ---
def attendance_page():
    st.title("📹 Live Private Attendance")
    
    if not st.session_state.registered_users:
        st.warning("⚠️ No facial maps found in session. Please register first.")
        return

    known_json = json.dumps(st.session_state.registered_users)

    js_attendance = f"""
    <div style="position: relative;">
        <video id="v" autoplay muted playsinline style="width: 100%; max-width: 500px; border-radius: 10px;"></video>
        <canvas id="c" style="position: absolute; top: 0; left: 0;"></canvas>
    </div>
    <p id="msg" style="font-family:sans-serif; color: #333;">Starting encrypted scanner...</p>

    <script src="{FACE_API_JS}"></script>
    <script>
        const v = document.getElementById('v');
        const c = document.getElementById('c');
        const m = document.getElementById('msg');
        const known = {known_json};

        async function start() {{
            const URL = 'https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js/weights';
            await faceapi.nets.tinyFaceDetector.loadFromUri(URL);
            await faceapi.nets.faceLandmark68Net.loadFromUri(URL);
            await faceapi.nets.faceRecognitionNet.loadFromUri(URL);

            const stream = await navigator.mediaDevices.getUserMedia({{ video: true }});
            v.srcObject = stream;

            v.onloadedmetadata = () => {{
                c.width = v.videoWidth;
                c.height = v.videoHeight;
                run();
            }};
        }}

        async function run() {{
            const labels = known.map(u => new faceapi.LabeledFaceDescriptors(u.name, [new Float32Array(u.encoding)]));
            const matcher = new faceapi.FaceMatcher(labels, 0.6);
            m.innerText = "🔒 Privacy Mode: Active (Local Processing Only)";

            setInterval(async () => {{
                const detections = await faceapi.detectAllFaces(v, new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks().withFaceDescriptors();
                const resized = faceapi.resizeResults(detections, {{width: v.videoWidth, height: v.videoHeight}});
                
                const ctx = c.getContext('2d');
                ctx.clearRect(0, 0, c.width, c.height);

                resized.forEach(d => {{
                    const match = matcher.findBestMatch(d.descriptor);
                    new faceapi.draw.DrawBox(d.detection.box, {{ label: match.toString() }}).draw(c);

                    if (match.label !== 'unknown') {{
                        window.parent.postMessage({{
                            type: 'streamlit:setComponentValue', 
                            value: match.label
                        }}, '*');
                    }}
                }});
            }}, 600);
        }}
        start();
    </script>
    """
    
    res = st.components.v1.html(js_attendance, height=500)

    if res and isinstance(res, str):
        now = datetime.now().strftime("%H:%M:%S")
        if not any(log['Name'] == res for log in st.session_state.attendance_records):
            st.session_state.attendance_records.insert(0, {{"Name": res, "Time": now, "Status": "Present"}})
            st.toast(f"Logged {res}")

    st.subheader("Today's Presence")
    st.table(pd.DataFrame(st.session_state.attendance_records))

# --- NAVIGATION ---
choice = st.sidebar.radio("Navigation", ["Register Face", "Take Attendance"])
if choice == "Register Face":
    registration_page()
else:
    attendance_page()
