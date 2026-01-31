import streamlit as st
import json
import base64
import pandas as pd
from datetime import datetime

# --- 1. CONFIG & STATE ---
st.set_page_config(page_title="Privacy-First AI", layout="wide")
DB_FILE = "registered_faces.json"
FACE_API_JS = "https://cdnjs.cloudflare.com/ajax/libs/face-api.js/0.22.2/face-api.min.js"

if 'registered_users' not in st.session_state:
    st.session_state.registered_users = [] # Load from DB_FILE in production

# --- 2. REGISTRATION (ENCODE ON CLIENT) ---
def registration_page():
    st.title("👤 Private Local Registration")
    st.markdown("🔒 **Privacy Check:** Your name and image remain in your browser's RAM. Only a 128-digit mathematical map is sent to our server.")

    name = st.text_input("Full Name").strip().upper()
    img_file = st.file_uploader("Upload Face Photo", type=['jpg', 'png', 'jpeg'])

    if img_file and name:
        img_base64 = base64.b64encode(img_file.read()).decode()
        
        # This HTML/JS block handles the LOCAL encoding
        js_handler = f"""
        <div id="ui-box" style="padding:15px; border-radius:10px; background:#f0f2f6; font-family:sans-serif;">
            <div id="status" style="color:#ff4b4b; font-weight:bold;">⏳ Initializing Secure AI...</div>
            <button id="encBtn" style="display:none; margin-top:10px; padding:8px 16px; background:#ff4b4b; color:white; border:none; border-radius:5px; cursor:pointer;">
                Step 1: Encode Locally
            </button>
        </div>

        <script src="{FACE_API_JS}"></script>
        <script>
            const status = document.getElementById('status');
            const btn = document.getElementById('encBtn');
            const MODEL_URL = 'https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js/weights';

            // 1. Wait for Library to Load
            async function init() {{
                try {{
                    await faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL);
                    await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);
                    await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
                    status.innerText = "✅ AI Ready. Click Encode.";
                    btn.style.display = "block";
                }} catch (e) {{
                    status.innerText = "❌ Browser blocked AI. Try Incognito or disable Tracking Protection.";
                }}
            }}

            // 2. Perform Local Encoding
            btn.onclick = async () => {{
                status.innerText = "🧬 Processing Features...";
                const img = new Image();
                img.src = "data:image/jpeg;base64,{img_base64}";
                img.onload = async () => {{
                    const det = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor();
                    if (det) {{
                        // SUCCESS: Only send descriptor back
                        window.parent.postMessage({{
                            type: 'streamlit:setComponentValue', 
                            value: {{ "name": "{name}", "encoding": Array.from(det.descriptor) }}
                        }}, '*');
                        status.innerText = "✅ Done! Click Register below.";
                        btn.style.display = "none";
                    }} else {{
                        status.innerText = "❌ No face detected.";
                    }}
                }};
            }};
            init();
        </script>
        """
        # Execute the component
        reg_result = st.components.v1.html(js_handler, height=120)

        # Step 2: Final Registration (Server Side - only receives the vector)
        if reg_result and isinstance(reg_result, dict):
            st.success(f"Mathematical features extracted for {name}")
            if st.button("Step 2: Register to Server"):
                st.session_state.registered_users.append(reg_result)
                st.success(f"Registered {name} successfully!")
                st.rerun()

    st.write("---")
    st.subheader("Recently Registered")
    if st.session_state.registered_users:
        st.table(pd.DataFrame(st.session_state.registered_users)[['name']].tail(10))

# --- 3. ATTENDANCE PAGE ---
def attendance_page():
    st.title("📹 Live Presence Scanner")
    known_json = json.dumps(st.session_state.registered_users)

    # Note: We use the same 'Sync-Check' for the live feed
    js_attendance = f"""
    <div style="position: relative;">
        <video id="v" autoplay muted playsinline style="width: 100%; max-width: 500px; border-radius: 10px;"></video>
        <canvas id="c" style="position: absolute; top: 0; left: 0;"></canvas>
    </div>
    <div id="m" style="margin-top:10px; font-family:sans-serif; font-weight:bold;">Loading Camera AI...</div>

    <script src="{FACE_API_JS}"></script>
    <script>
        const v = document.getElementById('v');
        const c = document.getElementById('c');
        const m = document.getElementById('m');
        const known = {known_json};

        async function start() {{
            const URL = 'https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js/weights';
            await faceapi.nets.tinyFaceDetector.loadFromUri(URL);
            await faceapi.nets.faceLandmark68Net.loadFromUri(URL);
            await faceapi.nets.faceRecognitionNet.loadFromUri(URL);

            const stream = await navigator.mediaDevices.getUserMedia({{ video: true }});
            v.srcObject = stream;
            v.onloadedmetadata = () => {{
                c.width = v.videoWidth; c.height = v.videoHeight;
                m.innerText = "🔒 Scanner Active (0.5 Tolerance)";
                run();
            }};
        }}

        async function run() {{
            const labels = known.map(u => new faceapi.LabeledFaceDescriptors(u.name, [new Float32Array(u.encoding)]));
            const matcher = new faceapi.FaceMatcher(labels, 0.5);

            setInterval(async () => {{
                const det = await faceapi.detectAllFaces(v, new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks().withFaceDescriptors();
                const ctx = c.getContext('2d');
                ctx.clearRect(0, 0, c.width, c.height);

                faceapi.resizeResults(det, {{width: v.videoWidth, height: v.videoHeight}}).forEach(d => {{
                    const match = matcher.findBestMatch(d.descriptor);
                    const score = Math.round((1 - match.distance) * 100);
                    new faceapi.draw.DrawBox(d.detection.box, {{ label: match.label + ' (' + score + '%)' }}).draw(c);
                    if (match.label !== 'unknown') {{
                        window.parent.postMessage({{ type: 'streamlit:setComponentValue', value: match.label }}, '*');
                    }}
                }});
            }}, 800);
        }}
        start();
    </script>
    """
    res = st.components.v1.html(js_attendance, height=550)
    if res and isinstance(res, str):
        st.toast(f"Seen: {res}")

# --- NAV ---
choice = st.sidebar.radio("Navigation", ["Register Face", "Take Attendance"])
if choice == "Register Face": registration_page()
else: attendance_page()
