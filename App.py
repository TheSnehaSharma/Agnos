import streamlit as st
import pandas as pd
from datetime import datetime
import json
import base64
import os

# --- 1. INITIALIZATION ---
st.set_page_config(page_title="AI Attendance System", layout="wide")

FACE_API_JS = "https://cdn.jsdelivr.net/npm/face-api.js@0.22.2/dist/face-api.min.js"
DB_FILE = "registered_faces.json"

# Persistence Logic
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
    st.title("👤 Local Face Registration")
    st.info("Face encoding happens entirely in your browser. No images are sent to the server.")
    
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Enter Full Name").strip().upper()
        img_file = st.file_uploader("Upload Profile Photo", type=['jpg', 'png', 'jpeg'])
    
    if img_file and name:
        img_base64 = base64.b64encode(img_file.read()).decode()
        
        raw_js_reg = """
        <div id="status" style="font-family:sans-serif; color:#ff4b4b; font-weight:bold;">🧬 AI Engine Loading...</div>
        <script src="FACE_API_URL"></script>
        <script>
            async function process() {
                try {
                    const URL = 'https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js/weights';
                    await faceapi.nets.ssdMobilenetv1.loadFromUri(URL);
                    await faceapi.nets.faceRecognitionNet.loadFromUri(URL);
                    await faceapi.nets.faceLandmark68Net.loadFromUri(URL);
                    
                    document.getElementById('status').innerText = "Processing Image...";
                    const img = new Image();
                    img.src = "data:image/jpeg;base64,IMG_DATA";
                    img.onload = async () => {
                        const det = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor();
                        if (det) {
                            window.parent.postMessage({
                                type: 'streamlit:setComponentValue', 
                                value: { "name": "USER_NAME", "encoding": Array.from(det.descriptor) }
                            }, '*');
                            document.getElementById('status').innerText = "✅ Done! Mapping received.";
                        } else {
                            document.getElementById('status').innerText = "❌ Error: No face detected.";
                        }
                    };
                } catch (e) { document.getElementById('status').innerText = "❌ Load Error."; }
            }
            process();
        </script>
        """
        js_component = raw_js_reg.replace("FACE_API_URL", FACE_API_JS).replace("IMG_DATA", img_base64).replace("USER_NAME", name)
        reg_result = st.components.v1.html(js_component, height=50)

        if reg_result and isinstance(reg_result, dict):
            # Verify and Save
            if not any(u['name'] == reg_result['name'] for u in st.session_state.registered_users):
                st.session_state.registered_users.append(reg_result)
                with open(DB_FILE, "w") as f:
                    json.dump(st.session_state.registered_users, f)
                st.success(f"Successfully registered {name}!")
                st.balloons()

# --- 3. ATTENDANCE PAGE ---
def attendance_page():
    st.title("📹 Live Presence Scanner")
    
    # Show status but don't stop the feed
    if not st.session_state.registered_users:
        st.warning("⚠️ No users registered. Showing feed in 'Detection Only' mode.")
    
    known_json = json.dumps(st.session_state.registered_users)

    raw_js_attendance = """
    <div style="position: relative; display: inline-block;">
        <video id="v" autoplay muted playsinline style="width: 100%; max-width: 600px; border-radius: 10px;"></video>
        <canvas id="c" style="position: absolute; top: 0; left: 0;"></canvas>
    </div>
    <p id="msg" style="font-family:sans-serif; color: #666;">Starting Privacy-Safe Scanner...</p>

    <script src="FACE_API_URL"></script>
    <script>
        const v = document.getElementById('v');
        const c = document.getElementById('c');
        const m = document.getElementById('msg');
        const known = KNOWN_DATA_JSON;

        async function start() {
            const URL = 'https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js/weights';
            await faceapi.nets.tinyFaceDetector.loadFromUri(URL);
            await faceapi.nets.faceLandmark68Net.loadFromUri(URL);
            await faceapi.nets.faceRecognitionNet.loadFromUri(URL);

            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            v.srcObject = stream;

            v.onloadedmetadata = () => {
                c.width = v.videoWidth;
                c.height = v.videoHeight;
                m.innerText = "🔒 Scanner Active (Edge Processing)";
                run();
            };
        }

        async function run() {
            // Create matcher only if users exist, otherwise default to empty
            const labels = known.length > 0 
                ? known.map(u => new faceapi.LabeledFaceDescriptors(u.name, [new Float32Array(u.encoding)]))
                : [new faceapi.LabeledFaceDescriptors("Unknown", [new Float32Array(128).fill(0)])];
            
            const matcher = new faceapi.FaceMatcher(labels, 0.5); // 0.5 Tolerance as requested

            setInterval(async () => {
                const detections = await faceapi.detectAllFaces(v, new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks().withFaceDescriptors();
                const displaySize = { width: v.videoWidth, height: v.videoHeight };
                const resized = faceapi.resizeResults(detections, displaySize);
                
                const ctx = c.getContext('2d');
                ctx.clearRect(0, 0, c.width, c.height);

                resized.forEach(d => {
                    const match = matcher.findBestMatch(d.descriptor);
                    
                    // Calculate Match Percentage
                    // Distance 0 = 100% match. Distance 0.5 = 50% match (at tolerance).
                    const matchScore = Math.round((1 - match.distance) * 100);
                    const labelText = `${match.label} (${matchScore}%)`;

                    // Draw Overlay
                    const drawBox = new faceapi.draw.DrawBox(d.detection.box, { label: labelText });
                    drawBox.draw(c);

                    if (match.label !== 'unknown' && match.label !== 'Unknown') {
                        window.parent.postMessage({ type: 'streamlit:setComponentValue', value: match.label }, '*');
                    }
                });
            }, 800);
        }
        start();
    </script>
    """
    js_attendance = raw_js_attendance.replace("FACE_API_URL", FACE_API_JS).replace("KNOWN_DATA_JSON", known_json)
    res = st.components.v1.html(js_attendance, height=520)

    # Attendance Logging
    if res and isinstance(res, str):
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        log_key = f"{res}_{date_str}"
        
        if 'logged_today' not in st.session_state:
            st.session_state.logged_today = set()
            
        if log_key not in st.session_state.logged_today:
            st.session_state.logged_today.add(log_key)
            new_log = {"Name": res, "Time": now.strftime("%H:%M:%S"), "Date": date_str}
            st.session_state.attendance_records.insert(0, new_log)
            st.toast(f"Verified: {res}!")

    st.subheader("Recent Attendance")
    st.dataframe(pd.DataFrame(st.session_state.attendance_records), use_container_width=True)

# --- NAVIGATION ---
choice = st.sidebar.radio("Navigation", ["Register Face", "Take Attendance"])
if choice == "Register Face":
    registration_page()
else:
    attendance_page()
