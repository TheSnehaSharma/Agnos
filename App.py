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
    
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Enter Full Name").strip().upper()
        img_file = st.file_uploader("Upload Profile Photo", type=['jpg', 'png', 'jpeg'])
    
    # Hidden placeholder for the AI result
    if img_file and name:
        img_bytes = img_file.read()
        img_base64 = base64.b64encode(img_bytes).decode()
        
        raw_js_reg = """
        <div id="status" style="font-family:sans-serif; color:#ff4b4b; font-weight:bold; padding:10px; border:1px solid #eee; border-radius:5px;">🧬 AI Engine Loading...</div>
        <script src="FACE_API_URL"></script>
        <script>
            async function process() {
                try {
                    const URL = 'https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js/weights';
                    await faceapi.nets.ssdMobilenetv1.loadFromUri(URL);
                    await faceapi.nets.faceRecognitionNet.loadFromUri(URL);
                    await faceapi.nets.faceLandmark68Net.loadFromUri(URL);
                    
                    document.getElementById('status').innerText = "Processing Image... Stay on this page.";
                    const img = new Image();
                    img.src = "data:image/jpeg;base64,IMG_DATA";
                    img.onload = async () => {
                        const det = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor();
                        if (det) {
                            window.parent.postMessage({
                                type: 'streamlit:setComponentValue', 
                                value: { "name": "USER_NAME", "encoding": Array.from(det.descriptor) }
                            }, '*');
                            document.getElementById('status').innerText = "✅ Extraction Complete! Click 'Confirm Registration' below.";
                        } else {
                            document.getElementById('status').innerText = "❌ Error: No face detected.";
                        }
                    };
                } catch (e) { document.getElementById('status').innerText = "❌ Model Load Error."; }
            }
            process();
        </script>
        """
        js_component = raw_js_reg.replace("FACE_API_URL", FACE_API_JS).replace("IMG_DATA", img_base64).replace("USER_NAME", name)
        reg_data = st.components.v1.html(js_component, height=100)

        # Show the actual Register button only after AI returns data
        if reg_data and isinstance(reg_data, dict):
            if st.button(f"Confirm Registration for {name}"):
                if not any(u['name'] == reg_data['name'] for u in st.session_state.registered_users):
                    st.session_state.registered_users.append(reg_data)
                    with open(DB_FILE, "w") as f:
                        json.dump(st.session_state.registered_users, f)
                    st.success(f"Successfully registered {name}!")
                    st.rerun()

    st.write("---")
    st.subheader("Recently Registered Users")
    if st.session_state.registered_users:
        # Show last 10 registered
        recent_df = pd.DataFrame(st.session_state.registered_users).tail(10)
        st.table(recent_df[['name']])
    else:
        st.info("No users registered yet.")

# --- 3. ATTENDANCE PAGE ---
def attendance_page():
    st.title("📹 Live Presence Scanner")
    
    # Ensure variables exist even if empty
    known_json = json.dumps(st.session_state.registered_users) if st.session_state.registered_users else "[]"

    raw_js_attendance = """
    <div style="position: relative; display: inline-block;">
        <video id="v" autoplay muted playsinline style="width: 100%; max-width: 600px; border-radius: 10px; background:#000;"></video>
        <canvas id="c" style="position: absolute; top: 0; left: 0;"></canvas>
    </div>
    <p id="msg" style="font-family:sans-serif; color: #666; margin-top:10px;">Initializing Camera...</p>

    <script src="FACE_API_URL"></script>
    <script>
        const v = document.getElementById('v');
        const c = document.getElementById('c');
        const m = document.getElementById('msg');
        const known = KNOWN_DATA_JSON;

        async function start() {
            try {
                const URL = 'https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js/weights';
                // Load faster models for live feed
                await faceapi.nets.tinyFaceDetector.loadFromUri(URL);
                await faceapi.nets.faceLandmark68Net.loadFromUri(URL);
                await faceapi.nets.faceRecognitionNet.loadFromUri(URL);

                const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
                v.srcObject = stream;

                v.onloadedmetadata = () => {
                    c.width = v.videoWidth;
                    c.height = v.videoHeight;
                    m.innerText = "🔒 Scanner Active (0.5 Tolerance)";
                    run();
                };
            } catch(e) { 
                m.innerText = "Error: " + e.message; 
                console.error(e);
            }
        }

        async function run() {
            // Setup matcher
            let faceMatcher;
            if (known.length > 0) {
                const labels = known.map(u => new faceapi.LabeledFaceDescriptors(u.name, [new Float32Array(u.encoding)]));
                faceMatcher = new faceapi.FaceMatcher(labels, 0.5);
            }

            setInterval(async () => {
                const detections = await faceapi.detectAllFaces(v, new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks().withFaceDescriptors();
                const displaySize = { width: v.videoWidth, height: v.videoHeight };
                const resized = faceapi.resizeResults(detections, displaySize);
                
                const ctx = c.getContext('2d');
                ctx.clearRect(0, 0, c.width, c.height);

                resized.forEach(d => {
                    let labelText = "Searching...";
                    let nameMatch = "unknown";

                    if (faceMatcher) {
                        const match = faceMatcher.findBestMatch(d.descriptor);
                        const score = Math.round((1 - match.distance) * 100);
                        labelText = `${match.label} (${score}%)`;
                        nameMatch = match.label;
                    } else {
                        labelText = "Unknown (No Database)";
                    }

                    // Draw Box
                    new faceapi.draw.DrawBox(d.detection.box, { label: labelText }).draw(c);

                    if (nameMatch !== 'unknown' && nameMatch !== 'Unknown') {
                        window.parent.postMessage({ type: 'streamlit:setComponentValue', value: nameMatch }, '*');
                    }
                });
            }, 600);
        }
        start();
    </script>
    """
    js_attendance = raw_js_attendance.replace("FACE_API_URL", FACE_API_JS).replace("KNOWN_DATA_JSON", known_json)
    res = st.components.v1.html(js_attendance, height=550)

    if res and isinstance(res, str):
        date_str = datetime.now().strftime("%Y-%m-%d")
        log_key = f"{res}_{date_str}"
        
        if 'logged_today' not in st.session_state:
            st.session_state.logged_today = set()
            
        if log_key not in st.session_state.logged_today:
            st.session_state.logged_today.add(log_key)
            new_log = {"Name": res, "Time": datetime.now().strftime("%H:%M:%S"), "Date": date_str}
            st.session_state.attendance_records.insert(0, new_log)
            st.toast(f"Verified: {res}!")

    st.subheader("Today's Logs")
    st.dataframe(pd.DataFrame(st.session_state.attendance_records), use_container_width=True)

# --- NAVIGATION ---
choice = st.sidebar.radio("Navigation", ["Register Face", "Take Attendance"])
if choice == "Register Face":
    registration_page()
else:
    attendance_page()
