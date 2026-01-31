import streamlit as st
import pandas as pd
import json
import base64
import os

# --- 1. INITIALIZATION ---
DB_FILE = "registered_faces.json"

if 'registered_users' not in st.session_state:
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            st.session_state.registered_users = json.load(f)
    else:
        st.session_state.registered_users = []

# --- 2. REGISTRATION PAGE ---
def registration_page():
    st.title("👤 Local Privacy Registration")
    
    name = st.text_input("Enter Full Name").strip().upper()
    img_file = st.file_uploader("Upload Profile Photo", type=['jpg', 'png', 'jpeg'])

    if img_file and name:
        img_base64 = base64.b64encode(img_file.read()).decode()
        
        # This component handles LOCAL encoding
        js_code = f"""
        <div id="status-box" style="padding:10px; background:#f0f2f6; border-radius:8px; font-family:sans-serif;">
            <div id="status-text" style="color:#ff4b4b; font-weight:bold;">⏳ Initializing Local AI...</div>
        </div>

        <script type="module">
            import * as faceapi from 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api/dist/face-api.esm.js';

            const status = document.getElementById('status-text');
            const MODEL_URL = 'https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js/weights';

            async function init() {{
                try {{
                    status.innerText = "Loading Models...";
                    await faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL);
                    await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);
                    await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
                    
                    status.innerText = "🧬 Vectorizing locally...";
                    const img = new Image();
                    img.src = "data:image/jpeg;base64,{img_base64}";
                    img.onload = async () => {{
                        const det = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor();
                        if (det) {{
                            // Send ONLY the 128-digit vector back to Streamlit
                            window.parent.postMessage({{
                                type: 'streamlit:setComponentValue', 
                                value: Array.from(det.descriptor)
                            }}, '*');
                            status.innerText = "✅ Encoding Complete! Click 'Register' below.";
                        } else {{
                            status.innerText = "❌ Error: No face detected.";
                        }}
                    }};
                }} catch (e) {{
                    status.innerText = "❌ AI Blocked. Please check browser shield.";
                }}
            }}
            init();
        </script>
        """
        # THE FIX: Capture the value returned from the component
        extracted_vector = st.components.v1.html(js_code, height=100)

        # Show the actual Register button only if the vector exists
        if extracted_vector and isinstance(extracted_vector, list):
            st.success(f"Mathematical features extracted for {name}")
            
            if st.button(f"Confirm Registration for {name}"):
                # 1. Update Session State
                new_user = {"name": name, "encoding": extracted_vector}
                st.session_state.registered_users.append(new_user)
                
                # 2. Save to permanent JSON file
                with open(DB_FILE, "w") as f:
                    json.dump(st.session_state.registered_users, f)
                
                st.success(f"✅ {name} added to database!")
                st.rerun()

    st.write("---")
    st.subheader("Recently Registered Users")
    if st.session_state.registered_users:
        # Show table of registered names
        df = pd.DataFrame(st.session_state.registered_users)
        st.table(df[['name']].tail(10))
    else:
        st.info("No users registered in local database.")
# --- 3. ATTENDANCE PAGE ---
def attendance_page():
    st.title("📹 Live Privacy Scanner")
    
    if not st.session_state.registered_users:
        st.warning("⚠️ No users registered. Showing feed in 'Detection Only' mode.")
    
    known_json = json.dumps(st.session_state.registered_users)

    # We use a standard string and .replace() to avoid f-string syntax errors
    raw_js_attendance = """
    <div style="position: relative; display: inline-block; width: 100%;">
        <video id="v" autoplay muted playsinline style="width: 100%; max-width: 600px; border-radius: 10px; background:#000;"></video>
        <canvas id="c" style="position: absolute; top: 0; left: 0;"></canvas>
    </div>
    <p id="msg" style="font-family:sans-serif; color: #666; margin-top:10px;">Initializing Camera & AI...</p>

    <script type="module">
        // Importing the modern ESM version of the AI
        import * as faceapi from 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api/dist/face-api.esm.js';

        const v = document.getElementById('v');
        const c = document.getElementById('c');
        const m = document.getElementById('msg');
        const known = KNOWN_DATA_JSON;

        async function start() {
            try {
                const URL = 'https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js/weights';
                
                m.innerText = "Step 1: Loading AI Models...";
                await faceapi.nets.tinyFaceDetector.loadFromUri(URL);
                await faceapi.nets.faceLandmark68Net.loadFromUri(URL);
                await faceapi.nets.faceRecognitionNet.loadFromUri(URL);

                m.innerText = "Step 2: Accessing Camera...";
                const stream = await navigator.mediaDevices.getUserMedia({ 
                    video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } } 
                });
                v.srcObject = stream;

                v.onloadedmetadata = () => {
                    c.width = v.videoWidth;
                    c.height = v.videoHeight;
                    m.innerText = "🔒 Scanner Active (Privacy-Enforced)";
                    run();
                };
            } catch(e) { 
                m.innerText = "❌ Error: " + e.message;
                if (e.name === "NotAllowedError") {
                    m.innerText += " - Please allow camera access in your browser settings.";
                }
            }
        }

        async function run() {
            // Setup local matcher from the vector data provided by Python
            let faceMatcher = null;
            if (known.length > 0) {
                const labels = known.map(u => new faceapi.LabeledFaceDescriptors(u.name, [new Float32Array(u.encoding)]));
                faceMatcher = new faceapi.FaceMatcher(labels, 0.5); // 0.5 Tolerance
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
                    }

                    // Local Drawing
                    new faceapi.draw.DrawBox(d.detection.box, { label: labelText }).draw(c);

                    // If a match is found, notify Streamlit (Python)
                    if (nameMatch !== 'unknown') {
                        window.parent.postMessage({ type: 'streamlit:setComponentValue', value: nameMatch }, '*');
                    }
                });
            }, 700);
        }
        start();
    </script>
    """
    js_attendance = raw_js_attendance.replace("KNOWN_DATA_JSON", known_json)
    res = st.components.v1.html(js_attendance, height=550)

    # Python Logging Logic
    if res and isinstance(res, str):
        # (Your existing logging code here...)
        st.toast(f"Verified: {res}!")

# (Ensure your navigation logic calls attendance_page())

# --- NAV ---
choice = st.sidebar.radio("Navigation", ["Register Face", "Take Attendance"])
if choice == "Register Face": registration_page()
else: attendance_page()
