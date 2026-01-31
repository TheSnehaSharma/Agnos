import streamlit as st
import pandas as pd
from datetime import datetime
import json
import base64

# --- 1. INITIALIZATION ---
st.set_page_config(page_title="Privacy-First AI Attendance", layout="wide")

FACE_API_JS = "https://cdn.jsdelivr.net/npm/face-api.js@0.22.2/dist/face-api.min.js"

# We use session state to bridge the Registration and Attendance pages
if 'registered_users' not in st.session_state:
    st.session_state.registered_users = [] 
if 'attendance_records' not in st.session_state:
    st.session_state.attendance_records = []

# --- 2. REGISTRATION (ENCODE ON CLIENT) ---
def registration_page():
    st.title("👤 Private Registration")
    st.info("Photo is processed locally. Only a 'Face Map' (128 numbers) is stored.")
    
    # We use a container to manage the flow
    reg_container = st.container()
    
    with reg_container:
        name = st.text_input("Full Name").strip().upper()
        img_file = st.file_uploader("Upload Face Photo", type=['jpg', 'png', 'jpeg'])
        
        # When user clicks, we "prime" the state to start extraction
        if img_file and name:
            img_bytes = img_file.read()
            img_base64 = base64.b64encode(img_bytes).decode()
            
            # THE KEY: The component must be visible to send data back
            st.write("---")
            st.caption("AI Engine Status:")
            
            js_component = f"""
            <div id="status" style="font-family:sans-serif; color:#ff4b4b; font-size:14px; font-weight:bold;">🧬 Initializing Local AI...</div>
            <script src="{FACE_API_JS}"></script>
            <script>
                async function process() {{
                    try {{
                        const MODEL_URL = 'https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js/weights';
                        await faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL);
                        await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);
                        await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
                        
                        document.getElementById('status').innerText = "Analyzing Image...";
                        const img = new Image();
                        img.src = "data:image/jpeg;base64,{img_base64}";
                        img.onload = async () => {{
                            const det = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor();
                            if (det) {{
                                document.getElementById('status').innerText = "✅ Extraction Complete!";
                                // Send data back to Streamlit
                                window.parent.postMessage({{
                                    type: 'streamlit:setComponentValue', 
                                    value: {{ "name": "{name}", "encoding": Array.from(det.descriptor) }}
                                }}, '*');
                            }} else {{
                                document.getElementById('status').innerText = "❌ No face detected!";
                                window.parent.postMessage({{type: 'streamlit:setComponentValue', value: "ERR_NO_FACE"}}, '*');
                            }}
                        }};
                    }} catch (e) {{
                        document.getElementById('status').innerText = "❌ Model Load Error";
                        window.parent.postMessage({{type: 'streamlit:setComponentValue', value: "ERR_LOAD"}}, '*');
                    }}
                }}
                process();
            </script>
            """
            # The return value from the HTML component
            returned_data = st.components.v1.html(js_component, height=70)

            # Check if the component returned data
            if returned_data:
                if isinstance(returned_data, dict):
                    # Check if already added
                    if not any(u['name'] == returned_data['name'] for u in st.session_state.registered_users):
                        st.session_state.registered_users.append(returned_data)
                        st.success(f"✅ Facial map for {name} saved to system!")
                        st.balloons()
                    else:
                        st.warning(f"User {name} is already registered.")
                elif returned_data == "ERR_NO_FACE":
                    st.error("No face found in that image.")

# --- 3. ATTENDANCE (COMPARE ON CLIENT) ---
def attendance_page():
    st.title("📹 Live Private Attendance")
    
    # Check if we have data
    if not st.session_state.registered_users:
        st.warning("⚠️ No facial maps found. Please go to 'Register Face' and ensure 'Extraction Complete' is shown.")
        return

    st.write(f"Users in Database: **{len(st.session_state.registered_users)}**")
    known_json = json.dumps(st.session_state.registered_users)

    js_attendance = f"""
    <div style="position: relative; display: inline-block;">
        <video id="v" autoplay muted playsinline style="width: 100%; max-width: 500px; border-radius: 10px; border: 2px solid #ff4b4b;"></video>
        <canvas id="c" style="position: absolute; top: 0; left: 0;"></canvas>
    </div>
    <p id="msg" style="font-family:sans-serif; color: #555; font-weight: bold;">Starting scanner...</p>

    <script src="{FACE_API_JS}"></script>
    <script>
        const v = document.getElementById('v');
        const c = document.getElementById('c');
        const m = document.getElementById('msg');
        const known = {known_json};

        async function start() {{
            try {{
                const URL = 'https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js/weights';
                await faceapi.nets.tinyFaceDetector.loadFromUri(URL);
                await faceapi.nets.faceLandmark68Net.loadFromUri(URL);
                await faceapi.nets.faceRecognitionNet.loadFromUri(URL);

                const stream = await navigator.mediaDevices.getUserMedia({{ video: true }});
                v.srcObject = stream;

                v.onloadedmetadata = () => {{
                    c.width = v.videoWidth;
                    c.height = v.videoHeight;
                    m.innerText = "🔒 Scanner Active (Encrypted)";
                    run();
                }};
            }} catch(e) {{ m.innerText = "Camera Error: " + e.message; }}
        }

        async function run() {{
            // Convert the array back to Float32 for the AI
            const labels = known.map(u => new faceapi.LabeledFaceDescriptors(u.name, [new Float32Array(u.encoding)]));
            const matcher = new faceapi.FaceMatcher(labels, 0.55);

            setInterval(async () => {{
                const detections = await faceapi.detectAllFaces(v, new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks().withFaceDescriptors();
                const resized = faceapi.resizeResults(detections, {{width: v.videoWidth, height: v.videoHeight}});
                
                const ctx = c.getContext('2d');
                ctx.clearRect(0, 0, c.width, c.height);

                resized.forEach(d => {{
                    const match = matcher.findBestMatch(d.descriptor);
                    // Draw box locally
                    new faceapi.draw.DrawBox(d.detection.box, {{ label: match.toString() }}).draw(c);

                    if (match.label !== 'unknown') {{
                        window.parent.postMessage({{
                            type: 'streamlit:setComponentValue', 
                            value: match.label
                        }}, '*');
                    }}
                }});
            }}, 800);
        }}
        start();
    </script>
    """
    
    res = st.components.v1.html(js_attendance, height=520)

    # Logging logic
    if res and isinstance(res, str):
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        log_key = f"{res}_{date_str}"
        
        # Simple daily duplicate prevention
        if 'logged_today' not in st.session_state:
            st.session_state.logged_today = set()
            
        if log_key not in st.session_state.logged_today:
            st.session_state.logged_today.add(log_key)
            new_log = {"Name": res, "Time": now.strftime("%H:%M:%S"), "Status": "Present"}
            st.session_state.attendance_records.insert(0, new_log)
            st.toast(f"Logged: {res} at {new_log['Time']}")

    if st.session_state.attendance_records:
        st.subheader("Attendance Log")
        st.dataframe(pd.DataFrame(st.session_state.attendance_records), use_container_width=True)

# --- NAVIGATION ---
choice = st.sidebar.radio("Navigation", ["Register Face", "Take Attendance"])
if choice == "Register Face":
    registration_page()
else:
    attendance_page()
