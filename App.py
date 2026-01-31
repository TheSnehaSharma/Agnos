import streamlit as st
import pandas as pd
from datetime import datetime
import streamlit.components.v1 as components
import json
import base64

# --- CONFIGURATION & DATABASE ---
LOG_FILE = "attendance_log.csv"

if 'registered_users' not in st.session_state:
    st.session_state.registered_users = {} # {name: encoding_list}

if 'present_names' not in st.session_state:
    st.session_state.present_names = []

# --- PAGE 1: REGISTRATION ---
def registration_page():
    st.title("👤 Register New User")
    name = st.text_input("Full Name")
    img_file = st.file_uploader("Upload Face Photo", type=['jpg', 'png', 'jpeg'])
    
    if st.button("Register"):
        if name and img_file:
            # In a client-side system, we usually extract the descriptor in JS.
            # For this simple version, we store the name.
            st.session_state.registered_users[name.upper()] = True
            st.success(f"Registered {name.upper()}! (Note: Real-time matching requires JS-side descriptors)")
        else:
            st.error("Name and Image are required.")

# --- PAGE 2: LIVE FEED (Client-Side) ---
def live_feed_page():
    st.title("📹 Live Attendance")
    
    # JavaScript to handle Webcam, Face Detection, and Audio
    # This uses face-api.js (TensorFlow.js)
    js_code = """
    <div style="position: relative;">
        <video id="video" autoplay muted style="width: 100%; max-width: 640px; border-radius: 10px;"></video>
        <canvas id="overlay" style="position: absolute; top: 0; left: 0;"></canvas>
        <audio id="beep" src="https://www.soundjay.com/buttons/sounds/button-3.mp3"></audio>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/face-api.js@0.22.2/dist/face-api.min.js"></script>
    <script>
        const video = document.getElementById('video');
        const beep = document.getElementById('beep');

        async function setup() {
            await faceapi.nets.tinyFaceDetector.loadFromUri('https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights');
            const stream = await navigator.mediaDevices.getUserMedia({ video: {} });
            video.srcObject = stream;

            video.addEventListener('play', () => {
                setInterval(async () => {
                    const detections = await faceapi.detectAllFaces(video, new faceapi.TinyFaceDetectorOptions());
                    if (detections.length > 0) {
                        // Notify Python side
                        window.parent.postMessage({
                            type: 'streamlit:setComponentValue',
                            value: {status: 'detected', time: new Date().toLocaleTimeString()}
                        }, '*');
                        beep.play();
                    }
                }, 2000);
            });
        }
        setup();
    </script>
    """
    
    # Capture the value sent back from JavaScript
    detection_data = components.html(js_code, height=500)
    
    if detection_data:
        name = "USER" # In a full system, JS would send the matched name
        now = datetime.now()
        st.success(f"✅ {name} marked present at {now.strftime('%H:%M:%S')}")
        if name not in st.session_state.present_names:
            st.session_state.present_names.append(name)
            # Log to CSV
            with open(LOG_FILE, "a") as f:
                f.write(f"{name},{now.strftime('%Y-%m-%d')},{now.strftime('%H:%M:%S')}\n")

# --- PAGE 3: LOGS & CSV ---
def logs_page():
    st.title("📄 Attendance Logs")
    
    # Calculate Absentees
    all_registered = set(st.session_state.registered_users.keys())
    present = set(st.session_state.present_names)
    absent = all_registered - present
    
    # Build Display Dataframe
    data = []
    for p in present: data.append({"Name": p, "Status": "Present"})
    for a in absent: data.append({"Name": a, "Status": "Absent"})
    
    df = pd.DataFrame(data)
    st.table(df)
    
    # Download CSV
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Attendance Report", csv, "report.csv", "text/csv")

# --- NAVIGATION ---
page = st.sidebar.selectbox("Go to", ["Register", "Live Feed", "Logs"])

if page == "Register":
    registration_page()
elif page == "Live Feed":
    live_feed_page()
else:
    logs_page()
