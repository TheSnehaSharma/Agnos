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

# Load persistent data
if 'registered_users' not in st.session_state:
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            st.session_state.registered_users = json.load(f)
    else:
        st.session_state.registered_users = []

# Initialize state variables
if 'reg_step' not in st.session_state:
    st.session_state.reg_step = "idle"  # idle -> encoding -> ready
if 'current_encoding' not in st.session_state:
    st.session_state.current_encoding = None

# --- 2. REGISTRATION PAGE ---
def registration_page():
    st.title("👤 Local Face Registration")
    
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Enter Full Name", key="reg_name").strip().upper()
        img_file = st.file_uploader("Upload Profile Photo", type=['jpg', 'png', 'jpeg'])
        
        # Button 1: ENCODE
        if st.button("Encode"):
            if not name or not img_file:
                st.warning("Please provide both a name and a photo.")
            else:
                st.session_state.reg_step = "encoding"

    # Step 2: Trigger the JS Encoding component
    if st.session_state.reg_step == "encoding" and img_file:
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
                    
                    document.getElementById('status').innerText = "Processing Image...";
                    const img = new Image();
                    img.src = "data:image/jpeg;base64,IMG_DATA";
                    img.onload = async () => {
                        const det = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor();
                        if (det) {
                            window.parent.postMessage({
                                type: 'streamlit:setComponentValue', 
                                value: Array.from(det.descriptor)
                            }, '*');
                            document.getElementById('status').innerText = "✅ Encoding Complete!";
                        } else {
                            document.getElementById('status').innerText = "❌ No face detected.";
                        }
                    };
                } catch (e) { document.getElementById('status').innerText = "❌ Model Error."; }
            }
            process();
        </script>
        """
        js_component = raw_js_reg.replace("FACE_API_URL", FACE_API_JS).replace("IMG_DATA", img_base64)
        encoding_result = st.components.v1.html(js_component, height=100)

        if encoding_result and isinstance(encoding_result, list):
            st.session_state.current_encoding = encoding_result
            st.session_state.reg_step = "ready"
            st.rerun()

    # Step 3: Show the REGISTER button once encoding is captured
    if st.session_state.reg_step == "ready":
        st.success(f"Facial features extracted for {name}!")
        if st.button("Register"):
            new_user = {"name": name, "encoding": st.session_state.current_encoding}
            
            # Check for duplicates
            if not any(u['name'] == name for u in st.session_state.registered_users):
                st.session_state.registered_users.append(new_user)
                with open(DB_FILE, "w") as f:
                    json.dump(st.session_state.registered_users, f)
                st.success(f"Successfully registered {name}!")
            else:
                st.info(f"{name} is already registered.")
            
            # Reset for next registration
            st.session_state.reg_step = "idle"
            st.session_state.current_encoding = None
            st.rerun()

    st.write("---")
    st.subheader("Recently Registered Users")
    if st.session_state.registered_users:
        df = pd.DataFrame(st.session_state.registered_users).tail(10)
        st.table(df[['name']])
    else:
        st.info("No users registered yet.")

# --- 3. ATTENDANCE PAGE --- (Keep as is since you said feed works)
def attendance_page():
    st.title("📹 Live Presence Scanner")
    known_json = json.dumps(st.session_state.registered_users)
    
    # Using your working feed logic here...
    raw_js_attendance = """
    <div style="position: relative; display: inline-block;">
        <video id="v" autoplay muted playsinline style="width: 100%; max-width: 600px; border-radius: 10px; background:#000;"></video>
        <canvas id="c" style="position: absolute; top: 0; left: 0;"></canvas>
    </div>
    <script src="FACE_API_URL"></script>
    <script>
        const v = document.getElementById('v');
        const c = document.getElementById('c');
        const known = KNOWN_DATA_JSON;

        async function start() {
            const URL = 'https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js/weights';
            await faceapi.nets.tinyFaceDetector.loadFromUri(URL);
            await faceapi.nets.faceLandmark68Net.loadFromUri(URL);
            await faceapi.nets.faceRecognitionNet.loadFromUri(URL);
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            v.srcObject = stream;
            v.onloadedmetadata = () => {
                c.width = v.videoWidth; c.height = v.videoHeight;
                run();
            };
        }

        async function run() {
            const labels = known.length > 0 
                ? known.map(u => new faceapi.LabeledFaceDescriptors(u.name, [new Float32Array(u.encoding)]))
                : [new faceapi.LabeledFaceDescriptors("Searching", [new Float32Array(128).fill(0)])];
            const matcher = new faceapi.FaceMatcher(labels, 0.5);

            setInterval(async () => {
                const detections = await faceapi.detectAllFaces(v, new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks().withFaceDescriptors();
                const ctx = c.getContext('2d');
                ctx.clearRect(0, 0, c.width, c.height);

                detections.forEach(d => {
                    const match = matcher.findBestMatch(d.descriptor);
                    const score = Math.round((1 - match.distance) * 100);
                    const label = `${match.label} (${score}%)`;
                    new faceapi.draw.DrawBox(d.detection.box, { label }).draw(c);
                    if (match.label !== 'unknown' && match.label !== 'Searching') {
                        window.parent.postMessage({ type: 'streamlit:setComponentValue', value: match.label }, '*');
                    }
                });
            }, 600);
        }
        start();
    </script>
    """
    js_attendance = raw_js_attendance.replace("FACE_API_URL", FACE_API_JS).replace("KNOWN_DATA_JSON", known_json)
    res = st.components.v1.html(js_attendance, height=520)
    
    if res and isinstance(res, str):
        st.toast(f"Verified: {res}")

# --- NAVIGATION ---
choice = st.sidebar.radio("Navigation", ["Register Face", "Take Attendance"])
if choice == "Register Face":
    registration_page()
else:
    attendance_page()
