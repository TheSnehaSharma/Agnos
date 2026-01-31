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
    
    if 'reg_stage' not in st.session_state:
        st.session_state.reg_stage = "idle"
    
    name = st.text_input("Enter Full Name").strip().upper()
    img_file = st.file_uploader("Upload Profile Photo", type=['jpg', 'png', 'jpeg'])
    
    if st.session_state.reg_stage == "idle":
        if st.button("Step 1: Encode Face"):
            if name and img_file:
                st.session_state.reg_stage = "encoding"
                st.rerun()
            else:
                st.warning("Please provide a name and photo first.")

    elif st.session_state.reg_stage == "encoding":
        st.warning("🧬 Processing... If blocked by browser, click the 'Shield' icon in your URL bar and allow 'Tracking'.")
        img_base64 = base64.b64encode(img_file.read()).decode()
        
    raw_js = """
        <div id="status" style="font-family:sans-serif; font-size:13px; color:#ff4b4b; padding:10px; background:#f0f2f6; border-radius:5px;">
            Initializing Local AI Engine...
        </div>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/face-api.js/0.22.2/face-api.min.js"></script>
        <script>
            // This ensures faceapi is defined before we call it
            window.onload = async function() {
                const status = document.getElementById('status');
                
                async function process() {
                    try {
                        const URL = 'https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js/weights';
                        
                        status.innerText = "Step 1: Downloading AI Weights...";
                        // Wait for each model to load
                        await faceapi.nets.ssdMobilenetv1.loadFromUri(URL);
                        await faceapi.nets.faceRecognitionNet.loadFromUri(URL);
                        await faceapi.nets.faceLandmark68Net.loadFromUri(URL);
                        
                        status.innerText = "Step 2: Analyzing Facial Geometry...";
                        const img = new Image();
                        img.src = "data:image/jpeg;base64,IMG_DATA";
                        
                        img.onload = async () => {
                            const det = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor();
                            if (det) {
                                window.parent.postMessage({
                                    type: 'streamlit:setComponentValue', 
                                    value: Array.from(det.descriptor)
                                }, '*');
                                status.innerText = "✅ Encoding Ready!";
                            } else {
                                window.parent.postMessage({type: 'streamlit:setComponentValue', value: 'ERR_NO_FACE'}, '*');
                                status.innerText = "❌ No face detected.";
                            }
                        };
                    } catch (e) {
                        status.innerText = "ERROR: " + e.message;
                        console.error(e);
                    }
                }
                
                // Safety check: wait 200ms for library to register in global scope
                setTimeout(process, 200);
            };
        </script>
        """
        js_component = raw_js.replace("IMG_DATA", img_base64)
        data_from_js = st.components.v1.html(js_component, height=100)

        if data_from_js:
            if isinstance(data_from_js, list):
                st.session_state.temp_encoding = data_from_js
                st.session_state.reg_stage = "ready"
                st.rerun()
            elif data_from_js == "ERR_NO_FACE":
                st.error("❌ No face detected. Use a solo photo with good lighting.")
                st.session_state.reg_stage = "idle"

    elif st.session_state.reg_stage == "ready":
        st.success(f"✅ Encoding Complete for {name}!")
        if st.button(f"Step 2: Register {name} Now"):
            new_user = {"name": name, "encoding": st.session_state.temp_encoding}
            st.session_state.registered_users.append(new_user)
            with open(DB_FILE, "w") as f:
                json.dump(st.session_state.registered_users, f)
            st.session_state.reg_stage = "idle"
            st.success(f"User {name} added to local database!")
            st.rerun()
        
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
