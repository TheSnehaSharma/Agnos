import streamlit as st
import pandas as pd
import json
import base64
import os
import numpy as np
from datetime import datetime

# --- 1. INITIALIZATION ---
st.set_page_config(page_title="Google MediaPipe AI", layout="wide")
DB_FILE = "registered_vectors.json"

if 'registered_users' not in st.session_state:
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            st.session_state.registered_users = json.load(f)
    else:
        st.session_state.registered_users = []

# --- 2. SERVER-SIDE COMPARISON ---
def find_match_on_server(input_vector):
    if not st.session_state.registered_users:
        return "UNKNOWN"
    
    input_vec = np.array(input_vector)
    best_match = "UNKNOWN"
    min_dist = 0.45 # Threshold: 0.4 (Strict) to 0.6 (Loose)
    
    for user in st.session_state.registered_users:
        known_vec = np.array(user['vector'])
        # Cosine Distance logic
        dot_product = np.dot(input_vec, known_vec)
        norm_a = np.linalg.norm(input_vec)
        norm_b = np.linalg.norm(known_vec)
        dist = 1 - (dot_product / (norm_a * norm_b))
        
        if dist < min_dist:
            min_dist = dist
            best_match = base64.b64decode(user['name_encoded']).decode()
            
    return best_match

# --- 3. REGISTRATION PAGE ---
def registration_page():
    st.title("👤 Secure Registration")
    name = st.text_input("Full Name").strip().upper()
    img_file = st.file_uploader("Upload Profile", type=['jpg', 'png', 'jpeg'])

    if img_file and name:
        img_b64 = base64.b64encode(img_file.read()).decode()
        name_enc = base64.b64encode(name.encode()).decode()

        # Fixed JS: Explicit model path and task assignment
        js_mp_reg = f"""
        <div id="status" style="color:#4285F4; font-family:sans-serif; font-weight:bold;">🧬 Initializing Google AI...</div>
        <script type="module">
            import vision from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

            async function run() {{
                const status = document.getElementById("status");
                try {{
                    const fileset = await vision.FilesetResolver.forVisionTasks(
                        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
                    );
                    const faceEmbedder = await vision.FaceEmbedder.createFromOptions(fileset, {{
                        baseOptions: {{ 
                            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_embedder/face_embedder/float16/latest/face_embedder.tflite",
                            delegate: "GPU"
                        }}
                    }});

                    const img = new Image();
                    img.src = "data:image/jpeg;base64,{img_b64}";
                    img.onload = async () => {{
                        status.innerText = "🧬 Vectorizing Face...";
                        const result = await faceEmbedder.embed(img);
                        if (result.embeddings && result.embeddings.length > 0) {{
                            window.parent.postMessage({{
                                type: 'streamlit:setComponentValue',
                                value: {{ vector: Array.from(result.embeddings[0].floatVector), name: "{name_enc}" }}
                            }}, '*');
                            status.innerText = "✅ Done!";
                        }} else {{
                            status.innerText = "❌ No face detected.";
                        }}
                    }};
                }} catch (e) {{
                    status.innerText = "❌ Error: " + e.message;
                }}
            }}
            run();
        </script>
        """
        data = st.components.v1.html(js_mp_reg, height=100)

        if data and isinstance(data, dict):
            new_user = {"name_encoded": data['name'], "vector": data['vector']}
            st.session_state.registered_users.append(new_user)
            with open(DB_FILE, "w") as f:
                json.dump(st.session_state.registered_users, f)
            st.success("Successfully registered!")
            st.rerun()

# --- 4. ATTENDANCE PAGE ---
def attendance_page():
    st.title("📹 Live Privacy Scanner")
    
    js_mp_attendance = """
    <div style="position: relative;">
        <video id="webcam" autoplay playsinline style="width: 100%; max-width: 640px; border-radius: 12px; transform: scaleX(-1);"></video>
        <canvas id="output_canvas" style="position: absolute; top: 0; left: 0; transform: scaleX(-1);"></canvas>
    </div>
    <p id="msg" style="color:#666; font-family:sans-serif;">Waiting for camera...</p>

    <script type="module">
        import vision from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

        const video = document.getElementById("webcam");
        const canvas = document.getElementById("output_canvas");
        const msg = document.getElementById("msg");

        async function init() {
            try {
                const fileset = await vision.FilesetResolver.forVisionTasks(
                    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
                );
                
                const detector = await vision.FaceDetector.createFromOptions(fileset, {
                    baseOptions: { 
                        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/face_detector.tflite" 
                    },
                    runningMode: "VIDEO"
                });

                const embedder = await vision.FaceEmbedder.createFromOptions(fileset, {
                    baseOptions: { 
                        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_embedder/face_embedder/float16/latest/face_embedder.tflite" 
                    }
                });

                const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                video.srcObject = stream;

                video.addEventListener("loadeddata", async () => {
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                    msg.innerText = "🔒 Official Google AI Active";
                    
                    const drawUtils = new vision.DrawingUtils(canvas.getContext("2d"));

                    async function loop() {
                        const detections = await detector.detectForVideo(video, performance.now());
                        canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);

                        if (detections.detections && detections.detections.length > 0) {
                            for (const det of detections.detections) {
                                drawUtils.drawBoundingBox(det.boundingBox, { color: "#4285F4", lineWidth: 2 });
                            }

                            const result = await embedder.embed(video);
                            if (result.embeddings && result.embeddings.length > 0) {
                                window.parent.postMessage({
                                    type: 'streamlit:setComponentValue',
                                    value: Array.from(result.embeddings[0].floatVector)
                                }, '*');
                            }
                        }
                        requestAnimationFrame(loop);
                    }
                    loop();
                });
            } catch (e) { msg.innerText = "Error: " + e.message; }
        }
        init();
    </script>
    """
    
    live_vector = st.components.v1.html(js_mp_attendance, height=520)

    if live_vector:
        name = find_match_on_server(live_vector)
        if name == "UNKNOWN":
            st.warning("⚠️ Unknown Person")
        else:
            st.success(f"✅ Recognized: {name}")
            if st.button(f"Log Presence for {name}"):
                st.toast(f"Logged {name}!")

# --- NAV ---
choice = st.sidebar.radio("Navigation", ["Take Attendance", "Register Face"])
if choice == "Register Face": registration_page()
else: attendance_page()
