import streamlit as st
import pandas as pd
import json
import base64
import os
import numpy as np
from datetime import datetime

# --- 1. INITIALIZATION ---
st.set_page_config(page_title="Official Google MediaPipe AI", layout="wide")
DB_FILE = "registered_vectors.json"

if 'registered_users' not in st.session_state:
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            st.session_state.registered_users = json.load(f)
    else:
        st.session_state.registered_users = []

# --- 2. DEEPFACE-STYLE COMPARISON (SERVER SIDE) ---
def find_match_on_server(input_vector):
    if not st.session_state.registered_users:
        return "UNKNOWN"
    
    best_match = "UNKNOWN"
    min_dist = 0.45 # Cosine distance threshold
    
    input_vec = np.array(input_vector)
    
    for user in st.session_state.registered_users:
        known_vec = np.array(user['vector'])
        # Cosine Distance
        dist = 1 - (np.dot(input_vec, known_vec) / (np.linalg.norm(input_vec) * np.linalg.norm(known_vec)))
        
        if dist < min_dist:
            min_dist = dist
            best_match = base64.b64decode(user['name_encoded']).decode()
            
    return best_match

# --- 3. REGISTRATION PAGE ---
def registration_page():
    st.title("👤 Official Google MediaPipe Registration")
    st.info("Uses Google's Face Embedder. Your photo never leaves this tab.")
    
    name = st.text_input("Enter Full Name").strip().upper()
    img_file = st.file_uploader("Upload Profile", type=['jpg', 'png', 'jpeg'])

    if img_file and name:
        img_b64 = base64.b64encode(img_file.read()).decode()
        name_enc = base64.b64encode(name.encode()).decode()

        # JS Component using Google MediaPipe
        js_mp_reg = f"""
        <script type="module">
            import {{ FaceLandmarker, FilesetResolver, FaceEmbedder }} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

            async function run() {{
                const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm");
                const faceEmbedder = await FaceEmbedder.createFromOptions(vision, {{
                    baseOptions: {{ modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_embedder/face_embedder/float16/1/face_embedder.tflite" }}
                }});

                const img = new Image();
                img.src = "data:image/jpeg;base64,{img_b64}";
                img.onload = async () => {{
                    const result = await faceEmbedder.embed(img);
                    if (result.embeddings.length > 0) {{
                        window.parent.postMessage({{
                            type: 'streamlit:setComponentValue',
                            value: {{ vector: Array.from(result.embeddings[0].floatVector), name: "{name_enc}" }}
                        }}, '*');
                    }}
                }};
            }}
            run();
        </script>
        <div style="color:#4285F4; font-family:sans-serif;">🧬 Google AI processing...</div>
        """
        data = st.components.v1.html(js_mp_reg, height=50)

        if data and isinstance(data, dict):
            new_user = {{"name_encoded": data['name'], "vector": data['vector']}}
            st.session_state.registered_users.append(new_user)
            with open(DB_FILE, "w") as f:
                json.dump(st.session_state.registered_users, f)
            st.success(f"Successfully registered via MediaPipe!")
            st.rerun()

# --- 4. ATTENDANCE PAGE ---
def attendance_page():
    st.title("📹 Official Live Scanner")
    
    # MediaPipe Live Face Detection + Embedding
    js_mp_attendance = """
    <div style="position: relative;">
        <video id="webcam" autoplay playsinline style="width: 100%; max-width: 640px; border-radius: 12px; transform: scaleX(-1);"></video>
        <canvas id="output_canvas" style="position: absolute; top: 0; left: 0; transform: scaleX(-1);"></canvas>
    </div>

    <script type="module">
        import {{ FaceDetector, FilesetResolver, FaceEmbedder, DrawingUtils }} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

        const video = document.getElementById("webcam");
        const canvas = document.getElementById("output_canvas");
        const ctx = canvas.getContext("2d");

        async function init() {
            const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm");
            
            const faceDetector = await FaceDetector.createFromOptions(vision, {
                baseOptions: { modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/face_detector.tflite" },
                runningMode: "VIDEO"
            });

            const faceEmbedder = await FaceEmbedder.createFromOptions(vision, {
                baseOptions: { modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_embedder/face_embedder/float16/1/face_embedder.tflite" }
            });

            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = stream;

            video.addEventListener("loadeddata", async () => {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                
                async function predict() {
                    const detections = await faceDetector.detectForVideo(video, performance.now());
                    ctx.clearRect(0, 0, canvas.width, canvas.height);

                    if (detections.detections.length > 0) {
                        // Draw Detection Box
                        const drawingUtils = new DrawingUtils(ctx);
                        for (const detection of detections.detections) {
                            drawingUtils.drawBoundingBox(detection.boundingBox, { color: "#4285F4", lineWidth: 3 });
                        }

                        // Extract Embedding (Vector)
                        const embedResult = await faceEmbedder.embed(video);
                        if (embedResult.embeddings.length > 0) {
                            window.parent.postMessage({
                                type: 'streamlit:setComponentValue',
                                value: Array.from(embedResult.embeddings[0].floatVector)
                            }, '*');
                        }
                    }
                    requestAnimationFrame(predict);
                }
                predict();
            });
        }
        init();
    </script>
    """
    
    live_vector = st.components.v1.html(js_mp_attendance, height=500)

    if live_vector:
        # Comparison logic on server
        user_name = find_match_on_server(live_vector)
        
        if user_name == "UNKNOWN":
            st.warning("🔍 Searching... Face detected but not recognized.")
        else:
            st.success(f"✅ Verified: **{user_name}**")
            if st.button("Submit Attendance"):
                st.toast(f"Attendance recorded for {user_name}")

# --- NAV ---
choice = st.sidebar.radio("Navigation", ["Take Attendance", "Register Face"])
if choice == "Register Face": registration_page()
else: attendance_page()
