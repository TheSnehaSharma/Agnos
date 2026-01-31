import streamlit as st
import pandas as pd
import json
import numpy as np
import os
import base64
from datetime import datetime

# --- PERSISTENCE ---
DB_FILE = "registered_faces.json"
LOG_FILE = "attendance_log.csv"

st.set_page_config(page_title="Privacy Face Auth", layout="wide")

if "db" not in st.session_state:
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f: st.session_state.db = json.load(f)
    else: st.session_state.db = {}

if "logs" not in st.session_state:
    if os.path.exists(LOG_FILE):
        try: st.session_state.logs = pd.read_csv(LOG_FILE)
        except: st.session_state.logs = pd.DataFrame(columns=["Name", "Time"])
    else: st.session_state.logs = pd.DataFrame(columns=["Name", "Time"])

# --- FRONTEND CODE ---
def get_component_html(img_b64=None):
    # If an image is provided, JS will process the static image instead of the webcam
    image_logic = f"const staticImgSrc = 'data:image/jpeg;base64,{img_b64}';" if img_b64 else "const staticImgSrc = null;"
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ margin:0; background: #0e1117; color: #00FF00; font-family: monospace; }}
            #view {{ position: relative; width: 100%; height: 400px; border-radius: 12px; overflow: hidden; background: #000; border: 1px solid #333; }}
            video, canvas, img {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover; }}
            #status-bar {{ position: absolute; top: 0; left: 0; right: 0; background: rgba(0,0,0,0.8); padding: 8px; font-size: 11px; z-index: 100; }}
        </style>
    </head>
    <body>
        <div id="view">
            <div id="status-bar">INITIALIZING ENGINE...</div>
            <video id="webcam" autoplay playsinline muted style="display: {'none' if img_b64 else 'block'}"></video>
            <img id="static-img" src="" style="display: {'block' if img_b64 else 'none'}">
            <canvas id="overlay"></canvas>
        </div>

        <script type="module">
            import {{ FaceLandmarker, FilesetResolver }} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

            const video = document.getElementById("webcam");
            const staticImg = document.getElementById("static-img");
            const canvas = document.getElementById("overlay");
            const ctx = canvas.getContext("2d");
            const log = document.getElementById("status-bar");
            {image_logic}
            let faceLandmarker;

            async function init() {{
                try {{
                    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm");
                    faceLandmarker = await FaceLandmarker.createFromOptions(vision, {{
                        baseOptions: {{
                            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
                            delegate: "GPU"
                        }},
                        runningMode: staticImgSrc ? "IMAGE" : "VIDEO",
                        numFaces: 1
                    }});

                    if (staticImgSrc) {{
                        log.innerText = "STATUS: Processing Uploaded Image...";
                        staticImg.src = staticImgSrc;
                        staticImg.onload = async () => {{
                            const results = await faceLandmarker.detect(staticImg);
                            processResults(results);
                        }};
                    }} else {{
                        log.innerText = "STATUS: Starting Camera...";
                        const stream = await navigator.mediaDevices.getUserMedia({{ video: true }});
                        video.srcObject = stream;
                        video.onloadeddata = () => {{
                            log.innerText = "SYSTEM ONLINE";
                            predictVideo();
                        }};
                    }}
                } catch (err) {{
                    log.innerText = "ERROR: " + err.message;
                }}
            }}

            function processResults(results) {{
                canvas.width = staticImgSrc ? staticImg.naturalWidth : video.videoWidth;
                canvas.height = staticImgSrc ? staticImg.naturalHeight : video.videoHeight;
                ctx.clearRect(0, 0, canvas.width, canvas.height);

                if (results.faceLandmarks && results.faceLandmarks.length > 0) {{
                    const landmarks = results.faceLandmarks[0];
                    drawRect(landmarks);
                    window.parent.postMessage({{
                        type: "streamlit:setComponentValue",
                        value: JSON.stringify(landmarks)
                    }}, "*");
                    log.innerText = "✅ FACE ENCODED SUCCESSFULLY";
                }} else {{
                    log.innerText = "❌ NO FACE DETECTED";
                }}
            }}

            async function predictVideo() {{
                const results = faceLandmarker.detectForVideo(video, performance.now());
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                if (results.faceLandmarks && results.faceLandmarks.length > 0) {{
                    drawRect(results.faceLandmarks[0]);
                    window.parent.postMessage({{
                        type: "streamlit:setComponentValue",
                        value: JSON.stringify(results.faceLandmarks[0])
                    }}, "*");
                }}
                window.requestAnimationFrame(predictVideo);
            }}

            function drawRect(landmarks) {{
                const xs = landmarks.map(p => p.x * canvas.width);
                const ys = landmarks.map(p => p.y * canvas.height);
                const minX = Math.min(...xs), maxX = Math.max(...xs);
                const minY = Math.min(...ys), maxY = Math.max(...ys);
                ctx.strokeStyle = "#00FF00";
                ctx.lineWidth = 5;
                ctx.strokeRect(minX, minY, maxX - minX, maxY - minY);
            }}

            init();
        </script>
    </body>
    </html>
    """

# --- UI LOGIC ---
page = st.sidebar.radio("Navigate", ["Register", "Live Feed", "Log"])

if page == "Register":
    st.header("👤 Register via Photo Upload")
    name = st.text_input("Name").upper()
    uploaded_file = st.file_uploader("Upload a clear face photo", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Convert image to b64 for the JS component
        bytes_data = uploaded_file.getvalue()
        b64_img = base64.b64encode(bytes_data).decode()
        
        # Load component with the image data
        val = st.components.v1.html(get_component_html(b64_img), height=420)
        
        if val and isinstance(val, str):
            st.session_state.buffered_encoding = val
            st.success(f"✅ Facial features extracted for {name if name else 'user'}")
    
    if st.button("Save to Database"):
        if name and st.session_state.get('buffered_encoding'):
            st.session_state.db[name] = json.loads(st.session_state.buffered_encoding)
            with open(DB_FILE, "w") as f:
                json.dump(st.session_state.db, f)
            st.success(f"Registered {name}!")
            st.rerun()
        else:
            st.error("Upload a photo and enter a name first.")

elif page == "Live Feed":
    st.header("📹 Attendance Feed")
    col1, col2 = st.columns([3, 1])
    with col1:
        feed_val = st.components.v1.html(get_component_html(), height=420)
    with col2:
        if feed_val and isinstance(feed_val, str):
            current_face = json.loads(feed_val)
            identified = "Unknown"
            for db_name, saved_face in st.session_state.db.items():
                curr_arr = np.array([[p['x'], p['y']] for p in current_face[:30]])
                save_arr = np.array([[p['x'], p['y']] for p in saved_face[:30]])
                dist = np.mean(np.linalg.norm(curr_arr - save_arr, axis=1))
                if dist < 0.05:
                    identified = db_name
                    break
            st.subheader(f"Status: {identified}")
            if identified != "Unknown":
                if identified not in st.session_state.logs["Name"].values:
                    now = datetime.now().strftime("%H:%M:%S")
                    new_entry = pd.DataFrame({"Name": [identified], "Time": [now]})
                    st.session_state.logs = pd.concat([st.session_state.logs, new_entry], ignore_index=True)
                    st.session_state.logs.to_csv(LOG_FILE, index=False)
                    st.toast(f"Logged {identified}")

elif page == "Log":
    st.header("📊 History")
    st.dataframe(st.session_state.logs, use_container_width=True)
    st.download_button("Export CSV", st.session_state.logs.to_csv(index=False), "attendance.csv")
