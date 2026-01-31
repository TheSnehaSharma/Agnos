import streamlit as st
import cv2
import numpy as np
import base64
import os
import pickle
from deepface import DeepFace
from streamlit_javascript import st_javascript

# --- CONFIG ---
DB_FOLDER = "registered_faces"
MODEL_NAME = "Facenet512"
if not os.path.exists(DB_FOLDER): os.makedirs(DB_FOLDER)

st.set_page_config(page_title="Bi-Directional Auth", layout="wide")

# --- THE JS BRIDGE SCRIPT ---
# This script initializes the camera once and returns a single frame when called
JS_CAPTURE = """
async () => {
    // 1. Setup persistent video element if it doesn't exist
    if (!window.cameraStream) {
        window.cameraStream = await navigator.mediaDevices.getUserMedia({ video: { width: 320, height: 240 } });
        window.videoEl = document.createElement('video');
        window.videoEl.srcObject = window.cameraStream;
        await window.videoEl.play();
        window.canvasEl = document.createElement('canvas');
    }
    
    // 2. Capture a frame
    const ctx = window.canvasEl.getContext('2d');
    window.canvasEl.width = 320;
    window.canvasEl.height = 240;
    ctx.drawImage(window.videoEl, 0, 0, 320, 240);
    
    // 3. Return the data to Python
    return window.canvasEl.toDataURL('image/jpeg', 0.5);
}
"""

st.header("📹 Real-Time Bridge")
col_v, col_s = st.columns([2, 1])

with col_v:
    st.write("Camera is active in the background.")
    # THE KEY: This function executes JS and WAITS for the return value in Python
    raw_frame = st_javascript(JS_CAPTURE)
    
    if raw_frame and len(str(raw_frame)) > 100:
        st.image(raw_frame, caption="Last Captured Frame", use_container_width=True)
    else:
        st.info("Initializing camera bridge...")

with col_s:
    if raw_frame:
        try:
            # Decode the captured frame
            encoded_data = raw_frame.split(",")[1]
            nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Run DeepFace
            res = DeepFace.find(img_path=frame, db_path=DB_FOLDER, 
                               model_name=MODEL_NAME, enforce_detection=False, 
                               detector_backend="opencv", silent=True)
            
            if len(res) > 0 and not res[0].empty:
                name = os.path.basename(res[0].iloc[0]['identity']).split('.')[0]
                st.success(f"Verified: {name}")
            else:
                st.warning("Identity Unknown")
        except Exception as e:
            st.error(f"Processing Error: {e}")

# Force a rerun to keep the "live" feel
if st.button("Manual Refresh"):
    st.rerun()
