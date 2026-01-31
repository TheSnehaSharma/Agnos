import streamlit as st
import pandas as pd
import cv2
import numpy as np
import base64
import os
import pickle
from deepface import DeepFace
from datetime import datetime

# --- CONFIG ---
DB_FOLDER = "registered_faces"
PKL_LOG = "attendance_data.pkl"
if not os.path.exists(DB_FOLDER):
    os.makedirs(DB_FOLDER)

st.set_page_config(page_title="DeepFace Bio-Auth", layout="wide")

# --- UI ASSETS ---
JS_CODE = """
<script>
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');

    async function setupCamera() {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
    }

    function captureFrame() {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0);
        const dataURL = canvas.toDataURL('image/jpeg', 0.5);
        
        // Send the image string to Streamlit
        window.parent.postMessage({
            type: "streamlit:setComponentValue",
            value: dataURL
        }, "*");
    }

    setupCamera();
    setInterval(captureFrame, 2000); // Send frame every 2 seconds for matching
</script>
"""

# --- LOGIC ---
def verify_face(img_base64):
    try:
        # Decode the image
        encoded_data = img_base64.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Save temp file for DeepFace
        temp_path = "temp_live.jpg"
        cv2.imwrite(temp_path, img)
        
        # Search the database folder
        results = DeepFace.find(img_path=temp_path, db_path=DB_FOLDER, 
                               model_name="Facenet", enforce_detection=False, silent=True)
        
        if len(results) > 0 and not results[0].empty:
            # Get the filename of the match (which is the user's name)
            match_path = results[0].iloc[0]['identity']
            user_name = os.path.basename(match_path).split('.')[0]
            return user_name
    except Exception as e:
        return None
    return "Unknown"

# --- UI NAVIGATION ---
page = st.sidebar.radio("Navigate", ["Register", "Live Feed", "Log"])

if page == "Register":
    st.header("👤 High-Accuracy Registration")
    name = st.text_input("Name").upper()
    img_file = st.file_uploader("Upload a clear face photo", type=['jpg', 'png'])
    
    if img_file and name:
        if st.button("Save to Secure Vault"):
            file_path = os.path.join(DB_FOLDER, f"{name}.jpg")
            with open(file_path, "wb") as f:
                f.write(img_file.getbuffer())
            st.success(f"Registered {name} with DeepFace Embeddings.")

elif page == "Live Feed":
    st.header("📹 Deep Learning Scanner")
    col_v, col_m = st.columns([3, 1])
    
    with col_v:
        # Create a simple container for the webcam
        st.markdown('<video id="video" autoplay style="width:100%; border-radius:10px;"></video>', unsafe_allow_html=True)
        st.markdown('<canvas id="canvas" style="display:none;"></canvas>', unsafe_allow_html=True)
        img_data = st.components.v1.html(f"<html><body>{JS_CODE}</body></html>", height=0)

    with col_m:
        if isinstance(img_data, str) and len(img_data) > 100:
            with st.spinner("Analyzing..."):
                identity = verify_face(img_data)
                
            if identity and identity != "Unknown":
                st.success(f"Verified: {identity}")
                # Pickle logging logic here...
            else:
                st.warning("Identity: Unknown")

elif page == "Log":
    st.header("📊 Deep Match History")
    # (Previous Pickle loading logic)
