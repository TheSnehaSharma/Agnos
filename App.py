import streamlit as st
import pandas as pd
import cv2
import numpy as np
import base64
import os
import pickle
from deepface import DeepFace
from datetime import datetime

# --- CONFIG & DIRECTORIES ---
DB_FOLDER = "registered_faces"
PKL_LOG = "attendance_data.pkl"

if not os.path.exists(DB_FOLDER):
    os.makedirs(DB_FOLDER)

st.set_page_config(page_title="DeepFace Auth System", layout="wide")

# --- SESSION STATE INITIALIZATION ---
if "logged_set" not in st.session_state:
    st.session_state.logged_set = set()

# --- HELPER: PICKLE ENGINE ---
def save_attendance_pkl(name):
    logs = []
    if os.path.exists(PKL_LOG):
        with open(PKL_LOG, "rb") as f:
            try: logs = pickle.load(f)
            except: logs = []
    
    entry = {
        "Name": name,
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Date": datetime.now().strftime("%Y-%m-%d")
    }
    logs.append(entry)
    with open(PKL_LOG, "wb") as f:
        pickle.dump(logs, f)

# --- FRONTEND ASSETS ---
# JavaScript to capture video frame as Base64 and send to Streamlit
JS_CODE = """
<script>
    const video = document.createElement('video');
    video.autoplay = true;
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    async function startCamera() {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        document.getElementById('video-container').appendChild(video);
        video.style.width = "100%";
        video.style.borderRadius = "12px";
    }

    function sendFrame() {
        if (video.readyState === video.HAVE_ENOUGH_DATA) {
            canvas.width = 400; // Lower res for faster server processing
            canvas.height = 300;
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            const dataURL = canvas.toDataURL('image/jpeg', 0.6);
            window.parent.postMessage({
                type: "streamlit:setComponentValue",
                value: dataURL
            }, "*");
        }
    }

    startCamera();
    setInterval(sendFrame, 3000); // Analyze face every 3 seconds
</script>
<div id="video-container"></div>
"""

# --- UI NAVIGATION ---
page = st.sidebar.radio("Navigate", ["Register", "Live Feed", "Log History"])

if page == "Register":
    st.header("👤 Face Registration (Deep Embeddings)")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        name = st.text_input("Enter Full Name").upper()
        img_file = st.file_uploader("Upload clear face photo", type=['jpg', 'jpeg', 'png'])
        
        if st.button("Register User") and name and img_file:
            img_path = os.path.join(DB_FOLDER, f"{name}.jpg")
            with open(img_path, "wb") as f:
                f.write(img_file.getbuffer())
            st.success(f"Successfully registered {name}!")
            # DeepFace needs to refresh its internal represent_filenames if we add new images
            if os.path.exists(os.path.join(DB_FOLDER, "ds_model_vgg_face.pkl")):
                os.remove(os.path.join(DB_FOLDER, "ds_model_vgg_face.pkl"))

    with col2:
        st.subheader("Manage Database")
        all_files = [f for f in os.listdir(DB_FOLDER) if f.endswith(('.jpg', '.png'))]
        for f in all_files:
            c1, c2 = st.columns([3, 1])
            user_label = f.split('.')[0]
            c1.write(f"✅ {user_label}")
            if c2.button("🗑️", key=f"del_{f}"):
                os.remove(os.path.join(DB_FOLDER, f))
                st.rerun()

elif page == "Live Feed":
    st.header("📹 DeepFace Live Terminal")
    col_v, col_s = st.columns([2, 1])
    
    with col_v:
        # The return value from the HTML component is the Base64 image string
        img_b64 = st.components.v1.html(JS_CODE, height=350)

    with col_s:
        st.subheader("Identification")
        if isinstance(img_b64, str) and len(img_b64) > 1000:
            try:
                # Decode Base64 for DeepFace
                header, data = img_b64.split(',')
                img_bytes = base64.b64decode(data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Perform Face Search
                dfs = DeepFace.find(img_path=frame, db_path=DB_FOLDER, 
                                   model_name="Facenet", enforce_detection=False, 
                                   silent=True, detector_backend="opencv")
                
                if len(dfs) > 0 and not dfs[0].empty:
                    # Get top match
                    identity_path = dfs[0].iloc[0]['identity']
                    identified_name = os.path.basename(identity_path).split('.')[0]
                    confidence = 100 - int(dfs[0].iloc[0]['distance'] * 100)
                    
                    st.metric("Detected", identified_name, f"{confidence}% Accuracy")
                    
                    # LOGGING LOGIC
                    if identified_name not in st.session_state.logged_set:
                        save_attendance_pkl(identified_name)
                        st.session_state.logged_set.add(identified_name)
                        st.toast(f"✅ Attendance Logged: {identified_name}")
                else:
                    st.warning("Identity: Unknown")
            except Exception as e:
                st.error(f"Engine Error: {e}")
        else:
            st.info("Aligning camera...")

elif page == "Log History":
    st.header("📊 Attendance Log (Pickle)")
    
    if os.path.exists(PKL_LOG):
        with open(PKL_LOG, "rb") as f:
            data = pickle.load(f)
        
        df = pd.DataFrame(data)
        st.table(df)
        
        st.markdown("---")
        # ACTION BUTTONS
        c1, c2 = st.columns(2)
        with c1:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Report (CSV)", csv, "attendance_export.csv", "text/csv")
        
        with c2:
            if st.button("🔥 WIPE ALL SESSION DATA"):
                if os.path.exists(PKL_LOG):
                    os.remove(PKL_LOG)
                st.session_state.logged_set = set()
                st.success("Session data wiped. Starting fresh.")
                st.rerun()
    else:
        st.info("No logs found. Start the live feed to begin recording.")
