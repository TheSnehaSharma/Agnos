import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import face_recognition
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import av
import threading
import os

RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
LOG_FILE = "attendance_log.csv"

if 'database' not in st.session_state:
    st.session_state.database = {"encodings": [], "names": []}

if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=["Name", "Date", "Time"]).to_csv(LOG_FILE, index=False)

# --- PAGE 1: REGISTRATION ---
def registration_page():
    st.title("👤 User Registration")
    st.info("Upload a photo to register a new user.")
    
    with st.form("registration_form", clear_on_submit=True):
        name = st.text_input("Enter Full Name")
        upload = st.file_uploader("Upload Face Photo", type=['jpg', 'png', 'jpeg'])
        submit = st.form_submit_button("Register")

        if submit:
            if name and upload:
                try:
        
                    file_bytes = np.asarray(bytearray(upload.read()), dtype=np.uint8)
                    img = cv2.imdecode(file_bytes, 1)
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                
                    encodes = face_recognition.face_encodings(rgb_img)

                    if encodes:

                        st.session_state.database["encodings"].append(encodes[0])
                        st.session_state.database["names"].append(name.upper())
                        st.success(f"✅ Registered {name.upper()} successfully!")
                    else:
                        st.error("❌ No face detected. Please upload a clearer photo.")
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning("⚠️ Name and Image are required.")

    
    if st.session_state.database["names"]:
        st.write("### Registered Users:")
        st.write(", ".join(st.session_state.database["names"]))

# --- PAGE 2: LIVE FEED ---
class VideoProcessor:
    def __init__(self):
        
        self.known_encodings = st.session_state.database["encodings"]
        self.known_names = st.session_state.database["names"]
        self.frame_count = 0 
        self.last_detected = None

    def recv(self, frame):
        
        img = frame.to_ndarray(format="bgr24")
        
        
        self.frame_count += 1
        
        if self.known_encodings: 
            
           
            img_small = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
            rgb_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
            
            
            face_locs = face_recognition.face_locations(rgb_small)
            face_encodes = face_recognition.face_encodings(rgb_small, face_locs)

            for encodeFace, faceLoc in zip(face_encodes, face_locs):
                matches = face_recognition.compare_faces(self.known_encodings, encodeFace)
                face_dis = face_recognition.face_distance(self.known_encodings, encodeFace)
                
                match_index = np.argmin(face_dis)
                name = "UNKNOWN"
                color = (0, 0, 255)

                if matches[match_index] and face_dis[match_index] < 0.50:
                    name = self.known_names[match_index]
                    color = (0, 255, 0) # Green
                    
                    
                    now = datetime.now()
                    current_date = now.strftime("%Y-%m-%d")
                    current_time = now.strftime("%H:%M:%S")
                    
                
                    try:
                        with open(LOG_FILE, "a") as f:
                             
                             f.write(f"{name},{current_date},{current_time}\n")
                    except:
                        pass

                
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

def live_feed_page():
    st.title("📷 Live Attendance")
    
    if not st.session_state.database["names"]:
        st.warning("Please register a face first.")
        return

    st.markdown("**Note:** Detection runs every few frames to maintain performance.")
    
    webrtc_streamer(
        key="attendance",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False}
    )

# --- PAGE 3: DATA & LOGS ---
def attendance_page():
    st.title("📋 Attendance Logs")
    
    if os.path.exists(LOG_FILE):
        
        try:
            df = pd.read_csv(LOG_FILE, names=["Name", "Date", "Time"])
            df = df[df["Name"] != "Name"] 
        except:
            df = pd.DataFrame(columns=["Name", "Date", "Time"])
    else:
        df = pd.DataFrame(columns=["Name", "Date", "Time"])

    st.dataframe(df, use_container_width=True)

    if not df.empty:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", csv, "attendance.csv", "text/csv")
    
    if st.button("🗑️ Clear Logs"):
        pd.DataFrame(columns=["Name", "Date", "Time"]).to_csv(LOG_FILE, index=False)
        st.rerun()

# --- NAVIGATION ---
pg = st.navigation({
    "System": [
        st.Page(registration_page, title="Register", icon="👤"),
        st.Page(live_feed_page, title="Live Camera", icon="📹"),
        st.Page(attendance_page, title="View Logs", icon="📄")
    ]
})

pg.run()
