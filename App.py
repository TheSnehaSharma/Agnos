import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import face_recognition
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os

# --- SHARED CONFIGURATION ---
RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# Initialize Session State
if 'database' not in st.session_state:
    st.session_state.database = {"encodings": [], "names": []}
if 'attendance_log' not in st.session_state:
    st.session_state.attendance_log = pd.DataFrame(columns=["Name", "Date", "Time"])

# --- PAGE 1: REGISTRATION ---
def registration_page():
    st.title("👤 User Registration")
    st.markdown("Use the form below to add a new person to the system. The data is processed only when you click **Submit**.")
    
    with st.form("registration_form", clear_on_submit=True):
        name = st.text_input("Enter Full Name")
        upload = st.file_uploader("Upload Clear Face Photo", type=['jpg', 'png', 'jpeg'])
        submit_button = st.form_submit_button("Register & Encode Face")

        if submit_button:
            if name and upload:
                try:

                    file_bytes = np.asarray(bytearray(upload.read()), dtype=np.uint8)
                    img = cv2.imdecode(file_bytes, 1)
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    encodes = face_recognition.face_encodings(rgb_img)
                    if encodes:
                        st.session_state.database["encodings"].append(encodes[0])
                        st.session_state.database["names"].append(name.upper())
                        st.success(f"✅ Successfully registered {name.upper()}!")
                    else:
                        st.error("❌ Could not find a face in that photo. Try again.")
                except Exception as e:
                    st.error(f"Error processing image: {e}")
            else:
                st.warning("⚠️ Both Name and Image are required.")

    with st.expander("📂 View Registered Users"):
        if st.session_state.database["names"]:
            st.write(", ".join(st.session_state.database["names"]))
        else:
            st.info("No users registered yet.")

# --- PAGE 2: LIVE FEED ---
class FaceProcessor(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if not st.session_state.database["encodings"]:
            return img

        img_small = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        face_locs = face_recognition.face_locations(img_small)
        face_encodes = face_recognition.face_encodings(img_small, face_locs)

        for encodeFace, faceLoc in zip(face_encodes, face_locs):

            face_distances = face_recognition.face_distance(st.session_state.database["encodings"], encodeFace)
            match_index = np.argmin(face_distances)
            distance = face_distances[match_index]
            

            if distance < 0.6:
                name = st.session_state.database["names"][match_index]
                match_perc = round((1 - distance) * 100, 1)
                color = (0, 255, 0)
                
                now = datetime.now()
                new_entry = {"Name": name, "Date": now.strftime("%Y-%m-%d"), "Time": now.strftime("%H:%M:%S")}

                if not ((st.session_state.attendance_log['Name'] == name) & 
                        (st.session_state.attendance_log['Date'] == new_entry['Date'])).any():
                    st.session_state.attendance_log = pd.concat([st.session_state.attendance_log, pd.DataFrame([new_entry])], ignore_index=True)
            else:
                name = "UNKNOWN"
                match_perc = 0
                color = (0, 0, 255)

            y1, x2, y2, x1 = [v*4 for v in faceLoc]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{name} {match_perc}%", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return img

def live_feed_page():
    st.title("📷 Live Attendance Feed")
    if not st.session_state.database["names"]:
        st.warning("No faces registered. Please go to the Registration page first.")
    
    webrtc_streamer(
        key="attendance-stream",
        video_transformer_factory=FaceProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False}
    )

# --- PAGE 3: ATTENDANCE LOGS ---
def attendance_page():
    st.title("📋 Attendance Records")
    
    if st.session_state.attendance_log.empty:
        st.info("No attendance records found yet.")
    else:
        st.dataframe(st.session_state.attendance_log, use_container_width=True)
        
        csv = st.session_state.attendance_log.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Attendance as CSV",
            data=csv,
            file_name=f"attendance_{datetime.now().strftime('%Y-%m-%d')}.csv",
            mime="text/csv",
        )
        
        if st.button("🗑️ Clear Logs"):
            st.session_state.attendance_log = pd.DataFrame(columns=["Name", "Date", "Time"])
            st.rerun()

# --- NAVIGATION ---
pg = st.navigation({
    "Manage": [st.Page(registration_page, title="Register Face", icon="👤")],
    "Live": [st.Page(live_feed_page, title="Live Feed", icon="📷")],
    "Data": [st.Page(attendance_page, title="View Attendance", icon="📋")]
})
pg.run()
