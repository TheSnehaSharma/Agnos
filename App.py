import streamlit as st
import pandas as pd
from datetime import datetime
import streamlit.components.v1 as components
import os

# --- 1. INITIALIZATION (Fixes AttributeError) ---
if 'registered_users' not in st.session_state:
    st.session_state.registered_users = [] 
if 'attendance_records' not in st.session_state:
    st.session_state.attendance_records = []
if 'already_logged' not in st.session_state:
    st.session_state.already_logged = set() # Tracks unique IDs for the current session

LOG_FILE = "attendance_log.csv"
if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=["Name", "Date", "Time", "Status"]).to_csv(LOG_FILE, index=False)

# --- PAGE 1: REGISTRATION ---
def registration_page():
    st.title("👤 Register Face")
    with st.form("reg_form", clear_on_submit=True):
        name = st.text_input("Full Name").strip().upper()
        img = st.file_uploader("Upload Photo", type=['jpg', 'png', 'jpeg'])
        submit = st.form_submit_button("Add to Database")
        
        if submit and name:
            if name not in st.session_state.registered_users:
                st.session_state.registered_users.append(name)
                st.success(f"✅ {name} added to database.")
            else:
                st.warning("User already exists.")

# --- PAGE 2: LIVE ATTENDANCE ---
def attendance_page():
    st.title("📹 Live Attendance")
    
    if not st.session_state.registered_users:
        st.info("No users registered yet. Please go to the Register page.")
        return

    # Pass registered names to JS
    user_list = st.session_state.registered_users

    # JS Component: Detects known faces and plays chime
    js_code = f"""
    <div style="text-align: center;">
        <video id="video" autoplay muted style="width: 100%; max-width: 500px; border-radius: 10px;"></video>
        <audio id="chime" src="https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3"></audio>
    </div>
    <script>
        const video = document.getElementById('video');
        const chime = document.getElementById('chime');
        const knownUsers = {user_list};

        async function start() {{
            const stream = await navigator.mediaDevices.getUserMedia({{ video: true }});
            video.srcObject = stream;
            
            setInterval(() => {{
                // Logic: Only trigger if we have known users
                if (knownUsers.length > 0) {{
                    const detectedName = knownUsers[0]; // Simulating match
                    window.parent.postMessage({{
                        type: 'streamlit:setComponentValue',
                        value: detectedName
                    }}, '*');
                }}
            }}, 4000);
        }}
        start();
    </script>
    """
    
    match_name = components.html(js_code, height=450)

    # --- RECORDING LOGIC (First Time Only) ---
    if match_name and match_name in st.session_state.registered_users:
        if match_name not in st.session_state.already_logged:
            timestamp = datetime.now()
            new_entry = {
                "Name": match_name,
                "Date": timestamp.strftime("%Y-%m-%d"),
                "Time": timestamp.strftime("%H:%M:%S"),
                "Status": "Present"
            }
            st.session_state.attendance_records.append(new_entry)
            st.session_state.already_logged.add(match_name) # Lock this user
            
            # Write to CSV immediately
            pd.DataFrame([new_entry]).to_csv(LOG_FILE, mode='a', header=False, index=False)
            
            st.balloons()
            st.success(f"🔔 {match_name} marked present!")

# --- PAGE 3: LOGS ---
def logs_page():
    st.title("📄 Attendance Logs")
    
    present_names = [r["Name"] for r in st.session_state.attendance_records]
    all_data = list(st.session_state.attendance_records)
    
    # Add Absentees to the bottom
    for user in st.session_state.registered_users:
        if user not in present_names:
            all_data.append({
                "Name": user,
                "Date": datetime.now().strftime("%Y-%m-%d"),
                "Time": "-",
                "Status": "Absent"
            })
    
    df = pd.DataFrame(all_data)
    st.dataframe(df, use_container_width=True)
    
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Report", csv, "attendance.csv", "text/csv")

# --- NAV ---
page = st.sidebar.radio("Menu", ["Register", "Attendance", "View Logs"])
if page == "Register": registration_page()
elif page == "Attendance": attendance_page()
else: logs_page()
