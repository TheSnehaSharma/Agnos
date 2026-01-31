import streamlit as st
import pandas as pd
from datetime import datetime
import streamlit.components.v1 as components
import os

# --- DATABASE & STATE ---
LOG_FILE = "attendance_log.csv"
if 'registered_users' not in st.session_state:
    st.session_state.registered_users = [] # List of Names
if 'attendance_records' not in st.session_state:
    st.session_state.attendance_records = [] # List of dicts

# Initialize Log File
if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=["Name", "Date", "Time", "Status"]).to_csv(LOG_FILE, index=False)

# --- PAGE 1: REGISTRATION ---
def registration_page():
    st.title("👤 Register Face")
    with st.form("reg_form", clear_on_submit=True):
        name = st.text_input("Full Name").strip().upper()
        img = st.file_uploader("Upload Reference Photo", type=['jpg', 'png', 'jpeg'])
        submit = st.form_submit_button("Add to Database")
        
        if submit and name and img:
            if name not in st.session_state.registered_users:
                st.session_state.registered_users.append(name)
                st.success(f"✅ {name} added to database.")
            else:
                st.warning("User already exists.")

# --- PAGE 2: LIVE ATTENDANCE ---
def attendance_page():
    st.title("📹 Live Attendance")
    
    if not st.session_state.registered_users:
        st.error("❌ No users in database. Detection disabled.")
        return

    user_list_json = st.session_state.registered_users

    js_code = f"""
    <div style="text-align: center;">
        <video id="video" autoplay muted style="width: 100%; max-width: 500px; border-radius: 15px; border: 3px solid #4CAF50;"></video>
        <audio id="chime" src="https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3"></audio>
    </div>

    <script>
        const video = document.getElementById('video');
        const chime = document.getElementById('chime');
        const knownUsers = {user_list_json};
        let lastMatch = "";

        async function start() {{
            const stream = await navigator.mediaDevices.getUserMedia({{ video: true }});
            video.srcObject = stream;
            
            // Simulation: In a full face-api implementation, detection happens here.
            // For this demo, we detect "Known Face" every 3 seconds if one exists.
            setInterval(() => {{
                if (knownUsers.length > 0) {{
                    const randomMatch = knownUsers[0]; // Logic: Match first person in list
                    if (randomMatch !== lastMatch) {{
                        lastMatch = randomMatch;
                        chime.play();
                        window.parent.postMessage({{
                            type: 'streamlit:setComponentValue',
                            value: randomMatch
                        }}, '*');
                    }}
                }}
            }}, 3000);
        }}
        start();
    </script>
    """
    
    match_name = components.html(js_code, height=400)

    if match_name:
        timestamp = datetime.now()
        record = {{
            "Name": str(match_name),
            "Date": timestamp.strftime("%Y-%m-%d"),
            "Time": timestamp.strftime("%H:%M:%S"),
            "Status": "Present"
        }}

        if not any(r['Name'] == record['Name'] for r in st.session_state.attendance_records):
            st.session_state.attendance_records.append(record)
            st.balloons()
            st.success(f"🎊 {record['Name']} marked PRESENT at {record['Time']}")

# --- PAGE 3: LOGS & REPORT ---
def logs_page():
    st.title("📄 Attendance Logs")
    
    # Prepare Final Report
    present_names = [r["Name"] for r in st.session_state.attendance_records]
    final_data = list(st.session_state.attendance_records)
    
    # Append Absentees
    for user in st.session_state.registered_users:
        if user not in present_names:
            final_data.append({
                "Name": user,
                "Date": datetime.now().strftime("%Y-%m-%d"),
                "Time": "-",
                "Status": "Absent"
            })
    
    df = pd.DataFrame(final_data)
    st.dataframe(df, use_container_width=True)

    if not df.empty:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV Report", csv, "attendance_report.csv", "text/csv")

# --- NAV ---
page = st.sidebar.radio("Navigation", ["Register", "Attendance", "View Logs"])
if page == "Register": registration_page()
elif page == "Attendance": attendance_page()
else: logs_page()
