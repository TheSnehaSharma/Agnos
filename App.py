import streamlit as st
import pandas as pd
import cv2
import numpy as np
import pickle
import hashlib
import os
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av

# --- 1. CONFIGURATION ---
DATA_DIR = "agnos_data"
if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
AUTH_FILE = os.path.join(DATA_DIR, "auth_registry.json")

# Required for Streamlit Cloud WebRTC
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.set_page_config(page_title="Agnos Enterprise", layout="wide", initial_sidebar_state="expanded")

# --- 2. BACKEND HELPERS ---
def load_auth():
    if os.path.exists(AUTH_FILE):
        import json
        with open(AUTH_FILE, "r") as f: return json.load(f)
    return {}

def save_auth(data):
    import json
    with open(AUTH_FILE, "w") as f: json.dump(data, f)

def get_file_paths(org_key):
    return {
        "db": os.path.join(DATA_DIR, f"faces_{org_key}.pkl"),
        "logs": os.path.join(DATA_DIR, f"logs_{org_key}.csv")
    }

# --- 3. SESSION STATE & LOGIN ---
if "auth_status" not in st.session_state: st.session_state.auth_status = False
if "org_key" not in st.session_state: st.session_state.org_key = None
if "known_names" not in st.session_state: st.session_state.known_names = []
if "known_encodings" not in st.session_state: st.session_state.known_encodings = []
if "logged_set" not in st.session_state: st.session_state.logged_set = set()

def load_org_data(org_key):
    paths = get_file_paths(org_key)
    st.session_state.known_names = []
    st.session_state.known_encodings = []
    st.session_state.logged_set = set()
    
    if os.path.exists(paths["db"]):
        try:
            with open(paths["db"], "rb") as f:
                data = pickle.load(f)
                st.session_state.known_names = data.get("names", [])
                st.session_state.known_encodings = data.get("encodings", [])
        except EOFError:
            pass
    
    if os.path.exists(paths["logs"]) and os.path.getsize(paths["logs"]) > 0:
        try:
            df = pd.read_csv(paths["logs"])
            today = datetime.now().strftime("%Y-%m-%d")
            todays_names = df[df['Date'] == today]['Name'].unique()
            st.session_state.logged_set = set(todays_names)
        except Exception:
            pass

def save_face_data():
    paths = get_file_paths(st.session_state.org_key)
    data = {"names": st.session_state.known_names, "encodings": st.session_state.known_encodings}
    with open(paths["db"], "wb") as f: pickle.dump(data, f)

def log_attendance(name):
    paths = get_file_paths(st.session_state.org_key)
    csv_path = paths["logs"]
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a") as f:
        if write_header: f.write("Name,Time,Date\n")
        f.write(f"{name},{time_str},{date_str}\n")
    
    st.session_state.logged_set.add(name)
    return True

# AUTO-LOGIN CHECK
if not st.session_state.auth_status:
    if "org" in st.query_params and "token" in st.query_params:
        q_org = st.query_params["org"]
        q_token = st.query_params["token"]
        auth_db = load_auth()
        if q_org in auth_db and auth_db[q_org] == q_token:
            st.session_state.auth_status = True
            st.session_state.org_key = q_org
            load_org_data(q_org)

# --- 4. UI START ---
if not st.session_state.auth_status:
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.title("🔐 Agnos Login")
        key_in = st.text_input("Org Key (5 Char)", max_chars=5).upper()
        auth_db = load_auth()
        is_known = (len(key_in) == 5 and key_in in auth_db)
        
        if len(key_in) == 5:
            if is_known: st.info(f"✅ Found: {key_in}")
            else: st.warning(f"🆕 Creating: {key_in}")

        with st.form("auth"):
            btn = "Sign In" if is_known else "Create Account"
            pw = st.text_input("Password", type="password")
            if st.form_submit_button(btn, type="primary"):
                h_pw = hashlib.sha256(pw.encode()).hexdigest()
                if is_known and auth_db[key_in] != h_pw:
                    st.error("Wrong Password")
                else:
                    if not is_known:
                        auth_db[key_in] = h_pw
                        save_auth(auth_db)
                    
                    st.session_state.auth_status = True
                    st.session_state.org_key = key_in
                    st.query_params["org"] = key_in
                    st.query_params["token"] = h_pw
                    load_org_data(key_in)
                    st.rerun()
else:
    with st.sidebar:
        st.title("🛡️ Agnos")
        st.caption(f"ORG: {st.session_state.org_key}")
        st.metric("Users", len(st.session_state.known_names))
        st.markdown("---")
        if st.button("Log Out"):
            st.query_params.clear()
            for k in list(st.session_state.keys()): del st.session_state[k]
            st.rerun()

    st.title("Agnos Enterprise (Simplified System)")
    tab1, tab2, tab3, tab4 = st.tabs(["🎥 Live Scanner", "👤 Register", "📊 Logs", "🗄️ Database"])

    with tab1:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown("### Camera Feed")
            
            # Simple Frame Processor
            def process_frame(frame: av.VideoFrame) -> av.VideoFrame:
                img = frame.to_ndarray(format="bgr24")
                height, width, _ = img.shape
                
                # Draw a static targeting box in the center
                cv2.rectangle(img, (width//4, height//4), (width*3//4, height*3//4), (0, 255, 0), 3)
                cv2.putText(img, "SCANNER ACTIVE", (width//4, height//4 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                return av.VideoFrame.from_ndarray(img, format="bgr24")

            webrtc_streamer(
                key="simple-scanner",
                mode=1,
                rtc_configuration=RTC_CONFIGURATION,
                video_frame_callback=process_frame,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True
            )

        with c2:
            st.subheader("Live Status")
            st.info("AI Face Recognition is temporarily disabled for UI testing.")
            
            st.markdown("---")
            st.markdown("**Test Logging System:**")
            test_name = st.text_input("Name to log")
            if st.button("Log Dummy Attendance", type="primary"):
                if test_name:
                    log_attendance(test_name.upper())
                    st.success(f"Logged {test_name.upper()} successfully!")
                else:
                    st.warning("Enter a name first.")

    with tab2:
        col1, col2 = st.columns([1, 1])
        with col1:
            new_name = st.text_input("Employee Name").upper()
            photo_file = st.camera_input("Take a photo") or st.file_uploader("Upload a photo", type=["jpg", "png", "jpeg"])
            
            if photo_file and new_name:
                if st.button("Register User", type="primary"):
                    with st.spinner("Saving..."):
                        # Save mock data since AI is off
                        st.session_state.known_names.append(new_name)
                        st.session_state.known_encodings.append([0.0] * 128) # Dummy array
                        save_face_data()
                        st.success(f"Successfully registered {new_name}!")
                        st.rerun()

    with tab3:
        paths = get_file_paths(st.session_state.org_key)
        if st.button("Refresh Logs"): load_org_data(st.session_state.org_key)
        
        if os.path.exists(paths["logs"]) and os.path.getsize(paths["logs"]) > 0:
            df = pd.read_csv(paths["logs"])
            st.dataframe(df.iloc[::-1], use_container_width=True)
            st.download_button("Download CSV", df.to_csv(index=False).encode('utf-8'), "logs.csv", "text/csv")
        else:
            st.info("No logs found.")

    with tab4:
        if st.session_state.known_names:
            for idx, name in enumerate(st.session_state.known_names):
                c1, c2 = st.columns([3, 1])
                c1.text(f"👤 {name}")
                if c2.button("Delete", key=f"del_{idx}_{name}"):
                    st.session_state.known_names.pop(idx)
                    st.session_state.known_encodings.pop(idx)
                    save_face_data()
                    st.rerun()
        else:
            st.info("Database empty.")
