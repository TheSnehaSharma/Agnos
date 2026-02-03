import streamlit as st
import pandas as pd
import json
import os
import base64
import pickle
import time
import hashlib
import uuid
from datetime import datetime

# --- CONFIGURATION ---
DATA_DIR = "agnos_data"
if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)

AUTH_FILE = os.path.join(DATA_DIR, "auth_registry.json")

# Helper to load external assets
def load_asset(path):
    with open(path, "r") as f: return f.read()

st.set_page_config(page_title="Agnos Enterprise", layout="wide", initial_sidebar_state="expanded")

# Inject CSS
st.markdown(f"<style>{load_asset('assets/style.css')}</style>", unsafe_allow_html=True)

# --- BACKEND HELPERS ---
def load_auth():
    if os.path.exists(AUTH_FILE):
        with open(AUTH_FILE, "r") as f: return json.load(f)
    return {}

def save_auth(data):
    with open(AUTH_FILE, "w") as f: json.dump(data, f)

def get_file_paths(org_key):
    return {
        "db": os.path.join(DATA_DIR, f"faces_{org_key}.json"),
        "logs": os.path.join(DATA_DIR, f"logs_{org_key}.csv") # Using CSV for speed
    }

# --- SESSION STATE INITIALIZATION ---
if "auth_status" not in st.session_state: st.session_state.auth_status = False
if "org_key" not in st.session_state: st.session_state.org_key = None
if "db" not in st.session_state: st.session_state.db = {}
if "logged_set" not in st.session_state: st.session_state.logged_set = set()
if "uploader_key" not in st.session_state: st.session_state.uploader_key = 0
if "component_key" not in st.session_state: st.session_state.component_key = str(uuid.uuid4())

# --- CORE LOGIC: PREVENT LEAKAGE ---
def load_org_data(org_key):
    # 1. CRITICAL: Wipe old logs from memory immediately
    st.session_state.logged_set = set()
    st.session_state.db = {}
    
    paths = get_file_paths(org_key)
    
    # 2. Load Faces
    if os.path.exists(paths["db"]):
        with open(paths["db"], "r") as f: st.session_state.db = json.load(f)
    
    # 3. Load Logs (Only for this specific Org)
    if os.path.exists(paths["logs"]):
        try:
            # Only read the required columns to speed up
            df = pd.read_csv(paths["logs"], usecols=["Name", "Date"])
            # Filter logic: We only care about caching today's attendance
            # Note: We use server time for initial cache, but client time for saving.
            # Ideally, this is just a quick cache.
            today = datetime.now().strftime("%Y-%m-%d") 
            todays_names = df[df['Date'] == today]['Name'].unique()
            st.session_state.logged_set = set(todays_names)
        except Exception:
            pass # File empty or new

def save_log(name, date_str, time_str):
    if not st.session_state.auth_status: return False
    
    # In-Memory Check (Fastest)
    if name in st.session_state.logged_set:
        return False # Already logged

    paths = get_file_paths(st.session_state.org_key)
    csv_path = paths["logs"]
    
    try:
        # Atomic Append (Fast CSV)
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a") as f:
            if write_header: f.write("Name,Time,Date\n")
            f.write(f"{name},{time_str},{date_str}\n")
            
        st.session_state.logged_set.add(name)
        return True
    except: return False

# --- COMPONENT BUILDER ---
def get_component_html(img_b64=None):
    if not st.session_state.auth_status: return "<div>Please Login</div>"
    try:
        with open("assets/script.html", "r") as f:
            html_template = f.read()
    except FileNotFoundError:
        return "<div>Error: assets/script.html not found.</div>"
    
    html_template = load_asset("assets/script.html")
    
    # Inject Python Data
    db_json = json.dumps(st.session_state.db)
    img_val = f"data:image/jpeg;base64,{img_b64}" if img_b64 else "null"
    run_mode = "IMAGE" if img_b64 else "VIDEO"
    display_video = "none" if img_b64 else "block"
    display_img = "block" if img_b64 else "none"

    # Replace placeholders
    html = html_template.replace("STATIC_IMG_PLACEHOLDER", img_val) \
                        .replace("RUN_MODE_PLACEHOLDER", run_mode) \
                        .replace("DB_JSON_PLACEHOLDER", db_json) \
                        .replace("{'none' if img_b64 else 'block'}", display_video) \
                        .replace("{'block' if img_b64 else 'none'}", display_img)
    return html

# --- AUTO-LOGGING CHECK ---
qp = st.query_params
if "detected_name" in qp:
    det_name = qp["detected_name"]
    c_date = qp.get("c_date", datetime.now().strftime("%Y-%m-%d"))
    c_time = qp.get("c_time", datetime.now().strftime("%H:%M:%S"))

    if st.session_state.auth_status and det_name and det_name != "Unknown":
        if save_log(det_name, c_date, c_time):
            st.toast(f"✅ Verified: {det_name}", icon="🔐")

# --- UI LOGIC ---

# 1. LOGIN SCREEN
if not st.session_state.auth_status:
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.title("🔐 Agnos Login")
        st.markdown("Enter your Organization Key to proceed.")
        
        key_in = st.text_input("Organization Key (5 Char)", max_chars=5, placeholder="e.g. ALPHA").upper()
        
        auth_db = load_auth()
        is_known = False
        if len(key_in) == 5:
            is_known = key_in in auth_db
            if is_known: st.info(f"✅ Organization found: **{key_in}**")
            else: st.warning(f"🆕 Organization **{key_in}** is available!")

            with st.form("auth_form"):
                btn_label = "Sign In" if is_known else "Sign Up & Create"
                pass_label = "Enter Password" if is_known else "Create New Password"
                pass_in = st.text_input(pass_label, type="password")
                submitted = st.form_submit_button(btn_label, type="primary")
                
                if submitted:
                    if len(pass_in) < 1:
                        st.error("Password cannot be empty.")
                    else:
                        hashed_pw = hashlib.sha256(pass_in.encode()).hexdigest()
                        if is_known:
                            if auth_db[key_in] == hashed_pw:
                                st.session_state.auth_status = True
                                st.session_state.org_key = key_in
                                st.session_state.component_key = str(uuid.uuid4())
                                load_org_data(key_in)
                                st.rerun()
                            else: st.error("❌ Incorrect Password")
                        else:
                            auth_db[key_in] = hashed_pw
                            save_auth(auth_db)
                            st.session_state.auth_status = True
                            st.session_state.org_key = key_in
                            st.session_state.component_key = str(uuid.uuid4())
                            load_org_data(key_in) 
                            st.success("Organization Created!")
                            st.rerun()

else:
    # 2. LOGGED IN DASHBOARD
    with st.sidebar:
        st.title("🛡️ Agnos")
        st.caption(f"Org: {st.session_state.org_key}")
        st.metric("Users", len(st.session_state.db))
        st.metric("Logs Today", len(st.session_state.logged_set))
        st.markdown("---")
        
        if st.button("Log Out"):
            for key in list(st.session_state.keys()): del st.session_state[key]
            st.query_params.clear()
            st.markdown("""<meta http-equiv="refresh" content="0"><script>parent.window.location.reload();</script>""", unsafe_allow_html=True)
            st.stop()
            
        with st.expander("⚠️ Danger Zone"):
            st.warning("Action is irreversible.")
            if st.button("DELETE ORGANIZATION", type="primary"):
                paths = get_file_paths(st.session_state.org_key)
                if os.path.exists(paths['db']): os.remove(paths['db'])
                if os.path.exists(paths['logs']): os.remove(paths['logs'])
                auth_db = load_auth()
                if st.session_state.org_key in auth_db:
                    del auth_db[st.session_state.org_key]
                    save_auth(auth_db)
                for key in list(st.session_state.keys()): del st.session_state[key]
                st.query_params.clear()
                st.markdown("""<meta http-equiv="refresh" content="0"><script>parent.window.location.reload();</script>""", unsafe_allow_html=True)
                st.stop()

    st.title("Agnos Enterprise Biometrics")
    
    tab_scan, tab_reg, tab_logs, tab_db = st.tabs(["🎥 Scanner", "👤 Register", "📊 Logs", "🗄️ Database"])

    with tab_scan:
        c_vid, c_stat = st.columns([2, 1])
        with c_vid:
            # Pass component_key to force rebuild on logout/login
            st.components.v1.html(get_component_html(), height=530)
        with c_stat:
            st.subheader("Live Feed")
            if "detected_name" in st.query_params:
                det = st.query_params["detected_name"]
                ts = float(st.query_params.get("ts", 0))
                # 3s Timeout to clear status
                if time.time() * 1000 - ts > 3000:
                    st.query_params.clear()
                    st.rerun()
                else:
                    st.success(f"**ACCESS GRANTED**")
                    st.markdown(f"<h1 style='font-size:3em;'>{det}</h1>", unsafe_allow_html=True)
                    if "c_time" in st.query_params:
                        st.caption(f"Logged at {st.query_params['c_time']}")
            else:
                st.info("System Active")
                st.markdown("*Waiting for personnel...*")

    with tab_reg:
        st.subheader("Register New Personnel")
        c1, c2 = st.columns([2, 1])
        with c1:
            name_in = st.text_input("Full Name", key=f"n_{st.session_state.uploader_key}").upper()
            file_in = st.file_uploader("Profile Photo", type=['jpg','png'], key=f"u_{st.session_state.uploader_key}")
        with c2:
            if file_in:
                st.image(file_in, width=200, caption="Preview")
                b64 = base64.b64encode(file_in.getvalue()).decode()
                st.components.v1.html(get_component_html(b64), height=0, width=0)
                
                if "face_data" in st.query_params:
                    if st.button("✅ Save to Org Database", type="primary", use_container_width=True):
                        data = json.loads(base64.b64decode(st.query_params["face_data"]).decode())
                        st.session_state.db[name_in] = data
                        
                        paths = get_file_paths(st.session_state.org_key)
                        with open(paths["db"], "w") as f: json.dump(st.session_state.db, f)
                        
                        st.session_state.uploader_key += 1
                        st.query_params.clear()
                        st.toast(f"Registered {name_in}!")
                        st.rerun()
                else: st.info("Analyzing biometrics...")

    with tab_logs:
        st.subheader(f"Logs for Org: {st.session_state.org_key}")
        paths = get_file_paths(st.session_state.org_key)
        
        c_ref, c_clr = st.columns([1, 1])
        with c_ref:
            if st.button("Refresh Logs"): st.rerun()
        with c_clr:
            if st.button("Clear History"):
                if os.path.exists(paths["logs"]): os.remove(paths["logs"])
                st.session_state.logged_set = set()
                st.rerun()
        
        if os.path.exists(paths["logs"]):
            try:
                df = pd.read_csv(paths["logs"])
                st.dataframe(df, use_container_width=True)
                st.download_button("Download CSV", df.to_csv(index=False), f"logs_{st.session_state.org_key}.csv", "text/csv")
            except: st.info("Log file is empty.")
        else: st.info("No logs found.")
        
    with tab_db:
        st.subheader("Manage Database")
        st.warning("Deleting a user here removes them permanently.")
        if st.session_state.db:
            for name in list(st.session_state.db.keys()):
                c1, c2 = st.columns([3,1])
                c1.text(f"👤 {name}")
                if c2.button("Delete", key=f"del_{name}"):
                    del st.session_state.db[name]
                    paths = get_file_paths(st.session_state.org_key)
                    with open(paths["db"], "w") as f: json.dump(st.session_state.db, f)
                    st.success(f"Deleted {name}")
                    st.rerun()
        else: st.info("Database is empty.")
