import streamlit as st
import pandas as pd
import json
import os
import base64
import time
import hashlib
import uuid
from datetime import datetime

# --- 1. CONFIGURATION & ASSETS ---
DATA_DIR = "agnos_data"
if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
AUTH_FILE = os.path.join(DATA_DIR, "auth_registry.json")

def load_asset(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "assets", filename)
    try:
        with open(file_path, "r") as f: return f.read()
    except FileNotFoundError: return ""

st.set_page_config(page_title="Agnos Enterprise", layout="wide", initial_sidebar_state="expanded")
st.markdown(f"<style>{load_asset('style.css')}</style>", unsafe_allow_html=True)

# --- 2. BACKEND HELPERS ---
def load_auth():
    if os.path.exists(AUTH_FILE):
        with open(AUTH_FILE, "r") as f: return json.load(f)
    return {}

def save_auth(data):
    with open(AUTH_FILE, "w") as f: json.dump(data, f)

def get_file_paths(org_key):
    return {
        "db": os.path.join(DATA_DIR, f"faces_{org_key}.json"),
        "logs": os.path.join(DATA_DIR, f"logs_{org_key}.csv")
    }

# --- 3. SESSION STATE ---
if "auth_status" not in st.session_state: st.session_state.auth_status = False
if "org_key" not in st.session_state: st.session_state.org_key = None
if "db" not in st.session_state: st.session_state.db = {}
if "logged_set" not in st.session_state: st.session_state.logged_set = set()
if "uploader_key" not in st.session_state: st.session_state.uploader_key = 0
if "component_key" not in st.session_state: st.session_state.component_key = str(uuid.uuid4())
if "last_toast" not in st.session_state: st.session_state.last_toast = 0

# --- 4. DATA LOGIC ---
def load_org_data(org_key):
    st.session_state.logged_set = set()
    st.session_state.db = {}
    
    paths = get_file_paths(org_key)
    
    # Load Faces
    if os.path.exists(paths["db"]):
        with open(paths["db"], "r") as f: st.session_state.db = json.load(f)
    
    # Load Logs Cache
    if os.path.exists(paths["logs"]):
        try:
            # Check if file is not empty
            if os.path.getsize(paths["logs"]) > 0:
                df = pd.read_csv(paths["logs"])
                # Cache today's names to prevent duplicate logging
                today = datetime.now().strftime("%Y-%m-%d")
                todays_names = df[df['Date'] == today]['Name'].unique()
                st.session_state.logged_set = set(todays_names)
        except Exception as e:
            print(f"Error loading logs: {e}")

def save_log(name, date_str, time_str):
    if not st.session_state.auth_status: return False
    
    # RAM Check: If already logged today, return False
    if name in st.session_state.logged_set:
        return False

    paths = get_file_paths(st.session_state.org_key)
    csv_path = paths["logs"]
    
    try:
        # Check if we need a header (File doesn't exist OR is empty)
        write_header = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)
        
        with open(csv_path, "a") as f:
            if write_header:
                f.write("Name,Time,Date\n")
            f.write(f"{name},{time_str},{date_str}\n")
            
        st.session_state.logged_set.add(name)
        return True
    except Exception as e:
        print(f"Save Error: {e}")
        return False

# --- 5. COMPONENT BUILDER ---
def get_component_html(img_b64=None):
    if not st.session_state.auth_status: return "<div>Please Login</div>"
    
    html_template = load_asset("script.html")
    db_json = json.dumps(st.session_state.db)
    img_val = f"data:image/jpeg;base64,{img_b64}" if img_b64 else "null"
    run_mode = "IMAGE" if img_b64 else "VIDEO"
    display_video = "none" if img_b64 else "block"
    display_img = "block" if img_b64 else "none"

    return html_template.replace("STATIC_IMG_PLACEHOLDER", img_val) \
                        .replace("RUN_MODE_PLACEHOLDER", run_mode) \
                        .replace("DB_JSON_PLACEHOLDER", db_json) \
                        .replace("{'none' if img_b64 else 'block'}", display_video) \
                        .replace("{'block' if img_b64 else 'none'}", display_img)

# --- 6. LOGIC TRIGGER ---
qp = st.query_params
if "detected_name" in qp:
    det_name = qp["detected_name"]
    c_date = qp.get("c_date", datetime.now().strftime("%Y-%m-%d"))
    c_time = qp.get("c_time", datetime.now().strftime("%H:%M:%S"))

    if st.session_state.auth_status and det_name and det_name != "Unknown":
        
        # 1. Try to Save
        saved_new = save_log(det_name, c_date, c_time)
        
        # 2. Feedback (Always show if present in URL)
        if saved_new:
            st.toast(f"✅ Attendance Marked: {det_name}", icon="⚡")
        else:
            # If already saved, show "Welcome Back"
            st.toast(f"👋 Welcome back: {det_name}", icon="👀")

    # Note: We do NOT clear query_params here. 
    # We let the JS clear them when the user walks away.
    # This keeps the "Toast" visible as long as the user stands there.
            
# --- 7. UI ---
if not st.session_state.auth_status:
    # LOGIN SCREEN
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
                if is_known:
                    if auth_db[key_in] == h_pw:
                        st.session_state.auth_status = True
                        st.session_state.org_key = key_in
                        st.session_state.component_key = str(uuid.uuid4())
                        load_org_data(key_in)
                        st.rerun()
                    else: st.error("Wrong Password")
                else:
                    auth_db[key_in] = h_pw
                    save_auth(auth_db)
                    st.session_state.auth_status = True
                    st.session_state.org_key = key_in
                    st.session_state.component_key = str(uuid.uuid4())
                    load_org_data(key_in)
                    st.rerun()
else:
    # DASHBOARD
    with st.sidebar:
        st.title("🛡️ Agnos")
        st.caption(f"ORG: {st.session_state.org_key}")
        st.metric("Users", len(st.session_state.db))
        st.metric("Logs Today", len(st.session_state.logged_set))
        st.markdown("---")
        
        if st.button("Log Out"):
            for k in list(st.session_state.keys()): del st.session_state[k]
            st.query_params.clear()
            st.markdown("""<meta http-equiv="refresh" content="0"><script>parent.window.location.reload();</script>""", unsafe_allow_html=True)
            st.stop()

        with st.expander("⚠️ Danger Zone"):
            if st.button("DELETE ORG", type="primary"):
                paths = get_file_paths(st.session_state.org_key)
                if os.path.exists(paths['db']): os.remove(paths['db'])
                if os.path.exists(paths['logs']): os.remove(paths['logs'])
                auth_db = load_auth()
                if st.session_state.org_key in auth_db:
                    del auth_db[st.session_state.org_key]
                    save_auth(auth_db)
                for k in list(st.session_state.keys()): del st.session_state[k]
                st.query_params.clear()
                st.markdown("""<meta http-equiv="refresh" content="0"><script>parent.window.location.reload();</script>""", unsafe_allow_html=True)
                st.stop()

    st.title("Agnos Enterprise")
    tab1, tab2, tab3, tab4 = st.tabs(["🎥 Scanner", "👤 Register", "📊 Logs", "🗄️ Database"])

    with tab1:
        c1, c2 = st.columns([2,1])
        with c1: st.components.v1.html(get_component_html(), height=530)
        with c2:
            st.subheader("Live Status")
            if "detected_name" in st.query_params:
                det = st.query_params["detected_name"]
                st.success("**ACCESS GRANTED**")
                st.markdown(f"# {det}")
                if "c_time" in st.query_params:
                    st.caption(f"Logged: {st.query_params['c_time']}")
            else:
                st.info("System Active")
                st.markdown("*Waiting for personnel...*")
    
    with tab2:
        c1, c2 = st.columns([2,1])
        with c1:
            name_in = st.text_input("Name", key=f"n_{st.session_state.uploader_key}").upper()
            file_in = st.file_uploader("Photo", type=['jpg','png'], key=f"u_{st.session_state.uploader_key}")
        with c2:
            if file_in:
                st.image(file_in, width=200)
                b64 = base64.b64encode(file_in.getvalue()).decode()
                st.components.v1.html(get_component_html(b64), height=0, width=0)
                if "face_data" in st.query_params:
                    if st.button("Save User", type="primary"):
                        data = json.loads(base64.b64decode(st.query_params["face_data"]).decode())
                        st.session_state.db[name_in] = data
                        paths = get_file_paths(st.session_state.org_key)
                        with open(paths["db"], "w") as f: json.dump(st.session_state.db, f)
                        st.session_state.uploader_key += 1
                        st.query_params.clear()
                        st.toast(f"Saved {name_in}!")
                        st.rerun()
                else: st.info("Processing...")

    with tab3:
        paths = get_file_paths(st.session_state.org_key)
        c1, c2 = st.columns([1,1])
        if c1.button("Refresh"): st.rerun()
        if c2.button("Clear History"):
            if os.path.exists(paths["logs"]): os.remove(paths["logs"])
            st.session_state.logged_set = set()
            st.rerun()
        
        if os.path.exists(paths["logs"]):
            try:
                # Debug read to ensure we see errors if CSV is malformed
                df = pd.read_csv(paths["logs"])
                if not df.empty:
                    st.dataframe(df, use_container_width=True)
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button("Download CSV", csv, "logs.csv", "text/csv")
                else:
                    st.info("Log file exists but is empty.")
            except Exception as e:
                st.error(f"Error reading logs: {e}")
        else: st.info("No logs found.")

    with tab4:
        if st.session_state.db:
            for name in list(st.session_state.db.keys()):
                c1, c2 = st.columns([3,1])
                c1.text(f"👤 {name}")
                if c2.button("Delete", key=f"d_{name}"):
                    del st.session_state.db[name]
                    paths = get_file_paths(st.session_state.org_key)
                    with open(paths["db"], "w") as f: json.dump(st.session_state.db, f)
                    st.rerun()
        else: st.info("Database empty.")
