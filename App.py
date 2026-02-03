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

st.set_page_config(page_title="Agnos Enterprise", layout="wide", initial_sidebar_state="expanded")

# --- AUTHENTICATION HELPERS ---
def load_auth():
    if os.path.exists(AUTH_FILE):
        with open(AUTH_FILE, "r") as f: return json.load(f)
    return {}

def save_auth(data):
    with open(AUTH_FILE, "w") as f: json.dump(data, f)

def get_file_paths(org_key):
    return {
        "db": os.path.join(DATA_DIR, f"faces_{org_key}.json"),
        "logs": os.path.join(DATA_DIR, f"logs_{org_key}.pkl")
    }

# --- SESSION STATE ---
if "auth_status" not in st.session_state: st.session_state.auth_status = False
if "org_key" not in st.session_state: st.session_state.org_key = None
if "db" not in st.session_state: st.session_state.db = {}
if "logged_set" not in st.session_state: st.session_state.logged_set = set()
if "uploader_key" not in st.session_state: st.session_state.uploader_key = 0
if "component_key" not in st.session_state: st.session_state.component_key = str(uuid.uuid4())

# --- BACKEND LOGIC ---
def load_org_data(org_key):
    paths = get_file_paths(org_key)
    
    if os.path.exists(paths["db"]):
        with open(paths["db"], "r") as f: st.session_state.db = json.load(f)
    else:
        st.session_state.db = {} 
    
    st.session_state.logged_set = set() 
    
    if os.path.exists(paths["logs"]):
        with open(paths["logs"], "rb") as f:
            try:
                logs = pickle.load(f)
                for entry in logs:
                    st.session_state.logged_set.add(entry["Name"])
            except Exception:
                pass 

def save_log(name, date_str, time_str):
    if not st.session_state.auth_status: return False
    
    paths = get_file_paths(st.session_state.org_key)
    logs = []
    if os.path.exists(paths["logs"]):
        with open(paths["logs"], "rb") as f: logs = pickle.load(f)
    
    if not any(e['Name'] == name and e['Date'] == date_str for e in logs):
        entry = {"Name": name, "Time": time_str, "Date": date_str}
        logs.append(entry)
        with open(paths["logs"], "wb") as f: pickle.dump(logs, f)
        st.session_state.logged_set.add(name)
        return True
    return False

# --- AUTO-LOGGING TRIGGER ---
qp = st.query_params
if "detected_name" in qp:
    det_name = qp["detected_name"]
    c_date = qp.get("c_date", datetime.now().strftime("%Y-%m-%d"))
    c_time = qp.get("c_time", datetime.now().strftime("%H:%M:%S"))

    if st.session_state.auth_status and det_name and det_name != "Unknown":
        if save_log(det_name, c_date, c_time):
            st.toast(f"✅ Verified: {det_name}", icon="🔐")

# --- CSS STYLING ---
st.markdown("""
<style>
    body { background:#0e1117; color:#fff; font-family: sans-serif; }
    #view { position: relative; width: 100%; height: 500px; background: #000; border-radius: 12px; overflow: hidden; border: 1px solid #333; }
    canvas { position:absolute; top:0; left:0; z-index:10; }
    video { position:absolute; top:0; left:0; z-index:5; }
    .stDeployButton {display:none;}
    div[data-testid="stForm"] { border: 1px solid #333; padding: 20px; border-radius: 10px; }
    
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        color: #fff;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f2937;
        color: #00FF00;
        border-bottom: 2px solid #00FF00;
    }
</style>
""", unsafe_allow_html=True)

# --- JAVASCRIPT LOGIC ---
JS_CODE = """
<script type="module">
import { FaceLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

const video = document.getElementById("webcam");
const staticImg = document.getElementById("static-img");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const triggerBtn = document.getElementById("trigger-btn"); 

const staticImgSrc = "STATIC_IMG_PLACEHOLDER";
const runMode = "RUN_MODE_PLACEHOLDER";
const registry = JSON.parse('DB_JSON_PLACEHOLDER');

let faceLandmarker;
let lastVideoTime = -1;
let currentMatch = "Unknown";
let matchStartTime = 0;

// --- GEOMETRIC MATH ---
function getDist3D(p1, p2) {
    return Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2) + Math.pow(p1.z - p2.z, 2));
}

// 2. SCALAR GEOMETRY (ROTATION INVARIANT)
// Instead of coordinates (which change when you tilt head),
// We measure the DISTANCE between key points. 
// Distance between nose and chin is the same even if you are upside down.
function getFaceVector(landmarks) {
    const P = (idx) => landmarks[idx]; 

    // 1. Anchor: Inter-Pupillary Distance (distance between eyes)
    // We divide all other distances by this to handle camera zoom.
    const eyeDist = getDist3D(P(468), P(473));

    // 2. Structural Measurements (The "Web")
    const features = [
        [1, 4],     // Nose bridge
        [1, 61],    // Nose Tip -> Mouth Corner L
        [1, 291],   // Nose Tip -> Mouth Corner R
        [61, 291],  // Mouth Width
        [33, 133],  // Eye Width L
        [263, 362], // Eye Width R
        [1, 152],   // Nose -> Chin
        [10, 152],  // Forehead -> Chin (Face Height)
        [234, 454], // Cheek -> Cheek (Face Width)
        [1, 468],   // Nose -> L Iris
        [1, 473],   // Nose -> R Iris
        [152, 234], // Chin -> L Cheek
        [152, 454], // Chin -> R Cheek
        [10, 1]     // Forehead -> Nose
    ];

    // Return Array of Ratios
    return features.map(pair => getDist3D(P(pair[0]), P(pair[1])) / eyeDist);
}

// Simple Manhattan Distance for Scalars
function calculateScore(vecA, vecB) {
    let diff = 0;
    for (let i = 0; i < vecA.length; i++) {
        diff += Math.abs(vecA[i] - vecB[i]);
    }
    return diff;
}

function findMatch(landmarks) {
    const currentVec = getFaceVector(landmarks);
    let bestMatch = { name: "Unknown", score: 1000.0 }; // Lower score is better

    for (const [name, savedLandmarks] of Object.entries(registry)) {
        const savedVec = getFaceVector(savedLandmarks);
        const score = calculateScore(currentVec, savedVec);
        if (score < bestMatch.score) bestMatch = { name: name, score: score };
    }
    
    // Threshold: Sum of ratio differences.
    // 0.8 is usually a good tight threshold for this many features.
    if (bestMatch.score < 0.65) return bestMatch.name;
    return "Unknown";
}

async function init() {
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm");
    faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            delegate: "GPU"
        },
        runningMode: runMode,
        numFaces: 1
    });

    if (staticImgSrc !== "null") {
        staticImg.src = staticImgSrc;
        staticImg.onload = async () => {
            const results = await faceLandmarker.detect(staticImg);
            if (results.faceLandmarks.length > 0) {
                const dataStr = btoa(JSON.stringify(results.faceLandmarks[0]));
                const url = new URL(window.parent.location.href);
                url.searchParams.set("face_data", dataStr);
                window.parent.history.replaceState({}, "", url);
                if(triggerBtn) triggerBtn.click(); 
            }
        }
    } else {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        video.addEventListener("loadeddata", predictVideo);
    }
}

async function predictVideo() {
    const now = performance.now();
    
    if (lastVideoTime !== video.currentTime) {
        lastVideoTime = video.currentTime;
        const results = faceLandmarker.detectForVideo(video, now);
        
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (results.faceLandmarks.length > 0) {
            const landmarks = results.faceLandmarks[0];
            const name = findMatch(landmarks);

            // --- STABILITY ---
            if (name !== currentMatch) {
                currentMatch = name;
                matchStartTime = now;
            }

            const timeElapsed = now - matchStartTime;
            const isUnknown = (currentMatch === "Unknown");
            const isVerified = (!isUnknown && timeElapsed > 1000); 

            if (isVerified) {
                try {
                    const url = new URL(window.parent.location.href);
                    if (url.searchParams.get("detected_name") !== currentMatch) {
                        url.searchParams.set("detected_name", currentMatch);
                        
                        // Client Time
                        const d = new Date();
                        const dateStr = d.getFullYear() + "-" + (d.getMonth()+1).toString().padStart(2, '0') + "-" + d.getDate().toString().padStart(2, '0');
                        const timeStr = d.getHours().toString().padStart(2, '0') + ":" + d.getMinutes().toString().padStart(2, '0') + ":" + d.getSeconds().toString().padStart(2, '0');
                        
                        url.searchParams.set("c_date", dateStr);
                        url.searchParams.set("c_time", timeStr);
                        url.searchParams.set("ts", Date.now()); 
                        
                        window.parent.history.replaceState({}, "", url);
                        if(triggerBtn) triggerBtn.click();
                    }
                } catch(e) {}
            }

            // --- DRAWING ---
            const xs = landmarks.map(p => p.x * canvas.width);
            const ys = landmarks.map(p => p.y * canvas.height);
            const x = Math.min(...xs), y = Math.min(...ys), w = Math.max(...xs)-x, h = Math.max(...ys)-y;
            
            let color = "#FF0000"; 
            let label = "UNKNOWN";

            if (!isUnknown) {
                if (isVerified) {
                    color = "#00FF00"; 
                    label = currentMatch; 
                } else {
                    color = "#FFFF00"; 
                    label = "VERIFYING..."; 
                }
            }

            ctx.strokeStyle = color; ctx.lineWidth = 4; ctx.strokeRect(x, y, w, h);
            ctx.fillStyle = color; ctx.fillRect(x, y - 30, w, 30);
            ctx.fillStyle = "#000"; ctx.font = "bold 16px sans-serif"; ctx.fillText(label, x + 5, y - 8);

            // Show "Web" for debug
            ctx.strokeStyle = "rgba(0, 255, 255, 0.2)";
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(landmarks[468].x*canvas.width, landmarks[468].y*canvas.height);
            ctx.lineTo(landmarks[473].x*canvas.width, landmarks[473].y*canvas.height); // Eye line
            ctx.stroke();
        }
    }
    window.requestAnimationFrame(predictVideo);
}

init();
</script>
"""

def get_component_html(img_b64=None):
    if not st.session_state.auth_status: return "<div>Please Login</div>"
    
    db_json = json.dumps(st.session_state.db)
    img_val = f"data:image/jpeg;base64,{img_b64}" if img_b64 else "null"
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ margin:0; background:black; overflow:hidden; }}
            #view {{ position:relative; width:100%; height:100vh; }}
            video, canvas, img {{ position:absolute; top:0; left:0; width:100%; height:100%; object-fit:contain; }}
        </style>
    </head>
    <body>
        <div id="view">
            <video id="webcam" autoplay muted playsinline style="display: {'none' if img_b64 else 'block'}"></video>
            <img id="static-img" style="display: {'block' if img_b64 else 'none'}">
            <canvas id="overlay"></canvas>
            <button id="trigger-btn" style="display:none;" onclick="window.parent.postMessage({{type: 'streamlit:setComponentValue', value: 'update'}}, '*')"></button>
        </div>
        {JS_CODE.replace("STATIC_IMG_PLACEHOLDER", img_val)
                .replace("RUN_MODE_PLACEHOLDER", "IMAGE" if img_b64 else "VIDEO")
                .replace("DB_JSON_PLACEHOLDER", db_json)}
    </body>
    </html>
    """
    return html

# --- LOGIN SCREEN ---
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
            
            if is_known:
                st.info(f"✅ Organization found: **{key_in}**")
            else:
                st.warning(f"🆕 Organization **{key_in}** is available!")

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
                            else:
                                st.error("❌ Incorrect Password")
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
    # --- LOGGED IN UI ---
    
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
            st.warning("This action cannot be undone.")
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
            st.components.v1.html(get_component_html(), height=500)
        with c_stat:
            st.subheader("Live Feed")
            if "detected_name" in st.query_params:
                det = st.query_params["detected_name"]
                ts = float(st.query_params.get("ts", 0))
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
            with open(paths["logs"],"rb") as f: df = pd.DataFrame(pickle.load(f))
            if not df.empty:
                st.dataframe(df, use_container_width=True)
                st.download_button("Download CSV", df.to_csv(index=False), f"logs_{st.session_state.org_key}.csv", "text/csv")
            else: st.info("No logs found.")
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
        else:
            st.info("Database is empty.")
