import streamlit as st
import pandas as pd
import json
import os
import base64
import pickle
from datetime import datetime

# --- CONFIG & STORAGE ---
DB_FILE = "registered_faces.json"
PKL_LOG = "attendance_data.pkl"

st.set_page_config(page_title="Agnos", layout="wide", initial_sidebar_state="expanded")

# --- SESSION STATE INITIALIZATION ---
if "db" not in st.session_state:
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f: st.session_state.db = json.load(f)
    else: st.session_state.db = {}

if "logged_set" not in st.session_state: st.session_state.logged_set = set()
if "uploader_key" not in st.session_state: st.session_state.uploader_key = 0

# --- BACKEND LOGIC (Moved to Top for Reliability) ---
def save_attendance_pkl(name):
    logs = []
    if os.path.exists(PKL_LOG):
        with open(PKL_LOG,"rb") as f: logs = pickle.load(f)
    
    # Check if already logged TODAY
    today = datetime.now().strftime("%Y-%m-%d")
    already_logged = any(entry['Name'] == name and entry['Date'] == today for entry in logs)
    
    if not already_logged:
        entry = {"Name":name, "Time":datetime.now().strftime("%H:%M:%S"), "Date":today}
        logs.append(entry)
        with open(PKL_LOG,"wb") as f: pickle.dump(logs, f)
        return True
    return False

# Check query params immediately upon rerun
qp = st.query_params
if "detected" in qp:
    det_name = qp["detected"]
    if det_name != "Unknown":
        if det_name not in st.session_state.logged_set:
            saved = save_attendance_pkl(det_name)
            if saved:
                st.toast(f"✅ Attendance Logged: {det_name}")
            st.session_state.logged_set.add(det_name)
    # Don't clear param immediately so UI can show the status, 
    # but the logic has already fired.

# --- CSS ---
CSS_CODE = """
<style>
body { margin:0; background:#0e1117; color:#fff; font-family: sans-serif; overflow:hidden; }
#view { position:relative; width:100%; height:450px; border-radius:12px; overflow:hidden; background:#000; border:1px solid #333; }
video, canvas, img { position:absolute; top:0; left:0; width:100%; height:100%; object-fit:cover; }
#status-bar { position:absolute; top:0; left:0; right:0; background:rgba(0,0,0,0.8); padding:8px; font-size:11px; z-index:100; color:#fff; display:flex; justify-content:space-between;}
</style>
"""

# --- JS CODE: Cosine Similarity (Device Agnostic) ---
JS_CODE = """
<script type="module">
import { FaceLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

const video = document.getElementById("webcam");
const staticImg = document.getElementById("static-img");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");

const staticImgSrc = "STATIC_IMG_PLACEHOLDER";
const runMode = "RUN_MODE_PLACEHOLDER";
const registry = JSON.parse('DB_JSON_PLACEHOLDER');

let faceLandmarker;
let currentMatchName = "Unknown";
let matchStartTime = 0;
let isConfirmed = false;

// 1. Calculate Vector Magnitude
function magnitude(vec) {
    let sum = 0;
    for (let val of vec) sum += val * val;
    return Math.sqrt(sum);
}

// 2. Calculate Dot Product
function dotProduct(vecA, vecB) {
    let product = 0;
    for (let i = 0; i < vecA.length; i++) product += vecA[i] * vecB[i];
    return product;
}

// 3. Cosine Similarity (The Fix for Different Cameras)
// Returns 1.0 (Perfect Match) to -1.0 (Opposite)
function cosineSimilarity(vecA, vecB) {
    return dotProduct(vecA, vecB) / (magnitude(vecA) * magnitude(vecB));
}

function getDist3D(p1, p2) {
    return Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2) + Math.pow(p1.z - p2.z, 2));
}

// Generate Geometric Feature Vector
function getFaceVector(landmarks) {
    const L_EYE = landmarks[468]; const R_EYE = landmarks[473]; 
    const NOSE  = landmarks[4];   const CHIN  = landmarks[152]; 
    const L_EAR = landmarks[234]; const R_EAR = landmarks[454]; 
    const M_LEFT = landmarks[61]; const M_RIGHT = landmarks[291];

    const eyeDist = getDist3D(L_EYE, R_EYE);

    // We use ratios to be scale-invariant
    // We add more features to make it robust
    return [
        getDist3D(NOSE, L_EYE) / eyeDist, 
        getDist3D(NOSE, R_EYE) / eyeDist,
        getDist3D(NOSE, CHIN)  / eyeDist, 
        getDist3D(L_EAR, R_EAR)/ eyeDist,
        getDist3D(L_EYE, CHIN) / eyeDist, 
        getDist3D(R_EYE, CHIN) / eyeDist,
        getDist3D(M_LEFT, M_RIGHT) / eyeDist // Mouth width
    ];
}

function findMatch(currentLandmarks) {
    const currentVec = getFaceVector(currentLandmarks);
    let bestMatch = { name: "Unknown", score: -1.0 }; // Higher score is better now

    for (const [name, savedLandmarks] of Object.entries(registry)) {
        const savedVec = getFaceVector(savedLandmarks);
        const sim = cosineSimilarity(currentVec, savedVec);
        
        if (sim > bestMatch.score) {
            bestMatch = { name: name, score: sim };
        }
    }

    // Threshold: 0.995 is usually good for geometric cosine sim
    // If it's too strict, lower to 0.99 or 0.985
    if (bestMatch.score > 0.992) {
        return { name: bestMatch.name }; 
    } else {
        return { name: "Unknown" };
    }
}

async function init() {
    try {
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
                if (results.faceLandmarks && results.faceLandmarks.length > 0) {
                    const dataString = btoa(JSON.stringify(results.faceLandmarks[0]));
                    const url = new URL(window.parent.location.href);
                    url.searchParams.set("face_data", dataString);
                    window.parent.history.replaceState({}, "", url);
                }
            }
        } else {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = stream;
            video.onloadeddata = () => predictVideo();
        }
    } catch (err) { console.error(err); }
}

async function predictVideo() {
    const now = performance.now();
    const results = faceLandmarker.detectForVideo(video, now);
    canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (results.faceLandmarks && results.faceLandmarks.length > 0) {
        const landmarks = results.faceLandmarks[0];
        const match = findMatch(landmarks);

        // --- TIME PERSISTENCE (1 Second) ---
        if (match.name !== currentMatchName) {
            currentMatchName = match.name;
            matchStartTime = now;
            isConfirmed = false;
        }

        if (currentMatchName !== "Unknown") {
            if ((now - matchStartTime) > 1000) isConfirmed = true;
        } else {
            isConfirmed = false;
        }

        // --- DRAWING ---
        const xs = landmarks.map(p => p.x * canvas.width);
        const ys = landmarks.map(p => p.y * canvas.height);
        const x = Math.min(...xs), y = Math.min(...ys), w = Math.max(...xs) - x, h = Math.max(...ys) - y;

        let boxColor = "#FF4B4B"; 
        let displayText = "Unknown";
        
        if (currentMatchName !== "Unknown") {
            if (isConfirmed) {
                boxColor = "#00FF00"; 
                displayText = currentMatchName;
            } else {
                boxColor = "#FFA500"; 
                const timeLeft = Math.ceil((1000 - (now - matchStartTime))/100);
                displayText = `Verifying... ${timeLeft}`;
            }
        }

        ctx.strokeStyle = boxColor; ctx.lineWidth = 4; ctx.strokeRect(x, y, w, h);
        ctx.fillStyle = boxColor; ctx.fillRect(x, y - 30, w, 30);
        ctx.fillStyle = "black"; ctx.font = "bold 16px sans-serif";
        ctx.fillText(displayText, x + 5, y - 10);

        // --- LOGGING ---
        if (isConfirmed && currentMatchName !== "Unknown") {
            const url = new URL(window.parent.location.href);
            if (url.searchParams.get("detected") !== currentMatchName) {
                url.searchParams.set("detected", currentMatchName);
                window.parent.location.href = url.toString(); // Force Reload
            }
        }
    } else {
        currentMatchName = "Unknown"; isConfirmed = false;
    }
    window.requestAnimationFrame(predictVideo);
}

init();
</script>
"""

def get_component_html(img_b64=None):
    db_json = json.dumps(st.session_state.db)
    img_val = f"data:image/jpeg;base64,{img_b64}" if img_b64 else "null"
    html = f"<!DOCTYPE html><html><head>{CSS_CODE}</head><body>"
    html += f'<div id="view"><div id="status-bar"><span>AGNOS SYSTEM</span><span>LIVE</span></div>'
    html += f'<video id="webcam" autoplay muted playsinline style="display: {"none" if img_b64 else "block"}"></video>'
    html += f'<img id="static-img" style="display: {"block" if img_b64 else "none"}">'
    html += f'<canvas id="overlay"></canvas></div>{JS_CODE}</body></html>'
    return html.replace("STATIC_IMG_PLACEHOLDER", img_val)\
               .replace("RUN_MODE_PLACEHOLDER", "IMAGE" if img_b64 else "VIDEO")\
               .replace("DB_JSON_PLACEHOLDER", db_json)

# --- UI LAYOUT ---

# 1. SIDEBAR
with st.sidebar:
    st.title("🛡️ Agnos Admin")
    st.markdown("---")
    
    total_logs = 0
    if os.path.exists(PKL_LOG):
        with open(PKL_LOG,"rb") as f: total_logs = len(pickle.load(f))
            
    st.markdown("### 📈 Live Stats")
    c1, c2 = st.columns(2)
    with c1: st.metric("Registered", len(st.session_state.db))
    with c2: st.metric("Present", len(st.session_state.logged_set))
    st.metric("Total Logs", total_logs)
    st.markdown("---")
    
    if st.button("🔄 Refresh System", use_container_width=True):
        st.rerun()

    if len(st.session_state.db) > 0:
        with st.expander("Manage Users"):
            for reg_name in list(st.session_state.db.keys()):
                c_name, c_del = st.columns([3,1])
                c_name.text(reg_name)
                if c_del.button("❌", key=f"del_{reg_name}"):
                    del st.session_state.db[reg_name]
                    with open(DB_FILE,"w") as f: json.dump(st.session_state.db,f,indent=4)
                    st.rerun()

# 2. MAIN TABS
st.title("Agnos Biometric System")
tab_live, tab_reg, tab_log = st.tabs(["🎥 Live Scanner", "👤 New Registration", "📊 Attendance Logs"])

# --- TAB 1: LIVE SCANNER ---
with tab_live:
    col_v, col_m = st.columns([3,1])
    det_name = st.query_params.get("detected", "Unknown")

    with col_v:
        st.components.v1.html(get_component_html(), height=460)

    with col_m:
        st.markdown("### Status")
        if det_name and det_name != "Unknown":
            st.success(f"**Match Confirmed**")
            st.markdown(f"### {det_name}") 
            st.caption(f"Logged at: {datetime.now().strftime('%H:%M:%S')}")
            # Visual checkmark 
        else:
            st.info("Scanning...")
            st.caption("Looking for faces...")

# --- TAB 2: REGISTRATION ---
with tab_reg:
    st.subheader("Enroll New Personnel")
    c_input, c_preview = st.columns([2, 1])
    
    with c_input:
        # Dynamic keys ensure inputs are cleared on reset
        name = st.text_input("Full Name", key=f"name_{st.session_state.uploader_key}").upper()
        uploaded = st.file_uploader("Upload Profile Image", type=['jpg','jpeg','png'], key=f"upl_{st.session_state.uploader_key}")
        st.info("Upload a photo to begin extraction.")
    
    with c_preview:
        if uploaded:
            st.image(uploaded, caption="Preview", use_container_width=True)
            b64 = base64.b64encode(uploaded.getvalue()).decode()
            st.components.v1.html(get_component_html(b64), height=0, width=0)
            
            url_data = st.query_params.get("face_data")
            if url_data and name:
                if st.button("💾 Save to Database", type="primary", use_container_width=True):
                    # Save Data
                    st.session_state.db[name] = json.loads(base64.b64decode(url_data).decode())
                    with open(DB_FILE,"w") as f: json.dump(st.session_state.db,f,indent=4)
                    
                    # Increment Key to Force Clear Inputs
                    st.session_state.uploader_key += 1
                    
                    # Clear URL params
                    st.query_params.clear()
                    
                    st.toast(f"Registered {name}!")
                    st.rerun()
            elif not url_data:
                st.caption("⏳ Extracting features...")
        else:
            st.markdown("""<div style="height:150px; border:1px dashed #444; border-radius:10px; display:flex; align-items:center; justify-content:center; color:#666;">No Image</div>""", unsafe_allow_html=True)

# --- TAB 3: LOGS ---
with tab_log:
    st.subheader("Access History")
    if os.path.exists(PKL_LOG):
        with open(PKL_LOG,"rb") as f: logs = pickle.load(f)
        df = pd.DataFrame(logs)
        if not df.empty:
            st.dataframe(df, use_container_width=True)
            c1, c2 = st.columns(2)
            with c1:
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download CSV", csv, "attendance.csv", "text/csv", use_container_width=True)
            with c2:
                if st.button("🗑️ Clear All Logs", use_container_width=True):
                    os.remove(PKL_LOG)
                    st.session_state.logged_set = set()
                    st.rerun()
        else: st.info("Log file is empty.")
    else: st.info("No logs found yet.")
