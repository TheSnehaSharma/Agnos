import streamlit as st
import pandas as pd
import json
import os
import base64
import pickle
from datetime import datetime

# --- CONFIGURATION ---
DB_FILE = "registered_faces.json"
PKL_LOG = "attendance_data.pkl"

st.set_page_config(page_title="Agnos Pro", layout="wide", initial_sidebar_state="expanded")

# --- SESSION STATE ---
if "db" not in st.session_state:
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f: st.session_state.db = json.load(f)
    else: st.session_state.db = {}

if "logged_set" not in st.session_state: st.session_state.logged_set = set()
if "uploader_key" not in st.session_state: st.session_state.uploader_key = 0

# --- BACKEND LOGIC ---
def save_log(name):
    logs = []
    if os.path.exists(PKL_LOG):
        with open(PKL_LOG,"rb") as f: logs = pickle.load(f)
    
    today = datetime.now().strftime("%Y-%m-%d")
    if not any(e['Name'] == name and e['Date'] == today for e in logs):
        entry = {"Name": name, "Time": datetime.now().strftime("%H:%M:%S"), "Date": today}
        logs.append(entry)
        with open(PKL_LOG,"wb") as f: pickle.dump(logs, f)
        return True
    return False

# Auto-Attendance Check
if "detected_name" in st.query_params:
    det_name = st.query_params["detected_name"]
    if det_name and det_name != "Unknown":
        if det_name not in st.session_state.logged_set:
            if save_log(det_name):
                st.toast(f"✅ Attendance Marked: {det_name}")
            st.session_state.logged_set.add(det_name)

# --- CSS STYLING ---
st.markdown("""
<style>
    body { background:#0e1117; color:#fff; font-family: sans-serif; }
    #view { 
        position: relative; 
        width: 100%; 
        height: 480px; 
        background: #000; 
        border-radius: 12px; 
        overflow: hidden; 
        border: 1px solid #333; 
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    }
    .status-overlay {
        position: absolute; top: 20px; right: 20px;
        background: rgba(0,0,0,0.7); padding: 10px 20px;
        border-radius: 8px; border-left: 4px solid #333;
        color: #fff; backdrop-filter: blur(4px);
    }
    .status-success { border-color: #00FF00; box-shadow: 0 0 15px rgba(0,255,0,0.2); }
    .status-scan { border-color: #0088FF; }
    .stDeployButton {display:none;}
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
let isLocked = false;

// --- MATH: Vector Geometry ---

function magnitude(vec) {
    let sum = 0;
    for (let val of vec) sum += val * val;
    return Math.sqrt(sum);
}

function dotProduct(vecA, vecB) {
    let product = 0;
    for (let i = 0; i < vecA.length; i++) product += vecA[i] * vecB[i];
    return product;
}

function cosineSimilarity(vecA, vecB) {
    return dotProduct(vecA, vecB) / (magnitude(vecA) * magnitude(vecB));
}

// THE "FINGERPRINT" GENERATOR
// We select 42 rigid points that define the skull structure
// This is device-agnostic because it uses internal ratios
function getFaceVector(landmarks) {
    // 1. Center Centroid
    let cx = 0, cy = 0, cz = 0;
    for(let p of landmarks) { cx+=p.x; cy+=p.y; cz+=p.z; }
    cx/=landmarks.length; cy/=landmarks.length; cz/=landmarks.length;

    // 2. Select Rigid Points (Brows, Nose Bridge, Eye Corners)
    // We avoid Mouth and Jaw points as they move too much
    const indices = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
    ];

    const vec = [];
    for(let i of indices) {
        // Create a vector from Centroid to each point
        // This captures the "Shape" of the face relative to its center
        vec.push(landmarks[i].x - cx);
        vec.push(landmarks[i].y - cy);
        vec.push(landmarks[i].z - cz);
    }
    return vec;
}

function findMatch(landmarks) {
    const currentVec = getFaceVector(landmarks);
    let bestMatch = { name: "Unknown", score: -1.0 };

    for (const [name, savedLandmarks] of Object.entries(registry)) {
        const savedVec = getFaceVector(savedLandmarks);
        const sim = cosineSimilarity(currentVec, savedVec);
        
        if (sim > bestMatch.score) {
            bestMatch = { name: name, score: sim };
        }
    }
    
    // Threshold: 0.99 is very strict (Identity)
    // 0.985 allows slight rotation
    if (bestMatch.score > 0.985) return bestMatch.name;
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
    let startTimeMs = performance.now();
    
    if (lastVideoTime !== video.currentTime) {
        lastVideoTime = video.currentTime;
        const results = faceLandmarker.detectForVideo(video, startTimeMs);
        
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (results.faceLandmarks.length > 0) {
            const landmarks = results.faceLandmarks[0];
            
            // DRAW MESH (Visual Feedback)
            ctx.lineWidth = 1;
            ctx.strokeStyle = "rgba(0, 255, 255, 0.3)";
            for(let i=0; i<landmarks.length; i+=10) { // Draw sparse mesh
                 const x = landmarks[i].x * canvas.width;
                 const y = landmarks[i].y * canvas.height;
                 ctx.beginPath(); ctx.arc(x,y,1,0,2*Math.PI); ctx.stroke();
            }

            const name = findMatch(landmarks);

            // --- STABILITY & LOGGING ---
            if (name !== currentMatch) {
                currentMatch = name;
                matchStartTime = now;
                isLocked = false;
            }

            if (currentMatch !== "Unknown" && (now - matchStartTime > 500) && !isLocked) {
                isLocked = true;
                const url = new URL(window.parent.location.href);
                url.searchParams.set("detected_name", currentMatch);
                window.parent.history.replaceState({}, "", url);
                if(triggerBtn) triggerBtn.click();
            }

            // --- BOUNDING BOX ---
            const xs = landmarks.map(p => p.x * canvas.width);
            const ys = landmarks.map(p => p.y * canvas.height);
            const x = Math.min(...xs), y = Math.min(...ys), w = Math.max(...xs)-x, h = Math.max(...ys)-y;
            
            const color = currentMatch === "Unknown" ? "#FF0000" : "#00FF00";
            
            ctx.strokeStyle = color;
            ctx.lineWidth = 3;
            ctx.strokeRect(x, y, w, h);
            
            if (currentMatch !== "Unknown") {
                ctx.fillStyle = color;
                ctx.fillRect(x, y - 30, w, 30);
                ctx.fillStyle = "#000";
                ctx.font = "bold 16px sans-serif";
                ctx.fillText(currentMatch, x + 10, y - 10);
            }
        }
    }
    window.requestAnimationFrame(predictVideo);
}

init();
</script>
"""

def get_component_html(img_b64=None):
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

# --- UI LAYOUT ---

# SIDEBAR
with st.sidebar:
    st.title("🛡️ Admin Panel")
    st.markdown("---")
    
    if os.path.exists(PKL_LOG):
        with open(PKL_LOG,"rb") as f: total = len(pickle.load(f))
    else: total = 0
    
    c1, c2 = st.columns(2)
    c1.metric("Users", len(st.session_state.db))
    c2.metric("Logs Today", len(st.session_state.logged_set))
    
    st.markdown("---")
    if st.button("Refresh System"): st.rerun()

    with st.expander("Database Operations"):
        if st.session_state.db:
            for name in list(st.session_state.db.keys()):
                c1, c2 = st.columns([3,1])
                c1.text(name)
                if c2.button("✖", key=f"del_{name}"):
                    del st.session_state.db[name]
                    with open(DB_FILE,"w") as f: json.dump(st.session_state.db,f)
                    st.rerun()
        else:
            st.caption("No users registered.")

# MAIN TABS
st.title("Agnos Biometric System")
tab_scan, tab_reg, tab_logs = st.tabs(["🎥 Scanner", "👤 Registration", "📊 Logs"])

# TAB 1: SCANNER
with tab_scan:
    c_vid, c_stat = st.columns([2, 1])
    
    with c_vid:
        st.components.v1.html(get_component_html(), height=480)

    with c_stat:
        st.subheader("Live Status")
        if "detected_name" in st.query_params:
            det = st.query_params["detected_name"]
            st.markdown(f"""
            <div class="status-overlay status-success" style="position:relative; top:0; right:0; margin-bottom:20px;">
                <h2 style="margin:0; color:#00FF00;">IDENTIFIED</h2>
                <h1 style="margin:0; font-size:3em;">{det}</h1>
                <p>Attendance Logged Successfully</p>
            </div>
            """, unsafe_allow_html=True)
            
        else:
            st.markdown("""
            <div class="status-overlay status-scan" style="position:relative; top:0; right:0; margin-bottom:20px;">
                <h2 style="margin:0; color:#0088FF;">SCANNING</h2>
                <p>Looking for registered faces...</p>
            </div>
            """, unsafe_allow_html=True)

# TAB 2: REGISTRATION
with tab_reg:
    st.subheader("New User Registration")
    
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
                if st.button("✅ Confirm Save", type="primary", use_container_width=True):
                    data = json.loads(base64.b64decode(st.query_params["face_data"]).decode())
                    st.session_state.db[name_in] = data
                    with open(DB_FILE,"w") as f: json.dump(st.session_state.db,f)
                    
                    st.session_state.uploader_key += 1
                    st.query_params.clear()
                    st.toast(f"Registered {name_in}!")
                    st.rerun()
            else:
                st.info("Processing biometrics...")

# TAB 3: LOGS
with tab_logs:
    st.subheader("Attendance History")
    if st.button("Refresh Logs"): st.rerun()
    
    if os.path.exists(PKL_LOG):
        with open(PKL_LOG,"rb") as f: df = pd.DataFrame(pickle.load(f))
        if not df.empty:
            st.dataframe(df, use_container_width=True)
            st.download_button("Download CSV", df.to_csv(index=False), "logs.csv", "text/csv")
            if st.button("Clear Logs"):
                os.remove(PKL_LOG)
                st.session_state.logged_set = set()
                st.rerun()
        else: st.info("Empty logs.")
    else: st.info("No logs found.")
