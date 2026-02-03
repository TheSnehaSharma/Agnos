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

st.set_page_config(
    page_title="Biometric Attendance",
    page_icon="👁️",
    layout="centered", # Centered looks better for mobile/focused view
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS FOR MODERN UI ---
st.markdown("""
<style>
    /* Remove default padding */
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    
    /* Custom Status Cards */
    .status-card {
        background-color: #1E1E1E;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin-top: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .status-success { border-left: 5px solid #00FF00; color: #00FF00; }
    .status-unknown { border-left: 5px solid #FF4B4B; color: #FF4B4B; }
    .status-waiting { border-left: 5px solid #FFFF00; color: #CCCCCC; }
    
    /* Header Styling */
    h1, h2, h3 { font-family: 'Segoe UI', sans-serif; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if "db" not in st.session_state:
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            st.session_state.db = json.load(f)
    else:
        st.session_state.db = {}

if "logged_set" not in st.session_state:
    st.session_state.logged_set = set()

# --- JAVASCRIPT: 3D BIOMETRIC ENGINE ---
JS_CODE = """
<script type="module">
import { FaceLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

const video = document.getElementById("webcam");
const staticImg = document.getElementById("static-img");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const log = document.getElementById("status-bar");

const staticImgSrc = "STATIC_IMG_PLACEHOLDER";
const runMode = "RUN_MODE_PLACEHOLDER";
const registry = JSON.parse('DB_JSON_PLACEHOLDER');

let faceLandmarker;

function getDist3D(p1, p2) {
    return Math.sqrt(
        Math.pow(p1.x - p2.x, 2) + 
        Math.pow(p1.y - p2.y, 2) + 
        Math.pow(p1.z - p2.z, 2)
    );
}

function getFaceVector(landmarks) {
    const L_EYE = landmarks[468]; 
    const R_EYE = landmarks[473]; 
    const NOSE  = landmarks[4];   
    const CHIN  = landmarks[152]; 
    const L_EAR = landmarks[234]; 
    const R_EAR = landmarks[454]; 
    
    const eyeDist = getDist3D(L_EYE, R_EYE);

    return [
        getDist3D(NOSE, L_EYE) / eyeDist,  
        getDist3D(NOSE, R_EYE) / eyeDist,  
        getDist3D(NOSE, CHIN)  / eyeDist,  
        getDist3D(L_EAR, R_EAR)/ eyeDist,  
        getDist3D(L_EYE, CHIN) / eyeDist,  
        getDist3D(R_EYE, CHIN) / eyeDist   
    ];
}

function findMatch(currentLandmarks) {
    const currentVec = getFaceVector(currentLandmarks);
    let bestMatch = { name: "Unknown", error: 100 };
    const weights = [2.0, 2.0, 1.0, 1.5, 1.0, 1.0]; 

    for (const [name, savedLandmarks] of Object.entries(registry)) {
        const savedVec = getFaceVector(savedLandmarks);
        let weightedError = 0;
        let totalWeight = 0;

        for(let i=0; i<currentVec.length; i++) {
            const diff = Math.abs(currentVec[i] - savedVec[i]);
            weightedError += diff * weights[i];
            totalWeight += weights[i];
        }

        const avgError = weightedError / totalWeight;
        if (avgError < bestMatch.error) {
            bestMatch = { name: name, error: avgError };
        }
    }

    if (bestMatch.error < 0.08) {
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
                    window.parent.postMessage({ type: 'streamlit:setComponentValue', value: 'READY' }, "*");
                }
            }
        } else {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = stream;
            video.onloadeddata = () => predictVideo();
        }
    } catch (err) { log.innerText = "ERROR: " + err.message; }
}

async function predictVideo() {
    const results = faceLandmarker.detectForVideo(video, performance.now());
    canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (results.faceLandmarks && results.faceLandmarks.length > 0) {
        const landmarks = results.faceLandmarks[0];
        const match = findMatch(landmarks);

        const xs = landmarks.map(p => p.x * canvas.width);
        const ys = landmarks.map(p => p.y * canvas.height);
        const x = Math.min(...xs), y = Math.min(...ys), w = Math.max(...xs) - x, h = Math.max(...ys) - y;

        const color = match.name === "Unknown" ? "#FF4B4B" : "#00FF00";
        ctx.strokeStyle = color; ctx.lineWidth = 4; ctx.strokeRect(x, y, w, h);
        
        // Only update python if the name changes to reduce flickering reruns
        if (match.name !== "Unknown") {
             const url = new URL(window.parent.location.href);
             if (url.searchParams.get("detected") !== match.name) {
                 url.searchParams.set("detected", match.name);
                 window.parent.history.replaceState({}, "", url);
                 // Trigger rerun immediately
                 window.parent.postMessage({ type: 'streamlit:setComponentValue', value: match.name }, "*");
             }
        }
    }
    window.requestAnimationFrame(predictVideo);
}

init();
</script>
"""

# --- COMPONENT WRAPPER ---
def get_component_html(img_b64=None):
    db_json = json.dumps(st.session_state.db)
    img_val = f"data:image/jpeg;base64,{img_b64}" if img_b64 else "null"
    
    # CSS for the embedded video
    css = """
    <style>
    body { margin:0; background:#000; overflow:hidden; }
    #view { position:relative; width:100%; height:400px; background:#000; border-radius:12px; overflow:hidden;}
    video, canvas, img { position:absolute; top:0; left:0; width:100%; height:100%; object-fit:cover; }
    #status-bar { position:absolute; bottom:0; left:0; right:0; background:rgba(0,0,0,0.6); color:#fff; padding:4px; font-size:10px; text-align:center;}
    </style>
    """
    
    html = f"<!DOCTYPE html><html><head>{css}</head><body>"
    html += f'<div id="view"><div id="status-bar">Waiting for camera...</div>'
    html += f'<video id="webcam" autoplay muted playsinline style="display: {"none" if img_b64 else "block"}"></video>'
    html += f'<img id="static-img" style="display: {"block" if img_b64 else "none"}">'
    html += f'<canvas id="overlay"></canvas></div>{JS_CODE}</body></html>'
    
    return html.replace("STATIC_IMG_PLACEHOLDER", img_val)\
               .replace("RUN_MODE_PLACEHOLDER", "IMAGE" if img_b64 else "VIDEO")\
               .replace("DB_JSON_PLACEHOLDER", db_json)

# --- BACKEND LOGIC ---
def save_attendance(name):
    logs = []
    if os.path.exists(PKL_LOG):
        with open(PKL_LOG, "rb") as f:
            logs = pickle.load(f)
    
    # Check if already logged TODAY
    today = datetime.now().strftime("%Y-%m-%d")
    already_logged = any(entry['Name'] == name and entry['Date'] == today for entry in logs)
    
    if not already_logged:
        entry = {
            "Name": name, 
            "Time": datetime.now().strftime("%H:%M:%S"), 
            "Date": today
        }
        logs.append(entry)
        with open(PKL_LOG, "wb") as f:
            pickle.dump(logs, f)
        return True, entry
    return False, None

# --- MAIN UI ---
st.title("👁️ Secure Entry System")

# Navigation Tabs (Cleaner than sidebar)
tab1, tab2, tab3 = st.tabs(["🎥 Live Scan", "👤 Register", "📊 Attendance Logs"])

# --- TAB 1: LIVE SCAN ---
with tab1:
    st.write("Look at the camera to mark attendance.")
    
    # The Video Component
    st.components.v1.html(get_component_html(), height=420)
    
    # The "Instant Feedback" Status Panel
    detected_name = st.query_params.get("detected")
    
    if detected_name and detected_name != "Unknown":
        # Save to DB
        is_new, entry = save_attendance(detected_name)
        
        if is_new:
            st.markdown(f"""
            <div class="status-card status-success">
                <h2>✅ Access Granted</h2>
                <h3>Welcome, {detected_name}</h3>
                <p>Logged at {entry['Time']}</p>
            </div>
            """, unsafe_allow_html=True)
            st.toast(f"Welcome {detected_name}!")
        else:
            # Already logged today
            st.markdown(f"""
            <div class="status-card status-success" style="opacity: 0.7;">
                <h3>👋 Hello, {detected_name}</h3>
                <p>Attendance already marked for today.</p>
            </div>
            """, unsafe_allow_html=True)
            
    else:
        st.markdown("""
        <div class="status-card status-waiting">
            <h3>🔭 Scanning...</h3>
            <p>Please face the camera directly.</p>
        </div>
        """, unsafe_allow_html=True)

# --- TAB 2: REGISTER ---
with tab2:
    st.header("Add New Member")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        name_input = st.text_input("Full Name (Required)", placeholder="e.g. John Doe").upper()
        uploaded_file = st.file_uploader("Upload Profile Photo", type=['jpg', 'png', 'jpeg'])

    with col2:
        if uploaded_file and name_input:
            # Convert to Base64 for processing
            b64_img = base64.b64encode(uploaded_file.getvalue()).decode()
            
            # Show preview
            st.image(uploaded_file, caption="Preview", use_container_width=True)
            
            # Hidden component to process the image for landmarks
            st.markdown("**Processing biometric data...**")
            st.components.v1.html(get_component_html(b64_img), height=0, width=0)
            
            # Check for return data
            face_data_str = st.query_params.get("face_data")
            
            if face_data_str:
                if st.button("💾 Save Registration", type="primary"):
                    face_data = json.loads(base64.b64decode(face_data_str).decode())
                    st.session_state.db[name_input] = face_data
                    with open(DB_FILE, "w") as f:
                        json.dump(st.session_state.db, f, indent=4)
                    
                    st.success(f"Successfully registered {name_input}!")
                    st.query_params.clear()
                    st.rerun()
        else:
            st.info("Enter name and upload photo to proceed.")

# --- TAB 3: LOGS & DOWNLOAD ---
with tab3:
    st.header("Attendance Records")
    
    if os.path.exists(PKL_LOG):
        with open(PKL_LOG, "rb") as f:
            logs = pickle.load(f)
        
        if logs:
            df = pd.DataFrame(logs)
            
            # Top Controls
            col_l, col_r = st.columns([4, 1])
            with col_l:
                st.dataframe(df, use_container_width=True)
            with col_r:
                # CSV Download Button
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"attendance_{datetime.now().strftime('%Y-%m-%d')}.csv",
                    mime="text/csv",
                    type="primary"
                )
                
                st.write("") # Spacer
                if st.button("🗑️ Clear Logs"):
                    os.remove(PKL_LOG)
                    st.rerun()
        else:
            st.info("Log file exists but is empty.")
    else:
        st.info("No attendance records found yet.")
