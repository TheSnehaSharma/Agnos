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

# --- SESSION STATE ---
if "db" not in st.session_state:
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            st.session_state.db = json.load(f)
    else:
        st.session_state.db = {}

if "logged_set" not in st.session_state:
    st.session_state.logged_set = set()
if "last_detected" not in st.session_state:
    st.session_state.last_detected = None

# --- CSS ---
CSS_CODE = """
<style>
body { margin:0; background:#0e1117; color:#00FF00; font-family: monospace; overflow:hidden; }
#view { position:relative; width:100%; height:400px; border-radius:12px; overflow:hidden; background:#000; border:1px solid #333; }
video, canvas, img { position:absolute; top:0; left:0; width:100%; height:100%; object-fit:cover; }
#status-bar { position:absolute; top:0; left:0; right:0; background:rgba(0,0,0,0.8); padding:8px; font-size:11px; z-index:100; }
</style>
"""

# --- JS CODE: 3D Biometric Fingerprinting ---
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

// --- 3D Euclidean Distance ---
function getDist3D(p1, p2) {
    return Math.sqrt(
        Math.pow(p1.x - p2.x, 2) + 
        Math.pow(p1.y - p2.y, 2) + 
        Math.pow(p1.z - p2.z, 2)
    );
}

// --- Generate Biometric Fingerprint ---
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

// --- Weighted Similarity Search ---
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
        const confidence = Math.max(0, Math.floor((1 - (bestMatch.error / 0.1)) * 100));
        return { name: bestMatch.name, conf: confidence }; 
    } else {
        return { name: "Unknown", conf: 0 };
    }
}

// --- Init MediaPipe ---
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
        ctx.fillStyle = color; ctx.fillRect(x, y - 25, w, 25);
        ctx.fillStyle = "white"; ctx.font = "bold 14px monospace";
        ctx.fillText(`${match.name}`, x + 5, y - 8);

        if (match.name !== "Unknown") {
            const url = new URL(window.parent.location.href);
            if (url.searchParams.get("detected") !== match.name) {
                url.searchParams.set("detected", match.name);
                window.parent.history.replaceState({}, "", url);
            }
        }
    }
    window.requestAnimationFrame(predictVideo);
}

init();
</script>
"""

# --- HTML component ---
def get_component_html(img_b64=None):
    db_json = json.dumps(st.session_state.db)
    img_val = f"data:image/jpeg;base64,{img_b64}" if img_b64 else "null"
    html = f"<!DOCTYPE html><html><head>{CSS_CODE}</head><body>"
    html += f'<div id="view"><div id="status-bar">AGNOS BIOMETRIC ACTIVE</div>'
    html += f'<video id="webcam" autoplay muted playsinline style="display: {"none" if img_b64 else "block"}"></video>'
    html += f'<img id="static-img" style="display: {"block" if img_b64 else "none"}">'
    html += f'<canvas id="overlay"></canvas></div>{JS_CODE}</body></html>'
    return html.replace("STATIC_IMG_PLACEHOLDER", img_val)\
               .replace("RUN_MODE_PLACEHOLDER", "IMAGE" if img_b64 else "VIDEO")\
               .replace("DB_JSON_PLACEHOLDER", db_json)

# --- Pickle logging ---
def save_attendance_pkl(name):
    logs = []
    if os.path.exists(PKL_LOG):
        with open(PKL_LOG,"rb") as f: logs = pickle.load(f)
    entry = {"Name":name, "Time":datetime.now().strftime("%H:%M:%S"), "Date":datetime.now().strftime("%Y-%m-%d")}
    logs.append(entry)
    with open(PKL_LOG,"wb") as
