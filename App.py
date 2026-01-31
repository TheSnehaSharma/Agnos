import streamlit as st
import pandas as pd
import json
import numpy as np
import os
import base64
from datetime import datetime

# --- CONFIG & STORAGE ---
DB_FILE = "registered_faces.json"
LOG_FILE = "attendance_log.csv"

st.set_page_config(page_title="Privacy Face Auth", layout="wide")

if "db" not in st.session_state:
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f: st.session_state.db = json.load(f)
    else: st.session_state.db = {}

if "logs" not in st.session_state:
    if os.path.exists(LOG_FILE):
        try: st.session_state.logs = pd.read_csv(LOG_FILE)
        except: st.session_state.logs = pd.DataFrame(columns=["Name", "Time"])
    else: st.session_state.logs = pd.DataFrame(columns=["Name", "Time"])

# --- FRONTEND ASSETS ---

CSS_CODE = """
<style>
    body { margin:0; background: #0e1117; color: #00FF00; font-family: monospace; }
    #view { position: relative; width: 100%; height: 400px; border-radius: 12px; overflow: hidden; background: #000; border: 1px solid #333; }
    video, canvas, img { position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover; }
    #status-bar { position: absolute; top: 0; left: 0; right: 0; background: rgba(0,0,0,0.8); padding: 8px; font-size: 11px; z-index: 100; }
</style>
"""

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
    
    let faceLandmarker;

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
                        const landmarks = results.faceLandmarks[0];
                        const dataString = btoa(JSON.stringify(landmarks));
                        const url = new URL(window.parent.location.href);
                        url.searchParams.set("face_data", dataString);
                        window.parent.history.replaceState({}, "", url);
                        window.parent.postMessage({type: 'streamlit:setComponentValue', value: 'READY'}, "*");
                    }
                };
            } else {
                const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                video.srcObject = stream;
                video.onloadeddata = () => { predictVideo(); };
            }
        } catch (err) { log.innerText = "ERROR: " + err.message; }
    }

    async function predictVideo() {
        const results = faceLandmarker.detectForVideo(video, performance.now());
        canvas.width = video.videoWidth; canvas.height = video.videoHeight;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (results.faceLandmarks && results.faceLandmarks.length > 0) {
            const landmarks = results.faceLandmarks[0];
            window.parent.postMessage({
                type: "streamlit:setComponentValue",
                value: JSON.stringify(landmarks)
            }, "*");
        }
        window.requestAnimationFrame(predictVideo);
    }
    init();
</script>
"""

def get_component_html(img_b64=None):
    html_template = f"<!DOCTYPE html><html><head>{CSS_CODE}</head><body>"
    html_template += f'<div id="view"><div id="status-bar">SYSTEM ONLINE</div>'
    html_template += f'<video id="webcam" autoplay muted playsinline style="display: {"none" if img_b64 else "block"}"></video>'
    html_template += f'<img id="static-img" style="display: {"block" if img_b64 else "none"}">'
    html_template += f'<canvas id="overlay"></canvas></div>{JS_CODE}</body></html>'
    img_val = f"data:image/jpeg;base64,{img_b64}" if img_b64 else "null"
    return html_template.replace("STATIC_IMG_PLACEHOLDER", img_val).replace("RUN_MODE_PLACEHOLDER", "IMAGE" if img_b64 else "VIDEO")

# --- UI NAVIGATION ---

page =
