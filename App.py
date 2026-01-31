import streamlit as st
import cv2
import numpy as np
import base64
import os
from insightface.app import FaceAnalysis

# --- 1. AI ENGINE ---
@st.cache_resource
def load_ai():
    # buffalo_s is the lightest pre-compiled model
    app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(320, 320))
    return app

# --- 2. JS BRIDGE (THE URL HACK) ---
# This script bypasses the "DeltaGenerator" by writing data to the URL hash
JS_URL_BRIDGE = """
<div style="background:#000; color:#0F0; font-family:monospace; padding:10px; border-radius:10px;">
    <video id="v" autoplay playsinline style="width:100%; height:auto; border:1px solid #333;"></video>
    <canvas id="c" style="display:none;"></canvas>
    <div id="msg">INITIALIZING CAMERA...</div>
</div>

<script>
    const v = document.getElementById('v');
    const c = document.getElementById('c');
    const ctx = c.getContext('2d');
    const msg = document.getElementById('msg');

    navigator.mediaDevices.getUserMedia({ video: { width: 320, height: 240 } })
        .then(s => { v.srcObject = s; msg.innerText = "SENSOR ONLINE"; })
        .catch(e => { msg.innerText = "ERROR: " + e.name; });

    function sync() {
        if (v.videoWidth > 0) {
            c.width = 320; c.height = 240;
            ctx.drawImage(v, 0, 0, 320, 240);
            const data = c.toDataURL('image/jpeg', 0.4);
            
            // WE SEND DATA TO THE URL HASH
            // Streamlit detects URL changes and reruns automatically
            if (data.length > 2000) {
                const b64 = data.split(',')[1];
                window.parent.location.hash = "frame=" + b64;
                msg.innerText = "SYNCING: " + b64.length + " bytes";
            }
        }
    }
    setInterval(sync, 2000); // Sync every 2 seconds
</script>
"""

# --- 3. MAIN UI ---
st.set_page_config(page_title="Agnos: URL Bridge", layout="wide")
st.title("🛰️ Agnos URL Bridge")

# Navigation Sidebar
page = st.sidebar.radio("Navigation", ["Live Feed", "Register Face", "Log History"])

if page == "Live Feed":
    col_cam, col_py = st.columns([1, 1])

    with col_cam:
        st.subheader("1. Browser Feed")
        # We render the component but DO NOT assign it to a variable
        st.components.v1.html(JS_URL_BRIDGE, height=300)

    with col_py:
        st.subheader("2. Python AI Engine")
        
        # We read directly from the URL hash
        # This is the only way to get true data back from a raw HTML component
        try:
            # Get the hash from the URL
            raw_hash = st.query_params.to_dict()
            # Note: Depending on your Streamlit version, we might need 
            # to use a hidden text_input + JS injection if query_params are limited.
            
            # Let's use the most compatible "Query" method:
            img_b64 = st.query_params.get("frame")

            if img_b64:
                # Fix padding
                img_b64 += "=" * ((4 - len(img_b64) % 4) % 4)
                img_bytes = base64.b64decode(img_b64)
                
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is not None:
                    st.image(frame, width=200, caption="Received via URL Bridge")
                    engine = load_ai()
                    faces = engine.get(frame)
                    
                    if len(faces) > 0:
                        st.success(f"🎯 MATCH FOUND: {len(faces)} Faces")
                    else:
                        st.warning("Scanning for face...")
                else:
                    st.error("Frame Corrupted in transit.")
            else:
                st.info("Awaiting first handshake from URL...")
        except Exception as e:
            st.error(f"Bridge Error: {e}")

elif page == "Register Face":
    st.header("👤 Registration")
    # ... (Your existing registration code)
    st.info("Register your face here to build the database.")

elif page == "Log History":
    st.header("📊 History")
    # ... (Your existing log history code)
    st.info("Attendance logs will appear here.")
