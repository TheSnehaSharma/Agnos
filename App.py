import streamlit as st
import cv2
import numpy as np
import base64
import os
from insightface.app import FaceAnalysis

st.set_page_config(page_title="Agnos Pipe Fix", layout="wide")

# --- AI ENGINE ---
@st.cache_resource
def load_insightface():
    try:
        # Using the smallest model for the fastest handshake
        app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(320, 320))
        return app
    except Exception as e:
        return None

# --- UPDATED JS BRIDGE (The Fix) ---
JS_FIXED_CODE = """
<div id="video-container" style="background:#000; border-radius:10px; overflow:hidden;">
    <video id="v" autoplay playsinline style="width:100%; height:auto;"></video>
    <canvas id="c" style="display:none;"></canvas>
</div>
<script>
    const v = document.getElementById('v');
    const c = document.getElementById('c');
    const ctx = c.getContext('2d');
    
    // Request camera
    navigator.mediaDevices.getUserMedia({ video: { width: 320, height: 240 } })
        .then(stream => { v.srcObject = stream; })
        .catch(err => console.error("Camera Error:", err));

    function capture() {
        // ONLY send if the video is actually playing and providing data
        if (v.readyState === 4) { 
            c.width = 320; c.height = 240;
            ctx.drawImage(v, 0, 0, 320, 240);
            
            const data = c.toDataURL('image/jpeg', 0.6);
            
            // Check if string is healthy (Longer than 1000 chars)
            if (data.length > 1000) {
                window.parent.postMessage({
                    type: "streamlit:setComponentValue",
                    value: data
                }, "*");
            }
        }
    }
    // Faster interval for debugging (1 second)
    setInterval(capture, 1000);
</script>
"""

st.title("🛰️ Pipe Repair Console")
col_cam, col_debug = st.columns([1, 1])

with col_cam:
    st.subheader("1. Video Source")
    img_data = st.components.v1.html(JS_FIXED_CODE, height=250)
    if not img_data:
        st.info("Waiting for first valid frame from camera...")

with col_debug:
    st.subheader("2. Python Debug")
    
    if img_data:
        try:
            # 1. Clean and Decode
            raw_str = str(img_data)
            if "," in raw_str:
                encoded = raw_str.split(",")[1]
                # Fix padding
                encoded += "=" * ((4 - len(encoded) % 4) % 4)
                img_bytes = base64.b64decode(encoded)
                
                st.write(f"📊 String Length: {len(raw_str)}")
                st.write(f"✅ Bytes Decoded: {len(img_bytes)}")
                
                # 2. Convert to CV2
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    st.image(frame, width=200, caption="Python Received Frame")
                    
                    # 3. AI Inference (ONLY if frame is valid)
                    engine = load_insightface()
                    if engine:
                        faces = engine.get(frame)
                        st.write(f"🎯 AI Status: Detected {len(faces)} faces")
                        if len(faces) > 0:
                            st.success("WE ARE LIVE! Bridge is fully functional.")
                else:
                    st.error("❌ OpenCV failed to read bytes. Bytes might be corrupted.")
            else:
                st.error("❌ Received malformed string (No Base64 header).")
        except Exception as e:
            st.error(f"💥 Error: {e}")

if st.button("Clear Cache & Rerun"):
    st.cache_resource.clear()
    st.rerun()
