import streamlit as st
import pandas as pd
import cv2
import numpy as np
import base64
import os
import pickle
from insightface.app import FaceAnalysis
from datetime import datetime

# --- CONFIG ---
DB_FOLDER = "registered_faces"
if not os.path.exists(DB_FOLDER):
    os.makedirs(DB_FOLDER)

st.set_page_config(page_title="Agnos Debugger", layout="wide")

# --- AI ENGINE (With Error Catching) ---
@st.cache_resource
def load_insightface():
    try:
        # We use the 'small' model first for faster debugging
        app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(320, 320))
        return app
    except Exception as e:
        st.error(f"❌ AI Engine failed to initialize: {e}")
        return None

# --- JS BRIDGE ---
JS_CODE = """
<script>
    const v = document.createElement('video');
    const c = document.createElement('canvas');
    const ctx = c.getContext('2d');
    
    navigator.mediaDevices.getUserMedia({ video: { width: 320, height: 240 } })
        .then(stream => {
            v.srcObject = stream;
            v.play();
        });

    function capture() {
        if (v.readyState === v.HAVE_ENOUGH_DATA) {
            c.width = 320; c.height = 240;
            ctx.drawImage(v, 0, 0, 320, 240);
            const data = c.toDataURL('image/jpeg', 0.5);
            window.parent.postMessage({
                type: "streamlit:setComponentValue",
                value: data
            }, "*");
        }
    }
    setInterval(capture, 2000); // Capture every 2 seconds
</script>
<div style="color: #0F0; font-family: monospace;">🛰️ SENSOR ACTIVE</div>
"""

# --- MAIN UI ---
st.title("🔍 Agnos Diagnostic Mode")
col_cam, col_debug = st.columns([1, 1])

with col_cam:
    st.subheader("1. Browser Component")
    img_data = st.components.v1.html(JS_CODE, height=100)
    st.info("The JS is running. Check 'Diagnostic Console' for updates.")

with col_debug:
    st.subheader("2. Diagnostic Console")
    
    if not img_data:
        st.write("⏳ Waiting for JS handshake...")
    else:
        # STEP 1: Check Data Reception
        st.write("✅ Step 1: Data Received from JS")
        st.write(f"📊 String Length: {len(str(img_data))}")
        
        try:
            # STEP 2: Decode Base64
            st.write("⏳ Step 2: Attempting Base64 Decode...")
            header, encoded = str(img_data).split(",", 1)
            # Fix potential padding issues
            encoded += "=" * ((4 - len(encoded) % 4) % 4)
            img_bytes = base64.b64decode(encoded)
            st.write(f"✅ Step 2: Decoded {len(img_bytes)} bytes")
            
            # STEP 3: Convert to CV2 Image
            st.write("⏳ Step 3: Converting to CV2 Frame...")
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is not None:
                st.write(f"✅ Step 3: Frame size: {frame.shape}")
                st.image(frame, width=150, caption="Python Side Preview")
            else:
                st.error("❌ Step 3: Frame is None after imdecode")

            # STEP 4: AI Inference
            st.write("⏳ Step 4: Passing to InsightFace...")
            engine = load_insightface()
            if engine:
                faces = engine.get(frame)
                st.write(f"✅ Step 4: AI returned {len(faces)} faces")
                
                if len(faces) > 0:
                    st.success("🎯 FACE DETECTED! System is working.")
                else:
                    st.warning("⚠️ No face found in this frame.")
            
        except Exception as e:
            st.error(f"💥 CRASH AT STEP {e}")
            st.exception(e)

# --- REFRESH BUTTON ---
if st.button("♻️ Hard Reset App"):
    st.cache_resource.clear()
    st.rerun()
