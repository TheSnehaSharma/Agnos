import streamlit as st
import cv2
import numpy as np
import base64
from insightface.app import FaceAnalysis

st.set_page_config(page_title="Pipe Repair", layout="wide")

@st.cache_resource
def load_ai():
    app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(320, 320))
    return app

# --- THE "WAIT-FOR-PIXELS" JS ---
JS_FINAL_FIX = """
<div style="background:#000; border-radius:10px; padding:10px; text-align:center;">
    <video id="v" autoplay playsinline style="width:240px; border:2px solid #333;"></video>
    <canvas id="c" style="display:none;"></canvas>
    <div id="stat" style="color:#0F0; font-family:monospace; margin-top:5px;">CAMERA: WAITING...</div>
</div>

<script>
    const v = document.getElementById('v');
    const c = document.getElementById('c');
    const ctx = c.getContext('2d');
    const stat = document.getElementById('stat');
    
    navigator.mediaDevices.getUserMedia({ video: { width: 320, height: 240 } })
        .then(stream => { 
            v.srcObject = stream;
            stat.innerText = "CAMERA: CONNECTED";
        })
        .catch(err => { stat.innerText = "ERROR: " + err.name; });

    function process() {
        // 1. Check if video is actually playing and has dimensions
        if (v.paused || v.ended || v.videoWidth === 0) return;

        c.width = 320;
        c.height = 240;
        ctx.drawImage(v, 0, 0, 320, 240);
        
        // 2. Extract frame
        const data = c.toDataURL('image/jpeg', 0.5);
        
        // 3. ONLY send if it's a real image (Real JPEGs are > 3000 chars)
        if (data.length > 2000) {
            stat.innerText = "STREAMING: " + data.length + " chars";
            window.parent.postMessage({
                type: "streamlit:setComponentValue",
                value: data
            }, "*");
        } else {
            stat.innerText = "STREAMING: DATA TOO SMALL";
        }
    }

    // Try to capture every 1.5 seconds
    setInterval(process, 1500);
</script>
"""

st.title("🛰️ Pipe Repair Console v2")
col_cam, col_debug = st.columns([1, 1])

with col_cam:
    st.subheader("1. Browser Component")
    img_data = st.components.v1.html(JS_FINAL_FIX, height=300)

with col_debug:
    st.subheader("2. Python Debug")
    
    if img_data:
        try:
            raw_str = str(img_data)
            # LOG THE RAW START TO SEE WHAT IT IS
            st.code(raw_str[:50] + "...", language="text")
            
            if "," in raw_str:
                encoded = raw_str.split(",")[1]
                # Fix padding
                encoded += "=" * ((4 - len(encoded) % 4) % 4)
                img_bytes = base64.b64decode(encoded)
                
                st.write(f"📊 String Length: {len(raw_str)}")
                st.write(f"✅ Bytes Decoded: {len(img_bytes)}")
                
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    st.image(frame, width=250, caption="SUCCESS: Frame received")
                    engine = load_ai()
                    faces = engine.get(frame)
                    st.success(f"🎯 AI DETECTED {len(faces)} FACES")
                else:
                    st.error("❌ OpenCV could not reconstruct the image.")
        except Exception as e:
            st.error(f"💥 Python Error: {e}")
    else:
        st.info("Awaiting JS Handshake...")
