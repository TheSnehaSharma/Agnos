import streamlit as st
import base64
import os
import cv2
import numpy as np

st.set_page_config(page_title="Bridge Debugger", layout="wide")

# --- JS BRIDGE (DEBUG VERSION) ---
JS_DEBUG_CODE = """
<div style="background:#000; border-radius:10px; overflow:hidden;">
    <video id="v" autoplay playsinline style="width:100%; height:auto;"></video>
    <canvas id="c" style="display:none;"></canvas>
</div>
<script>
    const v = document.getElementById('v');
    const c = document.getElementById('c');
    const ctx = c.getContext('2d');

    navigator.mediaDevices.getUserMedia({ video: { width: 160, height: 120 } })
        .then(s => v.srcObject = s);

    function send() {
        if (v.readyState === v.HAVE_ENOUGH_DATA) {
            c.width = 160; c.height = 120;
            ctx.drawImage(v, 0, 0, 160, 120);
            const data = c.toDataURL('image/jpeg', 0.1); // Ultra low quality for testing
            
            // Sending to Streamlit
            window.parent.postMessage({
                type: "streamlit:setComponentValue",
                value: data
            }, "*");
        }
    }
    setInterval(send, 2000); // Send every 2 seconds
</script>
"""

st.title("🛰️ Bridge Diagnostic")

col_cam, col_data = st.columns(2)

with col_cam:
    st.subheader("Browser Feed")
    # We call the component but DON'T assign it to a variable yet to avoid the TypeError
    st.components.v1.html(JS_DEBUG_CODE, height=250)

with col_data:
    st.subheader("Python Reception")
    
    # Workaround: Use a fragment or experimental state to catch the value
    # If the standard html() call is crashing, it's because it's not a bi-directional component.
    st.warning("If the data below is 'None', the standard HTML component is blocking the return path.")
    
    # Let's try to capture the data safely
    try:
        # In standard Streamlit, st.components.v1.html RETURNS NOTHING.
        # This is why your code was failing. 
        res = st.components.v1.html(JS_DEBUG_CODE, height=0) 
        st.write(f"Raw Component Return: {type(res)}") 
    except Exception as e:
        st.error(f"Handshake Error: {e}")

# --- THE REAL WORKAROUND ---
st.markdown("---")
st.info("💡 **The Diagnosis:** Standard `st.components.v1.html` is a one-way street. To get data back, we MUST use a custom component wrapper.")

# This is a lightweight bridge that works in Streamlit
from streamlit_gsheets import GSheetsConnection # Just checking environment

# Let's try the simplest possible bi-directional hack: st.text_input + JS Injection
st.subheader("Manual Bridge Test")
bridge_input = st.text_input("Hidden Bridge", key="my_bridge")

st.write(f"Current Bridge String Length: {len(bridge_input)}")

if len(bridge_input) > 100:
    st.success("✅ DATA DETECTED IN PYTHON!")
    # Try to decode just to prove it works
    try:
        header, encoded = bridge_input.split(",", 1)
        data = base64.b64decode(encoded)
        st.write(f"Byte count: {len(data)}")
    except:
        st.error("Data received but decoding failed.")
