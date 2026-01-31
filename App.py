import streamlit as st
import streamlit.components.v1 as components

st.title("🛡️ Privacy-First Attendance")
st.write("Processing happens on YOUR device. No video is sent to our servers.")

# This is a simplified "placeholder" for the JS implementation
# In a real app, we use a custom Streamlit Component to bridge the two.
def local_attendance_component():
    # JavaScript + TensorFlow.js code
    js_code = """
    <div id="container">
        <video id="webcam" autoplay playsinline width="640" height="480"></video>
        <canvas id="overlay" style="position: absolute; left: 0; top: 0;"></canvas>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/face_detection"></script>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>
    
    <script>
        const video = document.getElementById('webcam');
        navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
            video.srcObject = stream;
        });
        
        // Face detection logic runs here... 
        // When a face is matched, we use:
        // window.parent.postMessage({type: 'streamlit:setComponentValue', value: 'User_Name'}, '*');
    </script>
    """
    components.html(js_code, height=500)

local_attendance_component()

# --- Logic to receive the 'Log' from the browser ---
if "last_seen" not in st.session_state:
    st.session_state.last_seen = "Waiting..."

st.metric("Last Person Detected", st.session_state.last_seen)
