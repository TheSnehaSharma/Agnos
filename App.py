import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av

# Standard Google STUN servers to handle NAT traversal (connection)
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

def main():
    st.set_page_config(page_title="Live Stream", page_icon="📹")
    
    st.title("📹 Live Webcam Feed")
    st.write("This app uses WebRTC to stream your camera directly to the browser.")

    # A simple callback to handle the video frames
    # We use 'passthrough' logic here just to show the feed
    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        # You can add simple OpenCV filters here if you want!
        # e.g., img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="live-stream",
        video_frame_callback=video_frame_callback,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True, # Improves performance
    )

    st.info("💡 If the camera doesn't start, ensure you've granted browser permissions.")

if __name__ == "__main__":
    main()
