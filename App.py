import streamlit as st
import cv2
import numpy as np
import os
import pickle
from insightface.app import FaceAnalysis
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# ================= CONFIG =================
DB_FOLDER = "registered_faces"
PKL_LOG = "attendance_data.pkl"
SIM_THRESHOLD = 0.5
os.makedirs(DB_FOLDER, exist_ok=True)

st.set_page_config("Iron-Vision Biometric", layout="wide")

# ================= ENGINE =================
@st.cache_resource
def load_engine():
    app = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(320, 320))
    return app

@st.cache_resource
def load_face_db():
    engine = load_engine()
    db = {}
    for f in os.listdir(DB_FOLDER):
        if f.lower().endswith((".jpg", ".png", ".jpeg")):
            img = cv2.imread(os.path.join(DB_FOLDER, f))
            if img is None:
                continue
            faces = engine.get(img)
            if faces:
                db[f.split(".")[0]] = faces[0].normed_embedding
    return db

# ================= WEBCAM PROCESSOR =================
class FaceRecognizer(VideoTransformerBase):
    def __init__(self):
        self.engine = load_engine()
        self.db = load_face_db()
        self.logged_today = set()
        self.sim_threshold = SIM_THRESHOLD

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        faces = self.engine.get(img)
        for f in faces:
            bbox = f.bbox.astype(int)
            emb = f.normed_embedding
            best_name = "UNKNOWN"
            best_score = 0.0
            for name, ref in self.db.items():
                s = float(np.dot(emb, ref))
                if s > best_score:
                    best_name = name
                    best_score = s
            if best_score >= self.sim_threshold:
                text = f"{best_name} ({int(best_score*100)}%)"
                # Attendance logging
                today = datetime.now().strftime("%Y-%m-%d")
                if (best_name, today) not in self.logged_today:
                    self.logged_today.add((best_name, today))
                    logs = []
                    if os.path.exists(PKL_LOG):
                        with open(PKL_LOG, "rb") as f:
                            logs = pickle.load(f)
                    logs.append({"Name": best_name, "Time": datetime.now().strftime("%H:%M:%S"), "Date": today})
                    with open(PKL_LOG, "wb") as f:
                        pickle.dump(logs, f)
            else:
                text = f"UNKNOWN ({int(best_score*100)}%)"
            # Draw box + text
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 2)
            cv2.putText(img, text, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        return img

# ================= NAVIGATION =================
page = st.sidebar.radio("Navigation", ["Live Scanner", "Register Face", "Manage Registered Faces"])

# =========================================================
# ======================= LIVE SCANNER ====================
# =========================================================
if page == "Live Scanner":
    st.header("📹 Biometric Scanner (Live)")
    webrtc_streamer(
        key="biometric",
        video_transformer_factory=FaceRecognizer,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        video_html_attrs={"style": "width:100%; height:auto;"},
    )

# =========================================================
# ===================== REGISTER FACE =====================
# =========================================================
elif page == "Register Face":
    st.header("👤 Register New Face")
    name = st.text_input("Full Name").upper().strip()
    img_file = st.file_uploader("Upload face image", ["jpg","png","jpeg"])
    if st.button("Register"):
        if not name or not img_file:
            st.error("Please provide both name and image.")
        else:
            with open(os.path.join(DB_FOLDER, f"{name}.jpg"), "wb") as f:
                f.write(img_file.getbuffer())
            st.cache_resource.clear()
            st.success(f"Registered {name}")

# =========================================================
# ================= MANAGE REGISTERED FACES ===============
# =========================================================
elif page == "Manage Registered Faces":
    st.header("🗂️ Manage Registered Faces")
    files = [f for f in os.listdir(DB_FOLDER) if f.lower().endswith((".jpg",".png",".jpeg"))]
    if not files:
        st.info("No registered faces.")
    else:
        for f in files:
            col1, col2 = st.columns([4,1])
            col1.write(f"✅ {f.split('.')[0]}")
            if col2.button("Delete", key=f):
                os.remove(os.path.join(DB_FOLDER, f))
                st.cache_resource.clear()
                st.experimental_rerun()

# =========================================================
# ================= ATTENDANCE LOG =======================
# =========================================================
if page == "Live Scanner" or page == "Manage Registered Faces":
    st.subheader("📊 Attendance Log")
    if os.path.exists(PKL_LOG):
        with open(PKL_LOG, "rb") as f:
            data = pickle.load(f)
        st.table(data)
        if st.button("Clear Logs"):
            os.remove(PKL_LOG)
            st.experimental_rerun()
    else:
        st.info("No attendance records yet.")
