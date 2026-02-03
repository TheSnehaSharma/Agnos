import streamlit as st
import cv2
import numpy as np
import os
import base64
from insightface.app import FaceAnalysis

# ================= CONFIG =================
DB_FOLDER = "registered_faces"
SIM_THRESHOLD = 0.5
os.makedirs(DB_FOLDER, exist_ok=True)

st.set_page_config("Iron-Vision Biometric", layout="wide")

# ================= STATE =================
if "identity" not in st.session_state:
    st.session_state.identity = "SCANNING"
    st.session_state.score = 0

# ================= AI ENGINE =================
@st.cache_resource
def load_engine():
    engine = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
    engine.prepare(ctx_id=0, det_size=(320, 320))
    return engine

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

# ================= NAVIGATION =================
page = st.sidebar.radio(
    "Navigation",
    ["Live Scanner", "Register Face", "Manage Registered Faces"]
)

# =========================================================
# ======================= LIVE SCANNER ====================
# =========================================================
if page == "Live Scanner":
    st.header("📹 Live Biometric Scanner")

    # HTML + JS to stream camera feed and send base64 image to Streamlit
    JS = """
    <video autoplay playsinline id="video" style="width:100%;height:100%;object-fit:cover;"></video>
    <script>
    const video = document.getElementById('video');
    navigator.mediaDevices.getUserMedia({video:true})
      .then(stream => { video.srcObject = stream; });

    const canvas = document.createElement('canvas');
    setInterval(() => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video,0,0);
        if(window.Streamlit){
            Streamlit.setComponentValue(canvas.toDataURL('image/jpeg',0.8));
        }
    }, 500);
    </script>
    """
    img_data = st.components.v1.html(JS, height=600)

    # ---------------- Process frame in Python ----------------
    if isinstance(img_data, str) and img_data.startswith("data:image"):
        encoded = img_data.split(",")[1]
        nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        engine = load_engine()
        faces = engine.get(frame)

        if faces:
            emb = faces[0].normed_embedding
            db = load_face_db()
            best_name, best_score = "UNKNOWN", 0.0

            for name, ref in db.items():
                sim = float(np.dot(emb, ref))
                if sim > best_score:
                    best_name, best_score = name, sim

            if best_score >= SIM_THRESHOLD:
                st.session_state.identity = best_name
                st.session_state.score = int(best_score * 100)
            else:
                st.session_state.identity = "UNKNOWN"
                st.session_state.score = int(best_score * 100)

            # Draw rectangle + name over face for display
            for f in faces:
                bbox = f.bbox.astype(int)  # [x1,y1,x2,y2]
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                              (0,255,0), 2)
                cv2.putText(frame,
                            f"{st.session_state.identity} ({st.session_state.score}%)",
                            (bbox[0], bbox[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
        else:
            st.info("No face detected...")

# =========================================================
# ===================== REGISTER FACE =====================
# =========================================================
elif page == "Register Face":
    st.header("👤 Register New Face")

    name = st.text_input("Full Name").upper().strip()
    img_file = st.file_uploader("Upload face image", ["jpg","png","jpeg"])

    if st.button("Register"):
        if not name or not img_file:
            st.error("Please provide name and image.")
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
