# 👁️ Agnos

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**Agnos** is a lightweight, cloud-ready facial recognition system built for secure facility access and automated attendance logging. It transforms any standard webcam into a biometric scanner, processing video feeds in real-time to identify registered users and log their entry.

![Agnos Enterprise Demo](docs/hero-image-placeholder.png) *(Replace with a screenshot or GIF of the app in action)*

## ✨ Key Features

* **Real-Time Biometric Scanning:** Processes live webcam feeds asynchronously using WebRTC, providing immediate visual feedback (Green = Verified, Red = Unknown).
* **Frictionless Registration:** Add new employees in seconds via a live webcam snapshot or file upload.
* **Smart Attendance Logs:** Automatically records Name, Date, and Time to a CSV file. Includes built-in logic to prevent duplicate entries on the same day.
* **Multi-Tenant Architecture:** Secures data using a unique 5-character Organization Key (Org Key) and hashed passwords, completely isolating user databases and logs between different organizations.
* **CPU-Friendly:** Explicitly configured to run without dedicated GPUs, making it easy to deploy on standard cloud instances.

---

## 🛠️ Tech Stack

* **Frontend & UI:** `Streamlit`
* **Live Video Streaming:** `streamlit-webrtc` & `av`
* **Face Detection:** `OpenCV` (Haar Cascades)
* **Facial Embeddings (AI):** `keras-facenet` (TensorFlow)
* **Data Management:** `Pandas` (CSV logs), `Pickle` (Face encodings)

---

## 📖 Usage Guide

1. **Login/Create Workspace:** On launch, enter a 5-character **Org Key** (e.g., `ACME1`) and a password. If the key doesn't exist, it will create a new isolated workspace.
2. **Register Users:** Navigate to the **👤 Register User** tab. Enter a name and provide a clear, front-facing photo via webcam or file upload.
3. **Start Scanning:** Open the **🎥 Live Scanner** tab and allow camera permissions. The system will autonomously scan and verify faces.
4. **Export Logs:** Go to the **📊 Access Logs** tab to view attendance or download the data as a `.csv` file for payroll/HR integration.
5. **Manage Database:** Use the **🗄️ Database** tab to view all registered personnel or delete biometric profiles from the system.

---

## ⚠️ Known Limitations & Security Notes

This repository currently serves as a functional prototype. Before deploying to a production enterprise environment, contributors should be aware of the following:

* **No Liveness Detection:** The current OpenCV/FaceNet pipeline processes 2D images and is vulnerable to presentation attacks (e.g., holding up a photo to the camera). 
* **Pickle Serialization:** Face encodings are saved using `.pkl`. In a production environment with untrusted server access, Pickle can be a security vulnerability. Migrating to SQLite or a Vector Database (like Milvus or Pinecone) is recommended.
* **RBAC:** Currently, anyone with the Org Key and password has full admin rights (can clear logs and delete users). Role-Based Access Control should be implemented for standard users vs. admins.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/yourusername/agnos-enterprise/issues).

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.
