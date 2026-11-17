from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Storage
DATA_DIR = BASE_DIR / "agnos_data"
DB_FILE = DATA_DIR / "agnos.db"
DB_TIMEOUT_SECONDS = 30

# Security
PBKDF2_ITERATIONS = 310_000
SALT_BYTES = 16
SESSION_TOKEN_BYTES = 32

# Organizations
ORG_KEY_LENGTH = 5

# Attendance
TIMEZONE = "Asia/Kolkata"

# Face detection / recognition
UNKNOWN_NAME = "Unknown"
SCANNING_LABEL = "Finding Match..."
FACE_INPUT_SIZE = (160, 160)
MATCH_THRESHOLD = 0.40
DETECTION_INTERVAL_FRAMES = 3
DETECTION_SCALE_FACTOR = 1.1
DETECTION_MIN_NEIGHBORS = 5
MIN_FACE_SIZE = (60, 60)

# WebRTC
STUN_SERVER_URLS = ["stun:stun.l.google.com:19302"]
