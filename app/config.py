from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
APP_TITLE = os.getenv("APP_TITLE", "Face Liveness Detection")
APP_SUBTITLE = os.getenv("APP_SUBTITLE", "Real-time anti-spoofing with a polished Streamlit experience")
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "8"))
SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/png"}
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))
STREAMLIT_HOST = os.getenv("STREAMLIT_HOST", "0.0.0.0")
MODEL_PATH = os.getenv("MODEL_PATH", str(ROOT_DIR / "models" / "face_antispoof_quantized.onnx"))
