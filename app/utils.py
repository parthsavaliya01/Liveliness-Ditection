from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional, Tuple

import cv2
import numpy as np

from .config import MAX_UPLOAD_SIZE_MB, SUPPORTED_EXTENSIONS, SUPPORTED_IMAGE_TYPES


def validate_uploaded_image(uploaded_file: object) -> Tuple[bool, Optional[str]]:
    if uploaded_file is None:
        return False, "No file was provided."

    file_type = getattr(uploaded_file, "type", "") or ""
    file_name = getattr(uploaded_file, "name", "") or ""
    size = getattr(uploaded_file, "size", 0) or 0

    if file_type not in SUPPORTED_IMAGE_TYPES:
        return False, "Unsupported file type. Supported formats are JPG and PNG."

    if file_name and os.path.splitext(file_name)[1].lower() not in SUPPORTED_EXTENSIONS:
        return False, "Unsupported extension. Please upload .jpg, .jpeg, or .png."

    if size > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        return False, f"Image is larger than {MAX_UPLOAD_SIZE_MB}MB. Please choose a smaller image."

    return True, None


def read_image_from_upload(uploaded_file: object) -> Optional[np.ndarray]:
    is_valid, error = validate_uploaded_image(uploaded_file)
    if not is_valid:
        raise ValueError(error or "Unable to process the uploaded image.")

    file_bytes = uploaded_file.getvalue()
    if not file_bytes:
        raise ValueError("The selected file is empty.")

    array = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The selected file could not be decoded as an image.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_image_from_camera(image_bytes: bytes) -> Optional[np.ndarray]:
    if not image_bytes:
        raise ValueError("No image was captured.")

    array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The captured image could not be decoded.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def format_confidence(value: float) -> str:
    return f"{max(0.0, min(100.0, value * 100.0)):.2f}%"


def format_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
