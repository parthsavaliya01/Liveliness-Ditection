from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np
import streamlit as st


@st.cache_resource(show_spinner="Loading the liveness model...")
def get_detector() -> Any:
    from src.selfie_liveness import SingleSelfieLivenessDetector

    return SingleSelfieLivenessDetector()


def analyze_image(image_rgb: np.ndarray) -> Dict[str, Any]:
    detector = get_detector()
    image_bgr = np.ascontiguousarray(image_rgb[:, :, ::-1])

    start = time.perf_counter()
    result = detector.analyze(image_bgr)
    elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)

    status = str(result.get("status", "SPOOF ATTACK"))
    confidence = float(result.get("confidence", 0.0))
    message = result.get("message")

    if status == "REAL HUMAN":
        label = "LIVE HUMAN"
        tone = "success"
    else:
        label = "SPOOF ATTACK"
        tone = "error"

    return {
        "status": status,
        "label": label,
        "confidence": confidence,
        "confidence_percent": confidence * 100.0,
        "processing_time_ms": elapsed_ms,
        "message": message,
        "tone": tone,
    }
