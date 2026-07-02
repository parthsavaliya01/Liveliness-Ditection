from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from .config import (
    MIN_FACE_OCCUPANCY_RATIO,
    SINGLE_IMAGE_CROP_SCALES,
    SINGLE_IMAGE_THRESHOLD,
)
from .detector import AntiSpoofPredictor, FaceDetector, REAL_HUMAN, SPOOF_ATTACK

MOVE_CLOSER_MESSAGE = "Face too far from camera. Please move closer."


class SingleSelfieLivenessDetector:
    def __init__(
        self,
        threshold: float = SINGLE_IMAGE_THRESHOLD,
        min_face_occupancy_ratio: float = MIN_FACE_OCCUPANCY_RATIO,
        face_detector: Optional[FaceDetector] = None,
        predictor: Optional[AntiSpoofPredictor] = None,
    ) -> None:
        self.threshold = threshold
        self.min_face_occupancy_ratio = min_face_occupancy_ratio
        self.face_detector = face_detector or FaceDetector(crop_scales=SINGLE_IMAGE_CROP_SCALES)
        self.predictor = predictor or AntiSpoofPredictor()

    def analyze(self, selfie_bgr: np.ndarray) -> Dict[str, float | str]:
        if selfie_bgr is None or selfie_bgr.size == 0:
            return {"status": SPOOF_ATTACK, "confidence": 0.0}

        face = self.face_detector.detect_largest_face(selfie_bgr)
        if face is None:
            return {"status": SPOOF_ATTACK, "confidence": 0.0}

        if face.face_area_ratio < self.min_face_occupancy_ratio:
            return {"status": SPOOF_ATTACK, "confidence": 0.0, "message": MOVE_CLOSER_MESSAGE}

        crop_scores = [self.predictor.predict(crop.image).live_score for crop in face.anti_spoof_crops]
        live_score = max(crop_scores) if crop_scores else 0.0
        if live_score >= self.threshold:
            return {"status": REAL_HUMAN, "confidence": float(live_score)}

        spoof_confidence = 1.0 - live_score
        return {"status": SPOOF_ATTACK, "confidence": float(spoof_confidence)}

    def analyze_file(self, image_path: str | Path) -> Dict[str, float | str]:
        image = cv2.imread(str(image_path))
        if image is None:
            return {"status": SPOOF_ATTACK, "confidence": 0.0}
        return self.analyze(image)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run single-selfie liveness detection.")
    parser.add_argument("--image", required=True, help="Path to a captured selfie image.")
    args = parser.parse_args()

    detector = SingleSelfieLivenessDetector()
    result = detector.analyze_file(args.image)
    print(f"{result['status']}")
    print(f"Confidence: {result['confidence'] * 100:.2f}%")
    if "message" in result:
        print(result["message"])


if __name__ == "__main__":
    main()
