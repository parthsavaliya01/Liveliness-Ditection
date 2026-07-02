from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis

from .config import (
    ANTI_SPOOF_CROP_SCALES,
    DECISION_PERCENTILE,
    FACE_CONFIDENCE_THRESHOLD,
    INSIGHTFACE_DET_SIZE,
    MIN_FACE_SIZE,
    MIN_FRAME_SCORE,
    MODEL_INPUT_SIZE,
    NUM_FRAMES,
    ONNX_MODEL_INPUT_SIZE,
    ONNX_MODEL_PATH,
    THRESHOLD,
)
from .utils import align_face, bbox_area, bbox_size, crop_face, crop_face_by_scale


REAL_HUMAN = "REAL HUMAN"
SPOOF_ATTACK = "SPOOF ATTACK"


@dataclass(frozen=True)
class AntiSpoofCrop:
    scale: float
    image: np.ndarray


@dataclass(frozen=True)
class AntiSpoofPrediction:
    live_score: float
    real_logit: float
    spoof_logit: float


@dataclass(frozen=True)
class FaceDetection:
    bbox: np.ndarray
    landmarks: np.ndarray
    confidence: float
    aligned_face: np.ndarray
    anti_spoof_crops: Tuple[AntiSpoofCrop, ...]
    image_shape: Tuple[int, int]

    @property
    def face_width(self) -> float:
        return float(max(0.0, self.bbox[2] - self.bbox[0]))

    @property
    def face_height(self) -> float:
        return float(max(0.0, self.bbox[3] - self.bbox[1]))

    @property
    def face_area_ratio(self) -> float:
        image_height, image_width = self.image_shape
        image_area = max(1.0, float(image_height * image_width))
        return bbox_area(self.bbox) / image_area


@dataclass(frozen=True)
class FrameResult:
    face_detected: bool
    current_score: Optional[float]
    average_score: Optional[float]
    decision: Optional[str]
    valid_frames: int
    bbox: Optional[np.ndarray] = None


class FaceDetector:
    def __init__(
        self,
        confidence_threshold: float = FACE_CONFIDENCE_THRESHOLD,
        min_face_size: int = MIN_FACE_SIZE,
        det_size: Tuple[int, int] = INSIGHTFACE_DET_SIZE,
        ctx_id: int = -1,
        app: Optional[FaceAnalysis] = None,
        crop_scales: Tuple[float, ...] = ANTI_SPOOF_CROP_SCALES,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.min_face_size = min_face_size
        self.crop_scales = crop_scales
        self.app = app or FaceAnalysis(allowed_modules=["detection"])
        if app is None:
            self.app.prepare(ctx_id=ctx_id, det_size=det_size)

    def detect_largest_face(self, frame: np.ndarray) -> Optional[FaceDetection]:
        faces = self.app.get(frame)
        if not faces:
            return None

        candidates: List[Tuple[float, FaceDetection]] = []
        for face in faces:
            bbox = np.asarray(face.bbox, dtype=np.float32)
            confidence = float(getattr(face, "det_score", getattr(face, "score", 0.0)))
            width, height = bbox_size(bbox)
            if confidence < self.confidence_threshold:
                continue
            if width < self.min_face_size or height < self.min_face_size:
                continue

            landmarks = np.asarray(getattr(face, "kps", None), dtype=np.float32)
            aligned_face = align_face(frame, landmarks, MODEL_INPUT_SIZE)
            if aligned_face is None:
                aligned_face = crop_face(frame, bbox, MODEL_INPUT_SIZE)
            if aligned_face is None:
                continue

            anti_spoof_crops = []
            for scale in self.crop_scales:
                crop = crop_face_by_scale(frame, bbox, MODEL_INPUT_SIZE, scale)
                if crop is not None:
                    anti_spoof_crops.append(AntiSpoofCrop(scale=scale, image=crop))
            if not anti_spoof_crops:
                anti_spoof_crops.append(AntiSpoofCrop(scale=1.0, image=aligned_face))

            detection = FaceDetection(
                bbox=bbox,
                landmarks=landmarks,
                confidence=confidence,
                aligned_face=aligned_face,
                anti_spoof_crops=tuple(anti_spoof_crops),
                image_shape=frame.shape[:2],
            )
            candidates.append((bbox_area(bbox), detection))

        if not candidates:
            return None

        return max(candidates, key=lambda item: item[0])[1]


class AntiSpoofPredictor:
    def __init__(
        self,
        model_path: Path = ONNX_MODEL_PATH,
        input_size: int = ONNX_MODEL_INPUT_SIZE,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Anti-spoofing ONNX model not found: {model_path}")

        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")

        self.input_size = input_size
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.execution_provider = self.session.get_providers()[0]

    def preprocess(self, face_crop_bgr: np.ndarray) -> np.ndarray:
        if face_crop_bgr is None or face_crop_bgr.size == 0:
            raise ValueError("Face crop is empty.")

        face_crop = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
        height, width = face_crop.shape[:2]
        ratio = float(self.input_size) / max(height, width)
        scaled_width = max(1, int(width * ratio))
        scaled_height = max(1, int(height * ratio))
        interpolation = cv2.INTER_LANCZOS4 if ratio > 1.0 else cv2.INTER_AREA
        resized = cv2.resize(face_crop, (scaled_width, scaled_height), interpolation=interpolation)

        pad_w = self.input_size - scaled_width
        pad_h = self.input_size - scaled_height
        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_REFLECT_101)
        tensor = padded.transpose(2, 0, 1).astype(np.float32) / 255.0
        return np.expand_dims(tensor, axis=0)

    def predict(self, face_crop_bgr: np.ndarray) -> AntiSpoofPrediction:
        tensor = self.preprocess(face_crop_bgr)
        logits = self.session.run(None, {self.input_name: tensor})[0]
        if logits.shape[-1] != 2:
            raise ValueError(f"Unexpected anti-spoofing model output shape: {logits.shape}")

        real_logit = float(logits[0][0])
        spoof_logit = float(logits[0][1])
        live_score = 1.0 / (1.0 + np.exp(-(real_logit - spoof_logit)))
        return AntiSpoofPrediction(
            live_score=float(np.clip(live_score, 0.0, 1.0)),
            real_logit=real_logit,
            spoof_logit=spoof_logit,
        )

    def predict_live_score(self, face_crop_bgr: np.ndarray) -> float:
        return self.predict(face_crop_bgr).live_score


class LivenessDetector:
    def __init__(
        self,
        threshold: float = THRESHOLD,
        num_frames: int = NUM_FRAMES,
        face_detector: Optional[FaceDetector] = None,
        predictor: Optional[AntiSpoofPredictor] = None,
    ) -> None:
        if num_frames < 1:
            raise ValueError("num_frames must be at least 1.")

        self.threshold = threshold
        self.num_frames = num_frames
        self.face_detector = face_detector or FaceDetector()
        self.predictor = predictor or AntiSpoofPredictor()
        self._scores: List[float] = []

    @property
    def scores(self) -> List[float]:
        return list(self._scores)

    def reset(self) -> None:
        self._scores.clear()

    def analyze_frame(self, frame: np.ndarray) -> FrameResult:
        face = self.face_detector.detect_largest_face(frame)
        if face is None:
            return FrameResult(False, None, None, None, len(self._scores))

        crop_scores = [self.predictor.predict(crop.image).live_score for crop in face.anti_spoof_crops]
        frame_score = min(crop_scores)
        self._scores.append(frame_score)
        if len(self._scores) > self.num_frames:
            self._scores.pop(0)

        rolling_score = self._compute_rolling_score()
        decision = self.get_decision(rolling_score) if len(self._scores) == self.num_frames else None
        return FrameResult(True, frame_score, rolling_score, decision, len(self._scores), face.bbox)

    def _compute_rolling_score(self) -> float:
        mean_score = float(np.mean(self._scores))
        percentile_score = float(np.percentile(self._scores, DECISION_PERCENTILE))
        return min(mean_score, percentile_score)

    def get_decision(self, rolling_score: float) -> str:
        if len(self._scores) == self.num_frames and min(self._scores) < MIN_FRAME_SCORE:
            return SPOOF_ATTACK
        return REAL_HUMAN if rolling_score >= self.threshold else SPOOF_ATTACK
