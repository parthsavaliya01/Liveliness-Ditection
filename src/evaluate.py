from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from src.config import DATASET_CLASSES, DATASET_DIR, EVALUATION_REPORT_PATH, NUM_FRAMES, THRESHOLD
from src.detector import AntiSpoofPredictor, FaceDetector
from src.utils import collect_media_paths, compute_metrics, ensure_folder, is_image_file, is_video_file


@dataclass(frozen=True)
class SampleResult:
    path: Path
    label: int
    prediction: int
    score: float
    valid_frames: int


class DatasetEvaluator:
    def __init__(self, threshold: float = THRESHOLD, num_frames: int = NUM_FRAMES) -> None:
        self.threshold = threshold
        self.num_frames = num_frames
        self.face_detector = FaceDetector()
        self.predictor = AntiSpoofPredictor()

    def _score_frame(self, frame: np.ndarray) -> Optional[float]:
        face = self.face_detector.detect_largest_face(frame)
        if face is None:
            return None
        scores = [self.predictor.predict(crop.image).live_score for crop in face.anti_spoof_crops]
        return min(scores) if scores else None

    def _score_image(self, path: Path) -> Optional[float]:
        image = cv2.imread(str(path))
        if image is None:
            return None
        return self._score_frame(image)

    def _score_video(self, path: Path) -> Optional[float]:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            return None

        scores: List[float] = []
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        stride = max(1, frame_count // max(self.num_frames * 4, 1)) if frame_count else 1
        frame_index = 0

        try:
            while len(scores) < self.num_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index % stride == 0:
                    score = self._score_frame(frame)
                    if score is not None:
                        scores.append(score)
                frame_index += 1
        finally:
            capture.release()

        return float(np.mean(scores)) if scores else None

    def _result_from_score(self, path: Path, label: int, score: float, valid_frames: int) -> SampleResult:
        prediction = 1 if score >= self.threshold else 0
        return SampleResult(path, label, prediction, score, valid_frames)

    def evaluate_video_file(self, path: Path, label: int) -> SampleResult:
        if is_video_file(path):
            score = self._score_video(path)
            valid_frames = self.num_frames if score is not None else 0
        else:
            score = None
            valid_frames = 0

        if score is None:
            score = 0.0

        return self._result_from_score(path, label, score, valid_frames)

    def evaluate_image_sequence(self, paths: List[Path], label: int) -> List[SampleResult]:
        results: List[SampleResult] = []
        score_window: List[float] = []
        last_path: Optional[Path] = None

        for path in paths:
            score = self._score_image(path)
            if score is None:
                continue
            score_window.append(score)
            last_path = path
            if len(score_window) == self.num_frames:
                average_score = float(np.mean(score_window))
                results.append(self._result_from_score(last_path, label, average_score, self.num_frames))
                score_window.clear()

        return results

    def evaluate(self) -> Dict[str, object]:
        results: List[SampleResult] = []
        missing_folders: List[str] = []
        skipped_image_frames = 0

        for folder_name, label in DATASET_CLASSES.items():
            folder = DATASET_DIR / folder_name
            if not folder.exists():
                missing_folders.append(folder_name)
                continue
            media_paths = collect_media_paths(folder)
            image_paths = [path for path in media_paths if is_image_file(path)]
            video_paths = [path for path in media_paths if is_video_file(path)]

            image_results = self.evaluate_image_sequence(image_paths, label)
            results.extend(image_results)
            skipped_image_frames += len(image_paths) - len(image_results) * self.num_frames

            for video_path in video_paths:
                results.append(self.evaluate_video_file(video_path, label))

        metrics = compute_metrics([item.label for item in results], [item.prediction for item in results])
        metrics["samples"] = len(results)
        metrics["missing_folders"] = missing_folders
        metrics["skipped_image_frames"] = skipped_image_frames
        metrics["per_folder"] = self._per_folder_metrics(results)
        return metrics

    def _per_folder_metrics(self, results: List[SampleResult]) -> Dict[str, Dict[str, float]]:
        output: Dict[str, Dict[str, float]] = {}
        for folder_name in DATASET_CLASSES:
            folder_results = [item for item in results if item.path.parent.name == folder_name]
            if not folder_results:
                output[folder_name] = {"samples": 0, "average_score": 0.0}
                continue
            output[folder_name] = {
                "samples": len(folder_results),
                "average_score": float(np.mean([item.score for item in folder_results])),
            }
        return output


def save_report(metrics: Dict[str, object], output_path: Path = EVALUATION_REPORT_PATH) -> None:
    ensure_folder(output_path.parent)
    per_folder = metrics.get("per_folder", {})
    missing_folders = metrics.get("missing_folders", [])

    lines = [
        "Face Liveness Evaluation Report",
        "===============================",
        f"Threshold: {THRESHOLD:.4f}",
        f"Required Webcam Frames: {NUM_FRAMES}",
        f"Decision Samples Evaluated: {metrics.get('samples', 0)}",
        f"Skipped Image Frames: {metrics.get('skipped_image_frames', 0)}",
        f"Accuracy: {metrics['accuracy']:.4f}",
        f"Precision: {metrics['precision']:.4f}",
        f"Recall: {metrics['recall']:.4f}",
        f"F1 Score: {metrics['f1_score']:.4f}",
        f"False Accept Rate (FAR): {metrics['far']:.4f}",
        f"False Reject Rate (FRR): {metrics['frr']:.4f}",
        f"True Positives: {metrics['tp']}",
        f"True Negatives: {metrics['tn']}",
        f"False Positives: {metrics['fp']}",
        f"False Negatives: {metrics['fn']}",
        "",
        "Per Folder",
        "----------",
    ]

    for folder_name in DATASET_CLASSES:
        folder_metrics = per_folder.get(folder_name, {"samples": 0, "average_score": 0.0})
        lines.append(
            f"{folder_name}: samples={folder_metrics['samples']} "
            f"average_score={folder_metrics['average_score']:.4f}"
        )

    if missing_folders:
        lines.extend(["", "Missing Folders", "---------------", ", ".join(str(item) for item in missing_folders)])

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    evaluator = DatasetEvaluator()
    metrics = evaluator.evaluate()
    save_report(metrics)
    print(f"Evaluation complete. Report saved to {EVALUATION_REPORT_PATH}")


if __name__ == "__main__":
    main()
