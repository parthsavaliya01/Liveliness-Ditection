from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import FACE_CROP_SCALE, MODEL_INPUT_SIZE

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

REFERENCE_LANDMARKS = np.array(
    [
        [0.31556875, 0.4615741],
        [0.6826229, 0.4615741],
        [0.5002625, 0.6405054],
        [0.34947187, 0.82469195],
        [0.6534365, 0.82469195],
    ],
    dtype=np.float32,
)


def ensure_folder(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def collect_media_paths(folder: Path) -> List[Path]:
    extensions = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
    return [path for path in sorted(folder.rglob("*")) if path.is_file() and path.suffix.lower() in extensions]


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def is_video_file(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def bbox_area(bbox: Sequence[float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))


def bbox_size(bbox: Sequence[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return max(0.0, float(x2) - float(x1)), max(0.0, float(y2) - float(y1))


def align_face(
    image: np.ndarray,
    landmarks: np.ndarray,
    output_size: int = MODEL_INPUT_SIZE,
) -> Optional[np.ndarray]:
    if landmarks is None or np.asarray(landmarks).shape != (5, 2):
        return None

    source = np.asarray(landmarks, dtype=np.float32)
    target = REFERENCE_LANDMARKS * float(output_size)
    center = np.array([[output_size / 2.0, output_size / 2.0]], dtype=np.float32)
    target = (target - center) * FACE_CROP_SCALE + center

    matrix, _ = cv2.estimateAffinePartial2D(source, target, method=cv2.LMEDS)
    if matrix is None:
        return None

    aligned = cv2.warpAffine(
        image,
        matrix,
        (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return aligned if aligned.size else None


def crop_face(
    image: np.ndarray,
    bbox: Sequence[float],
    output_size: int = MODEL_INPUT_SIZE,
    margin: float = 0.25,
) -> Optional[np.ndarray]:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    box_width = x2 - x1
    box_height = y2 - y1
    if box_width <= 0 or box_height <= 0:
        return None

    pad_x = int(box_width * margin)
    pad_y = int(box_height * margin)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(width, x2 + pad_x)
    y2 = min(height, y2 + pad_y)

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    return cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_LINEAR)


def crop_face_by_scale(
    image: np.ndarray,
    bbox: Sequence[float],
    output_size: int = MODEL_INPUT_SIZE,
    scale: float = 2.7,
) -> Optional[np.ndarray]:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [float(value) for value in bbox]
    box_width = x2 - x1
    box_height = y2 - y1
    if box_width <= 0 or box_height <= 0:
        return None

    side = max(box_width, box_height) * scale
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    crop_x1 = int(round(center_x - side / 2.0))
    crop_y1 = int(round(center_y - side / 2.0))
    crop_x2 = int(round(center_x + side / 2.0))
    crop_y2 = int(round(center_y + side / 2.0))

    pad_left = max(0, -crop_x1)
    pad_top = max(0, -crop_y1)
    pad_right = max(0, crop_x2 - width)
    pad_bottom = max(0, crop_y2 - height)

    padded = image
    if pad_left or pad_top or pad_right or pad_bottom:
        padded = cv2.copyMakeBorder(
            image,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_REPLICATE,
        )

    crop_x1 += pad_left
    crop_x2 += pad_left
    crop_y1 += pad_top
    crop_y2 += pad_top
    crop = padded[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop.size == 0:
        return None

    return cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_LINEAR)


def compute_metrics(true_labels: Iterable[int], predicted_labels: Iterable[int]) -> Dict[str, float]:
    y_true = np.asarray(list(true_labels), dtype=np.int32)
    y_pred = np.asarray(list(predicted_labels), dtype=np.int32)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    total = int(y_true.size)

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1_score = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    far = fp / (fp + tn) if (fp + tn) else 0.0
    frr = fn / (fn + tp) if (fn + tp) else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "far": far,
        "frr": frr,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "total": total,
    }
