from pathlib import Path
from typing import Tuple

ROOT_DIR: Path = Path(__file__).resolve().parent.parent
MODELS_DIR: Path = ROOT_DIR / "models"
ONNX_MODEL_PATH: Path = MODELS_DIR / "face_antispoof_quantized.onnx"
DATASET_DIR: Path = ROOT_DIR / "dataset"
RESULTS_DIR: Path = ROOT_DIR / "results"
EVALUATION_REPORT_PATH: Path = RESULTS_DIR / "evaluation_report.txt"

THRESHOLD: float = 0.65
SINGLE_IMAGE_THRESHOLD: float = 0.55
NUM_FRAMES: int = 7
MIN_FRAME_SCORE: float = 0.50
DECISION_PERCENTILE: float = 20.0

WEBCAM_ID: int = 0
WEBCAM_WIDTH: int = 640
WEBCAM_HEIGHT: int = 480

FACE_CONFIDENCE_THRESHOLD: float = 0.50
MIN_FACE_SIZE: int = 80
INSIGHTFACE_DET_SIZE: Tuple[int, int] = (640, 640)

MODEL_INPUT_SIZE: int = 256
ONNX_MODEL_INPUT_SIZE: int = 128
FACE_CROP_SCALE: float = 1.25
ANTI_SPOOF_CROP_SCALES: Tuple[float, ...] = (1.50, 2.70)
SINGLE_IMAGE_CROP_SCALES: Tuple[float, ...] = (1.20, 1.50)
MIN_FACE_OCCUPANCY_RATIO: float = 0.025

DATASET_CLASSES = {
    "real": 1,
    "print_attack": 0,
    "mobile_attack": 0,
    "video_attack": 0,
    "amoled_attack": 0,
    "monitor_attack": 0,
}
