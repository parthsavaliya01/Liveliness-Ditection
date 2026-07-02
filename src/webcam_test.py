import time
from typing import Optional

import cv2

from src.config import NUM_FRAMES, WEBCAM_HEIGHT, WEBCAM_ID, WEBCAM_WIDTH
from src.detector import FrameResult, LivenessDetector, SPOOF_ATTACK


def _text(value: Optional[float]) -> str:
    return "--" if value is None else f"{value:.4f}"


def draw_overlay(frame, result: FrameResult, fps: float) -> None:
    face_status = "Detected" if result.face_detected else "Not Detected"
    average_score = result.average_score if result.valid_frames else None
    status = result.decision or "COLLECTING FRAMES"
    if not result.face_detected:
        status = "WAITING FOR FACE"

    color = (40, 210, 80) if result.decision and result.decision != SPOOF_ATTACK else (40, 40, 230)
    y = 34
    rows = [
        f"FPS: {fps:.1f}",
        f"Face: {face_status}",
        f"Current Score: {_text(result.current_score)}",
        f"Average Score: {_text(average_score)}",
        f"Frames: {result.valid_frames}/{NUM_FRAMES}",
        "STATUS:",
        status,
    ]
    for index, row in enumerate(rows):
        font_scale = 0.78 if index < 6 else 1.0
        thickness = 2 if index < 6 else 3
        row_color = color if index == 6 else (245, 245, 245)
        cv2.putText(frame, row, (22, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, row_color, thickness)
        y += 34 if index < 5 else 42

    if result.bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in result.bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)


def run_webcam() -> None:
    detector = LivenessDetector()
    capture = cv2.VideoCapture(WEBCAM_ID)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_HEIGHT)

    if not capture.isOpened():
        raise RuntimeError(f"Unable to open webcam id {WEBCAM_ID}.")

    window_name = "Face Liveness Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    previous_time = time.perf_counter()
    fps = 0.0
    latest_result = FrameResult(False, None, None, None, 0)

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Unable to read frame from webcam.")

            latest_result = detector.analyze_frame(frame)
            now = time.perf_counter()
            elapsed = now - previous_time
            previous_time = now
            if elapsed > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / elapsed) if fps else 1.0 / elapsed

            draw_overlay(frame, latest_result, fps)
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam()
