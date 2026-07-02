# Face Liveness Detection

A polished, deployable Streamlit application for detecting whether a face is a live human or a spoofed presentation using an existing ONNX-based anti-spoofing model.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

This project preserves the existing machine learning pipeline and model behavior while upgrading the experience into a modern, production-ready web app. Users can either capture a photo with their camera or upload a local image and receive a clear live/spoof prediction with confidence and timing details.

## Features

- Modern Streamlit UI with responsive layout
- Camera capture for instant inference
- Image upload support for JPG and PNG
- Prediction card with confidence, processing time, and timestamp
- Existing ONNX model and inference logic preserved
- Clean modular architecture for maintainability
- Docker and deployment support
- Friendly validation and error handling

## Project Structure

```text
face-liveness-detection/
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── services.py
│   ├── ui.py
│   └── utils.py
├── app.py
├── models/
│   └── face_antispoof_quantized.onnx
├── src/
│   ├── config.py
│   ├── detector.py
│   ├── evaluate.py
│   ├── selfie_liveness.py
│   ├── utils.py
│   └── webcam_test.py
├── tests/
│   └── test_app_utils.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── run.sh
├── .gitignore
└── README.md
```

## Installation

```bash
cd liveness_project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running Locally

### Streamlit app

```bash
./run.sh app
```

Then open your browser at:

```text
http://localhost:8501
```

### Existing CLI options

```bash
./run.sh webcam
./run.sh selfie /path/to/image.jpg
./run.sh evaluate
```

## Docker Setup

```bash
docker build -t face-liveness .
docker run -p 8501:8501 face-liveness
```

Or with Docker Compose:

```bash
docker compose up --build
```

## Deployment

This app is compatible with:

- Streamlit Community Cloud
- Render
- Railway
- Hugging Face Spaces
- Docker-based hosting

## Usage

1. Launch the app.
2. Choose either camera capture or image upload.
3. Preview the selected image.
4. Click the prediction button.
5. Review the live/spoof result with confidence and timing.

## Technologies Used

- Python
- Streamlit
- OpenCV
- InsightFace
- ONNX Runtime
- Docker

## Model Information

The app uses the pre-existing ONNX model at:

```text
models/face_antispoof_quantized.onnx
```

The model and inference behavior were preserved; only the application interface and deployment structure were improved.

## Screenshots

Add screenshots to the repository screenshots/ folder for a polished portfolio experience.

## Future Improvements

- Add result history and analytics
- Support batch processing
- Improve mobile camera handling
- Add dark/light theme toggle
- Add model confidence explanations

## License

This project is licensed under the MIT License.

If the detected face is too small in single-image mode, the output can include:

```text
Face too far from camera. Please move closer.
```

Run dataset evaluation:

```bash
./run.sh evaluate
```

Expected result:

- Media files under `dataset/` are scored.
- A metrics report is written to `results/evaluation_report.txt`.
- The terminal prints the report path after completion.

## Final Project Structure

The cleaned project keeps only source code, the anti-spoof model, dependency metadata, runtime entry points, evaluation folders, and the results placeholder:

```text
liveness_project/
├── README.md
├── requirements.txt
├── run.sh
├── dataset/
│   ├── amoled_attack/.gitkeep
│   ├── mobile_attack/.gitkeep
│   ├── monitor_attack/.gitkeep
│   ├── print_attack/.gitkeep
│   ├── real/.gitkeep
│   └── video_attack/.gitkeep
├── models/
│   └── face_antispoof_quantized.onnx
├── results/
│   └── .gitkeep
└── src/
    ├── __init__.py
    ├── config.py
    ├── detector.py
    ├── evaluate.py
    ├── selfie_liveness.py
    ├── utils.py
    └── webcam_test.py
```
