#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

case "${1:-}" in
  webcam)
    python3 -m src.webcam_test
    ;;
  evaluate)
    python3 -m src.evaluate
    ;;
  selfie)
    if [ $# -ne 2 ]; then
      echo "Usage: ./run.sh selfie /path/to/selfie.jpg"
      exit 1
    fi
    python3 -m src.selfie_liveness --image "$2"
    ;;
  app)
    streamlit run app.py --server.address "${STREAMLIT_HOST:-0.0.0.0}" --server.port "${STREAMLIT_PORT:-8501}"
    ;;
  *)
    echo "Usage: ./run.sh {webcam|evaluate|selfie|app}"
    echo "Examples:"
    echo "  ./run.sh webcam"
    echo "  ./run.sh evaluate"
    echo "  ./run.sh selfie /path/to/selfie.jpg"
    echo "  ./run.sh app"
    exit 1
    ;;
esac
