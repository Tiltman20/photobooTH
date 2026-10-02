"""Shared paths of the photo booth."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB_ROOT = ROOT / "web"
CAPTURE_DIR = ROOT / "captures"
FACE_MODEL_PATH = ROOT / "res" / "models" / "face_detection_yunet_2026may.onnx"
