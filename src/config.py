"""Shared paths of the photo booth."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB_ROOT = ROOT / "web"
MODEL_DIR = ROOT / "res" / "models"
FACE_MODEL_PATH = MODEL_DIR / "face_detection_yunet_2026may.onnx"
SETTINGS_FILE = ROOT / "settings.json"
