"""Neural network files: where they come from and one-time download.

Small models are part of the repository; the big CLIP model is downloaded on
the first start (about 180 MB). Run ``python src/models.py`` once with
internet access before the party, then the booth works offline.
"""

from __future__ import annotations

import os
import sys
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from config import MODEL_DIR

_CLIP_REPO = "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/d15189d7028b43f1d3e65039190477f6af591c2a"


@dataclass(frozen=True)
class ModelFile:
    name: str
    url: str
    description: str
    needed_at_runtime: bool = True  # False: only needed to compute new prompt embeddings

    @property
    def path(self) -> Path:
        return MODEL_DIR / self.name


HAND_LANDMARKER = ModelFile(
    "hand_landmarker.task",
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "MediaPipe Hand Landmarker (Hände & Finger)")
FACE_LANDMARKER = ModelFile(
    "face_landmarker.task",
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    "MediaPipe Face Landmarker (Gesichtsausdrücke)")
CLIP_VISION = ModelFile(
    "clip_vision_fp16.onnx", f"{_CLIP_REPO}/onnx/vision_model_fp16.onnx",
    "CLIP ViT-B/32 Bild-Encoder (Brille, Haare, Kleidung …)")
CLIP_TEXT = ModelFile(
    "clip_text_fp16.onnx", f"{_CLIP_REPO}/onnx/text_model_fp16.onnx",
    "CLIP Text-Encoder (nur für neue Prompts)", needed_at_runtime=False)
CLIP_TOKENIZER = ModelFile(
    "clip_tokenizer.json", f"{_CLIP_REPO}/tokenizer.json",
    "CLIP Tokenizer (nur für neue Prompts)", needed_at_runtime=False)

ALL_MODELS = (HAND_LANDMARKER, FACE_LANDMARKER, CLIP_VISION, CLIP_TEXT, CLIP_TOKENIZER)


def ensure(model: ModelFile) -> Path:
    """Return the local path of a model, downloading it once if it is missing."""
    if model.path.is_file():
        return model.path
    model.path.parent.mkdir(parents=True, exist_ok=True)
    temporary = model.path.with_suffix(model.path.suffix + ".download")
    print(f"Lade {model.description} herunter ...")
    try:
        urllib.request.urlretrieve(model.url, temporary, reporthook=_progress)
        print()
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"{model.name} konnte nicht geladen werden ({error}).\n"
                           f"Manuell von {model.url}\nladen und als {model.path} speichern.") from error
    temporary.replace(model.path)
    return model.path


def _progress(blocks: int, block_size: int, total: int) -> None:
    if total > 0:
        done = min(blocks * block_size / total, 1.0)
        print(f"\r  {done:6.1%} von {total / 1e6:.0f} MB", end="", flush=True)


@contextmanager
def quiet_native_stderr():
    """Hide the C++ warnings MediaPipe prints while loading (they cannot be switched off otherwise)."""
    sys.stderr.flush()
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)


def ensure_runtime_models() -> list[str]:
    """Download everything the booth needs; returns warnings for models that are unavailable."""
    warnings = []
    for model in ALL_MODELS:
        if not model.needed_at_runtime:
            continue
        try:
            ensure(model)
        except RuntimeError as error:
            warnings.append(str(error))
    return warnings


if __name__ == "__main__":
    # Pre-download every model (including the text encoder) so the booth runs offline.
    failed = False
    for model in ALL_MODELS:
        try:
            print(f"OK      {ensure(model).name}")
        except RuntimeError as error:
            print(f"FEHLER  {error}")
            failed = True
    sys.exit(1 if failed else 0)
