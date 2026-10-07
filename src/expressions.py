"""Facial expressions (smile, mouth wide open, kiss) with the MediaPipe Face Landmarker.

The landmarker returns 52 "blendshapes" per face: how strongly each facial
muscle group is activated, from 0 to 1. Each YuNet face is cut out upright and
analysed on its own, so groups and faces far from the camera work as well.
"""

from __future__ import annotations

import logging
import threading

import cv2
import numpy as np

import models
from face_geometry import FaceFrame, Region

logger = logging.getLogger("photobooth")

FACE_CROP = Region(-2.0, -2.2, 2.0, 2.6)
CROP_PIXELS_PER_UNIT = 64  # 256 px wide crops

# expression: blendshapes whose mean is the expression's score
EXPRESSIONS = {
    "smile": ("mouthSmileLeft", "mouthSmileRight"),
    "scream": ("jawOpen",),
    "kiss": ("mouthPucker",),
}


class ExpressionAnalyzer:
    """Thread-safe, lazily loaded Face Landmarker in image mode."""

    def __init__(self) -> None:
        self._landmarker = None
        self._lock = threading.Lock()

    def _get(self):
        if self._landmarker is None:
            from mediapipe.tasks.python import BaseOptions, vision

            logger.info("Lade Gesichtsausdruck-Modell …")
            # Passed as bytes: MediaPipe on Windows cannot open paths with special characters (OneDrive).
            options = vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_buffer=models.ensure(models.FACE_LANDMARKER).read_bytes()),
                running_mode=vision.RunningMode.IMAGE, num_faces=1, output_face_blendshapes=True,
                min_face_detection_confidence=0.3, min_face_presence_confidence=0.3,
            )
            self._landmarker = vision.FaceLandmarker.create_from_options(options)
        return self._landmarker

    def analyze(self, frame: np.ndarray, people: list[FaceFrame]) -> list[dict[str, float] | None]:
        """Per person: expression -> score (0..1), or None when the face could not be read."""
        import mediapipe as mp

        results = []
        with self._lock:
            landmarker = self._get()
            for face_frame in people:
                crop = face_frame.patch(frame, FACE_CROP, CROP_PIXELS_PER_UNIT).image
                rgb = np.ascontiguousarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                detection = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                if not detection.face_blendshapes:
                    results.append(None)
                    continue
                scores = {item.category_name: item.score for item in detection.face_blendshapes[0]}
                results.append({name: float(np.mean([scores.get(key, 0.0) for key in keys]))
                                for name, keys in EXPRESSIONS.items()})
        return results
