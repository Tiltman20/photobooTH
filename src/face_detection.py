"""Face detection with YuNet (OpenCV): boxes plus five landmarks per face."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import cv2
import numpy as np

from config import FACE_MODEL_PATH

Point = tuple[int, int]

# Lower than YuNet's usual 0.9 so faces further away (group photos) are still found.
FACE_SCORE_THRESHOLD = 0.8


@dataclass
class Face:
    """A detected face with the five YuNet landmarks (pixel coordinates)."""

    id: int
    x: int
    y: int
    width: int
    height: int
    left_eye: Point | None = None
    right_eye: Point | None = None
    nose: Point | None = None
    mouth_left: Point | None = None
    mouth_right: Point | None = None
    score: float = 1.0


class FaceDetector:
    """Thread-safe YuNet wrapper; the input size follows the frame size."""

    def __init__(self, model_path: str = str(FACE_MODEL_PATH), score_threshold: float = FACE_SCORE_THRESHOLD):
        # OpenCV 5 warns about an unsupported DNN target setting on every start; nothing to act on.
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
        self.detector = cv2.FaceDetectorYN.create(
            model=model_path, config="", input_size=(320, 320),
            score_threshold=score_threshold, nms_threshold=0.3, top_k=500,
        )
        self._lock = threading.Lock()

    def detect(self, frame: np.ndarray) -> list[Face]:
        height, width = frame.shape[:2]
        with self._lock:
            self.detector.setInputSize((width, height))
            _, detections = self.detector.detect(frame)
        if detections is None:
            return []
        # YuNet row: x, y, w, h, right eye, left eye, nose, right mouth corner, left mouth corner, score
        faces = []
        for face_id, row in enumerate(detections):
            x, y, w, h, rex, rey, lex, ley, nx, ny, mrx, mry, mlx, mly, score = row[:15]
            faces.append(Face(
                id=face_id, x=int(x), y=int(y), width=int(w), height=int(h),
                right_eye=(int(rex), int(rey)), left_eye=(int(lex), int(ley)), nose=(int(nx), int(ny)),
                mouth_right=(int(mrx), int(mry)), mouth_left=(int(mlx), int(mly)), score=float(score),
            ))
        return faces
