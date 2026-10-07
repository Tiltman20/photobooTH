"""Hand and finger detection with the MediaPipe Hand Landmarker.

A neural network finds up to ``max_hands`` hands and returns 21 landmarks per
hand (wrist, finger joints and finger tips). Which fingers are extended is then
decided with simple, rotation-invariant geometry on those landmarks.

Landmark layout (MediaPipe):
    0 wrist | 1-4 thumb | 5-8 index | 9-12 middle | 13-16 ring | 17-20 pinky
    per finger: MCP (base knuckle), PIP, DIP, TIP (thumb: CMC, MCP, IP, TIP)
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from enum import IntEnum

import cv2
import numpy as np

import models

# A finger counts as extended when its tip is this much farther from the wrist than its PIP joint.
FINGER_EXTENDED_RATIO = 1.15
# The thumb must be nearly straight, point away from the palm and not lie on the index finger.
THUMB_MIN_ANGLE_DEG = 150.0
THUMB_AWAY_FROM_PALM_RATIO = 1.1  # tip vs. IP joint distance to the pinky base
THUMB_MIN_DISTANCE_TO_INDEX = 0.45  # in palm sizes (wrist to middle-finger base)

Point = tuple[int, int]


class Landmark(IntEnum):
    WRIST = 0
    THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
    INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
    MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
    RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
    PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20


# (name, PIP joint, tip) for the four long fingers.
LONG_FINGERS = (
    ("index", Landmark.INDEX_PIP, Landmark.INDEX_TIP),
    ("middle", Landmark.MIDDLE_PIP, Landmark.MIDDLE_TIP),
    ("ring", Landmark.RING_PIP, Landmark.RING_TIP),
    ("pinky", Landmark.PINKY_PIP, Landmark.PINKY_TIP),
)
FINGERTIPS = {
    "thumb": Landmark.THUMB_TIP, "index": Landmark.INDEX_TIP, "middle": Landmark.MIDDLE_TIP,
    "ring": Landmark.RING_TIP, "pinky": Landmark.PINKY_TIP,
}
# Bone connections used to draw the hand skeleton.
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)


@dataclass(frozen=True)
class Hand:
    landmarks: np.ndarray  # shape (21, 2), pixel coordinates
    extended_fingers: tuple[str, ...]
    confidence: float

    @property
    def finger_count(self) -> int:
        return len(self.extended_fingers)

    @property
    def fingertips(self) -> tuple[Point, ...]:
        """Pixel positions of the extended fingertips."""
        return tuple(self.point(FINGERTIPS[name]) for name in self.extended_fingers)

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        x_min, y_min = self.landmarks.min(axis=0)
        x_max, y_max = self.landmarks.max(axis=0)
        return int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)

    def point(self, index: int) -> Point:
        x, y = self.landmarks[index]
        return int(x), int(y)


@dataclass(frozen=True)
class HandDetectionResult:
    hands: list[Hand]

    @property
    def total_fingers(self) -> int:
        return sum(hand.finger_count for hand in self.hands)


# ------------------------------------------------------------ finger logic
def find_extended_fingers(landmarks: np.ndarray) -> tuple[str, ...]:
    """Return the names of the extended fingers, from thumb to pinky."""
    extended = ["thumb"] if is_thumb_extended(landmarks) else []
    extended += [name for name, pip, tip in LONG_FINGERS if is_finger_extended(landmarks, pip, tip)]
    return tuple(extended)


def is_finger_extended(landmarks: np.ndarray, pip: int, tip: int) -> bool:
    """A long finger is extended when its tip lies clearly beyond its PIP joint."""
    wrist = landmarks[Landmark.WRIST]
    return _distance(wrist, landmarks[tip]) > FINGER_EXTENDED_RATIO * _distance(wrist, landmarks[pip])


def is_thumb_extended(landmarks: np.ndarray) -> bool:
    """The thumb is extended when it is almost straight and points away from the palm."""
    tip, ip = landmarks[Landmark.THUMB_TIP], landmarks[Landmark.THUMB_IP]
    straightness = _angle_deg(landmarks[Landmark.THUMB_MCP], ip, tip)
    pinky_base = landmarks[Landmark.PINKY_MCP]
    points_away = _distance(tip, pinky_base) > THUMB_AWAY_FROM_PALM_RATIO * _distance(ip, pinky_base)
    palm_size = max(_distance(landmarks[Landmark.WRIST], landmarks[Landmark.MIDDLE_MCP]), 1e-6)
    distance_to_index = _distance(tip, landmarks[Landmark.INDEX_MCP]) / palm_size
    return (straightness >= THUMB_MIN_ANGLE_DEG and points_away
            and distance_to_index >= THUMB_MIN_DISTANCE_TO_INDEX)


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.hypot(*(np.asarray(a, float) - np.asarray(b, float))))


def _angle_deg(a: np.ndarray, vertex: np.ndarray, b: np.ndarray) -> float:
    """Angle a-vertex-b in degrees (180 = straight line)."""
    first, second = np.asarray(a, float) - vertex, np.asarray(b, float) - vertex
    norm = np.linalg.norm(first) * np.linalg.norm(second)
    if norm == 0:
        return 0.0
    return math.degrees(math.acos(float(np.clip(np.dot(first, second) / norm, -1.0, 1.0))))


# ------------------------------------------------------------ detector
class HandDetector:
    """Thread-safe wrapper around the MediaPipe Hand Landmarker (video mode).

    Video mode tracks hands from one frame to the next instead of searching
    each frame from scratch, which makes the detection much more stable for a
    live camera stream.
    """

    def __init__(self, max_hands: int = 6, min_confidence: float = 0.4) -> None:
        # Imported here so the finger logic above stays usable without MediaPipe.
        from mediapipe.tasks.python import BaseOptions, vision

        # The model is passed as bytes: MediaPipe on Windows cannot open paths
        # with special characters (e.g. OneDrive folders), a buffer always works.
        model_bytes = models.ensure(models.HAND_LANDMARKER).read_bytes()
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=model_bytes),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._lock = threading.Lock()
        self._last_timestamp_ms = 0

    def detect(self, frame: np.ndarray) -> HandDetectionResult:
        """Detect hands in a BGR frame and count their extended fingers."""
        import mediapipe as mp

        height, width = frame.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        with self._lock:
            result = self._landmarker.detect_for_video(image, self._next_timestamp_ms())

        hands = []
        for normalized, handedness in zip(result.hand_landmarks, result.handedness):
            landmarks = np.array([(point.x * width, point.y * height) for point in normalized], dtype=float)
            confidence = handedness[0].score if handedness else 0.0
            hands.append(Hand(landmarks, find_extended_fingers(landmarks), float(confidence)))
        return HandDetectionResult(hands)

    def _next_timestamp_ms(self) -> int:
        """Video mode needs strictly increasing timestamps."""
        self._last_timestamp_ms = max(self._last_timestamp_ms + 1, int(time.monotonic() * 1000))
        return self._last_timestamp_ms


def run_self_check(camera_index: int = 0, frames: int = 90) -> None:
    """Print whether the model loads and how many hands the webcam sees (about 3 s)."""
    print("1/3 Lade Modell ...")
    detector = HandDetector()
    print("2/3 Öffne Kamera ... (Hand ins Bild halten)")
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise SystemExit("Kamera konnte nicht geöffnet werden (läuft der Browser noch mit Kamerazugriff?).")
    try:
        for index in range(frames):
            ok, frame = camera.read()
            if not ok:
                raise SystemExit("Kein Kamerabild erhalten.")
            result = detector.detect(frame)
            fingers = [hand.finger_count for hand in result.hands]
            print(f"   Bild {index + 1:2d}: {len(result.hands)} Hand/Hände, Finger je Hand: {fingers}")
    finally:
        camera.release()
    print("3/3 Fertig – das Modell läuft.")


if __name__ == "__main__":
    run_self_check()
