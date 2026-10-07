"""Tests for the landmark-based finger logic and the MediaPipe model wrapper."""

import math

import numpy as np
import pytest

import models
from hand_detection import find_extended_fingers

# Base knuckles (MCP) of index, middle, ring and pinky; wrist at the origin, fingers point up (-y).
FINGER_BASES_X = (-0.30, -0.05, 0.18, 0.38)
LONG_FINGER_NAMES = ("index", "middle", "ring", "pinky")


def make_hand(extended: set[str]) -> np.ndarray:
    """Build 21 synthetic landmarks for a right hand with the given fingers extended."""
    points = np.zeros((21, 2))
    thumb_open = "thumb" in extended
    points[1:5] = ([-0.25, -0.20], [-0.50, -0.40], [-0.70, -0.60], [-0.90, -0.80]) if thumb_open \
        else ([-0.25, -0.20], [-0.40, -0.45], [-0.30, -0.70], [-0.10, -0.80])
    for finger_index, (name, base_x) in enumerate(zip(LONG_FINGER_NAMES, FINGER_BASES_X)):
        first = 5 + 4 * finger_index
        if name in extended:
            joints = ([base_x, -1.00], [base_x, -1.40], [base_x, -1.65], [base_x, -1.90])
        else:  # curled into the palm
            joints = ([base_x, -1.00], [base_x, -1.30], [base_x, -1.05], [base_x, -0.85])
        points[first:first + 4] = joints
    return points


def transform(points: np.ndarray, angle_deg: float, scale: float, offset=(320, 240)) -> np.ndarray:
    angle = math.radians(angle_deg)
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    return points @ rotation.T * scale + offset


@pytest.mark.parametrize("extended", [
    set(),
    {"index"},
    {"index", "middle"},
    {"thumb"},
    {"thumb", "index", "pinky"},
    {"index", "middle", "ring", "pinky"},
    {"thumb", "index", "middle", "ring", "pinky"},
])
@pytest.mark.parametrize("angle", [0, 35, 90, 180])
def test_extended_fingers_independent_of_rotation(extended, angle):
    landmarks = transform(make_hand(extended), angle, scale=120)
    assert set(find_extended_fingers(landmarks)) == extended


def test_finger_order_is_thumb_to_pinky():
    landmarks = make_hand({"pinky", "thumb", "middle"}) * 100
    assert find_extended_fingers(landmarks) == ("thumb", "middle", "pinky")


@pytest.mark.skipif(not models.HAND_LANDMARKER.path.is_file(), reason="hand_landmarker.task not downloaded")
def test_model_runs_on_empty_frame():
    pytest.importorskip("mediapipe")
    from hand_detection import HandDetector

    result = HandDetector().detect(np.zeros((480, 640, 3), dtype=np.uint8))
    assert result.hands == []
