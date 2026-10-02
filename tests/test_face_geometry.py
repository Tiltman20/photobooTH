"""Tests for face-aligned coordinates."""

import math

import numpy as np
import pytest

from face_dataclass import Face
from face_geometry import FaceFrame, Region


def face_with_eyes(left, right) -> Face:
    return Face(id=0, x=0, y=0, width=10, height=10, left_eye=left, right_eye=right)


def test_eyes_map_to_unit_positions():
    frame = FaceFrame(face_with_eyes((100, 200), (160, 200)))
    assert frame.eye_distance == pytest.approx(60)
    assert frame.to_image(-0.5, 0) == pytest.approx((100, 200))
    assert frame.to_image(0.5, 0) == pytest.approx((160, 200))
    assert frame.to_image(0, 1) == pytest.approx((130, 260))  # one eye distance below, towards the mouth


def test_tilted_face_keeps_regions_attached():
    angle = math.radians(30)
    left = (200 - 30 * math.cos(angle), 200 - 30 * math.sin(angle))
    right = (200 + 30 * math.cos(angle), 200 + 30 * math.sin(angle))
    frame = FaceFrame(face_with_eyes(left, right))
    assert frame.roll_degrees == pytest.approx(30, abs=0.5)
    point = frame.to_image(0.3, 0.8)
    assert frame.to_face(point) == pytest.approx((0.3, 0.8))


def test_patch_is_upright_and_marks_pixels_outside_the_frame():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[:, 50:] = 255  # right half white
    frame = FaceFrame(face_with_eyes((40, 50), (60, 50)))
    patch = frame.patch(image, Region(-1, -1, 1, 1), pixels_per_unit=10)
    assert patch.image.shape[:2] == (20, 20)
    assert patch.image[10, 2, 0] == 0 and patch.image[10, 17, 0] == 255
    outside = frame.patch(image, Region(-5, -5, 0, 0), pixels_per_unit=10)
    assert 0 < outside.visible_share < 1
