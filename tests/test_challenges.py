"""Tests for the challenge catalogue, codes and synthetic trait checks."""

import cv2
import numpy as np
import pytest

from challenges import Challenge, KINDS, evaluate, parse_code, random_challenge
from face_checks import FrameContext, check_clothing_color, check_glasses, check_moustache
from face_dataclass import Face
from face_geometry import FaceFrame

SKIN = (120, 150, 210)
EYE_DISTANCE = 40


def draw_face(image: np.ndarray, center_x: int, eye_y: int = 120) -> Face:
    """Paint a simple skin-coloured face and return matching YuNet-style landmarks."""
    half = EYE_DISTANCE // 2
    cv2.ellipse(image, (center_x, eye_y + 25), (45, 60), 0, 0, 360, SKIN, -1)
    return Face(id=center_x, x=center_x - 45, y=eye_y - 35, width=90, height=120,
                left_eye=(center_x - half, eye_y), right_eye=(center_x + half, eye_y),
                nose=(center_x, eye_y + 28), mouth_left=(center_x - 16, eye_y + 50),
                mouth_right=(center_x + 16, eye_y + 50))


def blank(width: int = 640, height: int = 480) -> np.ndarray:
    return np.full((height, width, 3), (170, 170, 170), dtype=np.uint8)


def paint_shirt(image, face: Face, bgr) -> None:
    x = face.left_eye[0] + EYE_DISTANCE // 2
    cv2.rectangle(image, (x - 60, 120 + 2 * EYE_DISTANCE + 20), (x + 60, 479), bgr, -1)


def test_codes_round_trip():
    for challenge in [Challenge("glasses", 2), Challenge("color", 3, "blue"), Challenge("rainbow", 3)]:
        assert parse_code(challenge.code) == challenge


@pytest.mark.parametrize("code", ["", "nope", "glasses:9", "color:2", "color:2:pink", "group:x"])
def test_invalid_codes_are_rejected(code):
    assert parse_code(code) is None


def test_random_challenge_respects_kind_and_avoids_repeat():
    first = random_challenge("glasses")
    assert first.kind == "glasses" and first.required in KINDS["glasses"][1]
    for _ in range(30):
        assert random_challenge(avoid_code="group:3").code != "group:3"


def test_every_variant_has_texts():
    for kind, (_, sizes) in KINDS.items():
        for required in sizes:
            option = {"color": "red", "finger_sum": "12"}.get(kind)
            data = Challenge(kind, required, option).to_json()
            assert data["title"] and data["description"] and "None" not in data["title"]


def test_finger_sum_code_round_trip():
    challenge = random_challenge("finger_sum")
    assert parse_code(challenge.code) == challenge and 7 <= challenge.target <= 15
    assert parse_code("finger_sum:1:99") is None


def test_blue_shirt_is_recognised_red_is_not():
    image = blank()
    face = draw_face(image, 320)
    paint_shirt(image, face, (200, 60, 20))  # blue in BGR
    context = FrameContext(image)
    assert check_clothing_color(context, FaceFrame(face), "blue").passed
    assert not check_clothing_color(context, FaceFrame(face), "red").passed


def test_glasses_bridge_raises_score():
    plain, with_glasses = blank(), blank()
    face = draw_face(plain, 320)
    draw_face(with_glasses, 320)
    for eye_x in (300, 340):
        cv2.circle(with_glasses, (eye_x, 120), 14, (30, 30, 30), 3)
    cv2.line(with_glasses, (313, 118), (327, 118), (30, 30, 30), 3)
    plain_score = check_glasses(FrameContext(plain), FaceFrame(face))
    glasses_score = check_glasses(FrameContext(with_glasses), FaceFrame(face))
    assert glasses_score.passed and not plain_score.passed


def test_dark_band_under_nose_is_a_moustache():
    plain, moustache = blank(), blank()
    face = draw_face(plain, 320)
    draw_face(moustache, 320)
    cv2.rectangle(moustache, (302, 158), (338, 166), (40, 40, 50), -1)
    assert check_moustache(FrameContext(moustache), FaceFrame(face)).passed
    assert not check_moustache(FrameContext(plain), FaceFrame(face)).passed


def test_team_colour_needs_enough_people():
    image = blank()
    faces = [draw_face(image, x) for x in (140, 320, 500)]
    for face in faces[:2]:
        paint_shirt(image, face, (200, 60, 20))
    paint_shirt(image, faces[2], (30, 30, 200))  # red
    two = evaluate(Challenge("color", 2, "blue"), image, faces)
    three = evaluate(Challenge("color", 3, "blue"), image, faces)
    assert two.complete and not three.complete
    assert three.progress == {"met": 2, "required": 3}


def test_rainbow_counts_distinct_colours():
    image = blank()
    faces = [draw_face(image, x) for x in (140, 320, 500)]
    for face, bgr in zip(faces, [(200, 60, 20), (30, 30, 200), (40, 180, 40)]):
        paint_shirt(image, face, bgr)
    assert evaluate(Challenge("rainbow", 3), image, faces).complete
    assert not evaluate(Challenge("rainbow", 4), image, faces).complete


def test_group_status_asks_for_more_people():
    faces = [draw_face(blank(), 320)]
    result = evaluate(Challenge("group", 3), blank(), faces)
    assert not result.complete and "holt noch 2 Personen" in result.status_text
