"""Tests for hand-gesture challenges with synthetic hands (no camera, no model)."""

import json
from dataclasses import asdict

import numpy as np

from challenges import Challenge
from gestures import is_thumbs_up
from hand_challenges import GestureChallengeEvaluator
from hand_detection import Hand, HandDetectionResult, Landmark

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


def hand(fingers: tuple[str, ...], thumb_up: bool = False) -> Hand:
    landmarks = np.tile([300.0, 300.0], (21, 1))
    landmarks[Landmark.WRIST] = (300, 380)
    landmarks[Landmark.MIDDLE_MCP] = (300, 300)
    landmarks[Landmark.THUMB_MCP] = (270, 330)
    landmarks[Landmark.THUMB_TIP] = (265, 250) if thumb_up else (265, 360)
    landmarks[Landmark.PINKY_TIP] = (340, 260)
    return Hand(landmarks=landmarks, extended_fingers=fingers, confidence=.9)


class FakeDetector:
    def __init__(self, hands):
        self.hands = hands

    def get(self):
        return self

    def detect(self, frame):
        return HandDetectionResult(self.hands)


def evaluate(challenge, hands):
    return GestureChallengeEvaluator(FakeDetector(hands))(challenge, FRAME)


def test_thumbs_up_needs_thumb_pointing_up():
    assert is_thumbs_up(hand(("thumb",), thumb_up=True))
    assert not is_thumbs_up(hand(("thumb",), thumb_up=False))
    assert not is_thumbs_up(hand(("thumb", "index"), thumb_up=True))


def test_thumbs_up_counts_matching_hands():
    hands = [hand(("thumb",), thumb_up=True), hand(("thumb",), thumb_up=True), hand(("index",))]
    assert evaluate(Challenge("thumbs_up", 2), hands).complete
    result = evaluate(Challenge("thumbs_up", 3), hands)
    assert not result.complete and result.progress == {"met": 2, "required": 3}


def test_peace_and_open_hands():
    peace = hand(("index", "middle"))
    open_hand = hand(("thumb", "index", "middle", "ring", "pinky"))
    assert evaluate(Challenge("peace", 2), [peace, peace]).complete
    assert not evaluate(Challenge("high_five", 4), [open_hand] * 3).complete
    assert evaluate(Challenge("high_five", 4), [open_hand] * 4).complete


def test_finger_sum_must_match_exactly():
    five = hand(("thumb", "index", "middle", "ring", "pinky"))
    two = hand(("index", "middle"))
    assert evaluate(Challenge("finger_sum", 1, "12"), [five, five, two]).complete
    too_many = evaluate(Challenge("finger_sum", 1, "7"), [five, five])
    assert not too_many.complete and "zu viel" in too_many.status_text


def test_results_are_json_serialisable():
    """Regression: numpy.bool_ in a result crashed the server response."""
    hands = [hand(("thumb",), thumb_up=True), hand(("index", "middle"))]
    for challenge in [Challenge("thumbs_up", 2), Challenge("peace", 2), Challenge("finger_sum", 1, "7")]:
        result = evaluate(challenge, hands)
        json.dumps(asdict(result))
        assert type(result.complete) is bool


def test_server_json_default_handles_numpy():
    from web_server import json_default
    assert json.dumps({"a": np.bool_(True), "b": np.int64(3), "c": np.array([1, 2])}, default=json_default) \
        == '{"a": true, "b": 3, "c": [1, 2]}'
