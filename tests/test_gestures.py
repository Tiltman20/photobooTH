"""Tests for the air-drawing pen gestures."""

import numpy as np
import pytest

from gestures import PenGesture, classify_pen_gesture, pen_position, select_pen_hand
from hand_detection import Hand, Landmark


def make_hand(fingers: tuple[str, ...], confidence: float = .9) -> Hand:
    landmarks = np.arange(42, dtype=float).reshape(21, 2)
    return Hand(landmarks=landmarks, extended_fingers=fingers, confidence=confidence)


@pytest.mark.parametrize("fingers, expected", [
    (("index",), PenGesture.DRAW),
    (("thumb", "index"), PenGesture.DRAW),
    (("index", "middle"), PenGesture.HOVER),
    ((), PenGesture.HOVER),
    (("thumb",), PenGesture.HOVER),
    (("index", "middle", "ring", "pinky"), PenGesture.CLEAR),
    (("thumb", "index", "middle", "ring", "pinky"), PenGesture.CLEAR),
])
def test_classify_pen_gesture(fingers, expected):
    assert classify_pen_gesture(fingers) is expected


def test_drawing_hand_wins_over_more_confident_hand():
    drawing = make_hand(("index",), confidence=.6)
    other = make_hand(("index", "middle"), confidence=.99)
    assert select_pen_hand([other, drawing]) is drawing


def test_most_confident_hand_when_nobody_draws():
    weak, strong = make_hand((), .5), make_hand((), .8)
    assert select_pen_hand([weak, strong]) is strong


def test_no_hand_no_pen():
    assert select_pen_hand([]) is None


def test_pen_is_on_index_tip():
    hand = make_hand(("index",))
    assert pen_position(hand) == tuple(int(v) for v in hand.landmarks[Landmark.INDEX_TIP])


def moved_hand(dx: float, fingers=("index",)) -> Hand:
    hand = make_hand(fingers)
    return Hand(landmarks=hand.landmarks + (dx, 0), extended_fingers=fingers, confidence=.9)


def test_pen_stays_on_the_hand_that_draws():
    from gestures import PenTracker
    tracker = PenTracker(max_jump=0.25, release_after=0.7)
    first, other = moved_hand(100), moved_hand(500)
    assert tracker.select([first, other], 640, now=0.0) is first
    # other hand also points; the pen must stay with the (slightly moved) first hand
    assert tracker.select([moved_hand(500), moved_hand(130)], 640, now=0.1).landmarks[0, 0] == pytest.approx(130)


def test_pen_does_not_jump_when_locked_hand_disappears_briefly():
    from gestures import PenTracker
    tracker = PenTracker(max_jump=0.25, release_after=0.7)
    tracker.select([moved_hand(100)], 640, now=0.0)
    assert tracker.select([moved_hand(500)], 640, now=0.3) is None
    assert tracker.select([moved_hand(500)], 640, now=1.2) is not None  # released after 0.7 s


def test_challenge_gestures():
    from gestures import is_open_hand, is_peace_sign
    assert is_peace_sign(make_hand(("thumb", "index", "middle")))
    assert not is_peace_sign(make_hand(("index",)))
    assert is_open_hand(make_hand(("thumb", "index", "middle", "ring", "pinky")))
