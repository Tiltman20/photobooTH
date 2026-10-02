"""Hand gestures derived from the extended fingers of a hand.

Air-drawing pen:
    index finger only (thumb ignored)   -> DRAW   (pen down)
    all four long fingers (open hand)   -> CLEAR  (erase when held)
    anything else (fist, two fingers..) -> HOVER  (pen up, cursor follows)

Challenge gestures: thumbs up, peace sign, open hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from hand_detection import Hand, Landmark

LONG_FINGERS = frozenset({"index", "middle", "ring", "pinky"})


class PenGesture(str, Enum):
    DRAW = "draw"
    HOVER = "hover"
    CLEAR = "clear"


def classify_pen_gesture(extended_fingers: tuple[str, ...]) -> PenGesture:
    long_fingers = set(extended_fingers) & LONG_FINGERS
    if long_fingers == {"index"}:
        return PenGesture.DRAW
    if long_fingers == LONG_FINGERS:
        return PenGesture.CLEAR
    return PenGesture.HOVER


def select_pen_hand(hands: list[Hand]) -> Hand | None:
    """Pick the hand that controls the pen: a drawing hand first, then the most confident one."""
    if not hands:
        return None
    return max(hands, key=lambda hand: (classify_pen_gesture(hand.extended_fingers) is PenGesture.DRAW,
                                        hand.confidence))


def pen_position(hand: Hand) -> tuple[int, int]:
    """The pen sits on the tip of the index finger."""
    return hand.point(Landmark.INDEX_TIP)


# ------------------------------------------------------------ pen tracking
@dataclass
class _Lock:
    center: np.ndarray
    seen_at: float


class PenTracker:
    """Keeps the pen on one hand so it does not jump between people.

    The first drawing hand gets the pen. In the following frames only the hand
    closest to its last position (within ``max_jump`` of the frame width) may
    move the pen. When that hand has not been seen for ``release_after``
    seconds, any hand can take over.
    """

    def __init__(self, max_jump: float = 0.25, release_after: float = 0.7) -> None:
        self.max_jump = max_jump
        self.release_after = release_after
        self._lock: _Lock | None = None

    def has_owner(self, now: float) -> bool:
        """True while a hand holds the pen (shown to guests as 'Stift belegt')."""
        return self._lock is not None and now - self._lock.seen_at <= self.release_after

    def select(self, hands: list[Hand], frame_width: int, now: float) -> Hand | None:
        if self._lock and now - self._lock.seen_at > self.release_after:
            self._lock = None
        if self._lock is None:
            hand = select_pen_hand(hands)
            if hand is not None and classify_pen_gesture(hand.extended_fingers) is not PenGesture.DRAW:
                return hand  # only a drawing hand claims the pen; others just move the cursor
        else:
            hand = self._closest_to_lock(hands, frame_width)
        if hand is not None:
            self._lock = _Lock(hand_center(hand), now)
        return hand

    def _closest_to_lock(self, hands: list[Hand], frame_width: int) -> Hand | None:
        max_distance = self.max_jump * frame_width
        distances = [(float(np.hypot(*(hand_center(hand) - self._lock.center))), hand) for hand in hands]
        distances = [(distance, hand) for distance, hand in distances if distance <= max_distance]
        return min(distances, key=lambda item: item[0])[1] if distances else None


def hand_center(hand: Hand) -> np.ndarray:
    return hand.landmarks.mean(axis=0)


# ------------------------------------------------------- challenge gestures
def is_thumbs_up(hand: Hand) -> bool:
    """Only the thumb is extended and it points upwards (image y grows downwards)."""
    if set(hand.extended_fingers) != {"thumb"}:
        return False
    tip, base = hand.landmarks[Landmark.THUMB_TIP], hand.landmarks[Landmark.THUMB_MCP]
    palm_size = float(np.hypot(*(hand.landmarks[Landmark.WRIST] - hand.landmarks[Landmark.MIDDLE_MCP])))
    return bool((base[1] - tip[1]) > 0.4 * palm_size)


def is_peace_sign(hand: Hand) -> bool:
    """Index and middle finger up, ring and pinky folded (thumb does not matter)."""
    return set(hand.extended_fingers) & LONG_FINGERS == {"index", "middle"}


def is_open_hand(hand: Hand) -> bool:
    return set(hand.extended_fingers) & LONG_FINGERS == LONG_FINGERS
