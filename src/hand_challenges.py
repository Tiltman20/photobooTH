"""Hand-based modes: gesture challenges, air drawing and the finger debug view (MediaPipe hand landmarks)."""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

from face_dataclass import Face
from challenges import Challenge, challenge_texts
from face_tracking import FaceTracker, face_pose
from gestures import PenGesture, PenTracker, classify_pen_gesture, is_open_hand, is_peace_sign, is_thumbs_up, pen_position
from hand_detection import HAND_CONNECTIONS, Hand, HandDetectionResult, HandDetector
from overlay import Evaluation, box, circle, line

logger = logging.getLogger("photobooth")


FINGER_NAMES_DE = {"thumb": "Daumen", "index": "Zeigefinger", "middle": "Mittelfinger",
                   "ring": "Ringfinger", "pinky": "kleiner Finger"}


class LazyHandDetector:
    """Creates the neural hand detector on first use and shares it between challenges.

    One shared instance matters: the detector tracks hands across frames, and
    the server starts even when only face-based challenges are played.
    """

    def __init__(self) -> None:
        self._detector: HandDetector | None = None
        self._lock = threading.Lock()

    def get(self) -> HandDetector:
        with self._lock:
            if self._detector is None:
                logger.info("Lade Hand-Modell ...")
                self._detector = HandDetector()
                logger.info("Hand-Modell geladen.")
            return self._detector


class HandDebugEvaluator:
    """Debug view: shows detected hands and finger counts, never completes."""

    def __init__(self, hand_detector: LazyHandDetector) -> None:
        self.hand_detector = hand_detector

    def __call__(self, frame: np.ndarray, faces: list[Face], challenge: dict) -> Evaluation:
        started = time.perf_counter()
        result = self.hand_detector.get().detect(frame)
        analysis_ms = (time.perf_counter() - started) * 1000
        boxes, circles, lines = [], [], []
        for index, hand in enumerate(result.hands):
            x, y, width, height = hand.bbox
            label = f"Hand {index + 1} · {hand.finger_count} Finger"
            boxes.append(box(label, x, y, x + width, y + height, "peach"))
            lines.extend(line(hand.point(a), hand.point(b), "skeleton") for a, b in HAND_CONNECTIONS)
            circles.extend(circle(*hand.point(i), 4, "joint", filled=True) for i in range(len(hand.landmarks)))
            circles.extend(circle(*tip, 9, "tip", filled=True) for tip in hand.fingertips)
        return Evaluation(
            score=float(result.total_fingers),
            complete=False,
            boxes=boxes,
            circles=circles,
            lines=lines,
            status_text=_hand_status(result),
            debug={
                "hands": [{
                    "fingers": hand.finger_count,
                    "fingerNames": [FINGER_NAMES_DE[name] for name in hand.extended_fingers],
                    "confidence": round(hand.confidence, 2),
                } for hand in result.hands],
                "totalFingers": result.total_fingers,
                "analysisMs": round(analysis_ms, 1),
            },
        )


PEN_STATUS = {
    PenGesture.DRAW: "Malt – Zeigefinger führt den Stift",
    PenGesture.HOVER: "Stift oben – nur Zeigefinger ausstrecken zum Malen",
    PenGesture.CLEAR: "Hand offen halten zum Löschen …",
}


class AirDrawEvaluator:
    """Air drawing: reports where the index fingertip is and whether the pen is down.

    A ``PenTracker`` keeps the pen on the hand that started drawing.
    """

    def __init__(self, hand_detector: LazyHandDetector) -> None:
        self.hand_detector = hand_detector
        self.tracker = PenTracker()

    def __call__(self, frame: np.ndarray, faces: list[Face], challenge: dict) -> Evaluation:
        pointer = self.pen_pointer(frame)
        status = PEN_STATUS[PenGesture(pointer["gesture"])] if pointer else None
        return Evaluation(score=float(pointer is not None), complete=False, status_text=status, pointer=pointer)

    def pen_pointer(self, frame: np.ndarray) -> dict | None:
        """Fingertip position and pen gesture of the hand holding the pen, or None."""
        now = time.monotonic()
        hands = self.hand_detector.get().detect(frame).hands
        hand = self.tracker.select(hands, frame.shape[1], now)
        if hand is None:
            return None
        x, y = pen_position(hand)
        gesture = classify_pen_gesture(hand.extended_fingers)
        return {"x": x, "y": y, "gesture": gesture.value, "penOwned": self.tracker.has_owner(now)}


class DrawingChallengeEvaluator:
    """Stencil challenge: pen like air drawing plus tracked face poses.

    The browser maps the stencil drawings onto the faces and checks the
    progress; the server only delivers the measurements.
    """

    def __init__(self, pen: AirDrawEvaluator) -> None:
        self.pen = pen
        self.face_tracker = FaceTracker()

    def __call__(self, challenge: Challenge, frame: np.ndarray, faces: list[Face]) -> Evaluation:
        tracked = self.face_tracker.update(faces, time.monotonic())
        return Evaluation(score=0.0, complete=False, pointer=self.pen.pen_pointer(frame),
                          faces=[face_pose(track_id, face_frame) for track_id, face_frame in tracked])


def _hand_status(result: HandDetectionResult) -> str | None:
    if not result.hands:
        return None
    hand_word = "Hand" if len(result.hands) == 1 else "Hände"
    return f"{len(result.hands)} {hand_word} · {result.total_fingers} Finger gesamt"


# kind: (gesture test, label on a matching hand, progress noun)
GESTURE_CHECKS = {
    "thumbs_up": (is_thumbs_up, "Daumen hoch", "Daumen hoch"),
    "peace": (is_peace_sign, "Peace", "Peace-Zeichen"),
    "high_five": (is_open_hand, "Hand offen", "offene Hände"),
}


class GestureChallengeEvaluator:
    """Hand-gesture challenges: count matching hands, or add up all fingers."""

    def __init__(self, hand_detector: LazyHandDetector) -> None:
        self.hand_detector = hand_detector

    def __call__(self, challenge: Challenge, frame: np.ndarray) -> Evaluation:
        hands = self.hand_detector.get().detect(frame).hands
        if challenge.kind == "finger_sum":
            return self._finger_sum(challenge, hands)
        check, label, noun = GESTURE_CHECKS[challenge.kind]
        matches = [bool(check(hand)) for hand in hands]
        boxes, lines = [], []
        for hand, matched in zip(hands, matches):
            boxes.append(_hand_box(hand, f"{label} ✓" if matched else f"{label}?", matched))
            lines.extend(_skeleton(hand, "success" if matched else "muted"))
        met = int(min(sum(matches), challenge.required))
        complete = bool(met >= challenge.required)
        status = challenge_texts(challenge)["active"] if complete else f"{met}/{challenge.required} {noun}"
        if not complete and len(hands) < challenge.required:
            status += " – holt noch Hände dazu!" if hands else " – Hände in die Kamera!"
        return Evaluation(score=float(met), complete=complete, boxes=boxes, lines=lines, status_text=status,
                          progress={"met": met, "required": challenge.required})

    @staticmethod
    def _finger_sum(challenge: Challenge, hands: list[Hand]) -> Evaluation:
        total = int(sum(hand.finger_count for hand in hands))
        complete = bool(total == challenge.target)
        boxes, lines = [], []
        for hand in hands:
            boxes.append(_hand_box(hand, f"{hand.finger_count} Finger", complete))
            lines.extend(_skeleton(hand, "success" if complete else "lavender"))
        if complete:
            status = challenge_texts(challenge)["active"]
        elif total > challenge.target:
            status = f"{total} Finger – {total - challenge.target} zu viel!"
        else:
            status = f"{total}/{challenge.target} Finger – noch {challenge.target - total}!"
        return Evaluation(score=float(total), complete=complete, boxes=boxes, lines=lines, status_text=status,
                          progress={"met": int(complete), "required": 1})


def _hand_box(hand: Hand, label: str, passed: bool) -> dict:
    x, y, width, height = hand.bbox
    return box(label, x, y, x + width, y + height, "success" if passed else "muted", passed=passed)


def _skeleton(hand: Hand, color: str) -> list[dict]:
    return [line(hand.point(a), hand.point(b), color) for a, b in HAND_CONNECTIONS]


_shared_hand_detector = LazyHandDetector()
GESTURE_EVALUATOR = GestureChallengeEvaluator(_shared_hand_detector)
DRAWING_EVALUATOR = DrawingChallengeEvaluator(AirDrawEvaluator(_shared_hand_detector))
HAND_EVALUATORS = {
    "hands": HandDebugEvaluator(_shared_hand_detector),
    "air_draw": AirDrawEvaluator(_shared_hand_detector),
}
