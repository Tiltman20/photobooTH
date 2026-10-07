"""Face-based challenges: traits (CLIP) and expressions (Face Landmarker), counted per person.

Faces are tracked across frames and every network output is smoothed per
person, so a single uncertain frame does not break the countdown.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from itertools import permutations

import numpy as np

from attributes import TORSO, Answer, AttributeAnalyzer
from challenges import COLORS, HAIR_COLORS, Challenge, challenge_texts
from expressions import ExpressionAnalyzer
from face_detection import Face
from face_geometry import FaceFrame, Region
from face_tracking import FaceTracker
from overlay import Evaluation, box
from settings import Thresholds

HEAD_BOX = Region(-1.4, -1.8, 1.4, 2.0)
SMOOTHING = 0.5        # weight of the newest reading
FORGET_AFTER_S = 1.0   # older smoothed values are discarded

# kind: (question, answer that counts, threshold field, label on the person)
TRAIT_KINDS = {
    "glasses": ("glasses", "yes", "glasses", "Brille"),
    "beard": ("beard", "yes", "beard", "Bart"),
    "hat": ("hat", "yes", "hat", "Hut"),
}
# expression kind: (score name, threshold field, label)
EXPRESSION_KINDS = {
    "smile": ("smile", "smile", "Grinsen"),
    "scream": ("scream", "scream", "Schrei"),
    "kiss": ("kiss", "kiss", "Kussmund"),
}
MOOD_LABELS = {"smile": "Grinsen", "scream": "Schrei", "kiss": "Kussmund"}
QUESTIONS_FOR = {"glasses": {"glasses"}, "opposites": {"glasses"}, "beard": {"beard"}, "hat": {"hat"},
                 "hair": {"hair"}, "color": {"shirt"}, "rainbow": {"shirt"}}
PROGRESS_NOUNS = {"glasses": "mit Brille", "beard": "mit Bart", "hat": "mit Kopfbedeckung",
                  "hair": "mit passenden Haaren", "color": "in {color}", "rainbow": "Farben", "group": "Leute",
                  "opposites": "Rollen", "smile": "grinsen", "scream": "schreien", "kiss": "Kussmünder",
                  "mood_mix": "Gefühle"}


@dataclass
class Person:
    face: Face
    frame: FaceFrame
    track_id: int
    answers: dict[str, Answer]
    expressions: dict[str, float] | None


@dataclass(frozen=True)
class Reading:
    """What one person contributes: shown as a labelled box with a score bar."""

    label: str
    passed: bool
    meter: float | None = None   # score relative to the threshold (1.0 = threshold)
    color: str | None = None
    region: Region = HEAD_BOX


class FaceChallengeEvaluator:
    def __init__(self, attributes: AttributeAnalyzer | None = None, expressions: ExpressionAnalyzer | None = None):
        self.attributes = attributes or AttributeAnalyzer()
        self.expressions = expressions or ExpressionAnalyzer()
        self.tracker = FaceTracker()
        self._smoothed: dict[tuple[int, str, str], tuple[float, float]] = {}  # (track, group, key) -> (value, time)
        self._lock = threading.Lock()  # tracker and smoothing state are shared by all requests

    def __call__(self, challenge: Challenge, frame: np.ndarray, faces: list[Face], thresholds: Thresholds) -> Evaluation:
        if challenge.kind == "group":
            return _evaluate_group(challenge, faces)
        with self._lock:
            return self._evaluate(challenge, frame, faces, thresholds)

    def _evaluate(self, challenge: Challenge, frame: np.ndarray, faces: list[Face], thresholds: Thresholds) -> Evaluation:
        now = time.monotonic()
        tracked = self.tracker.update(faces, now)
        frames_by_face = {id(face_frame.face): (track_id, face_frame) for track_id, face_frame in tracked}
        small_boxes, people_frames = [], []
        for face in faces:
            if id(face) not in frames_by_face:
                continue
            track_id, face_frame = frames_by_face[id(face)]
            if face_frame.eye_distance < thresholds.min_eye_distance_px:
                small_boxes.append(_box_for(face_frame, Reading("Komm näher!", False), frame.shape))
            else:
                people_frames.append((face, track_id, face_frame))
        people = self._measure(challenge, frame, people_frames, thresholds, now)
        readings = [self._reading(challenge, person, thresholds) for person in people]
        met = _count_met(challenge, people, readings, thresholds)
        boxes = small_boxes + [_box_for(person.frame, reading, frame.shape) for person, reading in zip(people, readings)]
        hints = [reading.label for reading in readings if not reading.passed and reading.meter is None]
        return _finish(challenge, met, boxes, len(faces), hints, too_small=bool(small_boxes))

    # ------------------------------------------------------------ measuring
    def _measure(self, challenge: Challenge, frame: np.ndarray, people_frames, thresholds: Thresholds,
                 now: float) -> list[Person]:
        frames = [face_frame for _, _, face_frame in people_frames]
        questions = QUESTIONS_FOR.get(challenge.kind, set())
        answers = self.attributes.analyze(frame, frames, questions, thresholds.min_torso_visible) \
            if questions else [{} for _ in frames]
        expressions = self.expressions.analyze(frame, frames) if challenge.family == "expression" \
            else [None for _ in frames]
        people = []
        for (face, track_id, face_frame), person_answers, person_expressions in zip(people_frames, answers, expressions):
            smoothed_answers = {name: self._smooth_answer(track_id, name, answer, now)
                                for name, answer in person_answers.items()}
            if person_expressions is not None:
                person_expressions = self._smooth(track_id, "expr", person_expressions, now)
            people.append(Person(face, face_frame, track_id, smoothed_answers, person_expressions))
        self._forget(now)
        return people

    def _smooth_answer(self, track_id: int, question: str, answer: Answer, now: float) -> Answer:
        if not answer.usable:
            return answer
        return Answer(self._smooth(track_id, question, answer.probabilities, now))

    def _smooth(self, track_id: int, group: str, values: dict[str, float], now: float) -> dict[str, float]:
        result = {}
        for key, value in values.items():
            previous = self._smoothed.get((track_id, group, key))
            if previous is not None:
                value = SMOOTHING * value + (1 - SMOOTHING) * previous[0]
            self._smoothed[(track_id, group, key)] = (value, now)
            result[key] = value
        return result

    def _forget(self, now: float) -> None:
        self._smoothed = {key: item for key, item in self._smoothed.items() if now - item[1] <= FORGET_AFTER_S}

    # ------------------------------------------------------------ judging
    def _reading(self, challenge: Challenge, person: Person, thresholds: Thresholds) -> Reading:
        kind = challenge.kind
        if kind in TRAIT_KINDS:
            question, wanted, threshold_name, label = TRAIT_KINDS[kind]
            return _probability_reading(person.answers.get(question), wanted, getattr(thresholds, threshold_name), label)
        if kind == "hair":
            label = f"Haare {HAIR_COLORS[challenge.option]}"
            return _probability_reading(person.answers.get("hair"), challenge.option, thresholds.hair, label)
        if kind == "color":
            reading = _probability_reading(person.answers.get("shirt"), challenge.option, thresholds.shirt,
                                           f"{COLORS[challenge.option]}", region=TORSO)
            return Reading(reading.label, reading.passed, reading.meter,
                           challenge.option if reading.passed else None, TORSO)
        if kind == "rainbow":
            return _rainbow_reading(person.answers.get("shirt"), thresholds.shirt)
        if kind == "opposites":
            answer = person.answers.get("glasses")
            if answer is None or not answer.usable:
                return Reading(answer.reason if answer else "?", False)
            has_glasses = answer.probabilities["yes"] >= thresholds.glasses
            return Reading("Mit Brille" if has_glasses else "Ohne Brille", True,
                           color="success" if has_glasses else "lavender")
        if kind in EXPRESSION_KINDS:
            name, threshold_name, label = EXPRESSION_KINDS[kind]
            return _score_reading(person.expressions, name, getattr(thresholds, threshold_name), label)
        if kind == "mood_mix":
            moods = _moods(person, thresholds)
            if person.expressions is None:
                return Reading("Gesicht?", False)
            return Reading(" + ".join(MOOD_LABELS[mood] for mood in moods) or "Neutral", bool(moods),
                           color="success" if moods else "muted")
        raise ValueError(f"Keine Gesichts-Challenge: {kind}")


# ---------------------------------------------------------------- readings
def _probability_reading(answer: Answer | None, wanted: str, threshold: float, label: str,
                         region: Region = HEAD_BOX) -> Reading:
    if answer is None:
        return Reading("?", False, region=region)
    if not answer.usable:
        return Reading(answer.reason or "?", False, region=region)
    probability = answer.probabilities.get(wanted, 0.0)
    passed = probability >= threshold
    return Reading(label if passed else f"{label}?", passed, probability / threshold, region=region)


def _score_reading(scores: dict[str, float] | None, name: str, threshold: float, label: str) -> Reading:
    if scores is None:
        return Reading("Gesicht?", False)
    passed = scores[name] >= threshold
    return Reading(label if passed else f"{label}?", passed, scores[name] / threshold)


def _rainbow_reading(answer: Answer | None, threshold: float) -> Reading:
    if answer is None or not answer.usable:
        return Reading(answer.reason if answer else "?", False, region=TORSO)
    color, probability = answer.best()
    if probability >= threshold:
        return Reading(f"{COLORS[color]}", True, probability / threshold, color, TORSO)
    return Reading("Farbe?", False, probability / threshold, region=TORSO)


def _moods(person: Person, thresholds: Thresholds) -> list[str]:
    if person.expressions is None:
        return []
    return [name for name, (score_name, threshold_name, _) in EXPRESSION_KINDS.items()
            if person.expressions[score_name] >= getattr(thresholds, threshold_name)]


def _count_met(challenge: Challenge, people: list[Person], readings: list[Reading], thresholds: Thresholds) -> int:
    if challenge.kind == "rainbow":
        return len({reading.color for reading in readings if reading.passed})
    if challenge.kind == "opposites":  # one point each for "with" and "without" glasses
        return len({reading.label for reading in readings if reading.passed})
    if challenge.kind == "mood_mix":
        return _distinct_assignment([_moods(person, thresholds) for person in people])
    return sum(reading.passed for reading in readings)


def _distinct_assignment(options: list[list[str]]) -> int:
    """Largest number of people that can each show a *different* item from their list."""
    items = sorted({item for person_items in options for item in person_items})
    best = 0
    for order in permutations(range(len(options)), min(len(options), len(items))):
        best = max(best, sum(item in options[person] for person, item in zip(order, items)))
    return best


# ------------------------------------------------------------------ output
def _box_for(face_frame: FaceFrame, reading: Reading, frame_shape) -> dict:
    x1, y1, x2, y2 = face_frame.bounding_box(reading.region, frame_shape)
    color = reading.color or ("success" if reading.passed else "muted")
    return box(reading.label, x1, y1, x2, y2, color, meter=reading.meter, passed=reading.passed)


def _evaluate_group(challenge: Challenge, faces: list[Face]) -> Evaluation:
    boxes = [box(f"Person {index + 1}", face.x, face.y, face.x + face.width, face.y + face.height,
                 "success", passed=True) for index, face in enumerate(sorted(faces, key=lambda face: face.x))]
    return _finish(challenge, len(faces), boxes, len(faces), [], too_small=False)


def _finish(challenge: Challenge, met: int, boxes: list[dict], face_count: int, hints: list[str],
            too_small: bool) -> Evaluation:
    required = challenge.required
    met = min(met, required)
    complete = met >= required
    return Evaluation(score=float(met), complete=complete, boxes=boxes,
                      status_text=_status_text(challenge, met, face_count, hints, complete, too_small),
                      progress={"met": met, "required": required})


def _status_text(challenge: Challenge, met: int, face_count: int, hints: list[str], complete: bool,
                 too_small: bool) -> str:
    if complete:
        return challenge_texts(challenge)["active"]
    if face_count == 0:
        return "Niemand zu sehen – stellt euch vor die Kamera!"
    noun = PROGRESS_NOUNS[challenge.kind].format(color=COLORS.get(challenge.option or "", ""))
    text = f"{met}/{challenge.required} {noun}"
    if challenge.required > 1 and face_count < challenge.required:
        missing = challenge.required - face_count
        text += f" – holt noch {missing} {'Person' if missing == 1 else 'Leute'} dazu!"
    elif too_small:
        text += " – kommt näher ran!"
    elif "Oberkörper nicht im Bild" in hints:
        text += " – geht etwas zurück, das Oberteil muss ins Bild"
    return text
