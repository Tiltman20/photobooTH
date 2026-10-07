"""Tests for counting people in face challenges, with fake networks (no models needed)."""

import numpy as np
import pytest

from attributes import Answer, unusable
from challenges import Challenge
from face_challenges import FaceChallengeEvaluator, _distinct_assignment
from face_detection import Face
from settings import Thresholds

FRAME = np.zeros((480, 640, 3), np.uint8)


def face_at(x: int, eye_distance: int = 40) -> Face:
    half = eye_distance // 2
    return Face(id=x, x=x - 45, y=85, width=90, height=120, left_eye=(x - half, 120), right_eye=(x + half, 120),
                nose=(x, 148), mouth_left=(x - 16, 170), mouth_right=(x + 16, 170))


class FakeAttributes:
    """Answers per face x position; questions that are not listed get a clear 'no'."""

    def __init__(self, answers: dict[int, dict[str, Answer]]):
        self.answers = answers

    def analyze(self, frame, people, questions, min_torso_visible):
        return [{question: self.answers.get(int(person.origin[0]), {}).get(question, Answer({"yes": 0.0, "no": 1.0}))
                 for question in questions} for person in people]


class FakeExpressions:
    def __init__(self, scores: dict[int, dict[str, float]]):
        self.scores = scores

    def analyze(self, frame, people):
        return [self.scores.get(int(person.origin[0]), {"smile": 0.0, "scream": 0.0, "kiss": 0.0}) for person in people]


def evaluate(code, xs, answers=None, expressions=None):
    evaluator = FaceChallengeEvaluator(FakeAttributes(answers or {}), FakeExpressions(expressions or {}))
    return evaluator(Challenge(*code), FRAME, [face_at(x) for x in xs], Thresholds())


def yes(probability=0.9):
    return Answer({"yes": probability, "no": 1 - probability})


def test_glasses_count_people_above_threshold():
    answers = {100: {"glasses": yes()}, 300: {"glasses": yes(0.2)}, 500: {"glasses": yes()}}
    assert evaluate(("glasses", 2), [100, 300, 500], answers).complete
    result = evaluate(("glasses", 3), [100, 300, 500], answers)
    assert not result.complete and result.progress == {"met": 2, "required": 3}


def test_missing_people_are_asked_for():
    result = evaluate(("beard", 2), [100], {100: {"beard": yes()}})
    assert not result.complete and "holt noch 1 Person" in result.status_text


def test_shirt_colour_team_and_rainbow():
    def shirt(color):
        return {"shirt": Answer({"blue": 0.1, "red": 0.1, "green": 0.1, color: 0.8})}

    answers = {100: shirt("blue"), 300: shirt("blue"), 500: shirt("red")}
    assert evaluate(("color", 2, "blue"), [100, 300, 500], answers).complete
    assert not evaluate(("color", 3, "blue"), [100, 300, 500], answers).complete
    assert not evaluate(("rainbow", 3), [100, 300, 500], answers).complete
    answers[300] = shirt("green")
    assert evaluate(("rainbow", 3), [100, 300, 500], answers).complete


def test_hidden_torso_explains_itself():
    result = evaluate(("color", 1, "red"), [300], {300: {"shirt": unusable("Oberkörper nicht im Bild")}})
    assert not result.complete and "Oberteil" in result.status_text


def test_opposites_need_one_with_and_one_without():
    assert not evaluate(("opposites", 2), [100, 500], {100: {"glasses": yes()}, 500: {"glasses": yes()}}).complete
    assert evaluate(("opposites", 2), [100, 500], {100: {"glasses": yes()}, 500: {"glasses": yes(0.1)}}).complete


def test_expressions_and_mood_mix():
    scores = {100: {"smile": 0.9, "scream": 0.0, "kiss": 0.0},
              300: {"smile": 0.8, "scream": 0.7, "kiss": 0.0},
              500: {"smile": 0.9, "scream": 0.0, "kiss": 0.0}}
    assert evaluate(("smile", 3), [100, 300, 500], expressions=scores).complete
    assert not evaluate(("mood_mix", 3), [100, 300, 500], expressions=scores).complete
    scores[500] = {"smile": 0.0, "scream": 0.0, "kiss": 0.8}
    assert evaluate(("mood_mix", 3), [100, 300, 500], expressions=scores).complete


def test_distinct_assignment_finds_best_matching():
    assert _distinct_assignment([["smile", "scream"], ["smile"], ["kiss"]]) == 3
    assert _distinct_assignment([["smile"], ["smile"], ["smile"]]) == 1
    assert _distinct_assignment([]) == 0


def test_tiny_faces_are_not_judged():
    evaluator = FaceChallengeEvaluator(FakeAttributes({300: {"glasses": yes()}}), FakeExpressions({}))
    result = evaluator(Challenge("glasses", 1), FRAME, [face_at(300, eye_distance=8)], Thresholds())
    assert not result.complete and "näher" in result.status_text


def test_readings_are_smoothed_over_frames():
    answers = {300: {"glasses": yes(0.9)}}
    evaluator = FaceChallengeEvaluator(FakeAttributes(answers), FakeExpressions({}))
    assert evaluator(Challenge("glasses", 1), FRAME, [face_at(300)], Thresholds()).complete
    answers[300]["glasses"] = yes(0.1)  # one bad frame: the smoothed value stays above the threshold
    assert evaluator(Challenge("glasses", 1), FRAME, [face_at(300)], Thresholds()).complete
    assert not evaluator(Challenge("glasses", 1), FRAME, [face_at(300)], Thresholds()).complete


def test_group_counts_faces():
    result = evaluate(("group", 3), [100, 300])
    assert not result.complete and result.progress == {"met": 2, "required": 3}


@pytest.mark.parametrize("code", [("hat", 1), ("hair", 1, "red"), ("scream", 1), ("kiss", 2)])
def test_every_face_kind_evaluates(code):
    result = evaluate(code, [100, 300])
    assert result.status_text and result.boxes


def test_prompt_cache_covers_every_prompt():
    """The booth must not need the CLIP text encoder at runtime: run `python src/attributes.py` after prompt edits."""
    from attributes import _load_cache, all_prompts

    assert set(_load_cache()) == set(all_prompts())
