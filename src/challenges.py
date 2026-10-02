"""Photo challenges: catalogue, random variants and per-frame evaluation.

A challenge is identified by a compact code that the browser sends back with
every frame, e.g. ``glasses:2`` (two people with glasses), ``color:3:blue``
(three people in blue), ``rainbow:3`` or ``finger_sum:1:12`` (show exactly
12 fingers together). The server therefore needs no session state to know
what to check. Face challenges are evaluated here, hand-gesture challenges in
``hand_challenges``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

import numpy as np

from face_checks import (FrameContext, TraitResult, check_clothing_color, check_glasses, check_moustache,
                         check_red_hair, dominant_clothing_color, face_frame_or_none)
from face_dataclass import Face
from face_geometry import FaceFrame
from overlay import Evaluation, box

COLORS = {  # id: (display name, box colour key)
    "blue": "Blau", "red": "Rot", "green": "Grün", "yellow": "Gelb", "white": "Weiß", "black": "Schwarz",
}

# kind: (dropdown label, possible numbers of people / hands)
KINDS = {
    "glasses": ("Brillen-Challenge", (1, 2, 3)),
    "mustache": ("Schnurrbart-Challenge", (1, 2)),
    "red_hair": ("Rote Haare", (1,)),
    "color": ("Farb-Team", (1, 2, 3)),
    "rainbow": ("Regenbogen-Crew", (3, 4)),
    "opposites": ("Gegensätze: Brille & ohne", (2,)),
    "group": ("Gruppenfoto", (3, 4, 5)),
    "thumbs_up": ("Daumen hoch", (2, 3, 4)),
    "peace": ("Peace-Zeichen", (2, 3, 4)),
    "high_five": ("Alle Hände hoch", (4, 6)),
    "finger_sum": ("Finger-Summe", (1,)),
    "stencils": ("Masken-Werkstatt (Malen)", (2, 3)),
}
GESTURE_KINDS = frozenset({"thumbs_up", "peace", "high_five", "finger_sum"})
# Drawing challenges: the server reports pen and face poses, the browser checks the drawing.
DRAWING_KINDS = frozenset({"stencils"})
# theme id: (title, what to draw)
STENCIL_THEMES = {
    "hats": ("Hut-Party", "malt jeder Schablone einen Hut über den Kopf"),
    "horns": ("Teufelshörner", "malt Hörner auf die Stirn"),
    "moustache": ("Schnurrbart-Salon", "malt Schnurrbärte unter die Nase"),
    "free": ("Freestyle", "malt, was ihr wollt: Hüte, Hörner, Brillen, Bärte …"),
}
DRAWING_OPTIONS = {"airDraw": True, "analysisIntervalMs": 15, "analysisWidth": 480}
# More than ten fingers needs a second person, which is the point.
FINGER_SUM_TARGETS = tuple(range(7, 16))
MULTI_PERSON_WEIGHT = 2  # multi-person variants are drawn more often: meeting people is the point
KIND_WEIGHTS = {"red_hair": 0.4}  # rare trait: drawn less often so rounds stay solvable

# Modes without automatic photo; they are never drawn at random.
FREE_MODES = {
    "air_draw": {
        "id": "air_draw", "kind": "air_draw", "title": "Luftmalerei",
        "description": "Zeigefinger ausstrecken = malen. Zwei Finger oder Faust = Stift absetzen. "
                       "Hand öffnen und kurz halten = alles löschen. "
                       "Mit „Foto jetzt aufnehmen“ landet dein Kunstwerk im Foto.",
        "ready": "Kamera bereit – Zeigefinger ausstrecken", "waiting": "Zeig deinen Zeigefinger",
        "active": "Malt", "boothName": "air draw", "manualCapture": True,
        "autoCapture": False, "airDraw": True, "required": 1,
        # Fast, small analysis frames give more fingertip samples per second = rounder lines.
        "analysisIntervalMs": 15, "analysisWidth": 480,
    },
    "hands": {
        "id": "hands", "kind": "hands", "title": "Debug: Hand- und Fingererkennung",
        "description": "Halte eine oder mehrere Hände ins Bild und zeig eine Fingerzahl. "
                       "Diese Ansicht nimmt kein Foto automatisch auf.",
        "ready": "Kamera bereit – Hände ins Bild halten", "waiting": "Keine Hand erkannt",
        "active": "Hand erkannt", "boothName": "hand debug",
        "autoCapture": False, "showAnalysedFrame": True, "required": 1,
    },
}


@dataclass(frozen=True)
class Challenge:
    kind: str
    required: int = 1
    option: str | None = None  # colour id for "color", finger count for "finger_sum", theme for "stencils"

    @property
    def code(self) -> str:
        parts = [self.kind, str(self.required)] + ([self.option] if self.option else [])
        return ":".join(parts)

    @property
    def color(self) -> str | None:
        return self.option if self.kind == "color" else None

    @property
    def target(self) -> int | None:
        return int(self.option) if self.kind == "finger_sum" and self.option else None

    def to_json(self) -> dict:
        texts = challenge_texts(self)
        data = {"id": self.code, "kind": self.kind, "required": self.required, "color": self.color,
                "target": self.target, "boothName": f"{self.kind.replace('_', ' ')} booth", **texts}
        if self.kind in DRAWING_KINDS:
            # Drawing takes as long as it takes: the guests start the photo themselves.
            data |= DRAWING_OPTIONS | {"stencilMode": True, "manualCapture": True}
        return data


def options_for(kind: str) -> list[str | None]:
    if kind == "color":
        return list(COLORS)
    if kind == "finger_sum":
        return [str(target) for target in FINGER_SUM_TARGETS]
    if kind == "stencils":
        return list(STENCIL_THEMES)
    return [None]


def parse_code(code: str | None) -> Challenge | None:
    """Rebuild a challenge from its code; invalid codes return None."""
    if not code:
        return None
    parts = code.split(":")
    kind = parts[0]
    if kind not in KINDS:
        return None
    try:
        required = int(parts[1]) if len(parts) > 1 else KINDS[kind][1][0]
    except ValueError:
        return None
    if required not in KINDS[kind][1]:
        return None
    option = parts[2] if len(parts) > 2 else None
    if option not in options_for(kind):
        return None
    return Challenge(kind, required, option)


def random_challenge(kind: str | None = None, avoid_code: str | None = None) -> Challenge:
    """Draw a variant; multi-person variants are preferred and the previous code is avoided."""
    kinds = [kind] if kind in KINDS else list(KINDS)
    candidates, weights = [], []
    for name in kinds:
        options = options_for(name)
        for required in KINDS[name][1]:
            for option in options:
                candidates.append(Challenge(name, required, option))
                weight = (MULTI_PERSON_WEIGHT if required > 1 else 1) * KIND_WEIGHTS.get(name, 1.0)
                weights.append(weight / len(options))
    if len(candidates) > 1 and avoid_code:
        keep = [index for index, challenge in enumerate(candidates) if challenge.code != avoid_code]
        candidates, weights = [candidates[i] for i in keep], [weights[i] for i in keep]
    return random.choices(candidates, weights=weights, k=1)[0]


def challenge_catalogue() -> list[dict]:
    """Entries for the dropdown: all kinds plus the free modes."""
    entries = [{"id": kind, "title": label} for kind, (label, _) in KINDS.items()]
    entries += [{"id": mode["id"], "title": mode["title"]} for mode in FREE_MODES.values()]
    return entries


# ------------------------------------------------------------------ texts
def challenge_texts(challenge: Challenge) -> dict[str, str]:
    n = challenge.required
    color_name = COLORS.get(challenge.color or "", "")
    theme_title, theme_hint = STENCIL_THEMES.get(challenge.option or "", STENCIL_THEMES["free"])
    solo = n == 1
    texts = {
        "glasses": (
            "Zeig uns dein schönstes Brillengesicht." if solo else f"Brillen-Crew: {n} Leute mit Brille",
            "Sobald deine Brille erkannt wird, startet der Countdown." if solo
            else f"Findet {n} Leute mit Brille und stellt euch zusammen ins Bild – egal ob ihr euch kennt.",
            "Brille erkannt" if solo else "Crew komplett",
        ),
        "mustache": (
            "Zeig uns deinen Schnurrbart." if solo else f"Bart-Bande: {n} Schnurrbärte",
            "Sobald ein Schnurrbart erkannt wird, startet der Countdown." if solo
            else f"Sucht euch {n} Leute mit Schnurrbart zusammen. Aufgemalt zählt auch!",
            "Schnurrbart erkannt" if solo else "Bande komplett",
        ),
        "red_hair": (
            "Zeig uns deine rote Mähne.",
            "Sobald rote Haare erkannt werden, startet der Countdown. Kein Rotschopf dabei? Findet einen!",
            "Rote Haare erkannt",
        ),
        "color": (
            f"Farb-Challenge: {color_name}" if solo else f"Team {color_name}: {n} Leute",
            f"Zeig uns ein Kleidungsstück in {color_name}." if solo
            else f"Findet {n} Leute, die {color_name} tragen, und stellt euch zusammen ins Bild.",
            f"{color_name} erkannt" if solo else f"Team {color_name} komplett",
        ),
        "rainbow": (
            f"Regenbogen-Crew: {n} Farben",
            f"{n} Leute, {n} verschiedene Oberteil-Farben (Blau, Rot, Grün, Gelb, Weiß, Schwarz). "
            "Wer trägt was? Findet es heraus!",
            "Regenbogen komplett",
        ),
        "opposites": (
            "Gegensätze ziehen sich an",
            "Eine Person mit Brille und eine ohne – am besten jemand, den du noch nicht kennst.",
            "Gegensätze gefunden",
        ),
        "group": (
            f"Gruppenfoto: mindestens {n} Leute",
            f"Sobald mindestens {n} Personen im Bild sind, startet der Countdown.",
            "Gruppe vollständig",
        ),
        "thumbs_up": (
            f"{n} Daumen hoch!",
            f"Zeigt zusammen {n} Daumen nach oben – ab drei braucht ihr eine zweite Person.",
            "Daumen komplett",
        ),
        "peace": (
            f"Peace! {n} Victory-Zeichen",
            f"Zeigt zusammen {n}× das Peace-Zeichen (Zeige- und Mittelfinger).",
            "Peace komplett",
        ),
        "high_five": (
            f"Alle Hände hoch: {n} offene Hände",
            f"Streckt zusammen {n} offene Hände in die Kamera – das sind mindestens {(n + 1) // 2} Leute.",
            "Hände komplett",
        ),
        "stencils": (
            f"Masken-Werkstatt: {theme_title}",
            f"Im Bild sind {n} große Gesichts-Schablonen – {theme_hint}. Schablone 1 trägt die Person ganz links, "
            "Schablone 2 die nächste. Fertig? „Masken aufsetzen & Foto“ drücken – dann sitzen die Masken "
            "auf euren echten Gesichtern.",
            "Alle verkleidet!",
        ),
        "finger_sum": (
            f"Finger-Summe: genau {challenge.target}",
            f"Zeigt zusammen genau {challenge.target} Finger – nicht mehr, nicht weniger. "
            "Sprecht euch ab, wer wie viele zeigt!",
            f"Genau {challenge.target} Finger!",
        ),
    }
    title, description, active = texts[challenge.kind]
    return {"title": title, "description": description, "active": active,
            "ready": "Kamera bereit", "waiting": "Warte auf Gesichter"}


# ------------------------------------------------------------- evaluation
TraitCheck = Callable[[FrameContext, FaceFrame], TraitResult]
TRAIT_CHECKS: dict[str, tuple[TraitCheck, str]] = {  # kind: (check, label for a person who has it)
    "glasses": (check_glasses, "Brille"),
    "mustache": (check_moustache, "Schnurrbart"),
    "red_hair": (check_red_hair, "Rote Haare"),
}
PROGRESS_NOUNS = {"glasses": "mit Brille", "mustache": "mit Schnurrbart", "red_hair": "mit roten Haaren",
                  "color": "in {color}", "rainbow": "Farben", "group": "Personen", "opposites": "Rollen"}


def evaluate(challenge: Challenge, frame: np.ndarray, faces: list[Face]) -> Evaluation:
    if challenge.kind in GESTURE_KINDS | DRAWING_KINDS:
        raise ValueError("Hand-gesture and drawing challenges are evaluated in hand_challenges")
    if challenge.kind == "group":
        return _evaluate_group(challenge, faces)
    context = FrameContext(frame)
    people = [(face, face_frame) for face in faces if (face_frame := face_frame_or_none(face)) is not None]
    if challenge.kind == "rainbow":
        return _evaluate_rainbow(challenge, context, people)
    if challenge.kind == "opposites":
        return _evaluate_opposites(challenge, context, people)
    return _evaluate_trait(challenge, context, people)


def _trait_result(challenge: Challenge, context: FrameContext, face_frame: FaceFrame) -> tuple[TraitResult, str]:
    if challenge.kind == "color":
        return check_clothing_color(context, face_frame, challenge.color), COLORS[challenge.color]
    check, label = TRAIT_CHECKS[challenge.kind]
    return check(context, face_frame), label


def _evaluate_trait(challenge: Challenge, context: FrameContext, people: list[tuple[Face, FaceFrame]]) -> Evaluation:
    boxes, met, best, reasons = [], 0, 0.0, []
    for face, face_frame in people:
        result, label = _trait_result(challenge, context, face_frame)
        boxes.append(_result_box(result, label, face_frame, context.frame.shape))
        met += result.passed
        best = max(best, result.meter)
        if result.reason:
            reasons.append(result.reason)
    return _finish(challenge, met, best, boxes, reasons, face_count=len(people))


def _evaluate_rainbow(challenge: Challenge, context: FrameContext, people: list[tuple[Face, FaceFrame]]) -> Evaluation:
    boxes, colors, reasons = [], set(), []
    for face, face_frame in people:
        color, result = dominant_clothing_color(context, face_frame)
        x1, y1, x2, y2 = face_frame.bounding_box(result.region, context.frame.shape)
        if color:
            colors.add(color)
            boxes.append(box(COLORS[color], x1, y1, x2, y2, color, meter=result.meter, passed=True))
        else:
            boxes.append(box(result.reason or "Farbe?", x1, y1, x2, y2, "muted",
                             meter=result.meter if result.usable else None, passed=False))
            if result.reason:
                reasons.append(result.reason)
    return _finish(challenge, len(colors), float(len(colors)), boxes, reasons, face_count=len(people))


def _evaluate_opposites(challenge: Challenge, context: FrameContext, people: list[tuple[Face, FaceFrame]]) -> Evaluation:
    boxes, with_glasses, without_glasses, reasons = [], 0, 0, []
    for face, face_frame in people:
        result = check_glasses(context, face_frame)
        x1, y1, x2, y2 = face_frame.bounding_box(result.region, context.frame.shape)
        if not result.usable:
            boxes.append(box(result.reason or "?", x1, y1, x2, y2, "muted", passed=False))
            reasons.append(result.reason or "")
            continue
        with_glasses += result.passed
        without_glasses += not result.passed
        label = "Mit Brille" if result.passed else "Ohne Brille"
        boxes.append(box(label, x1, y1, x2, y2, "success" if result.passed else "lavender",
                         meter=result.meter, passed=True))
    met = int(with_glasses > 0) + int(without_glasses > 0)
    return _finish(challenge, met, float(met), boxes, reasons, face_count=len(people))


def _evaluate_group(challenge: Challenge, faces: list[Face]) -> Evaluation:
    boxes = [box(f"Person {index + 1}", face.x, face.y, face.x + face.width, face.y + face.height,
                 "success", passed=True) for index, face in enumerate(faces)]
    return _finish(challenge, len(faces), float(len(faces)), boxes, [], face_count=len(faces))


def _result_box(result: TraitResult, label: str, face_frame: FaceFrame, frame_shape) -> dict:
    x1, y1, x2, y2 = face_frame.bounding_box(result.region, frame_shape)
    if not result.usable:
        return box(result.reason or "?", x1, y1, x2, y2, "muted", passed=False)
    text = f"{label} ✓" if result.passed else f"{label}?"
    return box(text, x1, y1, x2, y2, "success" if result.passed else "muted", meter=result.meter, passed=result.passed)


def _finish(challenge: Challenge, met: int, score: float, boxes: list[dict], reasons: list[str],
            face_count: int) -> Evaluation:
    required = challenge.required
    complete = met >= required
    return Evaluation(
        score=score,
        complete=complete,
        boxes=boxes,
        status_text=_status_text(challenge, min(met, required), face_count, reasons, complete),
        progress={"met": min(met, required), "required": required},
    )


def _status_text(challenge: Challenge, met: int, face_count: int, reasons: list[str], complete: bool) -> str:
    if complete:
        return challenge_texts(challenge)["active"]
    if face_count == 0:
        return "Warte auf Gesichter – stellt euch vor die Kamera"
    noun = PROGRESS_NOUNS[challenge.kind].format(color=COLORS.get(challenge.color or "", ""))
    text = f"{met}/{challenge.required} {noun}"
    if challenge.required > 1 and face_count < challenge.required:
        missing = challenge.required - face_count
        text += f" – holt noch {missing} Person{'en' if missing > 1 else ''} dazu!"
    elif "zu weit weg" in reasons:
        text += " – kommt etwas näher"
    elif "Oberkörper nicht im Bild" in reasons:
        text += " – geht etwas zurück, Oberteil muss ins Bild"
    return text
