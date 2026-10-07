"""Photo challenges: catalogue, random variants and the texts shown to the guests.

A challenge is identified by a compact code that the browser sends back with
every frame, e.g. ``glasses:2`` (two people with glasses), ``color:3:blue``
(three people in blue), ``hair:1:red`` or ``finger_sum:1:12`` (exactly 12
fingers). The server therefore needs no session state to know what to check.

Evaluation lives in ``face_challenges`` (traits and expressions) and
``hand_challenges`` (gestures and drawing).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

COLORS = {"black": "Schwarz", "white": "Weiß", "blue": "Blau", "red": "Rot", "green": "Grün",
          "yellow": "Gelb", "pink": "Pink", "grey": "Grau"}
COLOR_DATIVE = {"black": "schwarzem", "white": "weißem", "blue": "blauem", "red": "rotem", "green": "grünem",
                "yellow": "gelbem", "pink": "pinkem", "grey": "grauem"}  # "mit ___ Oberteil"
HAIR_COLORS = {"red": "Rot", "blond": "Blond", "colorful": "Bunt"}
HAIR_DATIVE = {"red": "roten", "blond": "blonden", "colorful": "bunt gefärbten"}  # "mit ___ Haaren"
STENCIL_THEMES = {  # theme id: (title, what to draw)
    "hats": ("Hut-Party", "malt jeder Schablone einen Hut über den Kopf"),
    "horns": ("Teufelshörner", "malt Hörner auf die Stirn"),
    "moustache": ("Schnurrbart-Salon", "malt Schnurrbärte unter die Nase"),
    "free": ("Freestyle", "malt, was ihr wollt: Hüte, Hörner, Brillen, Bärte …"),
}
# More than ten fingers needs a second person, which is the point.
FINGER_SUM_TARGETS = tuple(str(target) for target in range(7, 16))
MULTI_PERSON_WEIGHT = 2.0  # multi-person variants are drawn more often: meeting people is the point


@dataclass(frozen=True)
class Kind:
    title: str
    family: str                        # "face", "expression", "hand" or "drawing"
    sizes: tuple[int, ...]             # possible numbers of people / hands
    options: tuple[str | None, ...] = (None,)
    option_sizes: dict[str, tuple[int, ...]] = field(default_factory=dict)  # rarer options: fewer people
    weight: float = 1.0                # how often the kind is drawn at random

    def sizes_for(self, option: str | None) -> tuple[int, ...]:
        return self.option_sizes.get(option, self.sizes)


KINDS: dict[str, Kind] = {
    "glasses": Kind("Brillen-Gang", "face", (1, 2, 3)),
    "opposites": Kind("Brille trifft Nicht-Brille", "face", (2,)),
    "beard": Kind("Bart-Bande", "face", (1, 2)),
    "hat": Kind("Hut ab!", "face", (1, 2), weight=0.6),
    "hair": Kind("Haarfarben-Jagd", "face", (1, 2), tuple(HAIR_COLORS),
                 option_sizes={"red": (1,), "colorful": (1,)}, weight=0.8),
    "color": Kind("Farb-Team", "face", (1, 2, 3), tuple(COLORS)),
    "rainbow": Kind("Regenbogen-Crew", "face", (3, 4)),
    "group": Kind("Gruppenfoto", "face", (3, 4, 5, 6)),
    "smile": Kind("Grinse-Kette", "expression", (2, 3, 4)),
    "scream": Kind("Schrei-Foto", "expression", (1, 2, 3)),
    "kiss": Kind("Kussmund-Crew", "expression", (2, 3)),
    "mood_mix": Kind("Gefühlschaos", "expression", (3,)),
    "thumbs_up": Kind("Daumen hoch", "hand", (2, 3, 4)),
    "peace": Kind("Peace!", "hand", (2, 3, 4)),
    "high_five": Kind("Hände hoch", "hand", (4, 6)),
    "finger_sum": Kind("Finger-Mathe", "hand", (1,), FINGER_SUM_TARGETS),
    "stencils": Kind("Masken-Werkstatt", "drawing", (2, 3), tuple(STENCIL_THEMES), weight=0.7),
}

# Drawing challenges analyse fast and small: more fingertip samples per second = rounder lines.
DRAWING_OPTIONS = {"airDraw": True, "analysisIntervalMs": 15, "analysisWidth": 480}
# CLIP judges small details (glasses, hair): a larger frame helps people further away.
TRAIT_OPTIONS = {"analysisWidth": 960}

# Modes without automatic photo; they are never drawn at random.
FREE_MODES = {
    "air_draw": {
        "id": "air_draw", "kind": "air_draw", "title": "Luftmalerei", "family": "free",
        "description": "Zeigefinger raus = malen. Zwei Finger oder Faust = Stift absetzen. "
                       "Hand offen halten = alles löschen. Mit „Foto aufnehmen“ landet euer Kunstwerk im Bild.",
        "ready": "Zeigefinger ausstrecken und loslegen", "waiting": "Zeig deinen Zeigefinger",
        "active": "Malt", "manualCapture": True, "autoCapture": False, "required": 1, **DRAWING_OPTIONS,
    },
    "hands": {
        "id": "hands", "kind": "hands", "title": "Debug: Hand-Erkennung", "family": "debug",
        "description": "Halte eine oder mehrere Hände ins Bild und zeig eine Fingerzahl. "
                       "Diese Ansicht nimmt kein Foto automatisch auf.",
        "ready": "Hände ins Bild halten", "waiting": "Keine Hand erkannt",
        "active": "Hand erkannt", "autoCapture": False, "showAnalysedFrame": True, "required": 1,
    },
}


@dataclass(frozen=True)
class Challenge:
    kind: str
    required: int = 1
    option: str | None = None  # colour, hair colour, finger target or stencil theme

    @property
    def code(self) -> str:
        return ":".join([self.kind, str(self.required)] + ([self.option] if self.option else []))

    @property
    def family(self) -> str:
        return KINDS[self.kind].family

    @property
    def target(self) -> int | None:
        return int(self.option) if self.kind == "finger_sum" and self.option else None

    def to_json(self) -> dict:
        kind = KINDS[self.kind]
        data = {"id": self.code, "kind": self.kind, "family": kind.family, "required": self.required,
                "option": self.option, "category": kind.title, **challenge_texts(self)}
        if kind.family == "drawing":
            # Drawing takes as long as it takes: the guests start the photo themselves.
            data |= DRAWING_OPTIONS | {"stencilMode": True, "manualCapture": True}
        elif kind.family == "face" and self.kind != "group":
            data |= TRAIT_OPTIONS
        return data


def parse_code(code: str | None) -> Challenge | None:
    """Rebuild a challenge from its code; invalid codes return None."""
    if not code:
        return None
    parts = code.split(":")
    kind = KINDS.get(parts[0])
    if kind is None or len(parts) > 3:
        return None
    option = parts[2] if len(parts) > 2 else None
    if option not in kind.options:
        return None
    try:
        required = int(parts[1]) if len(parts) > 1 else kind.sizes_for(option)[0]
    except ValueError:
        return None
    if required not in kind.sizes_for(option):
        return None
    return Challenge(parts[0], required, option)


def all_variants(kinds: list[str]) -> list[tuple[Challenge, float]]:
    """Every variant of the given kinds with its weight for the random draw."""
    variants = []
    for name in kinds:
        kind = KINDS[name]
        for option in kind.options:
            for required in kind.sizes_for(option):
                weight = kind.weight * (MULTI_PERSON_WEIGHT if required > 1 else 1.0)
                variants.append((Challenge(name, required, option), weight / len(kind.options)))
    return variants


def random_challenge(kind: str | None = None, avoid_code: str | None = None) -> Challenge:
    """Draw a variant; multi-person variants are preferred and the previous code is avoided."""
    variants = all_variants([kind] if kind in KINDS else list(KINDS))
    if len(variants) > 1 and avoid_code:
        variants = [(challenge, weight) for challenge, weight in variants if challenge.code != avoid_code]
    challenges, weights = zip(*variants)
    return random.choices(challenges, weights=weights, k=1)[0]


def challenge_catalogue() -> list[dict]:
    """Entries for the dropdown: all kinds plus the free modes."""
    entries = [{"id": name, "title": kind.title, "family": kind.family} for name, kind in KINDS.items()]
    entries += [{"id": mode["id"], "title": mode["title"], "family": mode["family"]} for mode in FREE_MODES.values()]
    return entries


# ------------------------------------------------------------------ texts
def challenge_texts(challenge: Challenge) -> dict[str, str]:
    """Title, description and success text of a variant, in party German."""
    n, option = challenge.required, challenge.option
    solo = n == 1
    color = COLORS.get(option or "", "")
    hair = HAIR_COLORS.get(option or "", "")
    theme_title, theme_hint = STENCIL_THEMES.get(option or "", STENCIL_THEMES["free"])
    texts = {
        "glasses": (
            "Zeig deine Brille!" if solo else f"{n} Brillen, ein Foto",
            "Brille auf, ab vor die Kamera." if solo
            else f"Schnappt euch {n} Leute mit Brille – egal ob ihr euch kennt. Sonnenbrillen zählen auch!",
            "Brille erkannt" if solo else "Brillen-Gang komplett",
        ),
        "opposites": (
            "Gegensätze ziehen sich an",
            "Eine Person mit Brille, eine ohne. Am besten jemand, den du noch nicht kennst.",
            "Gegensätze gefunden",
        ),
        "beard": (
            "Zeig uns deinen Bart!" if solo else f"Bart-Bande: {n} Bärte",
            "Schnurrbart, Vollbart, Ziegenbart – alles zählt." if solo
            else f"Findet {n} Leute mit Bart. Kein Bart weit und breit? Sucht weiter!",
            "Bart erkannt" if solo else "Bande komplett",
        ),
        "hat": (
            "Hut ab!" if solo else f"{n} Leute mit Kopfbedeckung",
            "Cap, Mütze, Hut oder Partyhut – Hauptsache was auf dem Kopf." if solo
            else f"Organisiert {n} Kopfbedeckungen. Ausleihen ist erlaubt!",
            "Hut erkannt" if solo else "Hut-Crew komplett",
        ),
        "hair": (
            f"Gesucht: Haarfarbe {hair}" if solo else f"Team {hair}: {n} Leute",
            f"Findet eine Person mit {HAIR_DATIVE.get(option or '')} Haaren." if solo
            else f"Stellt {n} Leute mit {HAIR_DATIVE.get(option or '')} Haaren zusammen ins Bild.",
            f"{hair} gefunden" if solo else f"Team {hair} komplett",
        ),
        "color": (
            f"Farbcode: {color}" if solo else f"Team {color}: {n} Leute",
            f"Wer trägt {color} obenrum? Ab ins Bild!" if solo
            else f"Findet {n} Leute mit {COLOR_DATIVE.get(option or '')} Oberteil und stellt euch zusammen.",
            f"{color} erkannt" if solo else f"Team {color} komplett",
        ),
        "rainbow": (
            f"Regenbogen-Crew: {n} Farben",
            f"{n} Leute, {n} verschiedene Oberteil-Farben. Wer trägt was? Findet's raus!",
            "Regenbogen komplett",
        ),
        "group": (
            f"Gruppenfoto: {n}+ Leute",
            f"Quetscht mindestens {n} Gesichter ins Bild. Je mehr, desto besser!",
            "Gruppe vollständig",
        ),
        "smile": (
            f"Grinse-Kette: {n} Leute",
            f"{n} Leute, {n} breite Grinsegesichter. Wer nicht lacht, fliegt raus!",
            "Alle grinsen!",
        ),
        "scream": (
            "Schrei-Foto!" if solo else f"Schrei-Foto: {n} Leute",
            "Mund weit auf, als wär der DJ grad dein Lieblingssong. Lauter!" if solo
            else f"{n} Leute reißen den Mund auf – wie auf der Achterbahn.",
            "AAAAH!",
        ),
        "kiss": (
            f"Kussmund-Crew: {n} Leute",
            f"{n} Leute machen den Duckface-Kussmund. Ja, auch du.",
            "Muah!",
        ),
        "mood_mix": (
            "Gefühlschaos",
            "Drei Leute, drei Gefühle: einer grinst, einer schreit, einer macht Kussmund.",
            "Oscar-reif!",
        ),
        "thumbs_up": (
            f"{n} Daumen hoch!",
            f"Zeigt zusammen {n} Daumen nach oben – ab drei braucht ihr eine zweite Person.",
            "Daumen komplett",
        ),
        "peace": (
            f"Peace! {n}× Victory",
            f"Zeigt zusammen {n}× das Peace-Zeichen (Zeige- und Mittelfinger).",
            "Peace komplett",
        ),
        "high_five": (
            f"Hände hoch: {n} offene Hände",
            f"Streckt zusammen {n} offene Hände in die Kamera – das sind mindestens {(n + 1) // 2} Leute.",
            "Hände komplett",
        ),
        "finger_sum": (
            f"Finger-Mathe: genau {challenge.target}",
            f"Zeigt zusammen genau {challenge.target} Finger – nicht mehr, nicht weniger. Absprechen!",
            f"Genau {challenge.target}!",
        ),
        "stencils": (
            f"Masken-Werkstatt: {theme_title}",
            f"Im Bild sind {n} Gesichts-Schablonen – {theme_hint}. Schablone 1 trägt die Person ganz links, "
            "Schablone 2 die nächste. Fertig? „Masken aufsetzen“ drücken.",
            "Alle verkleidet!",
        ),
    }
    title, description, active = texts[challenge.kind]
    return {"title": title, "description": description, "active": active,
            "ready": "Los geht's", "waiting": "Stellt euch vor die Kamera"}
