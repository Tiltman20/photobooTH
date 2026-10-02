"""Result of analysing one frame, plus the drawing primitives the browser renders."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Evaluation:
    """Outcome of checking one frame against a challenge."""

    score: float
    complete: bool
    boxes: list[dict] = field(default_factory=list)
    circles: list[dict] = field(default_factory=list)
    lines: list[dict] = field(default_factory=list)
    status_text: str | None = None
    debug: dict | None = None
    pointer: dict | None = None
    faces: list[dict] | None = None  # tracked face poses for drawing in face coordinates
    progress: dict | None = None  # {"met": int, "required": int}


def box(label: str, x1: int, y1: int, x2: int, y2: int, color: str,
        meter: float | None = None, passed: bool | None = None) -> dict:
    """A labelled rectangle; ``meter`` (1.0 = threshold) draws a score bar under the label."""
    result = {"label": label, "x": int(x1), "y": int(y1), "width": int(x2 - x1), "height": int(y2 - y1), "color": color}
    if meter is not None:
        result["meter"] = round(float(meter), 3)
    if passed is not None:
        result["passed"] = bool(passed)
    return result


def circle(x: int, y: int, radius: float, color: str, filled: bool = False) -> dict:
    return {"x": int(x), "y": int(y), "radius": round(float(radius), 1), "color": color, "filled": filled}


def line(start: tuple[int, int], end: tuple[int, int], color: str) -> dict:
    return {"x1": int(start[0]), "y1": int(start[1]), "x2": int(end[0]), "y2": int(end[1]), "color": color}
