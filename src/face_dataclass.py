from dataclasses import dataclass

import numpy as np

Point = tuple[int, int]


@dataclass
class Face:
    """A detected face with the five YuNet landmarks (pixel coordinates)."""

    id: int
    x: int
    y: int
    width: int
    height: int
    image: np.ndarray | None = None

    left_eye: Point | None = None
    right_eye: Point | None = None
    nose: Point | None = None
    mouth_left: Point | None = None
    mouth_right: Point | None = None
    score: float = 1.0

    @property
    def center(self) -> Point:
        return (
            self.x + self.width // 2,
            self.y + self.height // 2
        )

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (
            self.x,
            self.y,
            self.width,
            self.height
        )
