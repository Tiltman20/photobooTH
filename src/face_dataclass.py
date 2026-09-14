from dataclasses import dataclass
import numpy as np

@dataclass
class Face:
    id: int
    x: int
    y: int
    width: int
    height: int
    image: np.ndarray | None = None

    left_eye: tuple[int, int] | None = None
    right_eye: tuple[int, int] | None = None
    nose: tuple[int, int] | None = None

    @property
    def center(self):
        return (
            self.x + self.width // 2,
            self.y + self.height // 2
            )

    @property
    def bbox(self):
        return (
            self.x, 
            self.y,
            self.width,
            self.height
        )