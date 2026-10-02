"""Face-aligned coordinates built from the five YuNet landmarks.

All check regions are defined in a coordinate system that moves with the face:

* unit length = distance between the eyes
* origin      = midpoint between the eyes
* x axis      = from the left eye (in the image) to the right eye
* y axis      = perpendicular, pointing down towards the mouth

Because regions are defined relative to the eyes, they stay on the right spot
when a person tilts the head or stands closer to / further from the camera.
Patches are cut out upright with a fixed resolution, so edge and colour
measurements are comparable for near and far faces.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from face_dataclass import Face


@dataclass(frozen=True)
class Region:
    """Axis-aligned rectangle in face coordinates (eye distances)."""

    x0: float
    y0: float
    x1: float
    y1: float

    def shifted(self, dx: float = 0.0, dy: float = 0.0) -> Region:
        return Region(self.x0 + dx, self.y0 + dy, self.x1 + dx, self.y1 + dy)


@dataclass(frozen=True)
class Patch:
    """An upright crop of a face region plus the mask of pixels inside the camera frame."""

    image: np.ndarray
    valid: np.ndarray  # bool mask, False where the region left the camera frame

    @property
    def visible_share(self) -> float:
        return float(self.valid.mean()) if self.valid.size else 0.0


class FaceFrame:
    """Similarity transform between face coordinates and image pixels."""

    def __init__(self, face: Face) -> None:
        if face.left_eye is None or face.right_eye is None:
            raise ValueError("Face without eye landmarks")
        left, right = sorted((face.left_eye, face.right_eye), key=lambda point: point[0])
        left_eye, right_eye = np.asarray(left, float), np.asarray(right, float)
        self.face = face
        self.origin = (left_eye + right_eye) / 2
        axis = right_eye - left_eye
        self.eye_distance = float(max(np.hypot(*axis), 1.0))
        cos, sin = axis / self.eye_distance
        # Columns: image direction of the face x axis and y axis (both scaled to pixels).
        self.linear = self.eye_distance * np.array([[cos, -sin], [sin, cos]])

    @property
    def roll_degrees(self) -> float:
        return float(np.degrees(np.arctan2(self.linear[1, 0], self.linear[0, 0])))

    def to_image(self, x: float, y: float) -> tuple[float, float]:
        point = self.linear @ np.array([x, y]) + self.origin
        return float(point[0]), float(point[1])

    def to_face(self, point: tuple[float, float]) -> tuple[float, float]:
        local = np.linalg.solve(self.linear, np.asarray(point, float) - self.origin)
        return float(local[0]), float(local[1])

    def patch(self, frame: np.ndarray, region: Region, pixels_per_unit: float) -> Patch:
        """Cut the region out of the frame, upright and at a fixed resolution."""
        width = max(1, int(round((region.x1 - region.x0) * pixels_per_unit)))
        height = max(1, int(round((region.y1 - region.y0) * pixels_per_unit)))
        # Maps patch pixel (u, v) -> image pixel; used with WARP_INVERSE_MAP.
        patch_to_face = np.array([[1 / pixels_per_unit, 0, region.x0], [0, 1 / pixels_per_unit, region.y0]])
        patch_to_image = np.hstack([self.linear @ patch_to_face[:, :2],
                                    (self.linear @ patch_to_face[:, 2] + self.origin)[:, None]])
        flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
        image = cv2.warpAffine(frame, patch_to_image, (width, height), flags=flags, borderMode=cv2.BORDER_REPLICATE)
        inside = np.full(frame.shape[:2], 255, dtype=np.uint8)
        valid = cv2.warpAffine(inside, patch_to_image, (width, height), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
        return Patch(image=image, valid=valid)

    def bounding_box(self, region: Region, frame_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
        """Pixel box (x1, y1, x2, y2) around the (possibly rotated) region, clipped to the frame."""
        corners = [self.to_image(x, y) for x in (region.x0, region.x1) for y in (region.y0, region.y1)]
        xs, ys = [c[0] for c in corners], [c[1] for c in corners]
        height, width = frame_shape[:2]
        return (int(np.clip(min(xs), 0, width - 1)), int(np.clip(min(ys), 0, height - 1)),
                int(np.clip(max(xs), 0, width - 1)), int(np.clip(max(ys), 0, height - 1)))
