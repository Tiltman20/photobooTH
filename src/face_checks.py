"""Classical checks for visible traits: glasses, moustache, red hair, clothing colour.

Every check works on an upright, fixed-resolution patch in face coordinates
(see ``face_geometry``) and compares against the person's own skin where
possible, so results depend far less on distance, head tilt and lighting.

All tunable thresholds live in ``TraitThresholds`` and can be changed at runtime
(settings page in the browser, see ``settings.py``). The live score meter in
the web UI shows score and threshold per person to make tuning easy.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import cv2
import numpy as np

from face_dataclass import Face
from face_geometry import FaceFrame, Region


@dataclass(frozen=True)
class TraitThresholds:
    glasses: float = 0.55
    moustache: float = 0.14
    red_hair: float = 0.06
    clothing_color: float = 0.22
    # Faces with a smaller eye distance (pixels) are too small for trait checks.
    min_eye_distance_px: float = 14.0
    # Clothing checks need at least this share of the torso region inside the image.
    min_torso_visible: float = 0.35


_thresholds = TraitThresholds()


def thresholds() -> TraitThresholds:
    """The thresholds currently in use (adjustable at runtime via the settings page)."""
    return _thresholds


def set_thresholds(new_thresholds: TraitThresholds) -> None:
    global _thresholds
    _thresholds = new_thresholds

# Regions in face coordinates (1 unit = eye distance, origin between the eyes).
GLASSES_REGION = Region(-0.75, -0.38, 0.75, 0.38)
CHEEK_REGIONS = (Region(-0.85, 0.30, -0.45, 0.60), Region(0.45, 0.30, 0.85, 0.60))
HAIR_REGION = Region(-1.30, -2.00, 1.30, -0.80)
TORSO_REGION = Region(-1.40, 2.40, 1.40, 4.60)

# Patch resolutions (pixels per eye distance). 56 px matches a face of about
# 150 px width in a 640 px frame, the size the original thresholds were tuned on.
GLASSES_PPU = 56
DETAIL_PPU = 56
COARSE_PPU = 28

# Colour classes on white-balanced HSV (OpenCV hue 0-179).
COLOR_RANGES: dict[str, tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...]] = {
    "blue": (((92, 75, 45), (130, 255, 255)),),
    "red": (((0, 90, 45), (10, 255, 255)), ((170, 90, 45), (179, 255, 255))),
    "green": (((38, 65, 40), (88, 255, 255)),),
    "yellow": (((20, 90, 70), (37, 255, 255)),),
    "white": (((0, 0, 165), (179, 55, 255)),),
    "black": (((0, 0, 0), (179, 255, 65)),),
}


@dataclass(frozen=True)
class TraitResult:
    score: float
    threshold: float
    region: Region
    usable: bool = True
    reason: str | None = None  # why the check could not be done

    @property
    def passed(self) -> bool:
        return self.usable and self.score >= self.threshold

    @property
    def meter(self) -> float:
        """Score relative to the threshold: 1.0 means exactly at the threshold."""
        return self.score / self.threshold if self.threshold else 0.0


@dataclass(frozen=True)
class SkinReference:
    gray: float
    lab: np.ndarray  # median L, a, b of the cheeks


class FrameContext:
    """One camera frame plus lazily computed, shared conversions."""

    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame
        self.skin_cache: dict[int, SkinReference | None] = {}

    @cached_property
    def balanced(self) -> np.ndarray:
        """Gray-world white balance: removes colour casts from warm or coloured light."""
        small = cv2.resize(self.frame, (64, 48), interpolation=cv2.INTER_AREA).reshape(-1, 3).astype(float)
        means = small.mean(axis=0) + 1e-6
        gains = means.mean() / means
        gains = np.clip(1 + 0.6 * (gains - 1), 0.7, 1.4)  # partial correction avoids over-shooting
        return np.clip(self.frame.astype(np.float32) * gains, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- helpers
def face_frame_or_none(face: Face) -> FaceFrame | None:
    try:
        return FaceFrame(face)
    except ValueError:
        return None


def too_small(face_frame: FaceFrame, region: Region, threshold: float) -> TraitResult | None:
    if face_frame.eye_distance < thresholds().min_eye_distance_px:
        return TraitResult(0.0, threshold, region, usable=False, reason="zu weit weg")
    return None


def skin_reference(context: FrameContext, face_frame: FaceFrame) -> SkinReference | None:
    """Median cheek colour of this person (cached per frame)."""
    key = id(face_frame.face)
    if key not in context.skin_cache:
        context.skin_cache[key] = _measure_skin(context, face_frame)
    return context.skin_cache[key]


def _measure_skin(context: FrameContext, face_frame: FaceFrame) -> SkinReference | None:
    pixels = []
    for region in CHEEK_REGIONS:
        patch = face_frame.patch(context.balanced, region, COARSE_PPU)
        pixels.append(patch.image[patch.valid])
    pixels = np.concatenate(pixels) if pixels else np.empty((0, 3), np.uint8)
    if len(pixels) < 20:
        return None
    lab = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
    gray = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2GRAY).ravel()
    return SkinReference(gray=float(np.median(gray)), lab=np.median(lab, axis=0))


def skin_like_mask(patch_bgr: np.ndarray, skin: SkinReference | None, max_distance: float = 16.0) -> np.ndarray:
    """Pixels whose colour is close to the person's cheek skin (chroma only, so shading does not matter)."""
    if skin is None:
        return np.zeros(patch_bgr.shape[:2], dtype=bool)
    lab = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    chroma_distance = np.hypot(lab[..., 1] - skin.lab[1], lab[..., 2] - skin.lab[2])
    return chroma_distance < max_distance


def _clahe_gray(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)


# ---------------------------------------------------------------- glasses
def check_glasses(context: FrameContext, face_frame: FaceFrame) -> TraitResult:
    """Look for the glasses bridge: a wide, flat edge structure centred between the eyes."""
    threshold = thresholds().glasses
    if (small := too_small(face_frame, GLASSES_REGION, threshold)) is not None:
        return small
    patch = face_frame.patch(context.frame, GLASSES_REGION, GLASSES_PPU)
    gray = _clahe_gray(patch.image)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 80, 150)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    edges[~patch.valid] = 0
    return TraitResult(bridge_score(edges), threshold, GLASSES_REGION)


def bridge_score(edges: np.ndarray, min_component_area: int = 8) -> float:
    """Score the best edge component: centred, wide, flat and dense means 'bridge'."""
    roi_height, roi_width = edges.shape
    count, _, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
    best_score = 0.0
    for index in range(1, count):
        x, y, width, height, area = stats[index]
        if area < min_component_area or width < 3 or height < 2:
            continue
        centered = 1 - min(abs((x + width / 2) - roi_width / 2) / max(roi_width / 2, 1), 1)
        wide = min((width / max(roi_width, 1)) / .45, 1)
        flat = 1 - min(height / max(roi_height, 1), 1)
        dense = min((area / max(width * height, 1)) / .30, 1)
        score = .40 * centered + .35 * wide + .15 * flat + .10 * dense
        best_score = max(best_score, float(np.clip(score, 0, 1)))
    return best_score


# ---------------------------------------------------------------- moustache
def moustache_regions(face_frame: FaceFrame) -> tuple[Region, Region] | None:
    """Upper lip (between nose tip and mouth) and chin region, from the real landmarks."""
    face = face_frame.face
    if face.nose is None or face.mouth_left is None or face.mouth_right is None:
        return None
    _, nose_y = face_frame.to_face(face.nose)
    corners = sorted((face_frame.to_face(face.mouth_left), face_frame.to_face(face.mouth_right)))
    (mouth_x0, mouth_y0), (mouth_x1, mouth_y1) = corners
    mouth_y = (mouth_y0 + mouth_y1) / 2
    gap = mouth_y - nose_y
    if gap <= 0.1:
        return None
    lip = Region(mouth_x0 - 0.05, nose_y + 0.30 * gap, mouth_x1 + 0.05, mouth_y - 0.08 * gap)
    chin = Region(mouth_x0 - 0.05, mouth_y + 0.25, mouth_x1 + 0.05, mouth_y + 0.60)
    return lip, chin


def check_moustache(context: FrameContext, face_frame: FaceFrame) -> TraitResult:
    """A moustache is a compact band below the nose that is clearly darker than the cheeks."""
    threshold = thresholds().moustache
    regions = moustache_regions(face_frame)
    if regions is None:
        return TraitResult(0.0, threshold, GLASSES_REGION, usable=False, reason="Mund nicht erkannt")
    lip_region, chin_region = regions
    if (small := too_small(face_frame, lip_region, threshold)) is not None:
        return small
    skin = skin_reference(context, face_frame)
    if skin is None:
        return TraitResult(0.0, threshold, lip_region, usable=False, reason="Gesicht nicht sichtbar")
    dark_level = skin.gray * 0.75  # hair is darker than the person's own skin, whatever the light

    lip = face_frame.patch(context.balanced, lip_region, DETAIL_PPU)
    gray = cv2.cvtColor(lip.image, cv2.COLOR_BGR2GRAY)
    dark = (gray < dark_level) & lip.valid
    dark_ratio = float(dark.mean())
    column_coverage = float(np.mean(dark.sum(axis=0) >= max(1, int(dark.shape[0] * .22))))
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 130)
    edge_ratio = float(np.count_nonzero(edges) / edges.size)

    # A full beard continues below the mouth; penalising it keeps this a moustache check.
    chin = face_frame.patch(context.balanced, chin_region, DETAIL_PPU)
    chin_gray = cv2.cvtColor(chin.image, cv2.COLOR_BGR2GRAY)
    beard_ratio = float(((chin_gray < dark_level) & chin.valid).mean())

    raw = .50 * dark_ratio + .35 * column_coverage + .15 * min(edge_ratio / .18, 1.0)
    score = float(np.clip(raw - .95 * beard_ratio, 0.0, 1.0))
    return TraitResult(score, threshold, lip_region)


# ---------------------------------------------------------------- red hair
def check_red_hair(context: FrameContext, face_frame: FaceFrame) -> TraitResult:
    """Share of textured red/orange pixels above the forehead that do not look like skin."""
    threshold = thresholds().red_hair
    if (small := too_small(face_frame, HAIR_REGION, threshold)) is not None:
        return small
    patch = face_frame.patch(context.balanced, HAIR_REGION, COARSE_PPU)
    if patch.visible_share < 0.25:
        return TraitResult(0.0, threshold, HAIR_REGION, usable=False, reason="Haare nicht im Bild")
    hsv = cv2.cvtColor(patch.image, cv2.COLOR_BGR2HSV)
    red = (cv2.inRange(hsv, (0, 95, 35), (17, 255, 235)) | cv2.inRange(hsv, (170, 95, 35), (179, 255, 235))) > 0
    red &= patch.valid & ~skin_like_mask(patch.image, skin_reference(context, face_frame))
    red_pixels = int(red.sum())
    if red_pixels == 0:
        return TraitResult(0.0, threshold, HAIR_REGION)
    gray = cv2.cvtColor(patch.image, cv2.COLOR_BGR2GRAY)
    texture = float(gray[red].std())
    if texture < 13.0:  # flat red areas are walls or clothes, not hair
        return TraitResult(0.0, threshold, HAIR_REGION)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 45, 120) > 0
    edge_ratio = float((edges & red).sum() / red_pixels)
    texture_score = min(edge_ratio / .09, 1.0) * min(texture / 38.0, 1.0)
    score = red_pixels / max(int(patch.valid.sum()), 1) * texture_score
    return TraitResult(float(score), threshold, HAIR_REGION)


# ---------------------------------------------------------------- clothing
def clothing_color_shares(context: FrameContext, face_frame: FaceFrame) -> tuple[dict[str, float], TraitResult | None]:
    """Share of each colour class on the visible torso (skin excluded).

    Returns the shares and, if the torso cannot be checked, a failed result explaining why.
    """
    threshold = thresholds().clothing_color
    if (small := too_small(face_frame, TORSO_REGION, threshold)) is not None:
        return {}, small
    patch = face_frame.patch(context.balanced, TORSO_REGION, COARSE_PPU)
    if patch.visible_share < thresholds().min_torso_visible:
        return {}, TraitResult(0.0, threshold, TORSO_REGION, usable=False, reason="Oberkörper nicht im Bild")
    considered = patch.valid & ~skin_like_mask(patch.image, skin_reference(context, face_frame), max_distance=10.0)
    total = max(int(considered.sum()), 1)
    hsv = cv2.cvtColor(patch.image, cv2.COLOR_BGR2HSV)
    shares = {}
    for name, ranges in COLOR_RANGES.items():
        mask = np.zeros(hsv.shape[:2], dtype=bool)
        for lower, upper in ranges:
            mask |= cv2.inRange(hsv, lower, upper) > 0
        shares[name] = float((mask & considered).sum() / total)
    return shares, None


def check_clothing_color(context: FrameContext, face_frame: FaceFrame, color: str) -> TraitResult:
    shares, failure = clothing_color_shares(context, face_frame)
    if failure is not None:
        return failure
    return TraitResult(shares.get(color, 0.0), thresholds().clothing_color, TORSO_REGION)


def dominant_clothing_color(context: FrameContext, face_frame: FaceFrame) -> tuple[str | None, TraitResult]:
    """The clearly dominant colour class of a person's clothes, or None."""
    shares, failure = clothing_color_shares(context, face_frame)
    if failure is not None:
        return None, failure
    color, share = max(shares.items(), key=lambda item: item[1])
    result = TraitResult(share, thresholds().clothing_color, TORSO_REGION)
    return (color if result.passed else None), result
