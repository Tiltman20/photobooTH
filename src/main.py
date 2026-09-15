"""Photo booth that takes a picture after glasses are visible for three seconds."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from face_recognition import FaceDetector

CANNY_LOW, CANNY_HIGH = 80, 150
MIN_COMPONENT_AREA = 8
BRIDGE_THRESHOLD = 0.55
# A red candidate must also contain hair-like texture; colour alone can match scalp skin.
RED_HAIR_THRESHOLD = 0.06
RED_HAIR_MIN_SATURATION = 95
RED_HAIR_MIN_TEXTURE = 13.0
MUSTACHE_THRESHOLD = 0.14
MUSTACHE_DARK_VALUE = 125
GLASSES_HOLD_SECONDS = 3.0
WINDOW_NAME = "Glasses Photo Booth"
CAPTURE_DIR = Path(__file__).parent.parent / "captures"

# BGR palette: warm paper, lavender, peach and sage.
INK = (72, 60, 54)
MUTED_INK = (128, 111, 101)
PAPER = (244, 248, 255)
LAVENDER = (226, 210, 255)
PEACH = (204, 221, 255)
SAGE = (202, 232, 207)
ROSE = (211, 204, 255)


class GlassesPhotoBooth:
    """Camera UI and capture state, including protection from repeated captures."""

    def __init__(self, camera_index: int = 0) -> None:
        model = Path(__file__).parent.parent / "res" / "models" / "face_detection_yunet_2026may.onnx"
        self.camera_index = camera_index
        self.detector = FaceDetector(model_path=str(model))
        self.glasses_since: float | None = None
        self.armed = True
        self.last_capture_at: float | None = None
        self.message = "Warte auf eine Person mit Brille"

    def run(self) -> None:
        camera = cv2.VideoCapture(self.camera_index)
        if not camera.isOpened():
            raise RuntimeError("Kamera konnte nicht geoeffnet werden. Bitte Kamera pruefen.")
        CAPTURE_DIR.mkdir(exist_ok=True)
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1100, 700)
        try:
            while True:
                ok, frame = camera.read()
                if not ok:
                    self.message = "Kein Kamerabild empfangen"
                    break
                now = time.monotonic()
                display, glasses_found = self._analyze(frame)
                self._update_capture_state(frame, glasses_found, now)
                self._draw_interface(display, now)
                cv2.imshow(WINDOW_NAME, display)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("r"):
                    self._rearm()
                if key == ord("s"):
                    self._save_photo(frame, manual=True)
        finally:
            camera.release()
            cv2.destroyAllWindows()

    def _analyze(self, frame: np.ndarray) -> tuple[np.ndarray, bool]:
        display, glasses_found = frame.copy(), False
        for face in self.detector.detect(frame):
            bounds = get_glasses_region(frame, face)
            if bounds is None:
                continue
            x_min, y_min, x_max, y_max = bounds
            score, _ = check_glasses(frame[y_min:y_max, x_min:x_max])
            has_glasses = score >= BRIDGE_THRESHOLD
            glasses_found = glasses_found or has_glasses
            color = SAGE if has_glasses else LAVENDER
            cv2.rectangle(display, (face.x, face.y), (face.x + face.width, face.y + face.height), color, 2)
            cv2.rectangle(display, (x_min, y_min), (x_max, y_max), color, 1)
            label = "BRILLE ERKANNT" if has_glasses else "Gesicht erkannt"
            cv2.putText(display, f"{label}  {score:.0%}", (face.x, max(30, face.y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        return display, glasses_found

    def _update_capture_state(self, frame: np.ndarray, glasses_found: bool, now: float) -> None:
        if not glasses_found:
            self.glasses_since, self.armed = None, True
            self.message = "Warte auf eine Person mit Brille"
        elif not self.armed:
            self.message = "Foto gespeichert – kurz aus dem Bild gehen zum erneuten Ausloesen"
        elif self.glasses_since is None:
            self.glasses_since = now
            self.message = "Brille erkannt – bitte noch kurz stillhalten"
        elif now - self.glasses_since >= GLASSES_HOLD_SECONDS:
            self._save_photo(frame)
            self.armed, self.glasses_since = False, None
        else:
            self.message = f"Brille erkannt – Aufnahme in {GLASSES_HOLD_SECONDS - (now - self.glasses_since):.1f} Sekunden"

    def _save_photo(self, frame: np.ndarray, manual: bool = False) -> None:
        filename = CAPTURE_DIR / f"photo_{datetime.now():%Y-%m-%d_%H-%M-%S}.jpg"
        if cv2.imwrite(str(filename), frame):
            self.last_capture_at = time.monotonic()
            kind = "Manuell" if manual else "Automatisch"
            self.message = f"{kind} gespeichert: {filename.name}"
        else:
            self.message = "Foto konnte nicht gespeichert werden"

    def _rearm(self) -> None:
        self.armed, self.glasses_since = True, None
        self.message = "Bereit fuer die naechste Aufnahme"

    def _draw_interface(self, frame: np.ndarray, now: float) -> None:
        height, width = frame.shape[:2]
        header, footer, margin = 92, 122, 18
        overlay = frame.copy()
        draw_round_rect(overlay, (margin, margin), (width - margin, header), 18, PAPER, -1)
        draw_round_rect(overlay, (margin, height - footer), (width - margin, height - margin), 18, PAPER, -1)
        frame[:] = cv2.addWeighted(overlay, 0.90, frame, 0.10, 0)

        cv2.putText(frame, "glasses booth", (42, 53), cv2.FONT_HERSHEY_DUPLEX, 0.9, INK, 2, cv2.LINE_AA)
        cv2.putText(frame, "Ein Foto entsteht, wenn deine Brille 3 Sekunden sichtbar ist.", (43, 77),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, MUTED_INK, 1, cv2.LINE_AA)
        state, state_color = ("BEREIT", SAGE) if self.armed else ("FOTO ERSTELLT", PEACH)
        state_width = 142 if self.armed else 190
        chip_x = width - margin - state_width - 20
        draw_round_rect(frame, (chip_x, 39), (width - margin - 20, 70), 15, state_color, -1)
        cv2.putText(frame, state, (chip_x + 15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.47, INK, 1, cv2.LINE_AA)

        progress = 0.0 if self.glasses_since is None else min((now - self.glasses_since) / GLASSES_HOLD_SECONDS, 1.0)
        x, y, bar_width = 42, height - 65, max(200, width - 84)
        cv2.putText(frame, self.message, (42, height - 88), cv2.FONT_HERSHEY_SIMPLEX, 0.54, INK, 1, cv2.LINE_AA)
        draw_round_rect(frame, (x, y), (x + bar_width, y + 14), 7, LAVENDER, -1)
        if progress:
            draw_round_rect(frame, (x, y), (x + max(14, int(bar_width * progress)), y + 14), 7, SAGE, -1)
        cv2.putText(frame, "S  Speichern     R  Neu starten     ESC  Schliessen", (42, height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, MUTED_INK, 1, cv2.LINE_AA)
        if self.last_capture_at and now - self.last_capture_at < 1.4:
            card_width = 305
            draw_round_rect(frame, (width // 2 - card_width // 2, header + 24),
                            (width // 2 + card_width // 2, header + 78), 16, PEACH, -1)
            cv2.putText(frame, "Foto gespeichert!", (width // 2 - 119, header + 59),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, INK, 1, cv2.LINE_AA)


def draw_round_rect(image: np.ndarray, top_left: tuple[int, int], bottom_right: tuple[int, int],
                    radius: int, color: tuple[int, int, int], thickness: int) -> None:
    """Draw a rounded OpenCV panel without needing a GUI toolkit."""
    x1, y1 = top_left
    x2, y2 = bottom_right
    radius = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    cv2.rectangle(image, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(image, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    for center in ((x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
                   (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)):
        cv2.circle(image, center, radius, color, thickness, cv2.LINE_AA)


def get_glasses_region(frame: np.ndarray, face) -> tuple[int, int, int, int] | None:
    """Return a bounded region around both eyes, suited to a glasses bridge."""
    if face.left_eye is None or face.right_eye is None:
        return None
    left, right = sorted((face.left_eye, face.right_eye), key=lambda point: point[0])
    distance = max(1, right[0] - left[0])
    center_x, center_y = (left[0] + right[0]) // 2, (left[1] + right[1]) // 2
    x_min, x_max = max(0, center_x - int(distance * .75)), min(frame.shape[1], center_x + int(distance * .75))
    y_min, y_max = max(0, center_y - int(distance * .38)), min(frame.shape[0], center_y + int(distance * .38))
    return (x_min, y_min, x_max, y_max) if x_max > x_min and y_max > y_min else None


def check_glasses(frame: np.ndarray) -> tuple[float, np.ndarray]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), CANNY_LOW, CANNY_HIGH)
    return analyze_edges(cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))))


def check_red_hair(frame: np.ndarray, face) -> float:
    """Estimate textured red-hair coverage around the upper part of a face.

    The enlarged area includes the hairline and a small section above the face;
    restricting it to this region avoids treating red objects elsewhere in the
    image as hair. The returned value is the ratio of warm red/orange pixels.
    """
    bounds = get_red_hair_region(frame, face)
    if bounds is None:
        return 0.0
    x1, y1, x2, y2 = bounds
    region = frame[y1:y2, x1:x2]
    if region.size == 0:
        return 0.0
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    lower_red = cv2.inRange(hsv, (0, RED_HAIR_MIN_SATURATION, 35), (17, 255, 235))
    upper_red = cv2.inRange(hsv, (170, RED_HAIR_MIN_SATURATION, 35), (179, 255, 235))
    mask = cv2.bitwise_or(lower_red, upper_red)
    red_pixels = cv2.countNonZero(mask)
    if not red_pixels:
        return 0.0
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    texture = float(gray[mask > 0].std())
    if texture < RED_HAIR_MIN_TEXTURE:
        return 0.0
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 45, 120)
    textured_red_pixels = cv2.countNonZero(cv2.bitwise_and(edges, mask))
    edge_ratio = textured_red_pixels / red_pixels
    texture_score = min(edge_ratio / .09, 1.0) * min(texture / 38.0, 1.0)
    return float((red_pixels / mask.size) * texture_score)


def get_red_hair_region(frame: np.ndarray, face) -> tuple[int, int, int, int] | None:
    """Return the enlarged upper-head area used for the red-hair estimate."""
    x1 = max(0, face.x - int(face.width * .15))
    x2 = min(frame.shape[1], face.x + int(face.width * 1.15))
    y1 = max(0, face.y - int(face.height * .30))
    y2 = min(frame.shape[0], face.y + int(face.height * .13))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def check_mustache(frame: np.ndarray, face) -> float:
    """Return a moustache score without exposing diagnostic details."""
    return analyze_mustache(frame, face)[0]


def analyze_mustache(frame: np.ndarray, face) -> tuple[float, dict[str, float]]:
    """Estimate a distinct moustache and return local debugging metrics."""
    if face.nose is None:
        return 0.0, {"score": 0.0, "darkRatio": 0.0, "coverage": 0.0, "edgeRatio": 0.0, "beardRatio": 0.0}
    regions = get_mustache_regions(frame, face)
    if regions is None:
        return 0.0, {"score": 0.0, "darkRatio": 0.0, "coverage": 0.0, "edgeRatio": 0.0, "beardRatio": 0.0}
    (x1, mustache_y1, x2, mustache_y2), (_, beard_y1, _, beard_y2) = regions
    mustache_region = frame[mustache_y1:mustache_y2, x1:x2]
    beard_region = frame[beard_y1:beard_y2, x1:x2]
    if mustache_region.size == 0:
        return 0.0, {"score": 0.0, "darkRatio": 0.0, "coverage": 0.0, "edgeRatio": 0.0, "beardRatio": 0.0}
    gray = cv2.cvtColor(mustache_region, cv2.COLOR_BGR2GRAY)
    # A moustache should create a compact, dark horizontal band below the nose.
    # The tuned cutoff also includes lighter brown facial hair without treating normal skin
    # as hair in well-lit images. The beard check below remains the safeguard.
    mask = (gray < MUSTACHE_DARK_VALUE).astype(np.uint8)
    dark_ratio = float(mask.mean())
    column_coverage = float(np.mean(mask.sum(axis=0) >= max(1, int(mask.shape[0] * .22))))
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 130)
    edge_ratio = float(cv2.countNonZero(edges) / edges.size)

    # A beard continues well below the mouth. Penalising this separate region
    # distinguishes a standalone moustache from a full beard.
    beard_ratio = 0.0
    if beard_region.size:
        beard_gray = cv2.cvtColor(beard_region, cv2.COLOR_BGR2GRAY)
        beard_ratio = float(np.mean(beard_gray < MUSTACHE_DARK_VALUE))

    moustache_score = .50 * dark_ratio + .35 * column_coverage + .15 * min(edge_ratio / .18, 1.0)
    score = float(np.clip(moustache_score - .95 * beard_ratio, 0.0, 1.0))
    return score, {
        "score": round(score, 3),
        "darkRatio": round(dark_ratio, 3),
        "coverage": round(column_coverage, 3),
        "edgeRatio": round(edge_ratio, 3),
        "beardRatio": round(beard_ratio, 3),
    }


def get_mustache_regions(frame: np.ndarray, face) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    """Return the moustache and lower-beard areas used by the classifier."""
    if face.nose is None:
        return None
    nose_x, nose_y = face.nose
    x1 = max(0, nose_x - int(face.width * .38))
    x2 = min(frame.shape[1], nose_x + int(face.width * .38))
    mustache_y1 = max(0, nose_y)
    mustache_y2 = min(frame.shape[0], nose_y + int(face.height * .27))
    beard_y1 = min(frame.shape[0], nose_y + int(face.height * .35))
    beard_y2 = min(frame.shape[0], face.y + int(face.height * .93))
    if x2 <= x1 or mustache_y2 <= mustache_y1:
        return None
    return (x1, mustache_y1, x2, mustache_y2), (x1, beard_y1, x2, beard_y2)


def analyze_edges(edges: np.ndarray) -> tuple[float, np.ndarray]:
    roi_height, roi_width = edges.shape
    count, _, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
    best_score = 0.0
    for index in range(1, count):
        x, y, width, height, area = stats[index]
        if area < MIN_COMPONENT_AREA or width < 3 or height < 2:
            continue
        center = 1 - min(abs((x + width / 2) - roi_width / 2) / max(roi_width / 2, 1), 1)
        score = .40 * center + .35 * min((width / max(roi_width, 1)) / .45, 1) + .15 * (1 - min(height / max(roi_height, 1), 1)) + .10 * min((area / max(width * height, 1)) / .30, 1)
        best_score = max(best_score, float(np.clip(score, 0, 1)))
    return best_score, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def main() -> None:
    GlassesPhotoBooth().run()


if __name__ == "__main__":
    main()
