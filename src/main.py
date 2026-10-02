"""Photo booth that takes a picture after glasses are visible for three seconds."""

from __future__ import annotations

import time
from datetime import datetime

import cv2
import numpy as np

from config import CAPTURE_DIR, FACE_MODEL_PATH
from face_checks import FrameContext, check_glasses, face_frame_or_none
from face_recognition import FaceDetector

GLASSES_HOLD_SECONDS = 3.0
WINDOW_NAME = "Glasses Photo Booth"

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
        self.camera_index = camera_index
        self.detector = FaceDetector(model_path=str(FACE_MODEL_PATH))
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
        context = FrameContext(frame)
        for face in self.detector.detect(frame):
            face_frame = face_frame_or_none(face)
            if face_frame is None:
                continue
            result = check_glasses(context, face_frame)
            x_min, y_min, x_max, y_max = face_frame.bounding_box(result.region, frame.shape)
            score, has_glasses = result.score, result.passed
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


def main() -> None:
    GlassesPhotoBooth().run()


if __name__ == "__main__":
    main()
