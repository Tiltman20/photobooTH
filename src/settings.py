"""Adjustable settings (detection thresholds and round timing), saved to settings.json.

The settings drawer in the browser edits these values; every change applies
immediately and is written to disk so it survives a restart. The live score
bar of every person shows the current value and the threshold, which makes
tuning on site easy.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass, fields, replace

from config import SETTINGS_FILE

logger = logging.getLogger("photobooth")
# Bump when the meaning of saved values changes; older files are then ignored.
SETTINGS_VERSION = 2


@dataclass(frozen=True)
class Thresholds:
    """Network probabilities (0..1) a person needs to count for a challenge."""

    glasses: float = 0.45
    beard: float = 0.55
    hat: float = 0.5
    hair: float = 0.5
    shirt: float = 0.45
    smile: float = 0.5
    scream: float = 0.4
    kiss: float = 0.5
    # Faces with a smaller eye distance (pixels) are too small to judge.
    min_eye_distance_px: float = 14.0
    # Shirt questions need at least this share of the chest inside the image.
    min_torso_visible: float = 0.35


@dataclass(frozen=True)
class RoundTiming:
    countdown_seconds: float = 3.0
    photo_display_seconds: float = 20.0


@dataclass(frozen=True)
class SettingSpec:
    key: str
    group: str  # "thresholds", "faces" or "timing"
    label: str
    hint: str
    minimum: float
    maximum: float
    step: float


_STRICT = "Höher = strenger"
SPECS = (
    SettingSpec("glasses", "thresholds", "Brille", _STRICT, 0.1, 0.95, 0.01),
    SettingSpec("beard", "thresholds", "Bart", _STRICT, 0.1, 0.95, 0.01),
    SettingSpec("hat", "thresholds", "Hut / Cap", _STRICT, 0.1, 0.95, 0.01),
    SettingSpec("hair", "thresholds", "Haarfarbe", _STRICT, 0.1, 0.95, 0.01),
    SettingSpec("shirt", "thresholds", "Shirt-Farbe", _STRICT, 0.1, 0.95, 0.01),
    SettingSpec("smile", "thresholds", "Lächeln", _STRICT, 0.1, 0.95, 0.01),
    SettingSpec("scream", "thresholds", "Mund auf (Schrei)", _STRICT, 0.1, 0.95, 0.01),
    SettingSpec("kiss", "thresholds", "Kussmund", _STRICT, 0.1, 0.95, 0.01),
    SettingSpec("min_eye_distance_px", "faces", "Mindest-Gesichtsgröße",
                "Augenabstand in Pixeln; kleinere Gesichter werden ignoriert", 6, 40, 1),
    SettingSpec("min_torso_visible", "faces", "Oberkörper sichtbar",
                "Mindestanteil der Brust im Bild für Shirt-Challenges", 0.10, 0.90, 0.05),
    SettingSpec("countdown_seconds", "timing", "Countdown", "Sekunden bis zum Foto", 1, 10, 1),
    SettingSpec("photo_display_seconds", "timing", "Foto-Anzeige",
                "Sekunden Vollbild, danach wird das Foto gelöscht", 5, 60, 1),
)
SPECS_BY_KEY = {spec.key: spec for spec in SPECS}
_THRESHOLD_KEYS = {field.name for field in fields(Thresholds)}


class SettingsStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.thresholds = Thresholds()
        self.timing = RoundTiming()
        self._load()

    def to_json(self) -> dict:
        values = {**asdict(self.thresholds), **asdict(self.timing)}
        return {
            "values": {spec.key: values[spec.key] for spec in SPECS},
            "defaults": {**asdict(Thresholds()), **asdict(RoundTiming())},
            "specs": [asdict(spec) for spec in SPECS],
        }

    def update(self, changes: dict) -> dict:
        """Apply valid changes (clamped to their range), save them and return the new settings."""
        with self._lock:
            thresholds, timing = _validated(changes)
            self.thresholds = replace(self.thresholds, **thresholds)
            self.timing = replace(self.timing, **timing)
            self._save()
        return self.to_json()

    def reset(self) -> dict:
        with self._lock:
            self.thresholds, self.timing = Thresholds(), RoundTiming()
            self._save()
        return self.to_json()

    def _load(self) -> None:
        if not SETTINGS_FILE.is_file():
            return
        try:
            saved = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            if saved.get("version") != SETTINGS_VERSION:
                raise ValueError("Werte einer älteren Version")
            thresholds, timing = _validated(saved)
        except (OSError, ValueError, AttributeError) as error:
            logger.warning("settings.json ignoriert (%s) – Standardwerte aktiv.", error)
            return
        self.thresholds = replace(Thresholds(), **thresholds)
        self.timing = replace(RoundTiming(), **timing)
        logger.info("Einstellungen aus %s geladen.", SETTINGS_FILE.name)

    def _save(self) -> None:
        values = {"version": SETTINGS_VERSION, **asdict(self.thresholds), **asdict(self.timing)}
        try:
            SETTINGS_FILE.write_text(json.dumps(values, indent=2), encoding="utf-8")
        except OSError as error:
            logger.warning("Einstellungen konnten nicht gespeichert werden: %s", error)


def _validated(changes: dict) -> tuple[dict, dict]:
    """Known keys only, as numbers clamped to their range; split into thresholds and timing."""
    thresholds, timing = {}, {}
    for key, value in changes.items():
        spec = SPECS_BY_KEY.get(key)
        if spec is None:
            continue
        try:
            number = min(max(float(value), spec.minimum), spec.maximum)
        except (TypeError, ValueError):
            continue
        (thresholds if key in _THRESHOLD_KEYS else timing)[key] = number
    return thresholds, timing


SETTINGS = SettingsStore()
