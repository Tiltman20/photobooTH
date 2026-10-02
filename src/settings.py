"""Adjustable settings (detection thresholds and round timing), saved to settings.json.

The settings page in the browser edits these values; every change is applied
immediately and written to disk so it survives a restart.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass, fields, replace

import face_checks
from config import ROOT
from face_checks import TraitThresholds

logger = logging.getLogger("photobooth")
SETTINGS_FILE = ROOT / "settings.json"


@dataclass(frozen=True)
class RoundTiming:
    countdown_seconds: float = 3.0
    photo_display_seconds: float = 15.0


@dataclass(frozen=True)
class SettingSpec:
    key: str
    group: str  # "thresholds" or "timing"
    label: str
    hint: str
    minimum: float
    maximum: float
    step: float


SPECS = (
    SettingSpec("glasses", "thresholds", "Brille", "Höher = strenger (weniger Fehlalarme)", 0.30, 0.90, 0.01),
    SettingSpec("moustache", "thresholds", "Schnurrbart", "Höher = strenger", 0.05, 0.50, 0.01),
    SettingSpec("red_hair", "thresholds", "Rote Haare", "Anteil roter Haar-Pixel", 0.01, 0.30, 0.005),
    SettingSpec("clothing_color", "thresholds", "Kleidungsfarbe", "Anteil der Farbe am Oberkörper", 0.05, 0.60, 0.01),
    SettingSpec("min_eye_distance_px", "thresholds", "Mindest-Gesichtsgröße",
                "Augenabstand in Pixeln; kleinere Gesichter werden nicht geprüft", 6, 40, 1),
    SettingSpec("min_torso_visible", "thresholds", "Oberkörper sichtbar",
                "Mindestanteil des Oberkörpers im Bild für Farb-Challenges", 0.10, 0.90, 0.05),
    SettingSpec("countdown_seconds", "timing", "Countdown", "Sekunden bis zum Foto", 1, 10, 1),
    SettingSpec("photo_display_seconds", "timing", "Foto-Anzeige",
                "Sekunden Vollbild, danach wird das Foto gelöscht", 5, 60, 1),
)
SPECS_BY_KEY = {spec.key: spec for spec in SPECS}


class SettingsStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.timing = RoundTiming()
        self._load()

    @property
    def thresholds(self) -> TraitThresholds:
        return face_checks.thresholds()

    def to_json(self) -> dict:
        values = {**asdict(self.thresholds), **asdict(self.timing)}
        return {
            "values": {spec.key: values[spec.key] for spec in SPECS},
            "defaults": {**asdict(TraitThresholds()), **asdict(RoundTiming())},
            "specs": [asdict(spec) for spec in SPECS],
        }

    def update(self, changes: dict) -> dict:
        """Apply valid changes (clamped to their range), save them and return the new settings."""
        with self._lock:
            threshold_changes, timing_changes = self._validated(changes)
            face_checks.set_thresholds(replace(self.thresholds, **threshold_changes))
            self.timing = replace(self.timing, **timing_changes)
            self._save()
        return self.to_json()

    def reset(self) -> dict:
        with self._lock:
            face_checks.set_thresholds(TraitThresholds())
            self.timing = RoundTiming()
            self._save()
        return self.to_json()

    @staticmethod
    def _validated(changes: dict) -> tuple[dict, dict]:
        threshold_keys = {field.name for field in fields(TraitThresholds)}
        thresholds, timing = {}, {}
        for key, value in changes.items():
            spec = SPECS_BY_KEY.get(key)
            if spec is None:
                continue
            try:
                number = min(max(float(value), spec.minimum), spec.maximum)
            except (TypeError, ValueError):
                continue
            (thresholds if key in threshold_keys else timing)[key] = number
        return thresholds, timing

    def _load(self) -> None:
        if not SETTINGS_FILE.is_file():
            return
        try:
            saved = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            thresholds, timing = self._validated(saved)
            face_checks.set_thresholds(replace(TraitThresholds(), **thresholds))
            self.timing = replace(RoundTiming(), **timing)
            logger.info("Einstellungen aus %s geladen.", SETTINGS_FILE.name)
        except (OSError, ValueError) as error:
            logger.warning("settings.json ignoriert (%s) – Standardwerte aktiv.", error)

    def _save(self) -> None:
        values = {**asdict(self.thresholds), **asdict(self.timing)}
        try:
            SETTINGS_FILE.write_text(json.dumps(values, indent=2), encoding="utf-8")
        except OSError as error:
            logger.warning("Einstellungen konnten nicht gespeichert werden: %s", error)
