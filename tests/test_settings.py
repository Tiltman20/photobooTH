"""Tests for adjustable thresholds and timing."""

import json

import pytest

import face_checks
import settings
from face_checks import TraitThresholds


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "SETTINGS_FILE", tmp_path / "settings.json")
    face_checks.set_thresholds(TraitThresholds())
    yield settings.SettingsStore()
    face_checks.set_thresholds(TraitThresholds())


def test_update_applies_to_checks_and_is_saved(store):
    store.update({"glasses": 0.7, "photo_display_seconds": 20})
    assert face_checks.thresholds().glasses == pytest.approx(0.7)
    assert store.timing.photo_display_seconds == 20
    assert json.loads(settings.SETTINGS_FILE.read_text())["glasses"] == pytest.approx(0.7)


def test_values_are_clamped_and_unknown_keys_ignored(store):
    store.update({"glasses": 5, "moustache": "abc", "evil": 1})
    assert face_checks.thresholds().glasses == pytest.approx(0.9)
    assert face_checks.thresholds().moustache == TraitThresholds().moustache


def test_saved_settings_are_loaded_on_start(store):
    store.update({"red_hair": 0.1})
    face_checks.set_thresholds(TraitThresholds())
    settings.SettingsStore()
    assert face_checks.thresholds().red_hair == pytest.approx(0.1)


def test_reset_restores_defaults(store):
    store.update({"clothing_color": 0.4, "countdown_seconds": 5})
    data = store.reset()
    assert data["values"]["clothing_color"] == TraitThresholds().clothing_color
    assert data["values"]["countdown_seconds"] == 3
