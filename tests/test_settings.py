"""Tests for adjustable thresholds and timing."""

import json

import pytest

import settings
from settings import RoundTiming, Thresholds


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "SETTINGS_FILE", tmp_path / "settings.json")
    return settings.SettingsStore()


def test_update_applies_and_is_saved(store):
    store.update({"glasses": 0.7, "photo_display_seconds": 25})
    assert store.thresholds.glasses == pytest.approx(0.7)
    assert store.timing.photo_display_seconds == 25
    assert json.loads(settings.SETTINGS_FILE.read_text())["glasses"] == pytest.approx(0.7)


def test_values_are_clamped_and_unknown_keys_ignored(store):
    store.update({"glasses": 5, "beard": "abc", "evil": 1})
    assert store.thresholds.glasses == pytest.approx(0.95)
    assert store.thresholds.beard == Thresholds().beard


def test_saved_settings_are_loaded_on_start(store):
    store.update({"hair": 0.3})
    assert settings.SettingsStore().thresholds.hair == pytest.approx(0.3)


def test_settings_of_an_older_version_are_ignored(store):
    settings.SETTINGS_FILE.write_text(json.dumps({"glasses": 0.78, "countdown_seconds": 7}))
    loaded = settings.SettingsStore()
    assert loaded.thresholds == Thresholds() and loaded.timing == RoundTiming()


def test_reset_restores_defaults(store):
    store.update({"shirt": 0.8, "countdown_seconds": 5})
    data = store.reset()
    assert data["values"]["shirt"] == Thresholds().shirt
    assert data["values"]["countdown_seconds"] == RoundTiming().countdown_seconds
