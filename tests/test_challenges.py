"""Tests for the challenge catalogue and codes."""

import pytest

from challenges import KINDS, Challenge, all_variants, challenge_catalogue, parse_code, random_challenge


def test_codes_round_trip():
    for challenge in [Challenge("glasses", 2), Challenge("color", 3, "blue"), Challenge("hair", 1, "red"),
                      Challenge("rainbow", 3), Challenge("finger_sum", 1, "12")]:
        assert parse_code(challenge.code) == challenge


@pytest.mark.parametrize("code", ["", "nope", "glasses:9", "color:2", "color:2:purple", "group:x",
                                  "hair:2:red", "glasses:1:extra", "smile:2:x:y"])
def test_invalid_codes_are_rejected(code):
    assert parse_code(code) is None


def test_rare_options_need_fewer_people():
    assert parse_code("hair:1:red") is not None
    assert parse_code("hair:2:blond") is not None
    assert all(challenge.required == 1 for challenge, _ in all_variants(["hair"]) if challenge.option == "red")


def test_random_challenge_respects_kind_and_avoids_repeat():
    first = random_challenge("glasses")
    assert first.kind == "glasses" and first.required in KINDS["glasses"].sizes
    for _ in range(30):
        assert random_challenge(avoid_code="group:3").code != "group:3"


def test_every_variant_has_texts():
    for challenge, _ in all_variants(list(KINDS)):
        data = challenge.to_json()
        assert data["title"] and data["description"] and data["active"] and data["category"]
        assert "None" not in data["title"] + data["description"]


def test_trait_challenges_ask_for_larger_frames():
    assert Challenge("glasses", 1).to_json()["analysisWidth"] > 640
    assert "analysisWidth" not in Challenge("group", 3).to_json()
    assert Challenge("stencils", 2, "hats").to_json()["stencilMode"]


def test_catalogue_lists_kinds_and_free_modes():
    ids = {entry["id"] for entry in challenge_catalogue()}
    assert set(KINDS) <= ids and {"air_draw", "hands"} <= ids
