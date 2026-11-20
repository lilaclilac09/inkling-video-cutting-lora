import json

import pytest

import cutter
import onboarding


def test_welcome_shows_banner_and_every_slide(capsys):
    assert onboarding.welcome(interactive=False) == 0
    out = capsys.readouterr().out
    assert "i n k l i n g" in out
    for title, _, command in onboarding.SLIDES:
        assert title in out and command in out


def answers(monkeypatch, values):
    it = iter(values)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(it))


def windows(n):
    rows = []
    for i in range(n):
        rows += [f"clip{i}.mp4", "0:01", "0:06", f"shot {i}"]
    return rows


def test_init_writes_a_request_the_cutter_accepts(monkeypatch, tmp_path):
    out = tmp_path / "req.json"
    answers(monkeypatch, ["Demo Day", "", "2", "", *windows(4)])
    assert onboarding.init(out) == 0
    request = cutter.load_request(out)
    assert request["style"] == "fresh" and request["date"] == onboarding._today()
    assert len(request["candidates"]) == 4
    assert cutter.resolve("B: clip1.mp4 0:01-0:06", request) == [request["candidates"][1]]


def test_init_custom_style_and_beats(monkeypatch, tmp_path):
    out = tmp_path / "req.json"
    beats = ["Arrival", "sets the scene", "Demo", "the product", "Cheers", "the payoff"]
    answers(
        monkeypatch,
        ["Demo Day", "May 1", "6", "anime-summer", "Saturated colours, quick cuts.", "3", *beats, *windows(8)],
    )
    onboarding.init(out)
    request = cutter.load_request(out)
    assert len(request["beats"]) == 3 and len(request["candidates"]) == 8
    assert "Style: anime-summer — Saturated colours, quick cuts." in cutter.build_prompt(request)


def test_init_reasks_a_window_that_ends_before_it_starts(monkeypatch, tmp_path, capsys):
    out = tmp_path / "req.json"
    answers(monkeypatch, ["E", "", "1", "", "a.mp4", "0:09", "0:02", *windows(4)])
    onboarding.init(out)
    assert "try again" in capsys.readouterr().out
    assert json.loads(out.read_text())["candidates"][0]["file"] == "clip0.mp4"


def test_unknown_style_without_description_is_rejected(tmp_path):
    path = tmp_path / "req.json"
    path.write_text(json.dumps({"style": "noir", "beats": [], "candidates": []}))
    with pytest.raises(cutter.ResolveError, match="style_description"):
        cutter.load_request(path)
