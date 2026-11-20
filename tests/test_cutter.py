import json
from pathlib import Path

import pytest

import cutter

ROOT = Path(__file__).resolve().parent.parent
EVAL = {
    row["id"]: row
    for row in map(json.loads, (ROOT / "eval" / "eval-set-v2.jsonl").read_text().splitlines())
}


def request(name):
    return cutter.load_request(ROOT / "examples" / name)


@pytest.mark.parametrize(
    "name, item_id",
    [("request-single.json", "style-1-fresh"), ("request-multi.json", "multi-01")],
)
def test_prompt_matches_the_eval_prompt_byte_for_byte(name, item_id):
    assert cutter.build_prompt(request(name)) == EVAL[item_id]["prompt"]


@pytest.mark.parametrize("fields", [
    {}, {"style": None}, {"style": ""}, {"style": " "}, {"style": "Bad Style"},
    {"style": "my_style"}, {"style": "-calm"}, {"style": "calm-"},
    {"style": ["calm"]}, {"style": 42},
])
@pytest.mark.parametrize("brief_field", [None, "style_brief", "style_description"])
def test_request_requires_a_style_slug(tmp_path, fields, brief_field):
    req = request("request-single.json")
    del req["style"]
    req.update(fields)
    if brief_field:
        req[brief_field] = "Quiet human moments."
    path = tmp_path / "request.json"
    path.write_text(json.dumps(req), encoding="utf-8")
    with pytest.raises(cutter.ResolveError, match="style must be a non-empty lowercase slug"):
        cutter.load_request(path)


@pytest.mark.parametrize("brief_field", ["style_brief", "style_description"])
def test_custom_style_slug_with_brief_loads_and_builds(tmp_path, brief_field):
    req = request("request-single.json")
    req.update(style="my-calm-recap")
    req[brief_field] = "Quiet human moments."
    path = tmp_path / "request.json"
    path.write_text(json.dumps(req), encoding="utf-8")
    assert "Style: my-calm-recap — Quiet human moments." in cutter.build_prompt(cutter.load_request(path))


@pytest.mark.parametrize("style", ["fresh", "my-calm-recap"])
def test_explicit_brief_takes_precedence_over_description(tmp_path, style):
    req = request("request-single.json")
    req.update(style=style, style_brief="My new brief.", style_description="My old description.")
    path = tmp_path / "request.json"
    path.write_text(json.dumps(req), encoding="utf-8")
    assert f"Style: {style} — My new brief." in cutter.build_prompt(cutter.load_request(path))


def test_single_pick_resolves_the_letter():
    gold = EVAL["style-1-fresh"]["gold"]["clip"]
    clips = cutter.resolve("C: DJI_20260719101336_0037_D.MP4 0:01-0:07", request("request-single.json"))
    assert [(c["file"], c["start"], c["end"]) for c in clips] == [(gold["file"], gold["start"], gold["end"])]


def test_letter_and_named_window_must_agree():
    with pytest.raises(cutter.ResolveError, match="disagree"):
        cutter.resolve("C: DJI_20260719101632_0047_D.MP4 0:05-0:11", request("request-single.json"))


def test_dash_between_file_and_window_resolves():
    answer = "C: DJI_20260719101336_0037_D.MP4 - 0:01-0:07"
    assert cutter.resolve(answer, request("request-single.json"))


@pytest.mark.parametrize(
    "answer",
    [
        "C: DJI_20260719101336_0037_D.MP4 0:01.0-0:07.5",
        "C: DJI_20260719101336_0037_D.MP4 0:01-0:07.9",
        "C: DJI_20260719101632_0047_D.MP4 0:01-0:07",
        "C: invented.mp4 0:01-0:07",
        "C: old_DJI_20260719101336_0037_D.MP4 0:01-0:07",
    ],
)
def test_fractional_or_wrong_file_windows_are_checked(answer):
    if answer.endswith("0:07.5"):
        assert cutter.resolve(answer, request("request-single.json"))
    else:
        with pytest.raises(cutter.ResolveError, match="disagree"):
            cutter.resolve(answer, request("request-single.json"))


def test_file_names_with_leading_dot_resolve():
    req = request("request-single.json")
    req["candidates"][2]["file"] = "./intro.mp4"
    assert cutter.resolve("C: ./intro.mp4 0:01-0:07", req) == [req["candidates"][2]]


def test_file_names_with_spaces_resolve():
    req = request("request-single.json")
    req["candidates"][2]["file"] = "Day 1 exterior.mp4"
    assert cutter.resolve("C: Day 1 exterior.mp4 0:01-0:07", req) == [req["candidates"][2]]


def test_none_is_a_refusal():
    assert cutter.resolve("None", request("request-single.json")) == []


def test_multi_pick_resolves_in_beat_order():
    gold = EVAL["multi-01"]["gold"]["clips"]
    clips = cutter.resolve(json.dumps(gold), request("request-multi.json"))
    assert [c["file"] for c in clips] == [g["file"] for g in gold]


def test_multi_pick_rejects_an_invented_window():
    gold = [dict(c) for c in EVAL["multi-01"]["gold"]["clips"]]
    gold[1]["end"] = "0:30"
    with pytest.raises(cutter.ResolveError, match="not one of the offered"):
        cutter.resolve(json.dumps(gold), request("request-multi.json"))


def test_cut_dry_run_prints_ffmpeg_commands(capsys, tmp_path):
    code = cutter.main([
        "cut", str(ROOT / "examples" / "request-single.json"),
        "C: DJI_20260719101336_0037_D.MP4 0:01-0:07",
        "--footage", str(tmp_path), "--out", str(tmp_path / "clips"), "--dry-run",
    ])
    assert code == 0
    assert "ffmpeg" in capsys.readouterr().out
