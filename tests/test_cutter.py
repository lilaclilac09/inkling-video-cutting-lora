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


def test_single_pick_resolves_the_letter():
    gold = EVAL["style-1-fresh"]["gold"]["clip"]
    clips = cutter.resolve("C: DJI_20260719101336_0037_D.MP4 0:01-0:07", request("request-single.json"))
    assert [(c["file"], c["start"], c["end"]) for c in clips] == [(gold["file"], gold["start"], gold["end"])]


def test_letter_and_named_window_must_agree():
    with pytest.raises(cutter.ResolveError, match="disagree"):
        cutter.resolve("C: DJI_20260719101632_0047_D.MP4 0:05-0:11", request("request-single.json"))


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
