import json
import re
from collections import Counter
from pathlib import Path

import pytest

from eval.scoring import score_item, summarize


ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "eval"


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


ITEMS = read_rows(EVAL / "eval-set-v2.jsonl")


def key(clip):
    return tuple(clip[field] for field in ("file", "start", "end"))


def test_composition_and_ids():
    assert len(ITEMS) == 52
    assert len({item["id"] for item in ITEMS}) == 52
    assert Counter(item["format"] for item in ITEMS) == {
        "mc": 11, "sft": 11, "trap": 4, "style-mc": 16, "stability": 6, "multi": 4}


def test_gold_candidates_and_exact_training_exclusion():
    training = {key(row["clip"]) for path in (
        ROOT / "data/picks.jsonl", ROOT / "inkling-video-picks-27.jsonl")
        for row in read_rows(path)}
    for item in ITEMS:
        golds = item["gold"].get("clips", [item["gold"].get("clip")])
        candidates = set(re.findall(
            r"(?m)^[A-H]\. (\S+\.MP4) (\d+:\d+(?:\.\d+)?)-(\d+:\d+(?:\.\d+)?):",
            item["prompt"]))
        for gold in filter(None, golds):
            assert key(gold) not in training, item["id"]
            if item["format"] in ("mc", "style-mc", "stability", "multi"):
                assert key(gold) in candidates, item["id"]


def test_v1_byte_identical_prefix():
    v1 = EVAL / "eval-set-v1.jsonl"
    assert v1.read_bytes() == b"".join(
        (EVAL / "eval-set-v2.jsonl").read_bytes().splitlines(keepends=True)[:26])
    assert Counter(item["format"] for item in read_rows(v1)) == {"mc": 11, "sft": 11, "trap": 4}


@pytest.mark.parametrize("path", sorted(
    path for path in EVAL.glob("v2-*.json") if not path.name.endswith("-rescored.json")),
    ids=lambda path: path.stem)
def test_saved_raw_reproducibility(path):
    saved = path.with_name(path.stem + "-rescored.json")
    # v1/v2 have no separate rescored artifact; their original records are the reference.
    reference = json.loads((saved if saved.exists() else path).read_text())
    items = {item["id"]: item for item in ITEMS}
    records = [
        dict(record, **score_item(items[record["id"]], record["raw"]))
        for record in json.loads(path.read_text())
    ]
    actual = summarize("raw", records, verbose=False)
    expected = summarize("saved", reference, verbose=False)
    assert {fmt: metric["acc"] for fmt, metric in actual.items() if "acc" in metric} == {
        fmt: metric["acc"] for fmt, metric in expected.items() if "acc" in metric}
    if path.stem == "v2-base":
        original = summarize("live", json.loads(path.read_text()), verbose=False)
        assert original["multi"]["acc"] == 0.5
        assert actual["multi"]["acc"] == 0  # Historical truncated raws cannot recover late lists.
