import json

import pytest

from eval.build_eval_set import ROOT, load_jsonl
from eval.check_overlap import DEFAULT_EVAL, find_overlaps, main, training_windows
from eval.rescore_v2 import clean_summary, main as rescore_main, rescore


KNOWN_31 = {"mc-03", "sft-03", "multi-02", "style-3-fresh"}
EXPOSED = KNOWN_31 | {"style-3-silicon-valley"}


@pytest.mark.parametrize("path,expected", [
    ("inkling-video-picks-27.jsonl", set()),
    ("data/picks.jsonl", KNOWN_31),
    ("eval/sft-v3-data.jsonl", {"style-3-silicon-valley"}),
    ("eval/grpo-prompts.jsonl", {"style-3-silicon-valley"}),
    ("eval/grpo-prompts-v2.jsonl", set()),
])
def test_pinned_training_overlap(path, expected):
    assert set(find_overlaps(load_jsonl(DEFAULT_EVAL), load_jsonl(ROOT / path))) == expected


def clip(file="a.MP4", start="0:01", end="0:07"):
    return {"file": file, "start": start, "end": end}


def test_multi_style_candidates_and_endpoint_rules():
    items = [
        {"id": "multi", "gold": {"clips": [clip("other.MP4"), clip()]}},
        {"id": "style", "gold": {"clip": clip()}},
        {"id": "touch", "gold": {"clip": clip(start="0:07", end="0:12")}},
        {"id": "trap", "gold": {"letter": "NONE"}},
    ]
    rows = [{"clip": clip()}, {"gold": {"clip": clip()}},
            {"prompt": "A. a.MP4 0:01-0:07: shown in a refusal prompt"}]
    assert training_windows(rows) == [("a.MP4", 1, 7)]
    assert set(find_overlaps(items, rows)) == {"multi", "style"}
    assert find_overlaps(items, []) == {}
    assert find_overlaps([], rows) == {}
    with pytest.raises(ValueError):
        find_overlaps(items, [{"clip": clip(start="bad")}])


def test_overlap_cli_missing_and_empty(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        main([str(tmp_path / "missing.jsonl")])
    assert error.value.code == 2
    assert "missing.jsonl" in capsys.readouterr().err
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    main([str(path)])
    assert "0/52 eval items overlap" in capsys.readouterr().out


# mc, style-mc, stability, multi, sft, trap correct counts on the common 47 items.
EXPECTED = {
    "v2-base": (2, 7, 3, 0, 0, 0),
    "v2-v1": (8, 8, 6, 0, 2, 0),
    "v2-v2": (10, 8, 6, 2, 2, 0),
    "v2-grpo-20": (10, 10, 6, 2, 1, 0),
    "v2-grpo-40": (10, 11, 6, 2, 1, 0),
    "v2-grpo-60": (10, 10, 6, 2, 2, 0),
    "v2-v3": (9, 1, 5, 3, 0, 4),
    "v2-v3grpo-20": (10, 13, 6, 3, 0, 4),
    "v2-v3grpo-r2-20": (10, 11, 6, 3, 0, 4),
    "v2-v3grpo-r2-40": (10, 10, 6, 3, 0, 4),
    "v2-v3grpo-r2-60": (10, 11, 6, 3, 0, 4),
}


@pytest.mark.parametrize("path", sorted((ROOT / "eval").glob("v2-*.json")), ids=lambda path: path.stem)
def test_clean_scores_from_every_saved_raw(path):
    items = {item["id"]: item for item in load_jsonl(DEFAULT_EVAL)}
    recs = rescore(json.loads(path.read_text()), items)
    name = path.stem.removesuffix("-rescored")
    counts = EXPECTED[name]
    metrics = clean_summary(name, recs, EXPOSED, verbose=False)
    formats = ("mc", "style-mc", "stability", "multi", "sft", "trap")
    for fmt, n, correct in zip(formats, (10, 14, 6, 3, 10, 4), counts):
        assert metrics[fmt]["n"] == n
        assert metrics[fmt]["acc"] == round(correct / n, 3)
    assert metrics["overall"] == {"correct": sum(counts), "n": 47, "acc": round(sum(counts) / 47, 3)}


def test_rescore_cli_no_write_and_custom_clean_training(tmp_path, capsys):
    results = tmp_path / "results.json"
    results.write_text(json.dumps([{"id": "sft-03", "format": "sft", "raw": ""}]))
    rescore_main(["--no-write", "--clean-training", str(ROOT / "data/picks.jsonl"), str(results)])
    output = capsys.readouterr().out
    assert "CLEAN" in output and "OVERALL: 0/0" in output
    assert not results.with_name("results-rescored.json").exists()
    rescore_main(["--clean-training", str(ROOT / "inkling-video-picks-27.jsonl"), str(results)])
    assert results.with_name("results-rescored.json").exists()
    assert "Excluded eval IDs: (none)" in capsys.readouterr().out
    with pytest.raises(SystemExit) as error:
        rescore_main(["--no-write", "--clean-training", str(tmp_path / "missing"), str(results)])
    assert error.value.code == 2
    assert "Missing input file" in capsys.readouterr().err
