import json

import pytest

from eval.scoring import clip_exact, score_item, summarize


CLIP = {"file": "test.MP4", "start": "0:01", "end": "0:07"}


@pytest.mark.parametrize("fmt", ["mc", "style-mc", "stability"])
@pytest.mark.parametrize("answer", ["B", json.dumps(CLIP), "", "A"])
def test_single_pick(fmt, answer):
    item = {"format": fmt, "gold": {"letter": "B", "clip": CLIP}}
    result = score_item(item, answer)
    assert result["correct"] == int(answer in ("B", json.dumps(CLIP)))
    assert result["empty"] == int(not answer)


@pytest.mark.parametrize("answer,correct,hallucinated", [
    ("None", 1, 0), ("E", 1, 0), ("A", 0, 0), ("", 0, 0),
    ("None " + json.dumps(CLIP), 0, 1),
])
def test_trap(answer, correct, hallucinated):
    result = score_item({"format": "trap", "gold": {}}, answer)
    assert (result["correct"], result["hallucinated_clip"]) == (correct, hallucinated)


@pytest.mark.parametrize("answer, correct", [
    ("None", 1), (" none\n", 1), ("E", 0), ("A", 0), ("", 0),
    ("E: test.MP4 0:01-0:07", 0), ("None: test.MP4 0:01-0:07", 0),
    ("None " + json.dumps(CLIP), 0), (json.dumps([CLIP]), 0),
])
def test_trap_with_none_gold_requires_only_refusal(answer, correct):
    item = {"format": "trap", "gold": {"letter": "NONE", "clip": None}}
    assert score_item(item, answer)["correct"] == correct


@pytest.mark.parametrize("clip,correct,file_match,near_miss", [
    (CLIP, 1, 1, 1),
    (dict(CLIP, start="0:02"), 0, 1, 1),
    (dict(CLIP, start="0:20", end="0:26"), 0, 1, 0),
    (dict(CLIP, file="other.MP4"), 0, 0, 0),
    (None, 0, 0, 0),
])
def test_sft(clip, correct, file_match, near_miss):
    result = score_item({"format": "sft", "gold": {"clip": CLIP}},
                        json.dumps(clip) if clip else "unparseable")
    assert (result["correct"], result["file_match"], result["near_miss"]) == (
        correct, file_match, near_miss)


@pytest.mark.parametrize("offset,expected", [(0.5, 1), (0.501, 0)])
def test_tolerance(offset, expected):
    assert clip_exact(dict(CLIP, start=f"0:{1 + offset}", end=f"0:{7 + offset}"), CLIP) == expected
    assert clip_exact(dict(CLIP, start="bad"), CLIP) == 0


@pytest.mark.parametrize("clips,correct,slots", [
    ([CLIP] * 3, 1, 3),
    ([CLIP, CLIP, dict(CLIP, file="other.MP4")], 0, 2),
    ([CLIP] * 2, 0, 2), ([CLIP] * 4, 0, 3),
    ([], 0, 0), (None, 0, 0),
])
def test_multi(clips, correct, slots):
    result = score_item({"format": "multi", "gold": {"clips": [CLIP] * 3}},
                        json.dumps(clips))
    assert (result["correct"], result["slots_exact"]) == (correct, slots)


@pytest.mark.parametrize("letters", [True, False])
def test_stability_agrees_on_clip_not_letter_or_metadata(letters):
    records = []
    for index, letter in enumerate(("A", "C")):
        item = {"id": f"pair-{index}", "format": "stability",
                "prompt": f"{letter}. test.MP4 0:01-0:07: description",
                "gold": {"letter": letter, "clip": CLIP}}
        answer = letter if letters else json.dumps(dict(CLIP, title=str(index), reason=str(index)))
        records.append(dict(id=item["id"], format=item["format"], **score_item(item, answer)))
    metrics = summarize("test", records, verbose=False)
    assert metrics["stability"]["acc"] == 1
    assert metrics["stability_pairs"] == {"n_pairs": 1, "agree": 1}


def test_full_raw_and_late_answer():
    text = "padding " * 500 + json.dumps(CLIP)
    result = score_item({"format": "sft", "gold": {"clip": CLIP}}, text)
    assert result["raw"] == text
    assert result["correct"] == 1


def test_unknown_format_and_empty_summary():
    with pytest.raises(ValueError, match="unknown format"):
        score_item({"format": "unknown", "gold": {}}, "")
    assert summarize("empty", [], verbose=False)["overall"] == {"n": 0, "correct": 0, "acc": 0}


def test_grpo_import_and_reward_without_tinker():
    from eval.run_grpo import score_item as reward_score

    assert reward_score({"format": "mc", "gold": {"letter": "B", "clip": CLIP}}, "B")["correct"] == 1
