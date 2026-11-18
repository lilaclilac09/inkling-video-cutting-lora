"""Tests for the fixed infrastructure. These must pass before any experiment runs."""

from __future__ import annotations

import json

import pytest

import prepare


# ── timestamps ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("text", "seconds"),
    [("0:00", 0), ("0:06", 6), ("1:05", 65), ("12:00", 720), ("01:02:03", 3723)],
)
def test_parse_time_accepts_mm_ss_and_h_mm_ss(text, seconds):
    assert prepare.parse_time(text) == seconds


@pytest.mark.parametrize("text", ["6", "", "0:60", "0:0a", "1:2:3:4", "01:60:00"])
def test_parse_time_rejects_malformed_input(text):
    with pytest.raises(ValueError):
        prepare.parse_time(text)


def test_format_time_round_trips_through_parse_time():
    for seconds in (0, 6, 65, 720):
        assert prepare.parse_time(prepare.format_time(seconds)) == seconds


# ── extracting the object from a model answer ───────────────────────────────


def test_extract_json_handles_plain_fenced_and_prose_answers():
    payload = '{"file": "a.mp4", "start": "0:01", "end": "0:05"}'
    assert prepare.extract_json(payload) == json.loads(payload)
    assert prepare.extract_json(f"```json\n{payload}\n```") == json.loads(payload)
    assert prepare.extract_json(f"Here is the clip. {payload} Hope that helps.") == json.loads(
        payload
    )


def test_extract_json_survives_nested_braces_and_returns_none_without_json():
    assert prepare.extract_json('prefix {"a": {"b": 1}} suffix') == {"a": {"b": 1}}
    assert prepare.extract_json("no json at all") is None
    assert prepare.extract_json("") is None


# ── the contract ────────────────────────────────────────────────────────────

ALLOWED = {"DJI_0001.MP4"}


def answer(**overrides) -> dict:
    base = {
        "file": "DJI_0001.MP4",
        "start": "0:02",
        "end": "0:10",
        "title": "The room laughs",
        "reason": "A whole room laughing reads without context.",
    }
    base.update(overrides)
    return base


def test_validate_accepts_a_conforming_answer():
    assert prepare.validate(answer(), ALLOWED) == []


def test_validate_rejects_the_v1_label_key():
    # v1 emitted "label" instead of title/reason; the contract must catch that.
    broken = answer()
    del broken["title"]
    del broken["reason"]
    broken["label"] = "a person plays piano"
    errors = prepare.validate(broken, ALLOWED)
    assert any("missing keys" in e for e in errors)
    assert any("unexpected keys" in e for e in errors)


def test_validate_rejects_a_file_the_prompt_never_offered():
    errors = prepare.validate(answer(file="DJI_9999.MP4"), ALLOWED)
    assert any("not among the prompt's keyframes" in e for e in errors)


@pytest.mark.parametrize(
    ("start", "end", "fragment"),
    [
        ("0:10", "0:02", "not after start"),
        ("0:10", "0:10", "not after start"),
        ("0:02", "0:04", "minimum is 3s"),
        ("0:00", "0:25", "maximum is 20s"),
    ],
)
def test_validate_enforces_clip_length_and_order(start, end, fragment):
    errors = prepare.validate(answer(start=start, end=end), ALLOWED)
    assert any(fragment in e for e in errors)


def test_validate_rejects_empty_reason_and_long_title():
    assert any("reason" in e for e in prepare.validate(answer(reason="  "), ALLOWED))
    assert any("limit is 60" in e for e in prepare.validate(answer(title="x" * 61), ALLOWED))


def test_validate_rejects_a_missing_object():
    assert prepare.validate(None, ALLOWED) == ["no JSON object found"]


# ── scoring ─────────────────────────────────────────────────────────────────

REFERENCE = answer()


def test_score_is_zero_when_the_schema_is_invalid():
    result = prepare.score_one("I would pick the piano clip.", REFERENCE, ALLOWED)
    assert result["score"] == 0.0
    assert result["schema_valid"] is False


def test_valid_answer_with_the_wrong_file_scores_a_quarter():
    wrong = json.dumps(answer(file="DJI_0002.MP4"))
    result = prepare.score_one(wrong, REFERENCE, {"DJI_0001.MP4", "DJI_0002.MP4"})
    assert result["schema_valid"] is True
    assert result["file_match"] is False
    assert result["score"] == 0.25


def test_exact_answer_scores_one():
    result = prepare.score_one(json.dumps(REFERENCE), REFERENCE, ALLOWED)
    assert result["score"] == 1.0
    assert result["temporal_iou"] == 1.0


def test_partial_overlap_lands_between_the_floor_and_one():
    # reference 2-10 (8s), answer 6-14 (8s): overlap 4s, union 12s.
    result = prepare.score_one(json.dumps(answer(start="0:06", end="0:14")), REFERENCE, ALLOWED)
    assert result["temporal_iou"] == pytest.approx(4 / 12, abs=1e-4)
    assert 0.25 < result["score"] < 1.0


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ((0, 10, 0, 10), None, 1.0),
        ((0, 5, 10, 15), None, 0.0),
        ((0, 10, 5, 15), None, 5 / 15),
    ],
)
def test_interval_iou(a, b, expected):
    assert prepare.interval_iou(*a) == pytest.approx(expected, abs=1e-6)


# ── the split ───────────────────────────────────────────────────────────────


def test_split_is_deterministic_disjoint_and_complete():
    rows = prepare.load_rows()
    train, held_out = prepare.split(rows)
    assert (train, held_out) == prepare.split(rows)
    assert {r.clip_key for r in train}.isdisjoint({r.clip_key for r in held_out})
    assert len(train) + len(held_out) == len(rows)
    assert len(held_out) == len({r.clip_key for r in rows}) // 4


# ── the ledger ratchet ──────────────────────────────────────────────────────


def summary(score, valid_rate):
    return {
        "schema_valid_rate": valid_rate,
        "file_match_rate": 1.0,
        "mean_iou": score,
        "mean_score": score,
    }


def test_ratchet_keeps_only_improvements_above_the_schema_gate(tmp_path, monkeypatch):
    ledger = tmp_path / "results.tsv"
    monkeypatch.setattr(prepare, "RESULTS_PATH", ledger)
    config = {"run_id": "exp1", "model": "m", "rank": 16, "lr": 2e-4, "steps": 10, "train_rows": 24}

    assert prepare.record_result(summary(0.5, 1.0), config) is True  # first run
    assert prepare.record_result(summary(0.4, 1.0), config) is False  # worse score
    assert prepare.record_result(summary(0.9, 0.5), config) is False  # high score, below gate
    assert prepare.record_result(summary(0.6, 1.0), config) is True  # better and valid

    rows = ledger.read_text(encoding="utf-8").strip().splitlines()
    assert rows[0].split("\t") == list(prepare.LEDGER_COLUMNS)
    assert len(rows) == 5
    assert [line.split("\t")[11] for line in rows[1:]] == ["yes", "no", "no", "yes"]


# ── the data itself ─────────────────────────────────────────────────────────


def test_every_labelled_row_satisfies_the_contract():
    violations = prepare.audit(prepare.load_rows())["contract_violations"]
    assert violations == []


def test_held_out_rows_have_a_usable_reference():
    _, held_out = prepare.split(prepare.load_rows())
    assert held_out, "the held-out set must not be empty"
    for row in held_out:
        assert prepare.validate(row.clip, row.source_files) == []
