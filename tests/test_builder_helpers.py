import importlib

import pytest

import prepare
from eval import build_eval_set as helpers


@pytest.mark.parametrize("value", ["0:00", "12:59", "01:02:03", " 1:05 ", "123:01"])
def test_parse_matches_prepare(value):
    assert helpers.parse_ts(value) == prepare.parse_time(value)


@pytest.mark.parametrize("value", [None, 1, "", "1", "1:60", "1:60:00", "-1:03", "0:01.5", "a:00"])
def test_parse_rejects_like_prepare(value):
    with pytest.raises(ValueError):
        helpers.parse_ts(value)
    with pytest.raises(ValueError):
        prepare.parse_time(value)


@pytest.mark.parametrize("seconds", [0, 5, 60, 125, 1.4, 1.5, 2.5, 3599.9])
def test_format_matches_prepare(seconds):
    assert helpers.fmt_ts(seconds) == prepare.format_time(seconds)


def test_clip_key_ignores_metadata():
    assert helpers.clip_key({"file": "a.MP4", "start": "0:01", "end": "0:07", "reason": "x"}) == (
        "a.MP4", "0:01", "0:07")
    with pytest.raises(KeyError):
        helpers.clip_key({})


@pytest.mark.parametrize("other,start,end,expected", [
    ("a", 2, 4, True), ("a", 0, 2, True), ("a", 3, 5, False),
    ("a", 0, 1, False), ("b", 1, 3, False), ("a", 2, 2, False),
])
def test_overlap(other, start, end, expected):
    assert helpers.overlaps("a", 1, 3, other, start, end) is expected
    assert helpers.overlaps(other, start, end, "a", 1, 3) is expected


def test_jsonl_loads_empty_and_reports_bad_input(tmp_path):
    path = tmp_path / "input.jsonl"
    path.write_text('\n{"title": "你好"}\n\n')
    assert helpers.load_jsonl(path) == [{"title": "你好"}]
    path.write_text("")
    assert helpers.load_jsonl(path) == []
    for bad in ("broken", "[]"):
        path.write_text("{}\n" + bad)
        with pytest.raises(ValueError, match=r"input.jsonl:2:"):
            helpers.load_jsonl(path)
    missing = tmp_path / "v5-missing.jsonl"
    with pytest.raises(FileNotFoundError, match=str(missing)):
        helpers.load_jsonl(missing)


def test_input_environment_overrides(monkeypatch, tmp_path):
    with monkeypatch.context() as env:
        for name in ("INKLING_V5_PICKS_PATH", "INKLING_TRAIN_PATH", "INKLING_SHOT_PATH"):
            env.setenv(name, str(tmp_path / name))
        importlib.reload(helpers)
        assert helpers.PICKS_PATH == tmp_path / "INKLING_V5_PICKS_PATH"
        assert helpers.TRAIN_PATH == tmp_path / "INKLING_TRAIN_PATH"
        assert helpers.SHOT_PATH == tmp_path / "INKLING_SHOT_PATH"
    importlib.reload(helpers)


@pytest.mark.parametrize("module", [
    "build_eval_set_v2", "build_sft_v3_data", "build_grpo_prompts", "build_grpo_prompts_v2",
])
def test_builders_import_without_external_sources(module, monkeypatch):
    import sys
    monkeypatch.syspath_prepend(str(helpers.ROOT / "eval"))
    builder = importlib.import_module("eval." + module)
    assert callable(builder.main)
