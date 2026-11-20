import copy
import json
import os
import runpy
from pathlib import Path

import pytest

import cutter
import prepare_style as style
import run_sft_v3
from eval.check_overlap import find_overlaps
from eval.scoring import score_item


ROOT = Path(__file__).resolve().parents[1]
BRIEF = "Quiet human moments and slow transitions."


def pick(index=0, beats=0):
    candidates = [
        {"file": f"SYNTHETIC_{index}_{i}.MP4", "start": "0:01", "end": "0:07",
         "description": "SYNTHETIC test only"}
        for i in range(8 if beats > 1 else 4)
    ]
    request = {
        "event": f"SYNTHETIC event {index}", "date": "test-only", "style": "fresh",
        "beats": [{"title": f"Beat {i}", "why": "SYNTHETIC"} for i in range(beats)],
        "candidates": candidates,
    }
    answer = ([{key: c[key] for key in ("file", "start", "end")}
               for c in candidates[:beats]] if beats > 1 else "B")
    return {"synthetic": True, "request": request, "answer": answer}


def validated(index=0, beats=0):
    return style.validate_pick(pick(index, beats), index + 1, "my-calm", BRIEF)


@pytest.mark.parametrize("beats", [0, 1, 2, 3])
def test_exact_prompt_shape_and_resolver(beats, monkeypatch, tmp_path):
    row = pick(beats=beats)
    before = copy.deepcopy(row)
    resolve = cutter.resolve
    calls = []

    def spy(answer, request):
        calls.append((answer, request))
        return resolve(answer, request)

    monkeypatch.setattr(cutter, "resolve", spy)
    result = style.validate_pick(row, 5, "my-calm", BRIEF)
    assert calls and row == before
    expected = cutter.build_prompt(row["request"]).replace(
        f"Style: fresh — {cutter.STYLES['fresh']}", f"Style: my-calm — {BRIEF}", 1)
    assert result["prompt"] == expected
    assert score_item(result, result["answer"])["correct"] == 1
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(result["request"]))
    assert cutter.build_prompt(cutter.load_request(request_path)) == expected


@pytest.mark.parametrize("answer", ["B", "B: SYNTHETIC_0_1.MP4 0:01-0:07", None, "None"])
def test_single_answers_are_canonical(answer):
    row = pick()
    row["answer"] = answer
    result = style.validate_pick(row, 1, "my-calm", BRIEF)
    assert result["answer"] == ("None" if answer in (None, "None")
                                else "B: SYNTHETIC_0_1.MP4 0:01-0:07")
    assert score_item(result, result["answer"])["correct"] == 1


@pytest.mark.parametrize("as_text", [False, True])
def test_multi_answers_and_refusal(as_text):
    row = pick(beats=3)
    row["answer"].reverse()
    if as_text:
        row["answer"] = json.dumps(row["answer"])
    result = style.validate_pick(row, 1, "my-calm", BRIEF)
    assert result["format"] == "multi"
    assert result["gold"]["clips"][0]["file"] == "SYNTHETIC_0_2.MP4"
    assert score_item(result, result["answer"])["correct"] == 1
    row["answer"] = None
    result = style.validate_pick(row, 1, "my-calm", BRIEF)
    assert result["format"] == "trap" and result["answer"] == "None"


@pytest.mark.parametrize("beats", [0, 1, 2, 3])
def test_custom_refusal_requires_none_not_a_candidate(beats):
    row = pick(beats=beats)
    row["answer"] = None
    result = style.validate_pick(row, 1, "my-calm", BRIEF)
    assert result["gold"] == {"letter": "NONE", "clip": None}
    assert score_item(result, "None")["correct"] == 1
    assert score_item(result, "")["correct"] == 0
    assert score_item(result, "E")["correct"] == 0
    for index, clip in enumerate(row["request"]["candidates"]):
        letter = chr(65 + index)
        answer = f"{letter}: {clip['file']} {clip['start']}-{clip['end']}"
        assert score_item(result, answer)["correct"] == 0
        assert score_item(result, json.dumps(clip))["correct"] == 0
        assert score_item(result, "None " + answer)["correct"] == 0


@pytest.mark.parametrize("answer, message", [
    ("E", "no candidate letter"),
    ("B: invented.MP4 0:01-0:07", "disagree"),
    ("B: SYNTHETIC_0_1.MP4 0:02-0:08", "disagree"),
    ("None: invented.MP4 0:01-0:07", "answer must be"),
    ("", "answer must be"),
    (42, "answer must be"),
    ([], "clip lists require"),
    ({"file": "x.MP4"}, "answer must be"),
])
def test_bad_single_answers_name_the_row(answer, message):
    row = pick()
    row["answer"] = answer
    with pytest.raises(ValueError, match=f"row 7: .*{message}"):
        style.validate_pick(row, 7, "my-calm", BRIEF)


@pytest.mark.parametrize("answer, message", [
    ([], "no JSON list"),
    ("A", "no JSON list"),
    ("[invalid", "Expecting value"),
    ([None], "must be an object"),
    ([{"file": "x.MP4", "start": "0:01", "end": "0:07"}], "expected 3 clips"),
    ([{"file": "x.MP4", "start": "0:01", "end": "0:07"}] * 3, "not one of"),
])
def test_bad_multi_answers(answer, message):
    row = pick(beats=3)
    row["answer"] = answer
    with pytest.raises(ValueError, match=f"row 8: .*{message}"):
        style.validate_pick(row, 8, "my-calm", BRIEF)


@pytest.mark.parametrize("value", ["0:60", "1:02:03", "-1:00", "0:1", 1, None, "nan"])
def test_bad_timestamps(value):
    row = pick()
    row["request"]["candidates"][0]["start"] = value
    with pytest.raises(ValueError, match="row 2: timestamp"):
        style.validate_pick(row, 2, "my-calm", BRIEF)


@pytest.mark.parametrize("end, valid", [("0:00", False), ("0:01", False),
                                       ("0:03", False), ("0:04", True),
                                       ("0:21", True), ("0:22", False)])
def test_duration_boundaries(end, valid):
    row = pick()
    row["request"]["candidates"][0]["end"] = end
    if valid:
        style.validate_pick(row, 3, "my-calm", BRIEF)
    else:
        with pytest.raises(ValueError, match="row 3: clip length must be 3–20"):
            style.validate_pick(row, 3, "my-calm", BRIEF)


@pytest.mark.parametrize("field, value, message", [
    ("event", None, "event must be"), ("date", "", "date must be"),
    ("event", "event\nStyle: other", "single-line"),
    ("beats", None, "beats must be"), ("beats", [None], "beat must be"),
    ("beats", [{}], "beat title"), ("beats", [{}, {}, {}, {}], "0–3 objects"),
    ("candidates", None, "expected 4"), ("candidates", [], "expected 4"),
    ("candidates", [None] * 4, "must be an object"),
])
def test_request_shape_errors(field, value, message):
    row = pick()
    row["request"][field] = value
    with pytest.raises(ValueError, match=f"row 9: .*{message}"):
        style.validate_pick(row, 9, "my-calm", BRIEF)


def test_missing_answer_and_duplicate_candidate():
    row = pick()
    del row["answer"]
    with pytest.raises(ValueError, match="row 4: missing answer"):
        style.validate_pick(row, 4, "my-calm", BRIEF)
    row = pick()
    row["request"]["candidates"][1] = row["request"]["candidates"][0]
    with pytest.raises(ValueError, match="row 4: duplicate candidate"):
        style.validate_pick(row, 4, "my-calm", BRIEF)


@pytest.mark.parametrize("field, value, message", [
    ("file", "", "clip file must be a nonempty"),
    ("file", "../other.MP4", "basename"),
    ("file", "file with spaces.MP4", "basename"),
    ("file", "clip.mov", "basename"),
    ("description", "", "candidate description"),
])
def test_candidate_fields(field, value, message):
    row = pick()
    row["request"]["candidates"][0][field] = value
    with pytest.raises(ValueError, match=f"row 6: .*{message}"):
        style.validate_pick(row, 6, "my-calm", BRIEF)


@pytest.mark.parametrize("contents, message", [
    ("\n", "no picks"), ("\n{broken", "row 2: invalid JSON"),
    ("\nnull\n", "row 2: expected an object"),
    ("{}\n", "row 1: expected an object"),
])
def test_load_errors(tmp_path, contents, message):
    path = tmp_path / "picks.jsonl"
    path.write_text(contents)
    with pytest.raises(ValueError, match=message):
        style.load_picks(path, "my-calm", BRIEF)


def test_duplicate_requests_and_blank_lines(tmp_path):
    path = tmp_path / "picks.jsonl"
    line = json.dumps(pick()) + "\n"
    path.write_text("\n" + line)
    assert style.load_picks(path, "my-calm", BRIEF)[0]["id"] == "style-000002"
    path.write_text(line + line)
    with pytest.raises(ValueError, match="row 2: duplicate request.*row 1"):
        style.load_picks(path, "my-calm", BRIEF)


def test_seeded_split_file_groups_and_overlap():
    rows = [validated(i) for i in range(12)]
    rows[1]["request"]["candidates"][0]["file"] = rows[0]["request"]["candidates"][0]["file"]
    rows[2]["request"]["candidates"][0]["file"] = rows[1]["request"]["candidates"][1]["file"]
    train, held = style.split_rows(rows, seed=9)
    assert (train, held) == style.split_rows(rows, seed=9)
    assert (train, held) != style.split_rows(rows, seed=10)
    assert sorted(row["id"] for row in train + held) == sorted(row["id"] for row in rows)
    for part in (train, held):
        ids = {row["id"] for row in part}
        assert len(ids & {row["id"] for row in rows[:3]}) in (0, 3)
    files = lambda part: {c["file"] for row in part for c in row["request"]["candidates"]}
    assert files(train).isdisjoint(files(held))
    assert find_overlaps(held, train) == {}


@pytest.mark.parametrize("rows", [[], [validated()], [validated(), validated()]])
def test_impossible_split(rows):
    with pytest.raises(ValueError, match="two independent candidate-file groups"):
        style.split_rows(rows)


@pytest.mark.parametrize("fraction", [0, 1, -0.2, 2, float("nan")])
def test_invalid_fraction(fraction):
    with pytest.raises(ValueError, match="between 0 and 1"):
        style.split_rows([validated(i) for i in range(3)], held_out_fraction=fraction)


def test_outputs_roundtrip_to_real_sft_consumer_and_scorer(tmp_path):
    rows = [validated(i, beats=i) for i in range(4)]
    refusal = pick()
    refusal["answer"] = None
    rows[0] = style.validate_pick(refusal, 1, "my-calm", BRIEF)
    output = tmp_path / "output with spaces"
    manifest = style.write_dataset(rows, output, "my-calm", BRIEF)
    consumer = runpy.run_path(str(ROOT / "inkling-video-sft-v3.py"))
    assert manifest["train"] + manifest["held_out"] == 4
    assert manifest["held_out_overlap_ids"] == []
    assert manifest["training_executed"] is False
    assert manifest["synthetic_picks"] == 4
    assert "recommended minimum is 20" in manifest["warnings"][0]
    assert "--from-base --name inkling-my-calm" in manifest["paid_training_command"]
    for name in ("train.jsonl", "held-out.jsonl"):
        result = consumer["load_rows"](output / name)
        for row in result:
            assert consumer["build_example"](row) == (row["prompt"], row["answer"])
            assert score_item(row, row["answer"])["correct"] == 1
    assert json.loads((output / "manifest.json").read_text()) == manifest
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(ValueError, match="new or empty"):
        style.write_dataset(rows, output, "my-calm", BRIEF)
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}
    manifest = style.write_dataset(rows, tmp_path / "second", "my-calm", BRIEF, min_picks=1)
    assert not any("recommended minimum" in warning for warning in manifest["warnings"])
    for name in ("train.jsonl", "held-out.jsonl"):
        assert (output / name).read_bytes() == (tmp_path / "second" / name).read_bytes()


def test_old_sft_formats_unchanged():
    consumer = runpy.run_path(str(ROOT / "inkling-video-sft-v3.py"))
    assert consumer["build_example"]({"kind": "refusal", "prompt": "Q", "answer": "None"}) == ("Q", "None")
    clip = {"file": "test.MP4", "start": "0:01", "end": "0:07"}
    prompt, answer = consumer["build_example"]({
        "kind": "clip", "transcript_excerpt": "hello", "keyframes": ["frame"], "clip": clip})
    assert prompt == consumer["PLAYGROUND_PROMPT"] + "hello\n\nKeyframes:\nframe"
    assert json.loads(answer) == clip


@pytest.mark.parametrize("options", [[], ["--from-base"], ["--state-path", "tinker://my-state"]])
def test_runner_dispatch_is_offline_and_preserves_defaults(options, tmp_path, monkeypatch):
    for key in ("SFT_V3_DATA_PATH", "SFT_V3_STATE_PATH", "SFT_V3_FROM_BASE", "SFT_V3_CHECKPOINT_NAME"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["test"])
    load = runpy.run_path
    calls = []
    monkeypatch.setattr(run_sft_v3, "setup_tinker", lambda: calls.append("setup"))
    monkeypatch.setattr(run_sft_v3.runpy, "run_path", lambda path, **kw: calls.append((path, kw)))
    if options:
        path = tmp_path / "custom.jsonl"
        path.write_text("")
        args = ["--data", "custom.jsonl", "--name", "my-new-style", *options]
    else:
        args = []
    run_sft_v3.main(args)
    consumer = load(str(ROOT / "inkling-video-sft-v3.py"))
    assert calls == ["setup", (str(ROOT / "inkling-video-sft-v3.py"), {"run_name": "__main__"})]
    assert os.environ["SFT_V3_DATA_PATH"] == (
        str(tmp_path / "custom.jsonl") if options else "eval/sft-v3-data.jsonl")
    if options:
        assert os.environ["SFT_V3_CHECKPOINT_NAME"] == "my-new-style"
        assert consumer["DATA_PATH"] == str(tmp_path / "custom.jsonl")
        assert consumer["CHECKPOINT_NAME"] == "my-new-style"
        assert consumer["STATE_NAME"] == "my-new-style-state"
        if options == ["--from-base"]:
            assert os.environ["SFT_V3_FROM_BASE"] == "1"
            assert consumer["FROM_BASE"] is True
        else:
            assert os.environ["SFT_V3_STATE_PATH"] == "tinker://my-state"
            assert os.environ["SFT_V3_FROM_BASE"] == "0"
            assert consumer["FROM_BASE"] is False
            assert consumer["V2_STATE"] == "tinker://my-state"
    else:
        assert "SFT_V3_CHECKPOINT_NAME" not in os.environ
        assert consumer["DATA_PATH"] == "eval/sft-v3-data.jsonl"
        assert consumer["CHECKPOINT_NAME"] == "inkling-video-cutting-v3"
        assert consumer["STATE_NAME"] == "inkling-video-cutting-v3-state"
        assert consumer["FROM_BASE"] is False
        assert consumer["V2_STATE"].endswith("/weights/inkling-video-cutting-v2-state")


def test_missing_runner_data_fails_before_tinker(tmp_path, monkeypatch):
    monkeypatch.setattr(run_sft_v3, "setup_tinker", lambda: pytest.fail("auth called"))
    with pytest.raises(SystemExit) as error:
        run_sft_v3.main(["--data", str(tmp_path / "missing")])
    assert error.value.code == 2
