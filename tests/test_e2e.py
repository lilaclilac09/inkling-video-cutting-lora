"""CPU-only CLI integration tests; all synthetic data stays in tmp_path."""
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from eval.build_eval_set_v2 import STYLE_SETS


ROOT = Path(__file__).resolve().parents[1]
REAL_TRAINING = {
    "data/picks.jsonl": {"mc-03", "multi-02", "sft-03", "style-3-fresh"},
    "inkling-video-picks-27.jsonl": set(),
    "eval/sft-v3-data.jsonl": {"style-3-silicon-valley"},
    "eval/grpo-prompts.jsonl": {"style-3-silicon-valley"},
    "eval/grpo-prompts-v2.jsonl": set(),
}
CLEAN_COUNTS = {
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


def snapshot(root):
    files = {}
    for directory, dirs, names in os.walk(root):
        dirs[:] = [name for name in dirs if name not in (
            ".git", ".venv", "__pycache__", ".pytest_cache")]
        for name in names:
            path = Path(directory) / name
            if path.suffix != ".pyc":
                files[str(path.relative_to(root))] = (
                    hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
    return files


@pytest.fixture(autouse=True)
def repo_is_unchanged():
    before = snapshot(ROOT)
    yield
    assert snapshot(ROOT) == before


@pytest.fixture
def cli(tmp_path):
    guard = tmp_path / "guard"
    guard.mkdir()
    (guard / "sitecustomize.py").write_text(
        'import sys\n'
        'def offline(event, args):\n'
        '    if event in ("socket.connect", "socket.getaddrinfo", "socket.gethostbyname",\n'
        '                 "socket.gethostbyaddr", "socket.sendto"):\n'
        '        raise RuntimeError("E2E forbids network: " + event)\n'
        '    if event == "import" and args[0].split(".")[0] in ("tinker", "transformers"):\n'
        '        raise RuntimeError("E2E forbids model SDK imports")\n'
        'sys.addaudithook(offline)\n',
        encoding="utf-8",
    )
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("INKLING_", "TINKER_"))}
    env.update(PYTHONPATH=str(guard), PYTHONDONTWRITEBYTECODE="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")

    def run(script, *args, extra_env=None, code=0):
        result = subprocess.run(
            [sys.executable, str(ROOT / script), *map(str, args)],
            cwd=tmp_path, env=dict(env, **(extra_env or {})),
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == code, result.stdout + result.stderr
        return result

    return run


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8")


def read_rows(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def fixture_clip(file, start="0:01", end="0:07"):
    return {"file": file, "start": start, "end": end,
            "title": "TEST ONLY", "reason": "Synthetic fixture, not footage evidence."}


@pytest.fixture
def sources(tmp_path):
    picks = [dict(clip=fixture_clip(f"TEST_PICK_{i:02d}.MP4")) for i in range(20)]
    for style_set, indices in zip(STYLE_SETS[:2], ([0, 1, 7, 10], [18, 13, 15, 12])):
        for index, (file, start, end, _, _) in zip(indices, style_set["cands"]):
            picks[index] = {"clip": fixture_clip(file, f"{start // 60}:{start % 60:02d}",
                                             f"{end // 60}:{end % 60:02d}")}
    training = [
        {"clip": fixture_clip(f"TEST_TRAIN_{i:02d}.MP4"), "tags": ["test-only"],
         "node": "test", "transcript_excerpt": "", "keyframes": []}
        for i in range(27)
    ]
    shots = [
        {"clip": f"TEST_POOL_{i:02d}.MP4", "timestamp_sec": sec,
         "description": "TEST ONLY synthetic candidate"}
        for i in range(24) for sec in range(24)
    ]
    paths = [tmp_path / name for name in ("test-picks.jsonl", "test-train.jsonl", "test-shots.jsonl")]
    for path, rows in zip(paths, (picks, training, shots)):
        write_rows(path, rows)
    prefix_rows = []
    for fmt, count in (("mc", 11), ("sft", 11), ("trap", 4)):
        for index in range(count):
            clip = picks[index]["clip"]
            gold = {"letter": "E", "clip": None} if fmt == "trap" else {"letter": "A", "clip": clip}
            prefix_rows.append({
                "id": f"test-{fmt}-{index:02d}", "format": fmt, "gold": gold,
                "prompt": f"A. {clip['file']} {clip['start']}-{clip['end']}: TEST ONLY",
            })
    prefix = tmp_path / "test-v1.jsonl"
    prefix.write_bytes(b"".join(
        (json.dumps(row, ensure_ascii=False, separators=(", ", ":  ")) + "\r\n").encode("utf-8")
        for row in prefix_rows))
    return {
        "env": dict(zip(("INKLING_V5_PICKS_PATH", "INKLING_TRAIN_PATH", "INKLING_SHOT_PATH"),
                        map(str, paths))),
        "prefix": prefix, "training": paths[1],
    }


def overlap_ids(stdout):
    lines = stdout.splitlines()
    return [set() if line.strip() == "(none)" else set(line.strip().split(", "))
            for line in lines[1::2]]


def test_builders_to_overlap_cli(cli, sources, tmp_path):
    env = sources["env"]
    built = tmp_path / "test-v2.jsonl"
    cli("eval/build_eval_set_v2.py", "--v1-path", sources["prefix"], "--out", built,
        extra_env=env)
    rows = read_rows(built)
    assert built.read_bytes().startswith(sources["prefix"].read_bytes())
    assert Counter(row["format"] for row in rows) == {
        "mc": 11, "sft": 11, "trap": 4, "style-mc": 16, "stability": 6, "multi": 4}
    assert len({row["id"] for row in rows}) == 52
    assert all(row["prompt"] and "gold" in row for row in rows)
    assert all(len(row["gold"]["clips"]) == 3 for row in rows if row["format"] == "multi")
    assert all(row["gold"]["letter"] in "ABCD" for row in rows
               if row["format"] in ("mc", "style-mc", "stability"))
    sft = tmp_path / "test-sft.jsonl"
    grpo = tmp_path / "test-grpo.jsonl"
    for script, output in (("build_sft_v3_data.py", sft), ("build_grpo_prompts_v2.py", grpo)):
        cli("eval/" + script, "--eval-path", built, "--out", output, extra_env=env)
    sft_rows = read_rows(sft)
    assert Counter(row["kind"] for row in sft_rows) == {"clip": 27, "refusal": 8}
    assert all(row["answer"] == "None" for row in sft_rows if row["kind"] == "refusal")
    assert Counter(row["format"] for row in read_rows(grpo)) == {
        "mc": 11, "style-mc": 12, "trap": 16}
    result = cli("eval/check_overlap.py", "--eval-path", built, sources["training"], sft, grpo)
    assert overlap_ids(result.stdout) == [set(), set(), set()]
    assert result.stdout.count("0/52 eval items overlap") == 3
    exposed = tmp_path / "test-exposed.jsonl"
    known = rows[0]
    write_rows(exposed, [{"clip": known["gold"]["clip"]}])
    result = cli("eval/check_overlap.py", "--eval-path", built, exposed)
    assert known["id"] in overlap_ids(result.stdout)[0]


@pytest.mark.parametrize("variable", [
    "INKLING_V5_PICKS_PATH", "INKLING_TRAIN_PATH", "INKLING_SHOT_PATH",
])
def test_builder_missing_input_cli(cli, sources, tmp_path, variable):
    missing = tmp_path / f"missing-{variable}.jsonl"
    output = tmp_path / "must-not-exist.jsonl"
    result = cli(
        "eval/build_eval_set_v2.py", "--v1-path", sources["prefix"], "--out", output,
        extra_env=dict(sources["env"], **{variable: str(missing)}), code=1,
    )
    assert f"Missing input file: {missing}" in result.stderr
    assert not output.exists()


def test_real_overlap_cli(cli):
    result = cli("eval/check_overlap.py", *(ROOT / path for path in REAL_TRAINING))
    assert overlap_ids(result.stdout) == list(REAL_TRAINING.values())


def test_saved_rescoring_cli_no_write(cli, tmp_path):
    paths = sorted((ROOT / "eval").glob("v2-*.json"))
    assert {path.stem.removesuffix("-rescored") for path in paths} == set(CLEAN_COUNTS)
    before = snapshot(tmp_path)
    result = cli("eval/rescore_v2.py", "--no-write", *paths)
    assert snapshot(tmp_path) == before
    sections = re.findall(r"^--- ([^\n]+) CLEAN ---\n(.*?)(?=^--- |\Z)",
                          result.stdout, re.MULTILINE | re.DOTALL)
    assert len(sections) == len(paths)
    for name, body in sections:
        counts = CLEAN_COUNTS[Path(name).stem.removesuffix("-rescored")]
        assert f"OVERALL: {sum(counts)}/47" in body
        for fmt, count, n in zip(
            ("mc", "style-mc", "stability", "multi", "sft", "trap"), counts, (10, 14, 6, 3, 10, 4),
        ):
            match = re.search(rf"^  {fmt}: (.+)$", body, re.MULTILINE)
            assert match is not None
            metrics = ast.literal_eval(match.group(1))
            assert (metrics["n"], metrics["acc"]) == (n, round(count / n, 3))


def test_prepare_and_tinker_help_offline_cli(cli):
    audit = json.loads(cli("prepare.py", "--audit").stdout)
    assert (audit["rows"], audit["train_rows"], audit["held_out_rows"]) == (31, 23, 8)
    assert audit["contract_violations"] == []
    assert "self-check: pass" in cli("prepare.py", "--self-check").stdout
    for script in ("eval/run_eval.py", "eval/run_grpo.py"):
        assert "usage:" in cli(script, "--help").stdout


def test_subprocess_network_guard(cli, tmp_path):
    script = tmp_path / "test-guard.py"
    script.write_text(
        'import sys\n'
        'assert "sitecustomize" in sys.modules\n'
        'sys.audit("socket.connect", None, None)\n',
        encoding="utf-8",
    )
    assert "E2E forbids network: socket.connect" in cli(script, code=1).stderr
