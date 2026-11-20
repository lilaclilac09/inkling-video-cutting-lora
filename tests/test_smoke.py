"""Fast offline checks of public CLIs; outputs stay in tmp_path."""
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.smoke


def snapshot():
    files = {}
    directories = set()
    for directory, dirs, names in os.walk(ROOT):
        dirs[:] = [name for name in dirs if name not in (
            ".git", ".venv", "__pycache__", ".pytest_cache")]
        directories.add(str(Path(directory).relative_to(ROOT)))
        for name in names:
            path = Path(directory) / name
            if path.suffix != ".pyc":
                files[str(path.relative_to(ROOT))] = (
                    hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
    return files, directories


@pytest.fixture
def cli(tmp_path):
    before = snapshot()
    guard = tmp_path / "guard"
    guard.mkdir()
    (guard / "sitecustomize.py").write_text(
        'import sys\n'
        'def offline(event, args):\n'
        '    if event in ("socket.connect", "socket.getaddrinfo", "socket.gethostbyname",\n'
        '                 "socket.gethostbyaddr", "socket.sendto"):\n'
        '        raise RuntimeError("SMOKE forbids network: " + event)\n'
        '    if event == "import" and args[0].split(".")[0] in ("tinker", "transformers"):\n'
        '        raise RuntimeError("SMOKE forbids model SDK imports")\n'
        'sys.addaudithook(offline)\n',
        encoding="utf-8",
    )
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("INKLING_", "TINKER_", "SFT_V3_"))}
    env.update(PYTHONPATH=os.pathsep.join((str(guard), str(ROOT))),
               PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")

    def run(script, *args):
        result = subprocess.run(
            [sys.executable, str(ROOT / script), *map(str, args)],
            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout

    yield run
    assert snapshot() == before


@pytest.mark.parametrize("script,args,line", [
    ("prepare.py", ["--self-check"], "self-check: pass"),
    ("cutter.py", ["prompt", ROOT / "examples/request-single.json"], "Style: fresh —"),
    ("cutter.py", ["prompt", ROOT / "examples/request-multi.json"],
     "The recap needs 3 clips in one pass, one per beat, in order:"),
    ("cutter.py", [
        "cut", ROOT / "examples/request-single.json",
        "C: DJI_20260719101336_0037_D.MP4 0:01-0:07", "--dry-run",
    ], "ffmpeg -y"),
    ("run_sft_v2.py", ["--help"], "usage: run_sft_v2.py"),
    ("run_sft_v3.py", ["--help"], "usage: run_sft_v3.py"),
    ("eval/run_eval.py", ["--help"], "usage: run_eval.py"),
    ("eval/run_grpo.py", ["--help"], "usage: run_grpo.py"),
], ids=["self-check", "single-prompt", "multi-prompt", "cut-dry-run",
        "sft-v2-help", "sft-v3-help", "eval-help", "grpo-help"])
def test_entry_point(cli, script, args, line):
    assert line in cli(script, *args)


def test_train_dry_run(cli, tmp_path):
    # train.py anchors runs/ to __file__; keep the script unchanged but isolated.
    script = tmp_path / "train.py"
    shutil.copyfile(ROOT / "train.py", script)
    output = cli(script, "--dry-run", "--run-id", "smoke")
    assert "dry run: ledger not written" in output
    predictions = tmp_path / "runs/smoke/predictions.jsonl"
    assert predictions.is_file() and predictions.read_text(encoding="utf-8").strip()


def test_prepare_style(cli, tmp_path):
    output = tmp_path / "style"
    stdout = cli(
        "prepare_style.py", ROOT / "examples/my-style-picks.jsonl",
        "--style", "smoke-style", "--brief", "Synthetic smoke-test style.", "--out", output,
    )
    assert "Prepared 4 picks:" in stdout
    assert all((output / name).is_file()
               for name in ("train.jsonl", "held-out.jsonl", "manifest.json"))
