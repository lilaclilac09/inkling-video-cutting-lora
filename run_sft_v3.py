"""Run SFT v3 with my Tinker account.

Usage: uv run python run_sft_v3.py (after uv sync --extra train).
Set TINKER_API_KEY before running; training uses remote GPUs.
"""
import argparse
import os
import runpy
import sys
from pathlib import Path

from eval.tinker_env import setup_tinker


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="custom prompt/answer training JSONL")
    parser.add_argument("--name", help="new checkpoint name (training state adds -state)")
    start = parser.add_mutually_exclusive_group()
    start.add_argument("--from-base", action="store_true", help="create a fresh rank-16 LoRA")
    start.add_argument("--state-path", help="resume my Tinker training state, not sampler weights")
    args = parser.parse_args(argv)
    if args.data:
        data = args.data.resolve()
        if not data.is_file():
            parser.error(f"missing training data: {data}")
        os.environ["SFT_V3_DATA_PATH"] = str(data)
    else:
        os.environ.setdefault("SFT_V3_DATA_PATH", "eval/sft-v3-data.jsonl")
    if args.name:
        os.environ["SFT_V3_CHECKPOINT_NAME"] = args.name
    if args.from_base:
        os.environ["SFT_V3_FROM_BASE"] = "1"
    if args.state_path:
        os.environ["SFT_V3_STATE_PATH"] = args.state_path
        os.environ["SFT_V3_FROM_BASE"] = "0"
    setup_tinker()
    os.chdir(Path(__file__).resolve().parent)
    sys.argv = ["inkling-video-sft-v3.py"]
    runpy.run_path(
        str(Path(__file__).resolve().parent / "inkling-video-sft-v3.py"),
        run_name="__main__",
    )


if __name__ == "__main__":
    main()
