"""Run SFT v2 with my Tinker account.

Usage: uv run python run_sft_v2.py (after uv sync --extra train).
Set TINKER_API_KEY before running; training uses remote GPUs.
"""
import os
import runpy
import sys
from pathlib import Path

from eval.tinker_env import setup_tinker

setup_tinker()

os.environ.setdefault("INKLING_PICKS_PATH", "inkling-video-picks-27.jsonl")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.argv = ["inkling-video-sft-v2.py"]
runpy.run_path(
    str(Path(__file__).resolve().parent / "inkling-video-sft-v2.py"),
    run_name="__main__",
)
