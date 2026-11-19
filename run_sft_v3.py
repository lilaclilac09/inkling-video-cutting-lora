"""Run SFT v3 with my Tinker account.

Usage: uv run python run_sft_v3.py (after uv sync --extra train).
Set TINKER_API_KEY before running; training uses remote GPUs.
"""
import os
import runpy
import sys
from pathlib import Path

from eval.tinker_env import setup_tinker

setup_tinker()

os.environ.setdefault("SFT_V3_DATA_PATH", "eval/sft-v3-data.jsonl")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.argv = ["inkling-video-sft-v3.py"]
runpy.run_path(
    str(Path(__file__).resolve().parent / "inkling-video-sft-v3.py"),
    run_name="__main__",
)
