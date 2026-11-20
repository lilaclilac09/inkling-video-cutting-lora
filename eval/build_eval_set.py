"""Shared inputs and helpers for my frozen-eval and training-data builders.

I restored these interfaces from their callers, not from the lost builder.
External source data must be supplied; I never reconstruct it from eval prompts.
"""
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PICKS_PATH = Path(os.environ.get("INKLING_V5_PICKS_PATH", ROOT / "sources/v5-picks.jsonl"))
TRAIN_PATH = Path(os.environ.get("INKLING_TRAIN_PATH", ROOT / "inkling-video-picks-27.jsonl"))
SHOT_PATH = Path(os.environ.get("INKLING_SHOT_PATH", ROOT / "sources/shot-list.jsonl"))


def parse_ts(value):
    """Parse integer M:SS or H:MM:SS, with the same rules as prepare.py."""
    if not isinstance(value, str):
        raise ValueError(f"timestamp is not a string: {value!r}")
    parts = value.strip().split(":")
    if len(parts) not in (2, 3) or not all(p.isdigit() for p in parts):
        raise ValueError(f"bad timestamp: {value!r}")
    numbers = [int(p) for p in parts]
    if len(numbers) == 2:
        minutes, seconds = numbers
        if seconds >= 60:
            raise ValueError(f"seconds out of range: {value!r}")
        return minutes * 60 + seconds
    hours, minutes, seconds = numbers
    if minutes >= 60 or seconds >= 60:
        raise ValueError(f"minutes or seconds out of range: {value!r}")
    return hours * 3600 + minutes * 60 + seconds


def fmt_ts(seconds):
    total = int(round(seconds))
    return f"{total // 60}:{total % 60:02d}"


def clip_key(clip):
    return clip["file"], clip["start"], clip["end"]


def overlaps(file_a, start_a, end_a, file_b, start_b, end_b):
    """Positive-duration intersection; touching endpoints do not overlap."""
    return file_a == file_b and max(start_a, start_b) < min(end_a, end_b)


def load_jsonl(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing input file: {path}. Supply the original source; "
            "override paths with INKLING_V5_PICKS_PATH, INKLING_TRAIN_PATH "
            "or INKLING_SHOT_PATH. No source data is fabricated."
        )
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("expected a JSON object")
        except ValueError as error:
            raise ValueError(f"{path}:{number}: {error}") from error
        rows.append(row)
    return rows
