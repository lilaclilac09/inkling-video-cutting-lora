"""Audit my eval golds against training windows, without model calls.

Usage: uv run python eval/check_overlap.py data/picks.jsonl
Training prompt candidates count as exposure, including refusal distractors.
This measures temporal overlap, not shared topics, refusal moments or files.
"""
import argparse
import re
from pathlib import Path

try:
    from .build_eval_set import ROOT, load_jsonl, overlaps, parse_ts
except ImportError:
    from build_eval_set import ROOT, load_jsonl, overlaps, parse_ts


DEFAULT_EVAL = ROOT / "eval/eval-set-v2.jsonl"
DEFAULT_TRAINING = ROOT / "data/picks.jsonl"
AUDIT_TRAINING = (
    DEFAULT_TRAINING,
    ROOT / "inkling-video-picks-27.jsonl",
    ROOT / "eval/sft-v3-data.jsonl",
    ROOT / "eval/grpo-prompts.jsonl",
    ROOT / "eval/grpo-prompts-v2.jsonl",
)
OPTION = re.compile(
    r"(?m)^[A-H]\. (\S+\.MP4) (\d+:\d+(?::\d+)?)-(\d+:\d+(?::\d+)?):"
)


def gold_clips(row):
    gold = row.get("gold", {})
    return ([gold["clip"]] if gold.get("clip") else []) + gold.get("clips", [])


def training_windows(rows):
    windows = set()
    for row in rows:
        clips = ([row["clip"]] if row.get("clip") else []) + gold_clips(row)
        for clip in clips:
            windows.add((clip["file"], parse_ts(clip["start"]), parse_ts(clip["end"])))
        for file, start, end in OPTION.findall(row.get("prompt", "")):
            windows.add((file, parse_ts(start), parse_ts(end)))
    return sorted(windows)


def find_overlaps(items, rows):
    windows = training_windows(rows)
    found = {}
    for item in items:
        matches = set()
        for clip in gold_clips(item):
            for window in windows:
                if overlaps(clip["file"], parse_ts(clip["start"]), parse_ts(clip["end"]), *window):
                    matches.add(window)
        if matches:
            found[item["id"]] = sorted(matches)
    return found


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training", nargs="+", type=Path)
    parser.add_argument("--eval-path", type=Path, default=DEFAULT_EVAL)
    args = parser.parse_args(argv)
    try:
        items = load_jsonl(args.eval_path)
        for path in args.training:
            rows = load_jsonl(path)
            found = find_overlaps(items, rows)
            print(f"{path}: {len(rows)} rows; {len(found)}/{len(items)} eval items overlap")
            print("  " + (", ".join(sorted(found)) if found else "(none)"))
    except (OSError, ValueError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
