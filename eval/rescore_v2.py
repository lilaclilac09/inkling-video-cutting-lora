"""Re-score saved eval runs (v2) with the shared scorer.

Usage:
  uv run python eval/rescore_v2.py eval/v2-base.json [eval/v2-v1.json ...]
Reads the raw outputs back through scoring.score_item and prints the summary.
Writes <name>-rescored.json next to each input unless --no-write is set.
I also report a clean subset excluding overlap with --clean-training (default:
the union of my five checked-in training files). Repeat that flag to choose
the union across other training files.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import score_item, summarize  # noqa: E402
from check_overlap import AUDIT_TRAINING, DEFAULT_EVAL, find_overlaps  # noqa: E402
from build_eval_set import load_jsonl  # noqa: E402


def rescore(results, items):
    return [
        dict(
            {key: result[key] for key in ("id", "format", "pair", "style") if key in result},
            **score_item(items[result["id"]], result["raw"]),
        )
        for result in results
    ]


def clean_summary(name, recs, excluded, verbose=True):
    return summarize(name, [rec for rec in recs if rec["id"] not in excluded], verbose=verbose)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+")
    parser.add_argument("--eval-path", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--clean-training", type=Path, action="append")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)
    try:
        eval_items = load_jsonl(args.eval_path)
        training_paths = args.clean_training or AUDIT_TRAINING
        excluded = set()
        for training_path in training_paths:
            excluded.update(find_overlaps(eval_items, load_jsonl(training_path)))
    except (OSError, ValueError) as error:
        parser.exit(2, f"{error}\n")
    items = {item["id"]: item for item in eval_items}
    print("Clean exclusion training: " + ", ".join(str(path) for path in training_paths))
    print("Excluded eval IDs: " + (", ".join(sorted(excluded)) or "(none)"))
    for path in args.results:
        results = json.load(open(path, encoding="utf-8"))
        recs = rescore(results, items)
        out_path = str(Path(path).with_name(Path(path).stem + "-rescored.json"))
        if not args.no_write:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(recs, f, ensure_ascii=False, indent=2)
        summarize(os.path.basename(path), recs)
        clean_summary(os.path.basename(path) + " CLEAN", recs, excluded)
        # Per-style breakdown for style-mc.
        style_recs = [r for r in recs if r["format"] == "style-mc"]
        if style_recs:
            from collections import defaultdict
            by_style = defaultdict(list)
            for r in style_recs:
                by_style[r.get("style", "?")].append(r)
            for s, sub in sorted(by_style.items()):
                print(f"    style {s}: {sum(r['correct'] for r in sub)}/{len(sub)}")
        if not args.no_write:
            print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
