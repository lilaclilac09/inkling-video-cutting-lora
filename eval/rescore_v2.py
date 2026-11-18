"""Re-score saved eval runs (v2) with the shared scorer.

Usage:
  /home/hatch/tinker-venv/bin/python eval/rescore_v2.py eval/v2-base.json [eval/v2-v1.json ...]
Reads the raw outputs back through scoring.score_item and prints the summary.
Writes <name>-rescored.json next to each input.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import score_item, summarize  # noqa: E402


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: rescore_v2.py <results.json> [...]")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    items = {}
    for line in open(os.path.join(base_dir, "eval-set-v2.jsonl"), encoding="utf-8"):
        if line.strip():
            it = json.loads(line)
            items[it["id"]] = it
    for path in sys.argv[1:]:
        results = json.load(open(path, encoding="utf-8"))
        recs = []
        for r in results:
            item = items[r["id"]]
            rec = {"id": r["id"], "format": r["format"]}
            if "pair" in r:
                rec["pair"] = r["pair"]
            if "style" in r:
                rec["style"] = r["style"]
            rec.update(score_item(item, r["raw"]))
            recs.append(rec)
        out_path = path.replace(".json", "-rescored.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(recs, f, ensure_ascii=False, indent=2)
        summarize(os.path.basename(path), recs)
        # Per-style breakdown for style-mc.
        style_recs = [r for r in recs if r["format"] == "style-mc"]
        if style_recs:
            from collections import defaultdict
            by_style = defaultdict(list)
            for r in style_recs:
                by_style[r.get("style", "?")].append(r)
            for s, sub in sorted(by_style.items()):
                print(f"    style {s}: {sum(r['correct'] for r in sub)}/{len(sub)}")
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
