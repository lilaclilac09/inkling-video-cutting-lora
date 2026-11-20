"""Build SFT stage-2 (v3) data: replay + refusal teaching.

35 rows: the original 27 training examples (replay — prevents forgetting)
+ 8 refusal rows. Refusal rows teach the "None" answer on moments that are
verified absent from the footage but are NOT the eval's trap moments, so
the eval stays clean:
  fireworks / a dog on stage / snow / a birthday cake
Evidence of absence: zero mentions in any of the 1411 shot-list
descriptions, the 20 v5 picks, or the 27 training rows (text-level check;
the eval's own 4 moments were verified harder, by watching clips in
round 2 — see round2-findings.md). Each moment appears twice: once in the
vlog template, once in another style, so refusal is style-independent.
Distractors are clean windows excluding all eval-v2 windows.

Row shapes:
  {"kind": "clip", ...round-1 row...}      -> long SFT prompt, clip JSON answer
  {"kind": "refusal", "prompt": ..., "answer": "None", ...} -> mc trap prompt, "None"
"""
import argparse
import json
import random
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_eval_set import (  # noqa: E402
    PICKS_PATH, TRAIN_PATH, SHOT_PATH, fmt_ts, load_jsonl, overlaps, parse_ts,
)
from build_grpo_prompts import STYLE_BRIEFS  # noqa: E402

RNG = random.Random(45)

EVAL_PATH = "eval/eval-set-v2.jsonl"
OUT_PATH = "eval/sft-v3-data.jsonl"

REFUSALS = [
    ("fireworks", None),
    ("fireworks", "fresh"),
    ("a dog on stage", None),
    ("a dog on stage", "silicon-valley"),
    ("snow", None),
    ("snow", "glamour"),
    ("a birthday cake", None),
    ("a birthday cake", "hollywood"),
]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-path", default=EVAL_PATH)
    parser.add_argument("--out", default=OUT_PATH)
    args = parser.parse_args(argv)
    train = load_jsonl(TRAIN_PATH)
    shots = load_jsonl(SHOT_PATH)
    picks = load_jsonl(PICKS_PATH)

    desc = {}
    for r in shots:
        desc[(r["clip"], int(r["timestamp_sec"]))] = r.get("description", "")
    from collections import defaultdict
    clip_len = defaultdict(int)
    for r in shots:
        clip_len[r["clip"]] = max(clip_len[r["clip"]], int(r["timestamp_sec"]) + 1)

    def window_desc(clip_file, s0, s1):
        parts = []
        for sec in range(int(s0), int(s1)):
            d = desc.get((clip_file, sec), "")
            if d and (not parts or parts[-1] != d):
                parts.append(d)
        return " / ".join(parts) if parts else "(no shot description)"

    OPT_RE = re.compile(r"[A-H]\. (\S+\.MP4) (\d+:\d+(?:\.\d+)?)-(\d+:\d+(?:\.\d+)?):")
    eval_windows = set()
    for line in open(args.eval_path, encoding="utf-8"):
        if line.strip():
            it = json.loads(line)
            for m in OPT_RE.finditer(it["prompt"]):
                eval_windows.add((m.group(1), parse_ts(m.group(2)), parse_ts(m.group(3))))

    train_ranges = [(r["clip"]["file"], parse_ts(r["clip"]["start"]), parse_ts(r["clip"]["end"])) for r in train]
    pick_ranges = [(p["clip"]["file"], parse_ts(p["clip"]["start"]), parse_ts(p["clip"]["end"])) for p in picks]

    windows = []
    for clip_file, length in sorted(clip_len.items()):
        for s0 in range(0, max(1, length - 5)):
            s1 = s0 + 6
            if any(overlaps(clip_file, s0, s1, f, a, b) for f, a, b in pick_ranges):
                continue
            if any(overlaps(clip_file, s0, s1, f, a, b) for f, a, b in train_ranges):
                continue
            if (clip_file, s0, s1) in eval_windows:
                continue
            windows.append((clip_file, s0, s1))
    train_files = {r["clip"]["file"] for r in train}
    clean_windows = [w for w in windows if w[0] not in train_files]
    RNG.shuffle(clean_windows)
    _pos = [0]

    def draw(n):
        out = []
        tried = 0
        while len(out) < n and tried < len(clean_windows) * 2:
            w = clean_windows[_pos[0] % len(clean_windows)]
            _pos[0] += 1
            tried += 1
            if any(w[0] == d[0] and overlaps(w[0], w[1], w[2], d[0], d[1], d[2]) for d in out):
                continue
            out.append((w[0], w[1], w[2], window_desc(w[0], w[1], w[2])))
        assert len(out) == n
        return out

    letters = ["A", "B", "C", "D", "E"]

    def trap_prompt(moment, options, style):
        lines = ["You are cutting the Cafe Cursor Shanghai recap (July 19, 2026) into short clips.", ""]
        if style:
            lines.append(f"Style: {style} — {STYLE_BRIEFS[style]}")
            lines.append("")
        lines.append(
            f"The recap needs a clip of this moment: \"{moment}\". "
            "If the moment does not exist anywhere in the footage, answer \"None\" — do not invent a clip."
        )
        lines.append("Which candidate segment should be used? Answer with just the letter and the clip file + start/end, e.g. \"B: DJI_...MP4 0:05-0:15\". If none of them fits, answer \"None\".")
        lines.append("")
        for letter, (fname, s0, s1, d) in zip(letters, options):
            lines.append(f"{letter}. {fname} {fmt_ts(s0)}-{fmt_ts(s1)}: {d}")
        return "\n".join(lines)

    rows = []
    for r in train:
        row = dict(r)
        row["kind"] = "clip"
        rows.append(row)
    for moment, style in REFUSALS:
        ds = draw(4)
        options = ds + [("NONE", 0, 0, "None of these — the moment does not exist in the footage")]
        rows.append({
            "kind": "refusal",
            "moment": moment,
            "style": style,
            "prompt": trap_prompt(moment, options, style),
            "answer": "None",
            "evidence": "zero mentions in 1411 shot descriptions, 20 picks, 27 training rows",
        })

    with open(args.out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    from collections import Counter
    print(Counter(r["kind"] for r in rows), "rows ->", args.out)


if __name__ == "__main__":
    main()
