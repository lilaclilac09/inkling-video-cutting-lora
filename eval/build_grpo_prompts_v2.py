"""Build GRPO sweep-2 prompts (HARDER than sweep 1).

Sweep 1 saturated: by step 19 the model scored reward 1.000 on its batch
and most groups went degenerate (unanimous) — random-window distractors
are too easy to reject, and the model already masters the training moments.
Sweep 2 restores the learning signal without touching eval:
  * mc/style-mc distractors are OTHER TRAINING CLIPS (real picks, not random
    windows). The model must match the moment/style among 4 genuinely good
    clips — the choice is hard because every option is a plausible pick.
  * trap prompts come in all 4 styles (4 moments x 4 styles = 16): the
    style line must never override the refusal — a missing moment is missing
    in every style.
  * 39 prompts total: 11 mc + 12 style-mc + 16 trap.

Anti-leakage: golds and distractors are all training-example clips (never
Tier A eval golds, never eval windows). Same hand-picked style golds as
sweep 1 (STYLE_ROWS in build_grpo_prompts.py).
"""
import json
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_eval_set import (  # noqa: E402
    TRAIN_PATH, SHOT_PATH, fmt_ts, load_jsonl, overlaps, parse_ts,
)
from build_grpo_prompts import STYLE_BRIEFS, STYLE_ROWS  # noqa: E402

RNG = random.Random(44)

EVAL_PATH = "eval/eval-set-v2.jsonl"
OUT_PATH = "eval/grpo-prompts-v2.jsonl"

MISSING_MOMENTS = ["guitar", "piano", "rain", "group photo"]
VLOG_ROWS = [0, 1, 4, 6, 7, 9, 11, 13, 15, 19, 24]  # 11 distinct-file training rows


def main():
    train = load_jsonl(TRAIN_PATH)
    shots = load_jsonl(SHOT_PATH)

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

    # Random clean windows for trap distractors (same pool rules as sweep 1).
    import re
    from build_eval_set import PICKS_PATH
    picks = load_jsonl(PICKS_PATH)
    train_ranges = [(r["clip"]["file"], parse_ts(r["clip"]["start"]), parse_ts(r["clip"]["end"])) for r in train]
    pick_ranges = [(p["clip"]["file"], parse_ts(p["clip"]["start"]), parse_ts(p["clip"]["end"])) for p in picks]
    OPT_RE = re.compile(r"[A-H]\. (\S+\.MP4) (\d+:\d+(?:\.\d+)?)-(\d+:\d+(?:\.\d+)?):")
    eval_windows = set()
    for line in open(EVAL_PATH, encoding="utf-8"):
        if line.strip():
            it = json.loads(line)
            for m in OPT_RE.finditer(it["prompt"]):
                eval_windows.add((m.group(1), parse_ts(m.group(2)), parse_ts(m.group(3))))

    def candidate_windows():
        out = []
        for clip_file, length in sorted(clip_len.items()):
            for s0 in range(0, max(1, length - 5)):
                s1 = s0 + 6
                if any(overlaps(clip_file, s0, s1, f, a, b) for f, a, b in pick_ranges):
                    continue
                if any(overlaps(clip_file, s0, s1, f, a, b) for f, a, b in train_ranges):
                    continue
                if (clip_file, s0, s1) in eval_windows:
                    continue
                out.append((clip_file, s0, s1))
        return out

    windows = candidate_windows()
    train_files = {r["clip"]["file"] for r in train}
    clean_windows = [w for w in windows if w[0] not in train_files]
    RNG.shuffle(clean_windows)
    _pool_pos = [0]

    def draw_window(n, gold_file):
        out = []
        tried = 0
        while len(out) < n and tried < len(clean_windows) * 2:
            w = clean_windows[_pool_pos[0] % len(clean_windows)]
            _pool_pos[0] += 1
            tried += 1
            if w[0] == gold_file:
                continue
            if any(w[0] == d[0] and overlaps(w[0], w[1], w[2], d[0], d[1], d[2]) for d in out):
                continue
            out.append((w[0], w[1], w[2], window_desc(w[0], w[1], w[2])))
        assert len(out) == n
        return out

    letters = ["A", "B", "C", "D", "E"]

    def mc_prompt(slot_text, options, style=None):
        lines = [
            "You are cutting the Cafe Cursor Shanghai recap (July 19, 2026) into short clips.",
            "",
        ]
        if style:
            lines.append(f"Style: {style} — {STYLE_BRIEFS[style]}")
            lines.append("")
        lines.append(slot_text)
        lines.append("Which candidate segment should be used? Answer with just the letter and the clip file + start/end, e.g. \"B: DJI_...MP4 0:05-0:15\". If none of them fits, answer \"None\".")
        lines.append("")
        for letter, (fname, s0, s1, d) in zip(letters, options):
            lines.append(f"{letter}. {fname} {fmt_ts(s0)}-{fmt_ts(s1)}: {d}")
        return "\n".join(lines)

    def clip_opt(row):
        c = row["clip"]
        s0, e0 = parse_ts(c["start"]), parse_ts(c["end"])
        return (c["file"], s0, e0, window_desc(c["file"], s0, e0))

    items = []

    # --- 11 vlog mc: distractors are other training clips.
    for qi, ri in enumerate(VLOG_ROWS, start=1):
        row = train[ri]
        gold = clip_opt(row)
        others = [r for j, r in enumerate(train)
                  if j != ri and r["clip"]["file"] != row["clip"]["file"]]
        RNG.shuffle(others)
        ds, seen = [], set()
        for r in others:
            if r["clip"]["file"] in seen:
                continue
            seen.add(r["clip"]["file"])
            ds.append(clip_opt(r))
            if len(ds) == 3:
                break
        options = [gold] + ds
        RNG.shuffle(options)
        c = row["clip"]
        items.append({
            "id": f"grpo2-mc-{qi:02d}", "format": "mc", "style": None,
            "source": f"training example row {ri}: {c['file']} {c['start']}-{c['end']} (hard distractors: other training clips)",
            "prompt": mc_prompt(
                f"The recap needs a clip for this moment: \"{c['title']}\" — {c['reason']}", options),
            "gold": {"letter": letters[options.index(gold)],
                     "clip": {"file": c["file"], "start": c["start"], "end": c["end"]}},
        })

    # --- 12 style-mc: distractors are one training clip per OTHER style.
    for style, rows in STYLE_ROWS.items():
        others_styles = [s for s in STYLE_ROWS if s != style]
        for qi, ri in enumerate(rows, start=1):
            row = train[ri]
            gold = clip_opt(row)
            ds = []
            for os_ in others_styles:
                cands = [train[j] for j in STYLE_ROWS[os_]
                         if train[j]["clip"]["file"] != row["clip"]["file"]]
                ds.append(clip_opt(RNG.choice(cands)))
            options = [gold] + ds
            RNG.shuffle(options)
            c = row["clip"]
            items.append({
                "id": f"grpo2-style-{style}-{qi}", "format": "style-mc", "style": style,
                "source": f"training example row {ri}: {c['file']} {c['start']}-{c['end']} (hard distractors: one clip per other style)",
                "prompt": mc_prompt("The recap needs one more clip.", options, style=style),
                "gold": {"letter": letters[options.index(gold)],
                         "clip": {"file": c["file"], "start": c["start"], "end": c["end"]}},
            })

    # --- 16 trap: 4 moments x 4 styles.
    for moment in MISSING_MOMENTS:
        for style in ("fresh", "silicon-valley", "glamour", "hollywood"):
            ds = draw_window(4, "")
            options = ds + [("NONE", 0, 0, "None of these — the moment does not exist in the footage")]
            slot = (
                f"The recap needs a clip of this moment: \"{moment}\". "
                "If the moment does not exist anywhere in the footage, answer \"None\" — do not invent a clip."
            )
            items.append({
                "id": f"grpo2-trap-{moment.replace(' ', '-')}-{style}", "format": "trap", "style": style,
                "source": f"missing moment '{moment}' in style {style} — verified absent from all footage",
                "prompt": mc_prompt(slot, options, style=style),
                "gold": {"letter": "E", "clip": None},
            })

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    from collections import Counter
    print(Counter(it["format"] for it in items), "items ->", OUT_PATH)


if __name__ == "__main__":
    main()
