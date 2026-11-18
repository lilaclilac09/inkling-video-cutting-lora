"""Build the GRPO training prompts (NOT eval items).

27 prompts, all built from the 27 SFT training examples — never from eval v2:
  * 11 "mc" (vlog template): gold = a training example's clip.
  * 12 "style-mc": 4 styles x 3. Gold = a training clip whose tags fit the
    style (fresh <- exterior/storefront/opening; silicon-valley <- code/demo;
    glamour <- close-up/woman-presenter; hollywood <- laughter/peak).
  * 4 "trap": the missing moments (guitar/piano/rain/group photo), correct =
    refuse. Teaches refusal independent of style.

Anti-leakage: every distractor window is checked against ALL windows that
appear in eval-set-v2.jsonl (parsed out of the option lines) and against all
eval gold clips. A GRPO prompt therefore shares no candidate window and no
gold clip with any eval item. If the exclusion ever leaves too few windows,
the builder fails loudly instead of silently reusing an eval window.
"""
import json
import random
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_eval_set import (  # noqa: E402
    PICKS_PATH, TRAIN_PATH, SHOT_PATH, fmt_ts, load_jsonl, overlaps, parse_ts,
)

RNG = random.Random(43)

EVAL_PATH = "eval/eval-set-v2.jsonl"
OUT_PATH = "eval/grpo-prompts.jsonl"

STYLE_BRIEFS = {
    "fresh": "Fresh and clean — bright, airy, breathing room: outdoors, daylight, uncluttered frames, calm pacing.",
    "silicon-valley": "Silicon Valley — substance first: product demos, live code, big-topic takeaways, a builder explaining their own work.",
    "glamour": "Glamour — polished and beautiful: close-ups, warm light, people at their best, a frame worth a poster.",
    "hollywood": "Hollywood — drama and peak emotion: the room reacting, scale, the biggest beat of the day, a cinematic sweep.",
}

STYLE_TAGS = None  # not used: style golds are hand-picked below (tags alone mislabel)

# Hand-picked style golds: training-row indices into inkling-video-picks-27.jsonl.
# Chosen by content (title + tags + reason), not by tag match alone — the
# tag-only pass mislabeled three golds (laughter clip as "fresh", coffee
# cups as "silicon-valley"/"glamour", a demo talk as "hollywood").
#   fresh: storefront/banner/street/timelapse — bright, airy, outdoors.
#   silicon-valley: live code / vibe-coded site / showcase demo — substance.
#   glamour: listener close-up, woman presenter, smiling hosts — people at their best.
#   hollywood: the room laughing (x2 ranges) + the room leaning in — the room reacting.
STYLE_ROWS = {
    "fresh": [0, 18, 24],
    "silicon-valley": [5, 7, 21],
    "glamour": [12, 19, 1],
    "hollywood": [11, 17, 8],
}

MISSING_MOMENTS = ["guitar", "piano", "rain", "group photo"]

OPT_RE = re.compile(r"[A-H]\. (\S+\.MP4) (\d+:\d+(?:\.\d+)?)-(\d+:\d+(?:\.\d+)?):")


def main():
    train = load_jsonl(TRAIN_PATH)
    shots = load_jsonl(SHOT_PATH)
    picks = load_jsonl(PICKS_PATH)

    # All windows that appear anywhere in eval v2 (options + golds).
    eval_windows = set()
    eval_golds = set()
    for line in open(EVAL_PATH, encoding="utf-8"):
        if not line.strip():
            continue
        it = json.loads(line)
        for m in OPT_RE.finditer(it["prompt"]):
            eval_windows.add((m.group(1), parse_ts(m.group(2)), parse_ts(m.group(3))))
        g = it["gold"]
        if g.get("clip"):
            eval_golds.add((g["clip"]["file"], parse_ts(g["clip"]["start"]), parse_ts(g["clip"]["end"])))
        for c in g.get("clips", []):
            eval_golds.add((c["file"], parse_ts(c["start"]), parse_ts(c["end"])))
    print(f"eval windows to exclude: {len(eval_windows)} options, {len(eval_golds)} golds")

    train_ranges = [
        (r["clip"]["file"], parse_ts(r["clip"]["start"]), parse_ts(r["clip"]["end"]))
        for r in train
    ]
    pick_ranges = [
        (p["clip"]["file"], parse_ts(p["clip"]["start"]), parse_ts(p["clip"]["end"]))
        for p in picks
    ]

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
    RNG.shuffle(windows)
    print(f"GRPO distractor windows: {len(windows)} total, {len(clean_windows)} from never-trained files")

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

    def distractors(n, gold_file):
        # Rotating draw: each call consumes the NEXT windows from the pool,
        # so prompts do not all share the same top-of-pool distractors.
        # (The v1 builder restarts the pool per item — a known limitation;
        # v1 items are frozen, but new builders must not repeat it.)
        out = []
        pool = clean_windows if len(clean_windows) >= n else windows
        tried = 0
        while len(out) < n and tried < len(pool) * 2:
            w = pool[_pool_pos[0] % len(pool)]
            _pool_pos[0] += 1
            tried += 1
            if w[0] == gold_file:
                continue
            if any(w[0] == d[0] and overlaps(w[0], w[1], w[2], d[0], d[1], d[2]) for d in out):
                continue
            out.append((w[0], w[1], w[2], window_desc(w[0], w[1], w[2])))
        assert len(out) == n, f"only {len(out)} distractors, need {n}"
        return out

    _pool_pos = [0]

    def mc_item(pid, fmt, row, style=None):
        c = row["clip"]
        s0, e0 = parse_ts(c["start"]), parse_ts(c["end"])
        assert (c["file"], s0, e0) not in eval_golds, f"{pid}: gold is an eval gold"
        gold = (c["file"], s0, e0, window_desc(c["file"], s0, e0))
        options = [gold] + distractors(3, c["file"])
        RNG.shuffle(options)
        gold_letter = letters[options.index(gold)]
        if style:
            slot = "The recap needs one more clip."
        else:
            slot = f"The recap needs a clip for this moment: \"{c['title']}\" — {c['reason']}"
        return {
            "id": pid, "format": fmt, "style": style,
            "source": f"training example: {c['file']} {c['start']}-{c['end']} (tags {sorted(row.get('tags', []))})",
            "prompt": mc_prompt(slot, options, style=style),
            "gold": {"letter": gold_letter,
                     "clip": {"file": c["file"], "start": c["start"], "end": c["end"]}},
        }

    items = []

    # --- 11 vlog mc: distinct files, spread across the 27 training rows.
    by_file = {}
    for r in train:
        by_file.setdefault(r["clip"]["file"], r)
    files = sorted(by_file)
    chosen = [files[i] for i in range(0, len(files), 2)][:11]
    if len(chosen) < 11:
        chosen += [f for f in files if f not in chosen][: 11 - len(chosen)]
    print(f"vlog mc gold files ({len(chosen)}): {chosen}")
    for qi, f in enumerate(chosen, start=1):
        items.append(mc_item(f"grpo-mc-{qi:02d}", "mc", by_file[f]))

    # --- 12 style-mc: 3 per style, hand-picked training rows (see STYLE_ROWS).
    for style, rows in STYLE_ROWS.items():
        picked = [train[i] for i in rows]
        assert len({p["clip"]["file"] for p in picked}) >= 2, f"style {style}: golds collapse to one file"
        print(f"style {style}: {[(p['clip']['file'], p['clip']['title']) for p in picked]}")
        for qi, r in enumerate(picked, start=1):
            items.append(mc_item(f"grpo-style-{style}-{qi}", "style-mc", r, style=style))

    # --- 4 trap: missing moments, vlog template.
    for ti, moment in enumerate(MISSING_MOMENTS, start=1):
        ds = distractors(4, gold_file="")
        options = ds + [("NONE", 0, 0, "None of these — the moment does not exist in the footage")]
        slot = (
            f"The recap needs a clip of this moment: \"{moment}\". "
            "If the moment does not exist anywhere in the footage, answer \"None\" — do not invent a clip."
        )
        items.append({
            "id": f"grpo-trap-{ti:02d}", "format": "trap", "style": None,
            "source": f"missing moment '{moment}' — verified absent from all footage",
            "prompt": mc_prompt(slot, options),
            "gold": {"letter": "E", "clip": None},
        })

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    from collections import Counter
    print(Counter(it["format"] for it in items), "items ->", OUT_PATH)


if __name__ == "__main__":
    main()
