"""Build eval set v2 for the Inkling video-cutting LoRA.

v2 = v1 (26 items, copied verbatim) + 26 new items = 52 total.

New in v2 — the style dimension (Aileen's brief, 2026-11-18: the cut must be
a reusable TEMPLATE; style is a parameter, not a one-off; it must stay stable
across many files and run faster):

* "style-mc" (16 = 4 candidate sets x 4 styles). Same 4 candidates, same
  slot text, only the Style line changes; the gold differs by style. This is
  the rigorous test that the model conditions on the style parameter at all.
  Sets 1-2 use Tier A v5 picks as candidates (grounded in Aileen-approved
  pick reasons); sets 3-4 use 6s shot-list windows (grounded in the shot
  descriptions, quoted in `style_evidence`). Every gold is Tier A held-out
  (exact range not in the 27 training examples, no overlap with any training
  example); sets 3-4 windows are asserted non-overlapping with every pick
  and every training example.
* "stability" (6 = 3 Tier A golds x 2 distractor sets). Same slot text as
  the v1 mc item for that pick, different distractors. The pair must agree:
  a template that flips its pick when the distractors change is not stable.
* "multi" (4). One call, three beats, 8 candidates: pick one clip per beat,
  in order, as a JSON list. This is the "many files at once, faster" axis:
  the template must batch picks instead of one call per clip.

Styles (values of the template's `style` parameter):
  vlog           the reference recap style (v10)
  fresh          bright, airy, breathing room
  silicon-valley substance first: demos, live code, big takeaways
  glamour        polished and beautiful: close-ups, warm light
  hollywood      drama and peak emotion: the room reacting, scale

Deterministic: fixed seed 42. v1 items are copied byte-identical from
eval-set-v1.jsonl (never regenerated), so v1 scores stay comparable.
"""
import argparse
import json
import random
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_eval_set import (  # noqa: E402
    PICKS_PATH, TRAIN_PATH, SHOT_PATH, clip_key, fmt_ts, load_jsonl, overlaps,
    parse_ts,
)

RNG = random.Random(42)

V1_PATH = "eval/eval-set-v1.jsonl"
OUT_PATH = "eval/eval-set-v2.jsonl"

STYLE_BRIEFS = {
    "fresh": "Fresh and clean — bright, airy, breathing room: outdoors, daylight, uncluttered frames, calm pacing.",
    "silicon-valley": "Silicon Valley — substance first: product demos, live code, big-topic takeaways, a builder explaining their own work.",
    "glamour": "Glamour — polished and beautiful: close-ups, warm light, people at their best, a frame worth a poster.",
    "hollywood": "Hollywood — drama and peak emotion: the room reacting, scale, the biggest beat of the day, a cinematic sweep.",
    "vlog": "Vlog — the reference recap style: warm, human, one flowing story; a moment that works standalone.",
}

# Style sets: (file, start_sec, end_sec, style, evidence).
# Sets 1-2 are Tier A v5 picks; sets 3-4 are shot-list windows.
STYLE_SETS = [
    {
        "set": 1, "slot": "The recap needs one more clip.",
        "cands": [
            ("DJI_20260719101336_0037_D.MP4", 1, 7, "fresh",
             "v5 pick 0 'Arrival on Yanqing Road': whole storefront, awning and plane trees, people outside"),
            ("timelapse.MP4", 6, 14, "hollywood",
             "v5 pick 1 'The room fills up': timelapse of the audience arriving under the paper lanterns"),
            ("DJI_20260719101632_0047_D.MP4", 5, 11, "silicon-valley",
             "v5 pick 7 'The Agent era, on title': the most legible title slide of the day"),
            ("DJI_20260719092043_0007_D.MP4", 0, 4, "glamour",
             "v5 pick 10 'Audience with a gimbal': a quick audience close-up, energy in the seats"),
        ],
    },
    {
        "set": 2, "slot": "The recap needs one more clip.",
        "cands": [
            ("DJI_20260719104432_0054_D.MP4", 1, 6, "fresh",
             "v5 pick 18 'Chatting under the plane trees': a breather outside, four people talking over drinks"),
            ("DJI_20260719091529_0043_D.MP4", 18, 24, "hollywood",
             "v5 pick 13 'Your agents work': 'Your brand knows more than it remembers' on screen — the punchy line"),
            ("DJI_20260719103442_0049_D.MP4", 25, 32, "silicon-valley",
             "v5 pick 15 'Information Radar': a builder shows his AI-news dashboard, GitHub trending and tool rankings"),
            ("DJI_20260719084815_0042_D.MP4", 29, 36, "glamour",
             "v5 pick 12 'A woman demos her phone app': a woman presents her own phone app with the springboard screen behind her"),
        ],
    },
    {
        "set": 3, "slot": "The recap needs one more clip.",
        "cands": [
            ("DJI_20260719095456_0027_D.MP4", 8, 14, "fresh",
             "shot list: EXTERIOR: storefront entrance, glass doors, awning, people at the door"),
            ("DJI_20260719084815_0042_D.MP4", 18, 24, "hollywood",
             "shot list: Crowd/audience close-up; people standing and watching near the glass"),
            ("DJI_20260719103001_0048_D.MP4", 15, 21, "silicon-valley",
             "shot list: Carl Yang presents detailed Harness slides; the screen shows dense tables and architecture"),
            ("DJI_20260719104626_0056_D.MP4", 0, 6, "glamour",
             "shot list: Close-up of people at the bar (woman with a phone, men); a woman presenter (blonde, pink top) at the screen"),
        ],
    },
    {
        "set": 4, "slot": "The recap needs one more clip.",
        "cands": [
            ("DJI_20260719094910_0025_D.MP4", 0, 6, "fresh",
             "shot list: EXTERIOR street: close-up of people outside (woman with a phone, men chatting) under the plane trees"),
            ("timelapse.MP4", 14, 20, "hollywood",
             "shot list: timelapse continues — the room keeps filling, same sweeping scale as the opening beat"),
            ("DJI_20260719072352_0033_D.MP4", 13, 19, "silicon-valley",
             "shot list: man in striped shirt presents at the big screen; screen shows Chinese AI-native tools slides"),
            ("DJI_20260719073634_0037_D.MP4", 50, 56, "glamour",
             "shot list: camera pans to a close-up of audience members watching"),
        ],
    },
]

# Stability golds: Tier A pick indices (v5) x 2 distractor sets.
STABILITY_PICKS = [0, 8, 12]

# Multi-pick sequences: lists of Tier A v5 pick indices, in beat order.
MULTI_SEQS = [
    [0, 7, 18],
    [2, 11, 13],
    [1, 8, 12],
    [0, 15, 18],
]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-path", default=V1_PATH)
    parser.add_argument("--out", default=OUT_PATH)
    args = parser.parse_args(argv)
    picks = load_jsonl(PICKS_PATH)
    train = load_jsonl(TRAIN_PATH)
    shots = load_jsonl(SHOT_PATH)

    train_keys = {clip_key(r["clip"]) for r in train}
    train_ranges = [
        (r["clip"]["file"], parse_ts(r["clip"]["start"]), parse_ts(r["clip"]["end"]))
        for r in train
    ]
    pick_ranges = [
        (p["clip"]["file"], parse_ts(p["clip"]["start"]), parse_ts(p["clip"]["end"]))
        for p in picks
    ]

    # Tier A clean picks (same rule as v1).
    clean_idx = set()
    for i, p in enumerate(picks):
        c = p["clip"]
        if clip_key(c) in train_keys:
            continue
        s, e = parse_ts(c["start"]), parse_ts(c["end"])
        if any(overlaps(c["file"], s, e, f, s0, e0) for f, s0, e0 in train_ranges):
            continue
        clean_idx.add(i)
    print(f"Tier A clean picks: {sorted(clean_idx)}")

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

    # --- v1 items, copied verbatim.
    v1_bytes = Path(args.v1_path).read_bytes()
    v1 = [json.loads(l) for l in v1_bytes.splitlines() if l.strip()]
    items = list(v1)
    print(f"v1 items copied: {len(v1)}")

    letters = ["A", "B", "C", "D", "E", "F", "G", "H"]

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

    def candidate_windows():
        out = []
        for clip_file, length in sorted(clip_len.items()):
            for s0 in range(0, max(1, length - 5)):
                s1 = s0 + 6
                if any(overlaps(clip_file, s0, s1, f, a, b) for f, a, b in pick_ranges):
                    continue
                if any(overlaps(clip_file, s0, s1, f, a, b) for f, a, b in train_ranges):
                    continue
                out.append((clip_file, s0, s1))
        return out

    windows = candidate_windows()
    train_files = {r["clip"]["file"] for r in train}
    clean_windows = [w for w in windows if w[0] not in train_files]
    RNG.shuffle(clean_windows)
    RNG.shuffle(windows)
    print(f"distractor windows: {len(windows)} total, {len(clean_windows)} from never-trained files")

    # Rotating draw: each call consumes the NEXT windows from the pool, so
    # items do not all share the same top-of-pool distractors. (The v1
    # builder restarts the pool per item — a known limitation; v1 items are
    # frozen for comparability, but new v2 items must not repeat it. In
    # particular the two reps of a stability pair MUST get disjoint
    # distractors, or the pair would test nothing.)
    _pool_pos = [0]

    def draw(n, gold_file):
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

    # --- style-mc: same set, 4 styles, gold differs by style.
    for s in STYLE_SETS:
        cands = []
        for (fname, s0, s1, style, ev) in s["cands"]:
            # Held-out asserts.
            if (fname, fmt_ts(s0), fmt_ts(s1)) in train_keys or (
                fname, s0, s1) in [(f, a, b) for f, a, b in train_ranges]:
                raise AssertionError(f"style set {s['set']}: {fname} {s0}-{s1} IS a training example")
            if s["set"] in (3, 4):
                assert not any(overlaps(fname, s0, s1, f, a, b) for f, a, b in pick_ranges), \
                    f"style set {s['set']}: {fname} {s0}-{s1} overlaps a v5 pick"
                assert not any(overlaps(fname, s0, s1, f, a, b) for f, a, b in train_ranges), \
                    f"style set {s['set']}: {fname} {s0}-{s1} overlaps a training example"
            else:
                # Sets 1-2: candidates are Tier A picks — verify each is Tier A.
                idx = [i for i, p in enumerate(picks)
                       if p["clip"]["file"] == fname
                       and parse_ts(p["clip"]["start"]) == s0
                       and parse_ts(p["clip"]["end"]) == s1]
                assert idx and idx[0] in clean_idx, \
                    f"style set {s['set']}: {fname} {s0}-{s1} is not a Tier A pick"
            cands.append((fname, s0, s1, window_desc(fname, s0, s1), style, ev))
        files = [c[0] for c in cands]
        assert len(set(files)) == 4, f"style set {s['set']}: candidates must be 4 distinct files"
        for style in ("fresh", "silicon-valley", "glamour", "hollywood"):
            gold = [c for c in cands if c[4] == style][0]
            options = [(c[0], c[1], c[2], c[3]) for c in cands]
            RNG.shuffle(options)
            gold_t = (gold[0], gold[1], gold[2], gold[3])
            gold_letter = letters[options.index(gold_t)]
            items.append({
                "id": f"style-{s['set']}-{style}",
                "format": "style-mc",
                "style": style,
                "set": s["set"],
                "source": f"style set {s['set']}: gold {gold[0]} {fmt_ts(gold[1])}-{fmt_ts(gold[2])}",
                "held_out_evidence": "gold exact range not in 27 training examples and does not overlap any training example (Tier A pick or asserted-clean window)",
                "style_evidence": gold[5],
                "prompt": mc_prompt(s["slot"], options, style=style),
                "gold": {
                    "letter": gold_letter,
                    "clip": {"file": gold[0], "start": fmt_ts(gold[1]), "end": fmt_ts(gold[2])},
                },
            })
    print(f"style-mc items: 16")

    # --- stability: same gold, two different distractor sets, vlog template.
    for pi in STABILITY_PICKS:
        p = picks[pi]
        c = p["clip"]
        s0, e0 = parse_ts(c["start"]), parse_ts(c["end"])
        gold = (c["file"], s0, e0, window_desc(c["file"], s0, e0))
        slot = f"The recap needs a clip for this moment: \"{c['title']}\" — {c['reason']}"
        for rep in (1, 2):
            distractors = draw(3, c["file"])
            options = [gold] + distractors
            RNG.shuffle(options)
            gold_letter = letters[options.index(gold)]
            items.append({
                "id": f"stab-{pi:02d}-{rep}",
                "format": "stability",
                "pair": f"stab-{pi:02d}",
                "source": f"v5 pick {pi}: {c['file']} {c['start']}-{c['end']} (rep {rep})",
                "held_out_evidence": "Tier A pick; exact range not in 27 training examples, moment does not overlap any training example",
                "prompt": mc_prompt(slot, options),
                "gold": {
                    "letter": gold_letter,
                    "clip": {"file": c["file"], "start": c["start"], "end": c["end"]},
                },
            })
    print(f"stability items: 6")

    # --- multi: 3 beats in one call, 8 candidates.
    for mi, seq in enumerate(MULTI_SEQS, start=1):
        golds = []
        for pi in seq:
            p = picks[pi]
            assert pi in clean_idx, f"multi-{mi}: pick {pi} is not Tier A"
            c = p["clip"]
            golds.append((c["file"], parse_ts(c["start"]), parse_ts(c["end"]),
                          window_desc(c["file"], parse_ts(c["start"]), parse_ts(c["end"])), pi))
        # Distractors must not share a file with any gold (else the file name
        # alone cannot discriminate the beat), and must be disjoint across
        # multi items (rotating draw).
        gold_files = {g[0] for g in golds}
        distractors = []
        while len(distractors) < 5:
            w = draw(1, "")[0]
            if w[0] in gold_files:
                continue
            if any(w[0] == d[0] and overlaps(w[0], w[1], w[2], d[0], d[1], d[2]) for d in distractors):
                continue
            distractors.append(w)
        options = [(g[0], g[1], g[2], g[3]) for g in golds] + distractors
        RNG.shuffle(options)
        beats = "\n".join(
            f"Beat {k + 1}: \"{picks[pi]['clip']['title']}\" — {picks[pi]['clip']['reason']}"
            for k, pi in enumerate(seq)
        )
        lines = [
            "You are cutting the Cafe Cursor Shanghai recap (July 19, 2026) into short clips.",
            "",
            "Style: vlog — " + STYLE_BRIEFS["vlog"],
            "",
            "The recap needs 3 clips in one pass, one per beat, in order:",
            beats,
            "",
            "Candidates (pick one per beat, in order):",
        ]
        for letter, (fname, s0, s1, d) in zip(letters, options):
            lines.append(f"{letter}. {fname} {fmt_ts(s0)}-{fmt_ts(s1)}: {d}")
        lines.append("")
        lines.append("Answer as a JSON list of 3 clips in beat order, e.g. [{\"file\": \"DJI_...MP4\", \"start\": \"0:05\", \"end\": \"0:15\"}, ...]. No other text.")
        items.append({
            "id": f"multi-{mi:02d}",
            "format": "multi",
            "source": f"v5 picks {seq} in beat order",
            "held_out_evidence": "all 3 golds are Tier A picks; exact ranges not in 27 training examples, moments do not overlap any training example",
            "prompt": "\n".join(lines),
            "gold": {
                "clips": [
                    {"file": g[0], "start": fmt_ts(g[1]), "end": fmt_ts(g[2])}
                    for g in golds
                ]
            },
        })
    print(f"multi items: 4")

    with open(args.out, "wb") as f:
        f.write(v1_bytes)
        if v1_bytes and not v1_bytes.endswith(b"\n"):
            f.write(b"\n")
        for it in items[len(v1):]:
            f.write((json.dumps(it, ensure_ascii=False) + "\n").encode("utf-8"))

    from collections import Counter
    print(Counter(it["format"] for it in items), "items ->", args.out)


if __name__ == "__main__":
    main()
