"""Shared scoring helpers for the video-cutting eval sets (v1 + v2).

Formats:
  mc / style-mc / stability: pick one candidate. Correct iff the named clip
    (JSON or letter) matches the gold clip (file + start/end within 0.5s).
    v1 answers MC items with a JSON clip, never a letter — so clip wins.
  trap: correct iff the model refuses ("None"/letter E) and emits no clip.
  sft: parse a JSON clip; correct iff file AND start/end match (0.5s).
    Also reports file_match, window IoU, near_miss (IoU > 0.5).
  multi: parse a JSON list of 3 clips (one per beat, in order). Correct iff
    all 3 slots match exactly. Also reports per-slot file matches.

No training data, no prompts here — pure scoring.
"""
import json
import re


def parse_ts(s):
    parts = str(s).split(":")
    return float(parts[0]) * 60 + float(parts[1])


def iou(a0, a1, b0, b1):
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def extract_letter(text):
    """First standalone A-H letter, or 'NONE' if the model says none."""
    m = re.search(r"\b([A-H])\b", text)
    if m:
        return m.group(1)
    if re.search(r"\bnone\b", text, re.I):
        return "NONE"
    return None


def extract_clip_json(text):
    """First {...} that parses and has file/start/end."""
    for m in re.finditer(r"\{[^{}]*\}", text):
        try:
            obj = json.loads(m.group(0))
        except Exception:
            continue
        if isinstance(obj, dict) and "file" in obj and "start" in obj and "end" in obj:
            return obj
    return None


def extract_clip_list(text):
    """First [...] that parses as a list of clip dicts."""
    for m in re.finditer(r"\[.*?\]", text, re.DOTALL):
        try:
            obj = json.loads(m.group(0))
        except Exception:
            continue
        if isinstance(obj, list) and obj and all(
            isinstance(c, dict) and "file" in c and "start" in c and "end" in c
            for c in obj
        ):
            return obj
    return None


def clip_exact(clip, gold_clip, tol=0.5):
    """File matches and start/end within tol seconds."""
    if clip is None or gold_clip is None:
        return 0
    try:
        return (
            1
            if clip.get("file") == gold_clip["file"]
            and abs(parse_ts(clip.get("start")) - parse_ts(gold_clip["start"])) <= tol
            and abs(parse_ts(clip.get("end")) - parse_ts(gold_clip["end"])) <= tol
            else 0
        )
    except Exception:
        return 0


def clip_iou(clip, gold_clip):
    if clip is None or gold_clip is None:
        return 0.0
    try:
        return round(
            iou(
                parse_ts(clip.get("start")), parse_ts(clip.get("end")),
                parse_ts(gold_clip["start"]), parse_ts(gold_clip["end"]),
            ),
            3,
        )
    except Exception:
        return 0.0


def score_item(item, text):
    fmt = item["format"]
    gold = item["gold"]
    # Store the FULL raw output: re-scoring from a saved file must see exactly
    # what live scoring saw. (Truncating raw to 1500 chars lost base's late
    # multi-pick answers on re-score: live multi 0.50 vs re-scored 0.00.)
    out = {"raw": text}
    if fmt in ("mc", "style-mc", "stability"):
        clip = extract_clip_json(text)
        letter = extract_letter(text)
        by_letter = 1 if letter == gold["letter"] else 0
        by_clip = clip_exact(clip, gold["clip"])
        out["letter"] = letter
        out["by_letter"] = by_letter
        out["by_clip"] = by_clip
        out["pred_clip"] = clip
        # Resolve a letter answer to its option's clip. Letters are
        # position-dependent (the same pick sits at different letters when
        # options are shuffled across reps), so pair agreement must compare
        # clips, not letters. Without this, a letter-answering model that
        # picks correctly in both reps scores agree=0. Guarded: callers
        # that score without a prompt (e.g. the GRPO loop) simply skip
        # resolution — the letter itself is still scored above.
        if clip is None and letter and letter != "NONE" and item.get("prompt"):
            m = re.search(
                rf"(?m)^{re.escape(letter)}\. (\S+\.MP4) (\d+:\d+(?:\.\d+)?)-(\d+:\d+(?:\.\d+)?):",
                item["prompt"],
            )
            if m:
                out["pred_clip"] = {"file": m.group(1), "start": m.group(2), "end": m.group(3)}
        out["empty"] = 1 if not text.strip() else 0
        out["correct"] = 1 if (by_letter == 1 or by_clip == 1) else 0
    elif fmt == "trap":
        letter = extract_letter(text)
        clip = extract_clip_json(text)
        out["letter"] = letter
        out["hallucinated_clip"] = 1 if clip is not None else 0
        out["correct"] = 1 if (letter in ("E", "NONE") and clip is None) else 0
    elif fmt == "sft":
        clip = extract_clip_json(text)
        out["parsed_json"] = 1 if clip is not None else 0
        if clip is None:
            out["file_match"] = 0
            out["iou"] = 0.0
            out["near_miss"] = 0
            out["correct"] = 0
        else:
            out["file_match"] = 1 if clip.get("file") == gold["clip"]["file"] else 0
            out["iou"] = clip_iou(clip, gold["clip"])
            out["near_miss"] = 1 if (out["file_match"] == 1 and out["iou"] > 0.5) else 0
            out["correct"] = clip_exact(clip, gold["clip"])
    elif fmt == "multi":
        clips = extract_clip_list(text)
        out["parsed_list"] = 1 if clips is not None else 0
        gold_clips = gold["clips"]
        if clips is None:
            out["slots_exact"] = 0
            out["slots_file"] = 0
            out["correct"] = 0
        else:
            exact = sum(
                clip_exact(c, g) for c, g in zip(clips, gold_clips)
            )
            fmatch = sum(
                1 for c, g in zip(clips, gold_clips)
                if c.get("file") == g["file"]
            )
            out["slots_exact"] = exact
            out["slots_file"] = fmatch
            out["n_returned"] = len(clips)
            out["correct"] = 1 if len(clips) == len(gold_clips) and exact == len(gold_clips) else 0
    else:
        raise ValueError(f"unknown format {fmt}")
    return out


def summarize(name, recs, verbose=True):
    """Per-format accuracy + extras. Returns dict of metrics."""
    metrics = {}
    for fmt in ("mc", "style-mc", "stability", "multi", "sft", "trap"):
        sub = [r for r in recs if r["format"] == fmt]
        if not sub:
            continue
        acc = sum(r["correct"] for r in sub) / len(sub)
        m = {"n": len(sub), "acc": round(acc, 3)}
        if fmt in ("mc", "style-mc", "stability"):
            m["by_clip"] = round(sum(r.get("by_clip", 0) for r in sub) / len(sub), 3)
            m["by_letter"] = round(sum(r.get("by_letter", 0) for r in sub) / len(sub), 3)
            m["empty"] = sum(r.get("empty", 0) for r in sub)
        if fmt == "sft":
            m["file_match"] = round(sum(r.get("file_match", 0) for r in sub) / len(sub), 3)
            m["near_miss"] = round(sum(r.get("near_miss", 0) for r in sub) / len(sub), 3)
            m["mean_iou"] = round(sum(r.get("iou", 0) for r in sub) / len(sub), 3)
        if fmt == "trap":
            m["hallucinated_clip"] = round(sum(r.get("hallucinated_clip", 0) for r in sub) / len(sub), 3)
        if fmt == "multi":
            m["mean_slots_exact"] = round(sum(r.get("slots_exact", 0) for r in sub) / len(sub), 3)
            m["mean_slots_file"] = round(sum(r.get("slots_file", 0) for r in sub) / len(sub), 3)
        metrics[fmt] = m
    # Stability pair agreement: same PICK on both reps. Compare only
    # (file, start, end) — never the reason/title text, which varies
    # between calls even for the same pick. (Bug found 2026-11-18: comparing
    # the whole clip dict made v1 score agree=0 while acc=1.0, because its
    # two reps emitted the same clip with different reason text.)
    stab = [r for r in recs if r["format"] == "stability"]
    if stab:
        from collections import defaultdict

        def _pred_key(r):
            c = r.get("pred_clip")
            if c:
                try:
                    return (c.get("file"), parse_ts(c.get("start")), parse_ts(c.get("end")))
                except Exception:
                    return json.dumps(c, sort_keys=True)
            return r.get("letter")

        pairs = defaultdict(list)
        for r in stab:
            pairs[r.get("pair", r["id"].rsplit("-", 1)[0])].append(r)
        agree = 0
        total = 0
        for _, rs in pairs.items():
            if len(rs) == 2:
                total += 1
                if _pred_key(rs[0]) is not None and _pred_key(rs[0]) == _pred_key(rs[1]):
                    agree += 1
        metrics["stability_pairs"] = {"n_pairs": total, "agree": agree}
    metrics["overall"] = {
        "correct": sum(r["correct"] for r in recs),
        "n": len(recs),
        "acc": round(sum(r["correct"] for r in recs) / len(recs), 3) if recs else 0,
    }
    if verbose:
        print(f"--- {name} ---")
        for fmt, m in metrics.items():
            if fmt == "overall":
                continue
            print(f"  {fmt}: {m}")
        print(f"  OVERALL: {metrics['overall']['correct']}/{metrics['overall']['n']} = {metrics['overall']['acc']:.2f}")
    return metrics
