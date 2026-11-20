"""Prepare my custom cutting-style data locally; never train or call a model.

Each JSONL row is {"request": <cutter request>, "answer": <letter/list/null>}.
Run --help for split and minimum-picks controls.
"""
import argparse
import copy
import json
import math
import random
import re
import shlex
from pathlib import Path

import cutter
from eval.check_overlap import find_overlaps
from prepare import MAX_CLIP_SECONDS, MIN_CLIP_SECONDS, parse_time


def one_line(value, field):
    if not isinstance(value, str) or not value.strip() or any(
        c in value for c in ("\n", "\r", "\x00")
    ):
        raise ValueError(f"{field} must be a nonempty single-line string")
    return value


def timestamp(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]+:[0-5][0-9]", value):
        raise ValueError(f"timestamp must use M:SS (whole seconds): {value!r}")
    return parse_time(value)


def validate_clip(clip):
    if not isinstance(clip, dict):
        raise ValueError("each candidate/answer clip must be an object")
    file = one_line(clip.get("file"), "clip file")
    if not re.fullmatch(r"[^/\\\s]+\.MP4", file):
        raise ValueError("clip file must be a .MP4 basename without spaces")
    duration = timestamp(clip.get("end")) - timestamp(clip.get("start"))
    if not MIN_CLIP_SECONDS <= duration <= MAX_CLIP_SECONDS:
        raise ValueError(
            f"clip length must be {MIN_CLIP_SECONDS:g}–{MAX_CLIP_SECONDS:g} seconds: {file}"
        )


def validate_pick(row, number, style, brief):
    try:
        if not isinstance(row, dict) or not isinstance(row.get("request"), dict):
            raise ValueError("expected an object with request and answer")
        if "answer" not in row:
            raise ValueError("missing answer (use null for an approved refusal)")
        request = copy.deepcopy(row["request"])
        request.update(style=style, style_brief=brief)
        for field in ("event", "date"):
            one_line(request.get(field), field)
        beats = request.setdefault("beats", [])
        if not isinstance(beats, list) or len(beats) > 3:
            raise ValueError("beats must be a list of 0–3 objects")
        for beat in beats:
            if not isinstance(beat, dict):
                raise ValueError("each beat must be an object")
            for field in ("title", "why"):
                one_line(beat.get(field), f"beat {field}")
        candidates = request.get("candidates")
        expected = 8 if len(beats) > 1 else 4
        if not isinstance(candidates, list) or len(candidates) != expected:
            raise ValueError(f"expected {expected} candidates for {len(beats)} beats")
        windows = set()
        for candidate in candidates:
            validate_clip(candidate)
            one_line(candidate.get("description"), "candidate description")
            window = tuple(candidate[key] for key in ("file", "start", "end"))
            if window in windows:
                raise ValueError("duplicate candidate window")
            windows.add(window)
        answer = row["answer"]
        if answer is None:
            answer = "None"
        if isinstance(answer, list):
            if len(beats) <= 1:
                raise ValueError("clip lists require 2–3 beats")
            for clip in answer:
                validate_clip(clip)
            answer = json.dumps(answer, ensure_ascii=False)
        elif isinstance(answer, str):
            answer = answer.strip()
            if len(beats) > 1 and answer.startswith("["):
                clips = json.loads(answer)
                if not isinstance(clips, list):
                    raise ValueError("multi-pick answer must be a clip list")
                for clip in clips:
                    validate_clip(clip)
            elif not re.fullmatch(
                r"(?:None|[A-H](?::\s*[^/\\\s]+\.MP4 [0-9]+:[0-5][0-9]-[0-9]+:[0-5][0-9])?)",
                answer,
            ):
                raise ValueError("answer must be a letter, letter: file M:SS-M:SS, clip list, or None")
        else:
            raise ValueError("answer must be a string, clip list, or null")
        selected = cutter.resolve(answer, request)
        clips = [{key: clip[key] for key in ("file", "start", "end")} for clip in selected]
        if not clips:
            fmt, gold, answer = "trap", {"letter": "E", "clip": None}, "None"
        elif len(beats) > 1:
            fmt, gold = "multi", {"clips": clips}
            answer = json.dumps(clips, ensure_ascii=False)
        else:
            letter = chr(65 + candidates.index(selected[0]))
            fmt, gold = "mc", {"letter": letter, "clip": clips[0]}
            clip = clips[0]
            answer = f"{letter}: {clip['file']} {clip['start']}-{clip['end']}"
        return {
            "id": f"style-{number:06d}", "kind": "prompt", "style": style,
            "request": request, "prompt": cutter.build_prompt(request),
            "answer": answer, "format": fmt, "gold": gold,
            "synthetic": row.get("synthetic") is True,
        }
    except (ValueError, KeyError, TypeError) as error:
        raise ValueError(f"row {number}: {error}") from error


def load_picks(path, style, brief):
    rows = []
    prompts = {}
    with Path(path).open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as error:
                raise ValueError(f"row {number}: invalid JSON: {error}") from error
            item = validate_pick(row, number, style, brief)
            if item["prompt"] in prompts:
                raise ValueError(
                    f"row {number}: duplicate request (already on row {prompts[item['prompt']]})"
                )
            prompts[item["prompt"]] = number
            rows.append(item)
    if not rows:
        raise ValueError("picks file contains no picks")
    return rows


def split_rows(rows, seed=42, held_out_fraction=0.2):
    """Keep connected groups of candidate source files together, even distractors."""
    if not 0 < held_out_fraction < 1:
        raise ValueError("held-out fraction must be between 0 and 1")
    parents = list(range(len(rows)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    seen = {}
    for index, row in enumerate(rows):
        for candidate in row["request"]["candidates"]:
            file = candidate["file"]
            if file in seen:
                parents[root(index)] = root(seen[file])
            seen[file] = index
    groups = {}
    for index, row in enumerate(rows):
        groups.setdefault(root(index), []).append(row)
    groups = list(groups.values())
    if len(groups) < 2:
        raise ValueError(
            "need at least two independent candidate-file groups for train/held-out; "
            "shared source files (including distractors) must stay on the same side"
        )
    random.Random(seed).shuffle(groups)
    target = max(1, math.ceil(len(rows) * held_out_fraction))
    held = []
    while len(groups) > 1 and len(held) < target:
        held.extend(groups.pop())
    train = [row for group in groups for row in group]
    return sorted(train, key=lambda row: row["id"]), sorted(held, key=lambda row: row["id"])


def write_dataset(rows, output, style, brief, seed=42, held_out_fraction=0.2, min_picks=20):
    output = Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"output must be a new or empty directory: {output}")
    train, held = split_rows(rows, seed, held_out_fraction)
    overlap_ids = sorted(find_overlaps(held, train))
    if overlap_ids:
        raise ValueError(f"held-out golds overlap training exposure: {', '.join(overlap_ids)}")
    command = shlex.join([
        "uv", "run", "python", "run_sft_v3.py", "--data", str(output / "train.jsonl"),
        "--from-base", "--name", f"inkling-{style}",
    ])
    warnings = []
    if len(rows) < min_picks:
        warnings.append(f"only {len(rows)} picks; recommended minimum is {min_picks}")
    if any(row["synthetic"] for row in rows):
        warnings.append("contains synthetic picks; example data only, do not use for paid training")
    manifest = {
        "style": style, "style_brief": brief, "picks": len(rows),
        "train": len(train), "held_out": len(held),
        "synthetic_picks": sum(row["synthetic"] for row in rows),
        "seed": seed, "held_out_fraction_requested": held_out_fraction,
        "split_unit": "connected candidate-file groups, including distractors",
        "validation": "all rows resolved against offered candidates; M:SS, 3–20 second clips",
        "held_out_overlap_ids": overlap_ids, "min_picks": min_picks, "warnings": warnings,
        "files": {"train": "train.jsonl", "held_out": "held-out.jsonl"},
        "run_paid_command_from": str(Path(__file__).resolve().parent),
        "paid_training_command": command, "training_executed": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, items in (("train.jsonl", train), ("held-out.jsonl", held)):
        (output / name).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in items),
            encoding="utf-8",
        )
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("picks", type=Path)
    parser.add_argument("--style", required=True, help="custom style slug, e.g. my-calm-recap")
    parser.add_argument("--brief", required=True, help="one-line cutting-style description")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--held-out-fraction", type=float, default=0.2)
    parser.add_argument("--min-picks", type=int, default=20, help="warning threshold, not a hard minimum")
    args = parser.parse_args(argv)
    try:
        if not re.fullmatch(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*", args.style):
            raise ValueError("style must be a lowercase slug, e.g. my-calm-recap")
        one_line(args.brief, "style brief")
        if args.min_picks < 0:
            raise ValueError("min-picks must be nonnegative")
        rows = load_picks(args.picks, args.style, args.brief)
        manifest = write_dataset(
            rows, args.out, args.style, args.brief,
            args.seed, args.held_out_fraction, args.min_picks,
        )
    except (OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")
    for warning in manifest["warnings"]:
        print(f"WARNING: {warning}")
    print(f"Prepared {manifest['picks']} picks: {manifest['train']} train, {manifest['held_out']} held-out")
    print(f"Overlap check: {len(manifest['held_out_overlap_ids'])} held-out items exposed to training")
    print(f"Output: {args.out.resolve()}")
    print("Paid Tinker command (NOT RUN; run from repo root after uv sync --extra train and setting TINKER_API_KEY):")
    print(manifest["paid_training_command"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
