"""FIXED INFRASTRUCTURE — an experiment may never modify this file.

This is the integrity boundary of the research loop. It owns four things:

  1. CONTRACT      the output schema, clip length rule, and the one prompt template
  2. DATA          loading the picks, and the deterministic train / held-out split
  3. EVAL          scoring a model's answers against Aileen's picks
  4. LEDGER        appending a result to results.tsv and deciding keep / discard

train.py is the experiment target and is the only file the research loop edits.
If the eval could be edited, a run could look better without being better — the
whole point of pinning it down here is that a number in results.tsv means
something. See program.md for the loop rules.

Usage
    python prepare.py --audit            data report (rows, coverage, gaps)
    python prepare.py --split            print the held-out rows
    python prepare.py --score FILE       score a predictions .jsonl file
    python prepare.py --self-check       score the references themselves (must be 1.0)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "picks.jsonl"
RESULTS_PATH = ROOT / "results.tsv"

# ── CONTRACT ────────────────────────────────────────────────────────────────
# One clip per answer. Length bounds come from the data: the labelled clips run
# 3–12 s, so the contract allows 3–20 s and nothing longer. The old v1 prompt
# asked for "3 clips of 30–90 s", which no training row ever demonstrated.
REQUIRED_KEYS = ("file", "start", "end", "title", "reason")
MIN_CLIP_SECONDS = 3.0
MAX_CLIP_SECONDS = 20.0

PROMPT_TEMPLATE = """You are cutting a tech meetup video into one shareable clip.

You get a transcript excerpt with timestamps, and keyframe files from the source footage.

Pick ONE moment, 3 to 20 seconds long, that stands alone without the rest of the talk.

Rules:
- Return only a JSON object. No prose, no code fences.
- Keys: file, start, end, title, reason.
- "file" must be one of the source files named in the keyframes below.
- "start" and "end" are M:SS timestamps inside that file; end must be after start.
- "title" is under 60 characters. "reason" is one sentence on why it stands alone.
- Do not cut mid-sentence: move the boundary to the nearest sentence edge.

Transcript:
{transcript}

Keyframes:
{keyframes}
"""

# ── time ────────────────────────────────────────────────────────────────────


def parse_time(value: str) -> float:
    """'1:05' or '01:02:03' -> seconds. Raises ValueError on anything else."""
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


def format_time(seconds: float) -> str:
    total = int(round(seconds))
    return f"{total // 60}:{total % 60:02d}"


# ── output parsing ──────────────────────────────────────────────────────────


def extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of a model answer, ignoring prose and fences."""
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    for candidate in (stripped, *re.findall(r"```(?:json)?\s*(.*?)```", stripped, re.S)):
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    start = stripped.find("{")
    while start != -1:
        depth = 0
        for index in range(start, len(stripped)):
            char = stripped[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(stripped[start : index + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(value, dict):
                        return value
                    break
        start = stripped.find("{", start + 1)
    return None


def validate(answer: dict | None, allowed_files: set[str]) -> list[str]:
    """Every reason this answer breaks the contract. Empty list means valid."""
    if answer is None:
        return ["no JSON object found"]
    errors: list[str] = []
    missing = [key for key in REQUIRED_KEYS if key not in answer]
    if missing:
        errors.append(f"missing keys: {', '.join(missing)}")
    extra = [key for key in answer if key not in REQUIRED_KEYS]
    if extra:
        errors.append(f"unexpected keys: {', '.join(sorted(extra))}")
    if missing:
        return errors

    file_name = answer["file"]
    if not isinstance(file_name, str) or not file_name.strip():
        errors.append("file is not a non-empty string")
    elif allowed_files and file_name not in allowed_files:
        errors.append(f"file not among the prompt's keyframes: {file_name!r}")

    for key in ("title", "reason"):
        if not isinstance(answer[key], str) or not answer[key].strip():
            errors.append(f"{key} is not a non-empty string")
    if isinstance(answer.get("title"), str) and len(answer["title"]) > 60:
        errors.append(f"title is {len(answer['title'])} chars, limit is 60")

    try:
        start = parse_time(answer["start"])
        end = parse_time(answer["end"])
    except ValueError as exc:
        errors.append(str(exc))
        return errors

    duration = end - start
    if duration <= 0:
        errors.append(f"end {answer['end']} is not after start {answer['start']}")
    elif duration < MIN_CLIP_SECONDS:
        errors.append(f"clip is {duration:.0f}s, minimum is {MIN_CLIP_SECONDS:.0f}s")
    elif duration > MAX_CLIP_SECONDS:
        errors.append(f"clip is {duration:.0f}s, maximum is {MAX_CLIP_SECONDS:.0f}s")
    return errors


# ── data ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Row:
    index: int
    node: str
    tags: tuple[str, ...]
    transcript: str
    keyframes: tuple[str, ...]
    clip: dict
    synthetic: bool

    @property
    def clip_key(self) -> str:
        return f"{self.clip['file']}@{self.clip['start']}-{self.clip['end']}"

    @property
    def source_files(self) -> set[str]:
        return {Path(path).parts[0] for path in self.keyframes}

    def prompt(self) -> str:
        return PROMPT_TEMPLATE.format(
            transcript=self.transcript.strip() or "(no speech in this footage)",
            keyframes="\n".join(self.keyframes),
        )

    def answer(self) -> str:
        return json.dumps(self.clip, ensure_ascii=False)


def load_rows(path: Path = DATA_PATH) -> list[Row]:
    rows: list[Row] = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            raw = json.loads(line)
            clip = raw["clip"]
            for key in ("file", "start", "end"):
                if key not in clip:
                    raise ValueError(f"row {index}: clip is missing {key!r}")
            rows.append(
                Row(
                    index=index,
                    node=raw.get("node", "?"),
                    tags=tuple(raw.get("tags", ())),
                    transcript=raw.get("transcript_excerpt", ""),
                    keyframes=tuple(raw.get("keyframes", ())),
                    clip=clip,
                    synthetic=bool(raw.get("synthetic", False)),
                )
            )
    return rows


def split(rows: list[Row]) -> tuple[list[Row], list[Row]]:
    """Deterministic split: every fourth row by clip key is held out.

    No RNG, so the same rows are held out on every machine and every run. The
    held-out set is written into program.md and must not be changed to make a
    score look better.
    """
    keys = sorted(row.clip_key for row in rows)
    held_out_keys = set(keys[::4])
    train = [row for row in rows if row.clip_key not in held_out_keys]
    held_out = [row for row in rows if row.clip_key in held_out_keys]
    return train, held_out


# ── eval ────────────────────────────────────────────────────────────────────


def interval_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    overlap = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return overlap / union if union > 0 else 0.0


def score_one(answer_text: str, reference: dict, allowed_files: set[str]) -> dict:
    """Score one answer. Invalid schema scores 0; a valid answer starts at 0.25."""
    answer = extract_json(answer_text)
    errors = validate(answer, allowed_files)
    result = {
        "schema_valid": not errors,
        "file_match": False,
        "temporal_iou": 0.0,
        "score": 0.0,
        "errors": errors,
        "answer": answer,
    }
    if errors:
        return result

    assert answer is not None  # validate() guarantees a dict here
    if answer["file"] == reference["file"]:
        result["file_match"] = True
        iou = interval_iou(
            parse_time(answer["start"]),
            parse_time(answer["end"]),
            parse_time(reference["start"]),
            parse_time(reference["end"]),
        )
        result["temporal_iou"] = round(iou, 4)
    result["score"] = round(0.25 + 0.75 * result["temporal_iou"], 4)
    return result


def evaluate(predictions: dict[str, str], held_out: list[Row]) -> dict:
    """Score a mapping of clip_key -> answer text against the held-out rows."""
    details = []
    for row in held_out:
        text = predictions.get(row.clip_key)
        result = score_one(text or "", row.clip, row.source_files)
        result["clip_key"] = row.clip_key
        details.append(result)
    count = len(details) or 1
    valid = sum(1 for item in details if item["schema_valid"])
    return {
        "rows": len(details),
        "schema_valid_rate": round(valid / count, 4),
        "file_match_rate": round(sum(1 for d in details if d["file_match"]) / count, 4),
        "mean_iou": round(sum(d["temporal_iou"] for d in details) / count, 4),
        "mean_score": round(sum(d["score"] for d in details) / count, 4),
        "details": details,
    }


# ── ledger ──────────────────────────────────────────────────────────────────

LEDGER_COLUMNS = (
    "timestamp",
    "run_id",
    "model",
    "rank",
    "lr",
    "steps",
    "train_rows",
    "schema_valid_rate",
    "file_match_rate",
    "mean_iou",
    "mean_score",
    "keep",
    "note",
)


def best_score() -> float:
    """The best score of a run that was actually kept.

    A run that failed the schema gate never counts, even if its mean score was
    high, because it cannot be shipped. Without this, one lucky invalid run sets
    a bar that no honest run can clear.
    """
    if not RESULTS_PATH.exists():
        return -1.0
    best = -1.0
    with RESULTS_PATH.open(encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        if "mean_score" not in header or "keep" not in header:
            return -1.0
        score_column = header.index("mean_score")
        keep_column = header.index("keep")
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) <= max(score_column, keep_column):
                continue
            if fields[keep_column] != "yes":
                continue
            try:
                best = max(best, float(fields[score_column]))
            except ValueError:
                continue
    return best


def decide(summary: dict) -> bool:
    """Would this run be kept? Schema gate first, then it must beat the best kept score."""
    return summary["schema_valid_rate"] >= 0.9 and summary["mean_score"] > best_score()


def record_result(summary: dict, config: dict, note: str = "") -> bool:
    """Append a run to results.tsv. Returns True when it beats the best kept score.

    The gate is deliberate: a run must be schema-valid on at least 90% of the
    held-out rows before its score is allowed to count as an improvement.
    """
    keep = decide(summary)
    row = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_id": config.get("run_id", "manual"),
        "model": config.get("model", "?"),
        "rank": config.get("rank", "?"),
        "lr": config.get("lr", "?"),
        "steps": config.get("steps", "?"),
        "train_rows": config.get("train_rows", "?"),
        "schema_valid_rate": summary["schema_valid_rate"],
        "file_match_rate": summary["file_match_rate"],
        "mean_iou": summary["mean_iou"],
        "mean_score": summary["mean_score"],
        "keep": "yes" if keep else "no",
        "note": note.replace("\t", " ").replace("\n", " "),
    }
    new_file = not RESULTS_PATH.exists()
    with RESULTS_PATH.open("a", encoding="utf-8") as handle:
        if new_file:
            handle.write("\t".join(LEDGER_COLUMNS) + "\n")
        handle.write("\t".join(str(row[column]) for column in LEDGER_COLUMNS) + "\n")
    return keep


# ── audit / cli ─────────────────────────────────────────────────────────────


def audit(rows: list[Row]) -> dict:
    train, held_out = split(rows)
    durations = [
        parse_time(row.clip["end"]) - parse_time(row.clip["start"]) for row in rows
    ]
    tags = sorted({tag for row in rows for tag in row.tags})
    return {
        "rows": len(rows),
        "synthetic": sum(1 for row in rows if row.synthetic),
        "with_transcript": sum(1 for row in rows if row.transcript.strip()),
        "source_files": len({row.clip["file"] for row in rows}),
        "duplicate_clip_keys": len(rows) - len({row.clip_key for row in rows}),
        "duration_min": min(durations),
        "duration_max": max(durations),
        "duration_mean": round(sum(durations) / len(durations), 2),
        "nodes": sorted({row.node for row in rows}),
        "tag_count": len(tags),
        "train_rows": len(train),
        "held_out_rows": len(held_out),
        "held_out": [row.clip_key for row in held_out],
        "contract_violations": [
            (row.clip_key, error)
            for row in rows
            for error in validate(row.clip, row.source_files)
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="store_true", help="print a data report")
    parser.add_argument("--split", action="store_true", help="print the held-out rows")
    parser.add_argument("--score", metavar="FILE", help="score a predictions .jsonl")
    parser.add_argument("--self-check", action="store_true", help="score the references")
    args = parser.parse_args(argv)

    rows = load_rows()

    if args.audit:
        print(json.dumps(audit(rows), indent=2, ensure_ascii=False))
        return 0

    if args.split:
        _, held_out = split(rows)
        for row in held_out:
            print(row.clip_key)
        return 0

    if args.self_check:
        _, held_out = split(rows)
        summary = evaluate({row.clip_key: row.answer() for row in held_out}, held_out)
        print(json.dumps({k: v for k, v in summary.items() if k != "details"}, indent=2))
        ok = summary["mean_score"] == 1.0 and summary["schema_valid_rate"] == 1.0
        print("self-check:", "pass" if ok else "FAIL")
        return 0 if ok else 1

    if args.score:
        _, held_out = split(rows)
        predictions: dict[str, str] = {}
        with Path(args.score).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                predictions[item["clip_key"]] = item.get("answer", "")
        summary = evaluate(predictions, held_out)
        print(json.dumps({k: v for k, v in summary.items() if k != "details"}, indent=2))
        for detail in summary["details"]:
            mark = "ok " if detail["schema_valid"] else "BAD"
            print(
                f"{mark} {detail['clip_key']:<48} "
                f"iou={detail['temporal_iou']:.2f} score={detail['score']:.2f} "
                f"{'; '.join(detail['errors'])}"
            )
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
