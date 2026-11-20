"""Cutter adapter: build a STYLE_TEMPLATE prompt, resolve the model's answer, cut the clips.

The winner answers with a letter (`B: <file> <start>-<end>`), a JSON list of clips
(multi-pick) or `None`. This adapter turns that answer back into real candidate
windows, refuses anything that is not one of the offered candidates, and cuts each
window with ffmpeg.

    uv run python cutter.py prompt examples/request.json
    uv run python cutter.py cut examples/request.json "C: DJI_...MP4 0:01-0:07" \
        --footage footage/ --out clips/
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from eval.scoring import clip_exact, extract_clip_list, extract_letter, parse_ts

STYLES = {
    "vlog": "Vlog — the reference recap style: warm, human, one flowing story; a moment that works standalone.",
    "fresh": "Fresh and clean — bright, airy, breathing room: outdoors, daylight, uncluttered frames, calm pacing.",
    "silicon-valley": "Silicon Valley — substance first: product demos, live code, big-topic takeaways, a builder explaining their own work.",
    "glamour": "Glamour — polished and beautiful: close-ups, warm light, people at their best, a frame worth a poster.",
    "hollywood": "Hollywood — drama and peak emotion: the room reacting, scale, the biggest beat of the day, a cinematic sweep.",
}
LETTERS = "ABCDEFGH"


class ResolveError(ValueError):
    pass


def load_request(path: str | Path) -> dict:
    request = json.loads(Path(path).read_text(encoding="utf-8"))
    if request.get("style") not in STYLES and not request.get("style_description"):
        raise ResolveError(
            f"style must be one of {', '.join(STYLES)}, or set style_description for your own"
        )
    beats = request.setdefault("beats", [])
    candidates = request.get("candidates") or []
    if len(beats) > 3:
        raise ResolveError("beats must hold at most 3 beats")
    expected = 8 if len(beats) > 1 else 4
    if len(candidates) != expected:
        raise ResolveError(f"{len(beats)} beat(s) need {expected} candidates, got {len(candidates)}")
    for candidate in candidates:
        if parse_ts(candidate["end"]) <= parse_ts(candidate["start"]):
            raise ResolveError(f"candidate ends before it starts: {candidate}")
    return request


def _candidate_lines(candidates: list[dict]) -> str:
    return "\n".join(
        f"{LETTERS[i]}. {c['file']} {c['start']}-{c['end']}: {c['description']}"
        for i, c in enumerate(candidates)
    )


def build_prompt(request: dict) -> str:
    style = request["style"]
    head = (
        f"You are cutting the {request['event']} recap ({request['date']}) into short clips.\n\n"
        f"Style: {style} — {STYLES.get(style) or request['style_description']}\n\n"
    )
    beats = request["beats"]
    candidates = _candidate_lines(request["candidates"])
    if len(beats) <= 1:
        need = (
            f"The recap needs a clip for this moment: \"{beats[0]['title']}\" — {beats[0]['why']}\n"
            if beats
            else "The recap needs one more clip.\n"
        )
        return (
            head
            + need
            + "Which candidate segment should be used? Answer with just the letter and the clip "
            + 'file + start/end, e.g. "B: DJI_...MP4 0:05-0:15". If none of them fits, answer "None".\n\n'
            + candidates
        )
    beat_lines = "\n".join(
        f"Beat {i + 1}: \"{b['title']}\" — {b['why']}" for i, b in enumerate(beats)
    )
    return (
        head
        + f"The recap needs {len(beats)} clips in one pass, one per beat, in order:\n"
        + beat_lines
        + "\n\nCandidates (pick one per beat, in order):\n"
        + candidates
        + f"\n\nAnswer as a JSON list of {len(beats)} clips in beat order, e.g. "
        + '[{"file": "DJI_...MP4", "start": "0:05", "end": "0:15"}, ...]. No other text.'
    )


def _match(clip: dict, candidates: list[dict]) -> dict:
    for candidate in candidates:
        if clip_exact(clip, candidate):
            return candidate
    raise ResolveError(f"answer is not one of the offered candidates: {clip}")


def resolve(answer: str, request: dict) -> list[dict]:
    """Candidate windows the answer selects, in beat order. [] means the model refused."""
    candidates = request["candidates"]
    beats = request["beats"]
    text = answer.strip()
    if len(beats) > 1:
        clips = extract_clip_list(text)
        if clips is None:
            if re.fullmatch(r"\W*none\W*", text, re.I):
                return []
            raise ResolveError("multi-pick answer has no JSON list of clips")
        if len(clips) != len(beats):
            raise ResolveError(f"expected {len(beats)} clips, got {len(clips)}")
        return [_match(clip, candidates) for clip in clips]

    letter = extract_letter(text)
    if letter == "NONE":
        return []
    if letter is None or LETTERS.index(letter) >= len(candidates):
        raise ResolveError(f"no candidate letter in answer: {text!r}")
    chosen = candidates[LETTERS.index(letter)]
    timestamp = r"\d+:\d{2}(?:\.\d+)?"
    window = re.search(rf"({timestamp})\s*-\s*({timestamp})", text)
    named_file = ""
    if window:
        after_letter = re.search(rf"\b{letter}\b", text).end()
        named_file = re.sub(
            r"^\s*[:.)-]?\s*", "", text[after_letter : window.start()]
        ).rstrip(" \t,;-")
    clip = {
        "file": named_file or chosen["file"],
        "start": window.group(1) if window else chosen["start"],
        "end": window.group(2) if window else chosen["end"],
    }
    if not clip_exact(clip, chosen):
        raise ResolveError(f"letter {letter} and the named window disagree: {text!r}")
    return [chosen]


def ffmpeg_command(clip: dict, footage: Path, out: Path, index: int) -> list[str]:
    source = footage / clip["file"]
    target = out / f"{index:02d}_{Path(clip['file']).stem}_{clip['start'].replace(':', '')}.mp4"
    return [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", str(parse_ts(clip["start"])), "-to", str(parse_ts(clip["end"])),
        "-i", str(source), "-c:v", "libx264", "-c:a", "aac", str(target),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prompt", help="print the STYLE_TEMPLATE prompt for a request")
    p.add_argument("request")
    c = sub.add_parser("cut", help="resolve a model answer and cut the clips")
    c.add_argument("request")
    c.add_argument("answer", help="the model's raw answer text, or @file to read it")
    c.add_argument("--footage", default=".", help="directory holding the source files")
    c.add_argument("--out", default="clips", help="where cut clips are written")
    c.add_argument("--dry-run", action="store_true", help="print ffmpeg commands only")
    sub.add_parser("welcome", help="show the welcome screen and intro slides")
    i = sub.add_parser("init", help="answer a few questions and write a request JSON")
    i.add_argument("--out", default="my-request.json")
    args = parser.parse_args(argv)

    if args.command in {"welcome", "init"}:
        import onboarding

        if args.command == "welcome":
            return onboarding.welcome()
        try:
            return onboarding.init(args.out)
        except (EOFError, KeyboardInterrupt):
            print("\ncancelled", file=sys.stderr)
            return 1
    try:
        request = load_request(args.request)
        if args.command == "prompt":
            print(build_prompt(request))
            return 0
        answer = args.answer
        if answer.startswith("@"):
            answer = Path(answer[1:]).read_text(encoding="utf-8")
        clips = resolve(answer, request)
    except ResolveError as exc:
        print(f"rejected: {exc}", file=sys.stderr)
        return 2

    if not clips:
        print("model refused: no candidate fits, nothing to cut")
        return 0
    footage, out = Path(args.footage), Path(args.out)
    for index, clip in enumerate(clips, 1):
        command = ffmpeg_command(clip, footage, out, index)
        print(f"clip {index}: {clip['file']} {clip['start']}-{clip['end']} -> {command[-1]}")
        if args.dry_run:
            print("  " + " ".join(command))
            continue
        if shutil.which("ffmpeg") is None:
            print("ffmpeg not found; rerun with --dry-run or install ffmpeg", file=sys.stderr)
            return 1
        if not (footage / clip["file"]).exists():
            print(f"missing source file: {footage / clip['file']}", file=sys.stderr)
            return 1
        out.mkdir(parents=True, exist_ok=True)
        subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
