"""Welcome screen, intro slides and a guided request builder for new users.

    uv run python cutter.py welcome     # banner + four intro slides
    uv run python cutter.py init        # answer a few questions, get a request JSON
"""
from __future__ import annotations

import datetime
import json
import os
import re
import shutil
import sys
from pathlib import Path

from eval.scoring import parse_ts

INK = "\033[38;5;255m"
GREY = "\033[38;5;245m"
BLUE = "\033[38;5;75m"
BOLD = "\033[1m"
RESET = "\033[0m"

SLIDES = [
    (
        "Pick a style.",
        "Five trained looks: vlog, fresh, silicon-valley, glamour, hollywood.\n"
        "Or describe your own in one sentence.",
        "uv run python cutter.py init",
    ),
    (
        "List your moments.",
        "Give 4 candidate windows from your footage (8 for a 2–3 beat cut):\n"
        "file, start, end and what is on screen.",
        "examples/request-single.json",
    ),
    (
        "Ask the model.",
        "The prompt matches the one the LoRA was scored on.\n"
        "Send it to the winner checkpoint on Tinker.",
        "uv run python cutter.py prompt my-request.json",
    ),
    (
        "Cut.",
        "The answer is checked against your candidates, then ffmpeg cuts it.\n"
        "Invented windows are rejected. \"None\" cuts nothing.",
        'uv run python cutter.py cut my-request.json "<answer>" --footage footage/',
    ),
]


def _color() -> bool:
    return sys.stdout.isatty() and "NO_COLOR" not in os.environ


def _paint(code: str, text: str) -> str:
    return f"{code}{text}{RESET}" if _color() else text


def _center(text: str, width: int) -> str:
    return text.center(width).rstrip()


def banner(width: int | None = None) -> str:
    width = width or min(shutil.get_terminal_size((80, 24)).columns, 80)
    rule = "─" * min(40, width - 4)
    lines = [
        "",
        _paint(BOLD + INK, _center("i n k l i n g", width)),
        _paint(GREY, _center("Video cutting, in your style.", width)),
        "",
        _paint(GREY, _center(rule, width)),
        "",
    ]
    return "\n".join(lines)


def slide(index: int, width: int | None = None) -> str:
    width = width or min(shutil.get_terminal_size((80, 24)).columns, 80)
    title, body, command = SLIDES[index]
    dots = " ".join("●" if i == index else "○" for i in range(len(SLIDES)))
    lines = ["", _paint(BOLD + INK, _center(title, width)), ""]
    lines += [_paint(GREY, _center(line, width)) for line in body.splitlines()]
    lines += ["", _paint(BLUE, _center(command, width)), "", _paint(GREY, _center(dots, width)), ""]
    return "\n".join(lines)


def welcome(interactive: bool | None = None) -> int:
    interactive = sys.stdin.isatty() if interactive is None else interactive
    for index in range(len(SLIDES)):
        if interactive:
            print("\033[2J\033[H", end="")
        if interactive or index == 0:
            print(banner())
        print(slide(index))
        if interactive and index < len(SLIDES) - 1:
            input(_paint(GREY, "  return ↵ next"))
    print(_paint(INK, "  Ready. Run `uv run python cutter.py init` to set up your first cut.\n"))
    return 0


def _today() -> str:
    day = datetime.date.today()
    return f"{day:%B} {day.day}, {day.year}"


def _ask(prompt: str, default: str | None = None) -> str:
    hint = f" [{default}]" if default else ""
    while True:
        value = input(f"  {prompt}{hint}: ").strip()
        if value or default is not None:
            return value or default
        print("  (required)")


def _ask_window(label: str) -> dict:
    while True:
        file = _ask(f"{label} file")
        start = _ask(f"{label} start (M:SS)")
        end = _ask(f"{label} end (M:SS)")
        if all(re.fullmatch(r"\d+:\d{2}(?:\.\d+)?", t) for t in (start, end)) and (
            parse_ts(end) > parse_ts(start)
        ):
            return {
                "file": file,
                "start": start,
                "end": end,
                "description": _ask(f"{label} what is on screen"),
            }
        print("  end must be a later M:SS time than start; try again")


def init(out: str | Path = "my-request.json") -> int:
    from cutter import STYLES

    print(banner())
    request = {"event": _ask("Event name"), "date": _ask("Date", _today())}
    names = list(STYLES)
    for number, name in enumerate(names, 1):
        print(f"    {number}. {STYLES[name]}")
    print(f"    {len(names) + 1}. Your own style (describe it)")
    while True:
        pick = _ask("Style number", "1")
        if pick.isdigit() and 1 <= int(pick) <= len(names) + 1:
            break
    if int(pick) <= len(names):
        request["style"] = names[int(pick) - 1]
    else:
        request["style"] = _ask("Style name, e.g. anime-summer")
        request["style_description"] = _ask("Describe it in one sentence")
    while True:
        count = _ask("How many beats (0 = just one clip, up to 3)", "0")
        if count in {"0", "1", "2", "3"}:
            break
    request["beats"] = [
        {"title": _ask(f"Beat {i + 1} title"), "why": _ask(f"Beat {i + 1} why it matters")}
        for i in range(int(count))
    ]
    total = 8 if int(count) > 1 else 4
    print(f"  Now your {total} candidate windows from the footage.")
    request["candidates"] = [_ask_window(f"Candidate {'ABCDEFGH'[i]}") for i in range(total)]
    Path(out).write_text(json.dumps(request, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(_paint(INK, f"\n  Saved {out}. Next: uv run python cutter.py prompt {out}\n"))
    return 0
