"""Walkthrough 4: one-step install, intro slides, then the guided request builder.

Drives the real commands in a pseudo-terminal and types the answers, so it can be
recorded: asciinema rec -c "python3 docs/walkthroughs/04-install-and-init.py".
"""
import os
import pty
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROMPT = "$\033[0m "
STEPS = [
    (PROMPT, "./install.sh"),
    ("next", ""), ("next", ""), ("next", ""),
    (PROMPT, "uv run python cutter.py init --out my-request.json"),
    ("Event name", "Rooftop Launch Party"),
    ("Date", ""),
    ("Style number", "6"),
    ("Style name", "neon-night"),
    ("Describe it", "Night city glow, neon signs, fast beat-matched cuts."),
    ("How many beats", "0"),
]
for letter, (file, start, end, what) in zip("ABCD", [
    ("arrival.mp4", "0:02", "0:08", "Guests step out of the lift onto the lit rooftop"),
    ("skyline.mp4", "0:10", "0:16", "Slow pan across the neon skyline at dusk"),
    ("demo.mp4", "1:04", "1:12", "Founder shows the app on a phone, crowd leans in"),
    ("toast.mp4", "0:30", "0:35", "Everyone raises a glass under string lights"),
]):
    STEPS += [(f"{letter} file", file), (f"{letter} start", start), (f"{letter} end", end),
              (f"{letter} what", what)]
STEPS += [(PROMPT, "uv run python cutter.py prompt my-request.json"), (PROMPT, "exit")]


def type_out(fd, text):
    time.sleep(0.6)
    for ch in text:
        os.write(fd, ch.encode())
        time.sleep(0.035)
    time.sleep(0.3)
    os.write(fd, b"\r")


pid, fd = pty.fork()
if pid == 0:
    os.chdir(ROOT)
    os.execvpe("bash", ["bash", "--norc", "--noprofile", "-i"],
               {**os.environ, "PS1": "\033[1;32m$\033[0m "})
buffer, step = "", 0
while True:
    try:
        data = os.read(fd, 4096)
    except OSError:
        break
    if not data:
        break
    sys.stdout.write(data.decode(errors="replace"))
    sys.stdout.flush()
    buffer += data.decode(errors="replace")
    if step < len(STEPS) and STEPS[step][0] in buffer:
        buffer = ""
        wait = 2.5 if STEPS[step][0] == "next" else 0
        time.sleep(wait)
        type_out(fd, STEPS[step][1])
        step += 1
os.waitpid(pid, 0)
