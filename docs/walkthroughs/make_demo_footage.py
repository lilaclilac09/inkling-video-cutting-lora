"""Write synthetic stand-in footage (labelled test pattern) for every file a request names."""
import json
import subprocess
import sys
from pathlib import Path

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
files = sorted({c["file"] for path in sys.argv[2:] for c in json.loads(Path(path).read_text())["candidates"]})
for name in files:
    target = out / name
    if not target.exists():
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=duration=90:size=640x360:rate=25",
             "-f", "lavfi", "-i", "sine=frequency=440:duration=90", "-c:v", "libx264", "-c:a", "aac", "-shortest", str(target)],
            check=True,
        )
print(f"{len(files)} synthetic clips in {out}")
