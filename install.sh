#!/usr/bin/env bash
# One-step install: checks the tools, installs the harness, verifies it, shows the intro.
set -e
cd "$(dirname "$0")"
for tool in uv ffmpeg; do
  if ! command -v "$tool" >/dev/null; then
    case $tool in
      uv) echo "uv is missing: curl -LsSf https://astral.sh/uv/install.sh | sh" ;;
      ffmpeg) echo "ffmpeg is missing: brew install ffmpeg  (or: sudo apt install ffmpeg)" ;;
    esac
    exit 1
  fi
done
echo "Installing…"
uv sync --quiet
echo "Checking…"
uv run pytest -q >/dev/null
uv run python prepare.py --self-check >/dev/null
echo "Installed and verified."
uv run python cutter.py welcome
