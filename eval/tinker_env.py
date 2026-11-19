"""Portable Tinker setup shared by the training and sampling scripts."""
import os


def setup_tinker():
    """Require an explicit API key without logging it or patching SDK auth."""
    for name in ("no_proxy", "NO_PROXY"):
        value = os.environ.get(name, "")
        if value:
            os.environ[name] = ",".join(
                part for part in value.split(",") if not part.strip().startswith("[")
            )
    if not os.environ.get("TINKER_API_KEY", "").strip():
        raise SystemExit(
            "Set TINKER_API_KEY in your environment before running this script. "
            "Install training dependencies with: uv sync --extra train"
        )
