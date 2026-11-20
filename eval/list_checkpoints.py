"""List all Tinker checkpoints for the current user (name, path, size).

Read-only. Used to find what is consuming the free 100 GB checkpoint
storage after the v3 sampler save failed with "limit reached".

Usage (after uv sync --extra train): uv run python eval/list_checkpoints.py
"""
import asyncio
import json
import os
import sys

from tinker_env import setup_tinker


async def main():
    setup_tinker()

    import tinker

    rc = tinker.RestClient()
    resp = await rc.list_user_checkpoints(limit=100)
    print(f"total checkpoints: {resp.cursor.total_count if resp.cursor else '?'}")
    total = 0
    for c in resp.checkpoints:
        d = c.model_dump() if hasattr(c, "model_dump") else dict(c)
        size = d.get("size_bytes") or d.get("size")
        if size:
            total += size
        print(f"- {json.dumps(d, default=str)[:400]}")
    print(f"TOTAL bytes (listed): {total} ({total / 1e9:.2f} GB)")


if __name__ == "__main__":
    asyncio.run(main())
