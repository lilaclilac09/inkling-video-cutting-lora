"""Delete Tinker checkpoints by tinker path (IRREVERSIBLE).

Usage (after uv sync --extra train):
  uv run python eval/delete_checkpoints.py <tinker-path> [...]

Only delete checkpoints this project created tonight that are strictly
dominated by a kept checkpoint (worse eval score, same lineage) or a
finished sweep's final state that no later sweep resumes from. Never
delete the published round-1 (v1) or the current-best checkpoints.
Prints the remaining total afterwards.
"""
import asyncio
import json
import os
import sys

from tinker_env import setup_tinker


async def main():
    paths = sys.argv[1:]
    if not paths:
        raise SystemExit("usage: delete_checkpoints.py <tinker-path> [...]")

    setup_tinker()

    import tinker

    rc = tinker.RestClient()
    for p in paths:
        await rc.delete_checkpoint_from_tinker_path(p)
        print(f"DELETED {p}", flush=True)
    resp = await rc.list_user_checkpoints(limit=100)
    total = sum(
        (c.model_dump().get("size_bytes") or 0) for c in resp.checkpoints
    )
    print(f"remaining: {len(resp.checkpoints)} checkpoints, {total / 1e9:.2f} GB")


if __name__ == "__main__":
    asyncio.run(main())
