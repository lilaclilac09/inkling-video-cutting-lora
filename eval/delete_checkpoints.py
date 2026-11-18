"""Delete Tinker checkpoints by tinker path (IRREVERSIBLE).

Usage:
  /home/hatch/tinker-venv/bin/python eval/delete_checkpoints.py <tinker-path> [...]

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

sys.path.insert(0, "/opt/muse/skills/custom-connector-setup/bin")


async def main():
    paths = sys.argv[1:]
    if not paths:
        raise SystemExit("usage: delete_checkpoints.py <tinker-path> [...]")

    for _k in ("no_proxy", "NO_PROXY"):
        _v = os.environ.get(_k, "")
        if _v:
            os.environ[_k] = ",".join(p for p in _v.split(",") if not p.strip().startswith("["))

    from dynamic_credentials import dynamic_credential_entry

    _entry = dynamic_credential_entry("custom.tinker", "access_token")
    os.environ["TINKER_API_KEY"] = _entry["surrogate"]

    import tinker
    from tinker.lib import _auth_token_provider as auth_mod

    def _patched_init(self, api_key: str) -> None:
        self._token = api_key
        self._expires_at = auth_mod._credential_expiry(api_key)

    auth_mod.ApiKeyAuthProvider.__init__ = _patched_init

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
