"""List all Tinker checkpoints for the current user (name, path, size).

Read-only. Used to find what is consuming the free 100 GB checkpoint
storage after the v3 sampler save failed with "limit reached".
"""
import asyncio
import json
import os
import sys

sys.path.insert(0, "/opt/muse/skills/custom-connector-setup/bin")


async def main():
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
