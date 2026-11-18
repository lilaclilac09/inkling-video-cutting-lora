"""Run the SFT stage 2 (v3) on this VM using the stored Tinker credential.

Same two workarounds as run_lora_training.py / run_sft_v2.py:
1. httpx chokes on bracketed IPv6 entries in no_proxy — strip them.
2. The stored credential reaches this process only as an hsurr: surrogate;
   patch ApiKeyAuthProvider.__init__ to accept it.

The surrogate is never written to disk or logged; it lives only in this process.
"""
import os
import runpy
import sys

for _k in ("no_proxy", "NO_PROXY"):
    _v = os.environ.get(_k, "")
    if _v:
        os.environ[_k] = ",".join(p for p in _v.split(",") if not p.strip().startswith("["))

sys.path.insert(0, "/opt/muse/skills/custom-connector-setup/bin")
from dynamic_credentials import dynamic_credential_entry

_entry = dynamic_credential_entry("custom.tinker", "access_token")
_surrogate = _entry["surrogate"]
os.environ["TINKER_API_KEY"] = _surrogate
print("Tinker credential loaded (surrogate, prefix hsurr:).", flush=True)

import tinker
from tinker.lib import _auth_token_provider as auth_mod


def _patched_init(self, api_key: str) -> None:
    self._token = api_key
    self._expires_at = auth_mod._credential_expiry(api_key)


auth_mod.ApiKeyAuthProvider.__init__ = _patched_init

os.environ.setdefault("SFT_V3_DATA_PATH", "eval/sft-v3-data.jsonl")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.argv = ["inkling-video-sft-v3.py"]
runpy.run_path(
    "/home/hatch/workspace/inkling-recap/inkling-video-sft-v3.py",
    run_name="__main__",
)
