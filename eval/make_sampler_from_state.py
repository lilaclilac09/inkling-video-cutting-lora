"""Make a sampler checkpoint from a saved training state.

Usage:
  /home/hatch/tinker-venv/bin/python eval/make_sampler_from_state.py \
      <state-tinker-path> <sampler-name>

Resumes a training client from the state (no training steps) and saves
sampler weights under <sampler-name>. Used when a training run saved its
state but the sampler save failed (e.g. storage limit) — the weights are
already good; only the sampler copy is missing.
"""
import asyncio
import os
import sys

sys.path.insert(0, "/opt/muse/skills/custom-connector-setup/bin")


async def main():
    state_path, name = sys.argv[1:3]
    for _k in ("no_proxy", "NO_PROXY"):
        _v = os.environ.get(_k, "")
        if _v:
            os.environ[_k] = ",".join(p for p in _v.split(",") if not p.strip().startswith("["))

    from dynamic_credentials import dynamic_credential_entry

    _entry = dynamic_credential_entry("custom.tinker", "access_token")
    os.environ["TINKER_API_KEY"] = _entry["surrogate"]
    print("Tinker credential loaded (surrogate, prefix hsurr:).", flush=True)

    import tinker
    from tinker.lib import _auth_token_provider as auth_mod

    def _patched_init(self, api_key: str) -> None:
        self._token = api_key
        self._expires_at = auth_mod._credential_expiry(api_key)

    auth_mod.ApiKeyAuthProvider.__init__ = _patched_init

    service_client = tinker.ServiceClient(display_name="inkling-video-cutting-sampler")
    print(f"Resuming from state {state_path} ...", flush=True)
    tc = await service_client.create_training_client_from_state(state_path)
    print("Training client resumed.", flush=True)
    ckpt = await tc.save_weights_for_sampler(name=name)
    print(f"SAMPLER_PATH={ckpt.path}", flush=True)
    print(f"Playground: {ckpt.get_playground_url()}&reasoning_effort=none", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
