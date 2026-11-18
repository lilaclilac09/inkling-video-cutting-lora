"""Probe: can a training client be created from the v2 STATE path?

Round 1 proved v1's SAMPLER path cannot resume training. v2 saved a proper
training state (tinker://.../weights/...). This probe creates the training
client from it, snapshots an on-policy sampler, and runs one tiny sample —
the exact first three calls the GRPO loop makes. No training steps, no
meaningful spend. Exit 0 + OK lines = GRPO can start; anything else = stop
and report the error before burning the sweep budget.
"""
import asyncio
import os
import sys

sys.path.insert(0, "/opt/muse/skills/custom-connector-setup/bin")

STATE_PATH = "tinker://0059270a-eeab-5add-bf67-b9f39ee0a5cb:train:0/weights/inkling-video-cutting-v2-state"
BASE_MODEL = "thinkingmachines/Inkling-2-Small"


async def main():
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

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    service_client = tinker.ServiceClient(display_name="inkling-video-cutting-probe")
    print(f"Trying create_training_client_from_state({STATE_PATH}) ...", flush=True)
    try:
        tc = await service_client.create_training_client_from_state(STATE_PATH)
        print("OK: training client resumed from v2 state")
    except Exception as e:
        print(f"RESUME_FAILED: {type(e).__name__}: {e}")
        return
    try:
        sampler = await tc.save_weights_and_get_sampling_client()
        print("OK: on-policy sampler created from resumed client")
    except Exception as e:
        print(f"SAMPLER_FAILED: {type(e).__name__}: {e}")
        return
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Pick the clip: A. f1 0:01-0:07 B. f2 0:02-0:08. Answer with the letter."}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    ptokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    result = await sampler.sample(
        prompt=tinker.ModelInput.from_ints(ptokens),
        num_samples=1,
        sampling_params=tinker.SamplingParams(max_tokens=64, temperature=0.0),
    )
    seq = result.sequences[0]
    text = tokenizer.decode(seq.tokens, skip_special_tokens=True).strip()
    print(f"OK: sample works from resumed client: logprobs={'yes' if seq.logprobs is not None else 'NO'} text={text[:120]!r}")


if __name__ == "__main__":
    asyncio.run(main())
