"""Probe: are sample() logprobs tempered?

GRPO step 0 showed e_max_violation ~46, which would be impossible if the
sampling logprobs matched the training policy exactly (fresh on-policy
snapshot => ratio 1). One hypothesis: sample() returns temperature-scaled
logprobs. This probe samples one completion at temp 0.7 from the v2 sampler
and compares seq.logprobs against compute_logprobs() on the same full
sequence (untempered by definition). If they match, the GRPO datums are
correct and the violation metric has another cause; if they differ
systematically, GRPO must use compute_logprobs for sampling logprobs.
"""
import asyncio
import os
import sys

sys.path.insert(0, "/opt/muse/skills/custom-connector-setup/bin")

V2_PATH = "tinker://0059270a-eeab-5add-bf67-b9f39ee0a5cb:train:0/sampler_weights/inkling-video-cutting-v2"
BASE_MODEL = "thinkingmachines/Inkling-2-Small"


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

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    service_client = tinker.ServiceClient(display_name="inkling-video-cutting-probe")
    client = await service_client.create_sampling_client(model_path=V2_PATH)
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Pick the clip: A. f1 0:01-0:07 B. f2 0:02-0:08 C. f3 0:03-0:09 D. f4 0:04-0:10. Answer with the letter and the clip."}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    ptokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    result = await client.sample(
        prompt=tinker.ModelInput.from_ints(ptokens),
        num_samples=1,
        sampling_params=tinker.SamplingParams(max_tokens=64, temperature=0.7),
    )
    seq = result.sequences[0]
    full = ptokens + list(seq.tokens)
    lps = await client.compute_logprobs(
        tinker.ModelInput.from_ints(full), prompt_logprobs_last_n=len(seq.tokens)
    )
    # compute_logprobs returns len(full) entries; the last len(tokens) are the
    # completion positions (aligned with seq.tokens).
    comp_lps = lps[len(ptokens):]
    print(f"completion tokens: {len(seq.tokens)}")
    diffs = [abs(a - b) for a, b in zip(seq.logprobs, comp_lps) if b is not None]
    print(f"sample-logprobs vs compute_logprobs: max|diff|={max(diffs):.4f} mean|diff|={sum(diffs)/len(diffs):.4f}")
    print("first 5 pairs (sample, computed):")
    for a, b in list(zip(seq.logprobs, comp_lps))[:5]:
        print(f"  {a:.4f} vs {b if b is None else round(b, 4)}")
    if max(diffs) < 0.05:
        print("MATCH: sample() logprobs are untempered; GRPO datums are correct.")
    else:
        print("MISMATCH: sample() logprobs differ from compute_logprobs — GRPO should use compute_logprobs.")


if __name__ == "__main__":
    asyncio.run(main())
