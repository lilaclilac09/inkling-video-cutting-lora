"""Probe: does SamplingClient.sample() return per-token logprobs?

Samples 2 short completions from the v2 sampler and prints whether
seq.logprobs is populated. GRPO needs sampling logprobs for the
importance_sampling datum; if sample() does not return them, the loop
must call compute_logprobs(full_sequence) instead.

Usage (after uv sync --extra train): uv run python eval/probe_logprobs.py
"""
import asyncio
import os
import sys

from tinker_env import setup_tinker

V2_PATH = "tinker://0059270a-eeab-5add-bf67-b9f39ee0a5cb:train:0/sampler_weights/inkling-video-cutting-v2"
BASE_MODEL = "thinkingmachines/Inkling-2-Small"


async def main():
    setup_tinker()

    import tinker

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    service_client = tinker.ServiceClient(display_name="inkling-video-cutting-probe")
    client = await service_client.create_sampling_client(model_path=V2_PATH)
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Pick the clip: A. f1 0:01-0:07 B. f2 0:02-0:08. Answer with the letter."}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    result = await client.sample(
        prompt=tinker.ModelInput.from_ints(prompt_tokens),
        num_samples=2,
        sampling_params=tinker.SamplingParams(max_tokens=64, temperature=0.7),
    )
    for i, seq in enumerate(result.sequences):
        text = tokenizer.decode(seq.tokens, skip_special_tokens=True).strip()
        print(f"seq {i}: tokens={len(seq.tokens)} logprobs={'YES len=%d' % len(seq.logprobs) if seq.logprobs is not None else 'NO (None)'} text={text[:100]!r}")
    # Also test compute_logprobs on a full sequence.
    full = prompt_tokens + list(result.sequences[0].tokens)
    lps = await client.compute_logprobs(
        tinker.ModelInput.from_ints(full), prompt_logprobs_last_n=len(result.sequences[0].tokens)
    )
    print(f"compute_logprobs last_n: {len(lps)} entries, non-None={sum(1 for x in lps if x is not None)}")
    print(f"  tail values: {[round(x, 3) if x is not None else None for x in lps[-3:]]}")


if __name__ == "__main__":
    asyncio.run(main())
