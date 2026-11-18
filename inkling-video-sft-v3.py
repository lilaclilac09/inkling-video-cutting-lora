"""SFT stage 2 (v3) for Inkling 2 Small: replay + refusal teaching.

Starts from the v2 TRAINING STATE (not from base, not from sampler weights)
and runs 6 gentle steps (lr 1e-4) on 35 examples:
  * 27 original training examples (replay — prevents forgetting the picks),
  * 8 refusal examples: moments verified absent from the footage
    (fireworks / a dog on stage / snow / a birthday cake) in vlog + one
    other style each, answer "None".

Why SFT, not GRPO, for refusal: GRPO can only reinforce behaviour that
already appears in rollouts. v2 never refuses (trap 0.00, hallucinated
0.75-1.00), so trap groups are degenerate and carry no gradient — GRPO
cannot invent the refusal. SFT teaches the FORMAT directly. The refusal
moments are deliberately NOT the eval's trap moments (guitar / piano /
rain / group photo), so the eval still tests generalisation, not memory.

New names (v2 untouched):
  sampler checkpoint: inkling-video-cutting-v3
  training state:     inkling-video-cutting-v3-state

Run via run_sft_v3.py (auth workaround lives there, not here).
"""
import asyncio
import json
import os

DATA_PATH = os.environ.get("SFT_V3_DATA_PATH", "eval/sft-v3-data.jsonl")
BASE_MODEL = "thinkingmachines/Inkling-2-Small"
V2_STATE = "tinker://0059270a-eeab-5add-bf67-b9f39ee0a5cb:train:0/weights/inkling-video-cutting-v2-state"
CHECKPOINT_NAME = "inkling-video-cutting-v3"
STATE_NAME = "inkling-video-cutting-v3-state"
STEPS = 6
LEARNING_RATE = 1e-4

PLAYGROUND_PROMPT = """You are cutting a tech meetup talk into short clips for social media.

Below is the full transcript with timestamps, plus keyframes from the video.

Pick the 3 most shareable moments. Each clip 30-90 seconds. For each clip, give:
- start and end timestamp
- a short title
- one line on why it works standalone

Only pick moments that make sense without the rest of the talk. Prefer moments with a clear takeaway, a live demo, or a surprising point. Prefer moments where a guest is presenting their own work in their own words (a showcase monologue). Prefer moments that include audience response (laughter, applause, someone picking up the thread) or follow-up questions. Do not cut mid-sentence; extend to the nearest sentence boundary if needed. Timestamps must match the transcript.

Transcript:
"""

TEST_PROMPTS = [
    "Transcript excerpt: [14:02:11] someone sits at the piano and plays a short tune. [14:02:40] the room goes quiet, then applauds. Pick the clip.",
]


def load_rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def build_example(row):
    if row["kind"] == "refusal":
        return row["prompt"], row["answer"]
    prompt = (
        PLAYGROUND_PROMPT
        + row.get("transcript_excerpt", "")
        + "\n\nKeyframes:\n"
        + "\n".join(row.get("keyframes", []))
    )
    answer = json.dumps(row.get("clip", {}), ensure_ascii=False)
    return prompt, answer


async def main():
    rows = load_rows(DATA_PATH)
    from collections import Counter
    print(f"Training rows: {len(rows)} {Counter(r['kind'] for r in rows)}")

    import tinker

    service_client = tinker.ServiceClient(display_name="inkling-video-cutting-v3")
    print(f"Resuming training from v2 state ...", flush=True)
    training_client = await service_client.create_training_client_from_state(V2_STATE)
    print("Training client resumed from v2 state.", flush=True)

    base_client = await service_client.create_sampling_client(base_model=BASE_MODEL)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    examples = [build_example(r) for r in rows]

    def make_datum(prompt, answer):
        messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]
        prompt_text = tokenizer.apply_chat_template(
            messages[:1], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        full_text = tokenizer.apply_chat_template(messages, tokenize=False, enable_thinking=False)
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
        completion_tokens = tokenizer.encode(
            full_text[len(prompt_text):], add_special_tokens=False
        )
        tokens = prompt_tokens + completion_tokens
        weights = [0.0] * len(prompt_tokens) + [1.0] * len(completion_tokens)
        return tinker.Datum(
            model_input=tinker.ModelInput.from_ints(tokens[:-1]),
            loss_fn_inputs={"target_tokens": tokens[1:], "weights": weights[1:]},
        )

    training_data = [make_datum(p, a) for p, a in examples]

    for step in range(STEPS):
        fwdbwd_future = training_client.forward_backward(training_data, "cross_entropy")
        optim_future = training_client.optim_step(tinker.AdamParams(learning_rate=LEARNING_RATE))
        fwdbwd_result = await fwdbwd_future
        await optim_future
        logprobs = [
            lp for output in fwdbwd_result.loss_fn_outputs for lp in output["logprobs"].tolist()
        ]
        weights = [
            w for datum in training_data for w in datum.loss_fn_inputs["weights"].tolist()
        ]
        loss = -sum(lp * w for lp, w in zip(logprobs, weights)) / sum(weights)
        print(f"Step {step:2d}: loss = {loss:.4f}", flush=True)

    try:
        state_future = training_client.save_state(name=STATE_NAME)
        state = await state_future
        print(f"\nSTATE_PATH={state.path}", flush=True)
    except Exception as e:
        print(f"\nSTATE_SAVE_FAILED: {type(e).__name__}: {e}", flush=True)

    save_future = training_client.save_weights_for_sampler(name=CHECKPOINT_NAME)
    checkpoint = await save_future
    print(f"SAMPLER_PATH={checkpoint.path}", flush=True)
    trained_client = await service_client.create_sampling_client(model_path=checkpoint.path)

    print(f"\nTry it in the Playground: {checkpoint.get_playground_url()}&reasoning_effort=none")

    async def sample(client, prompt):
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
        result = await client.sample(
            prompt=tinker.ModelInput.from_ints(prompt_tokens),
            num_samples=1,
            sampling_params=tinker.SamplingParams(max_tokens=512, temperature=0.7),
        )
        return result.sequences[0]

    def decode(sequence):
        return tokenizer.decode(sequence.tokens, skip_special_tokens=True).strip()

    for prompt in TEST_PROMPTS:
        before = await sample(base_client, prompt)
        after = await sample(trained_client, prompt)
        print(f"\nQ: {prompt}")
        print(f"Before: {decode(before)}")
        print(f"After:  {decode(after)}")

    # Refusal check on a TRAINING moment (fireworks — not an eval moment).
    refusal_rows = [r for r in rows if r["kind"] == "refusal"]
    if refusal_rows:
        rp = refusal_rows[0]["prompt"]
        before = await sample(base_client, rp)
        after = await sample(trained_client, rp)
        print(f"\nRefusal check (training moment, fireworks):")
        print(f"Before: {decode(before)[:300]}")
        print(f"After:  {decode(after)[:300]}")


if __name__ == "__main__":
    asyncio.run(main())
