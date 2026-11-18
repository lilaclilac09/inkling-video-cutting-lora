"""GRPO loop for the Inkling video-cutting LoRA (v2 -> v2-grpo).

Starts from the SFT v2 training state (resumable weights path — v1 sampler
weights cannot resume training). Each step:
  1. refresh the on-policy sampler from current weights,
  2. sample G rollouts per prompt (temp 0.7) for a batch of prompts,
  3. reward 0/1 per rollout (right clip picked / refused a missing moment),
  4. group-relative advantages A = R - mean(R_group); constant-reward groups
     carry no signal and are skipped ("degenerate"),
  5. forward_backward with the built-in "importance_sampling" loss
     (datum: target_tokens + sampling logprobs + per-token advantages,
     prompt positions zero-padded — per the official Tinker SDK cheatsheet),
  6. optim_step (Adam, lr 1e-5, no KL term — the importance-sampling ratio
     is the off-policy correction, same as the Suiche RLVR reference:
     60 GRPO steps, pass@1 0.667 -> 0.708, $22).

Checkpoints by validation, never by step count: sampler checkpoints are saved
at steps 20/40/60 and each is scored on eval-set-v2 afterwards; the best one
wins. Hard cap $200 total (this run is expected to cost roughly Suiche-scale,
tens of dollars; token counts are logged per step so spend stays visible).
If any step fails 3 times in a row, or create_training_client_from_state
rejects the state path, the script stops and says exactly why.

Usage (tinker-venv python):
  /home/hatch/tinker-venv/bin/python eval/run_grpo.py \
      --steps 60 --group-size 8 --prompts-per-step 4 --lr 1e-5 \
      --state-path 'tinker://.../weights/inkling-video-cutting-v2-state'
"""
import argparse
import asyncio
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import score_item  # noqa: E402

sys.path.insert(0, "/opt/muse/skills/custom-connector-setup/bin")

BASE_MODEL = "thinkingmachines/Inkling-2-Small"
DEFAULT_STATE = "tinker://0059270a-eeab-5add-bf67-b9f39ee0a5cb:train:0/weights/inkling-video-cutting-v2-state"
PROMPTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grpo-prompts.jsonl")


def load_prompts(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--prompts-per-step", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--state-path", default=DEFAULT_STATE)
    ap.add_argument("--prompts", default=PROMPTS_PATH)
    ap.add_argument("--name-prefix", default="inkling-video-cutting-v2-grpo")
    ap.add_argument("--no-final-state", action="store_true",
                    help="Skip the final save_state (25 GB). Use when checkpoint "
                         "storage is tight; the best sampler checkpoint is the "
                         "deliverable and the start state remains resumable.")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    log_path = args.log or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "grpo-run.log"
    )
    logf = open(log_path, "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    for _k in ("no_proxy", "NO_PROXY"):
        _v = os.environ.get(_k, "")
        if _v:
            os.environ[_k] = ",".join(p for p in _v.split(",") if not p.strip().startswith("["))

    from dynamic_credentials import dynamic_credential_entry

    _entry = dynamic_credential_entry("custom.tinker", "access_token")
    os.environ["TINKER_API_KEY"] = _entry["surrogate"]
    log("Tinker credential loaded (surrogate, prefix hsurr:).")

    import tinker
    from tinker.lib import _auth_token_provider as auth_mod

    def _patched_init(self, api_key: str) -> None:
        self._token = api_key
        self._expires_at = auth_mod._credential_expiry(api_key)

    auth_mod.ApiKeyAuthProvider.__init__ = _patched_init

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    prompts = load_prompts(args.prompts)
    log(f"GRPO prompts: {len(prompts)} from {args.prompts} (mc/style-mc/trap)")

    service_client = tinker.ServiceClient(display_name="inkling-video-cutting-grpo")
    log(f"Resuming training from state: {args.state_path}")
    try:
        training_client = await service_client.create_training_client_from_state(
            args.state_path
        )
    except Exception as e:
        log(f"RESUME_FAILED: {type(e).__name__}: {e}")
        log("STOP: cannot resume from the state path; GRPO cannot start from sampler weights.")
        return
    log("Training client resumed from state.")

    rng = random.Random(args.seed)
    order = list(range(len(prompts)))
    rng.shuffle(order)
    cursor = [0]

    def next_batch():
        batch = []
        for _ in range(args.prompts_per_step):
            if cursor[0] >= len(order):
                rng.shuffle(order)
                cursor[0] = 0
            batch.append(prompts[order[cursor[0]]])
            cursor[0] += 1
        return batch

    def render_prompt(text):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )

    sampler = None
    total_sample_tokens = 0
    total_train_tokens = 0
    consecutive_failures = 0

    for step in range(args.steps):
        t0 = time.time()
        # 1. Refresh the on-policy sampler from current weights.
        try:
            sampler = await training_client.save_weights_and_get_sampling_client()
        except Exception as e:
            log(f"step {step}: SAMPLER_REFRESH_FAILED {type(e).__name__}: {e} (keeping previous sampler)")
            if sampler is None:
                log("STOP: no sampler available.")
                return

        batch = next_batch()

        # 2. Sample G rollouts per prompt (parallel across prompts).
        async def sample_prompt(prompt):
            ptext = render_prompt(prompt["prompt"])
            ptokens = tokenizer.encode(ptext, add_special_tokens=False)
            result = await sampler.sample(
                prompt=tinker.ModelInput.from_ints(ptokens),
                num_samples=args.group_size,
                sampling_params=tinker.SamplingParams(
                    max_tokens=args.max_tokens, temperature=args.temperature
                ),
            )
            return prompt, ptokens, result

        try:
            sampled = await asyncio.gather(*[sample_prompt(p) for p in batch])
        except Exception as e:
            consecutive_failures += 1
            log(f"step {step}: SAMPLE_FAILED {type(e).__name__}: {e} (consecutive={consecutive_failures})")
            if consecutive_failures >= 3:
                log("STOP: 3 consecutive sampling failures.")
                return
            continue
        consecutive_failures = 0

        # 3-4. Reward + group-relative advantages; build datums.
        datums = []
        step_rewards = []
        degenerate = 0
        for prompt, ptokens, result in sampled:
            rewards = []
            seqs = []
            for seq in result.sequences:
                text = tokenizer.decode(seq.tokens, skip_special_tokens=True).strip()
                rec = score_item(prompt, text)
                rewards.append(rec["correct"])
                seqs.append((seq, text))
                total_sample_tokens += len(ptokens) + len(seq.tokens)
            step_rewards.extend(rewards)
            mean_r = sum(rewards) / len(rewards) if rewards else 0.0
            if len(set(rewards)) <= 1:
                degenerate += 1
                continue
            n_prefix = len(ptokens) - 1
            for (seq, _text), r in zip(seqs, rewards):
                adv = r - mean_r
                if adv == 0 or len(seq.tokens) == 0:
                    continue
                if seq.logprobs is None or len(seq.logprobs) != len(seq.tokens):
                    log(f"step {step}: rollout without logprobs skipped ({prompt['id']})")
                    continue
                full = list(ptokens) + list(seq.tokens)
                datums.append(tinker.Datum(
                    model_input=tinker.ModelInput.from_ints(full[:-1]),
                    loss_fn_inputs={
                        "target_tokens": full[1:],
                        "logprobs": [0.0] * n_prefix + [float(x) for x in seq.logprobs],
                        "advantages": [0.0] * n_prefix + [float(adv)] * len(seq.tokens),
                    },
                ))
                total_train_tokens += len(full)

        mean_reward = sum(step_rewards) / len(step_rewards) if step_rewards else 0.0

        # 5-6. Train on this step's datums (skip if no signal).
        loss_str = "n/a (no datums)"
        if datums:
            try:
                fwdbwd_future = training_client.forward_backward(datums, "importance_sampling")
                optim_future = training_client.optim_step(
                    tinker.AdamParams(learning_rate=args.lr)
                )
                fwdbwd_result = await fwdbwd_future
                await optim_future
                loss_str = json.dumps(dict(fwdbwd_result.metrics)) if fwdbwd_result.metrics else "n/a"
            except Exception as e:
                consecutive_failures += 1
                log(f"step {step}: TRAIN_FAILED {type(e).__name__}: {e} (consecutive={consecutive_failures})")
                if consecutive_failures >= 3:
                    log("STOP: 3 consecutive training failures.")
                    return
                continue
            consecutive_failures = 0

        dt = time.time() - t0
        log(f"step {step:3d}: reward={mean_reward:.3f} degenerate_groups={degenerate}/{len(batch)} "
            f"datums={len(datums)} loss={loss_str} {dt:.1f}s "
            f"cum_sample_tokens={total_sample_tokens} cum_train_tokens={total_train_tokens}")

        # Checkpoint + eval points.
        if (step + 1) % args.eval_every == 0 or (step + 1) == args.steps:
            name = f"{args.name_prefix}-{step + 1}"
            try:
                ckpt = await training_client.save_weights_for_sampler(name=name)
                log(f"CHECKPOINT step {step + 1}: name={name} path={ckpt.path}")
                log(f"  playground: {ckpt.get_playground_url()}&reasoning_effort=none")
            except Exception as e:
                log(f"CHECKPOINT_FAILED step {step + 1}: {type(e).__name__}: {e}")

    if not args.no_final_state:
        try:
            state = await training_client.save_state(name=f"{args.name_prefix}-state")
            log(f"FINAL_STATE_PATH={state.path}")
        except Exception as e:
            log(f"FINAL_STATE_SAVE_FAILED: {type(e).__name__}: {e}")
    else:
        log("FINAL_STATE skipped (--no-final-state; storage discipline).")
    log(f"DONE: {args.steps} steps. cum_sample_tokens={total_sample_tokens} "
        f"cum_train_tokens={total_train_tokens}")
    logf.close()


if __name__ == "__main__":
    asyncio.run(main())
