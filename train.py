"""The experiment target — this is the only file the research loop edits.

Every experiment changes one thing in HYPERPARAMETERS (or one line of the
training body), runs this file, and reads the score it prints. prepare.py owns
the contract, the data split, the scoring and the ledger; nothing here may
re-implement or bypass them.

Three ways to run it:

    python train.py --dry-run                 # no Tinker key, no spend, stub model
    python train.py --dry-run --fake-accuracy 0.5
    python train.py                           # real LoRA SFT on Tinker, then score

A dry run exercises the whole path — data, prompt, scoring, ledger — with a stub
sampler, so the harness can be trusted before a single token is paid for.

Real runs need `TINKER_API_KEY` (or `tinker auth login`) and `pip install tinker`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import prepare

# ── HYPERPARAMETERS — the research loop edits these ─────────────────────────
BASE_MODEL = "thinkingmachines/Inkling-2-Small"
CHECKPOINT_NAME = "inkling-video-cutting-v2"
RANK = 16
STEPS = 40
LEARNING_RATE = 2e-4
BATCH_SIZE = 4
MAX_ANSWER_TOKENS = 256
TEMPERATURE = 0.2
REASONING = "low"  # Inkling effort: low / medium / high / xhigh / max
# ────────────────────────────────────────────────────────────────────────────

RUNS_DIR = Path(__file__).resolve().parent / "runs"


def build_predictions(rows, sampler) -> dict[str, str]:
    """Ask the sampler for one answer per row. The sampler takes a prompt."""
    return {row.clip_key: sampler(row.prompt()) for row in rows}


def stub_sampler(accuracy: float):
    """A fake model for dry runs.

    It answers with the reference clip, then degrades on purpose so the eval can
    be seen discriminating: with probability `1 - accuracy` it either wraps the
    JSON in prose (schema failure), names a file the prompt never offered, or
    shifts the range. Deterministic per clip key, so dry runs are reproducible.
    """
    rows = {row.clip_key: row for row in prepare.load_rows()}

    def sampler(prompt: str) -> str:
        row = next((r for r in rows.values() if r.prompt() == prompt), None)
        if row is None:
            return "I cannot find that clip."
        digest = int(hashlib.sha256(row.clip_key.encode()).hexdigest()[:8], 16)
        roll = (digest % 1000) / 1000
        if roll < accuracy:
            return row.answer()
        mode = digest % 3
        if mode == 0:
            return f"Here is the clip I would cut. {row.answer()}"
        if mode == 1:
            broken = dict(row.clip, file="DJI_00000000000000_9999_D.MP4")
            return json.dumps(broken, ensure_ascii=False)
        start = prepare.parse_time(row.clip["start"])
        end = prepare.parse_time(row.clip["end"])
        shifted = dict(
            row.clip,
            start=prepare.format_time(start + 6),
            end=prepare.format_time(end + 6),
        )
        return json.dumps(shifted, ensure_ascii=False)

    return sampler


def tinker_sampler(rows, dry_run: bool):
    """Train a LoRA on Tinker and return a sampler bound to the new checkpoint."""
    import tinker  # imported lazily so dry runs need no Tinker install

    service = tinker.ServiceClient(display_name="inkling-video-cutting")
    training = service.create_lora_training_client(base_model=BASE_MODEL, rank=RANK)
    tokenizer = training.get_tokenizer()

    def render(prompt: str) -> tuple[list[int], int]:
        messages = [{"role": "user", "content": prompt}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        return prompt_ids, len(prompt_ids)

    def make_datum(prompt: str, answer: str):
        prompt_ids, prompt_len = render(prompt)
        full_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}],
            tokenize=False,
        )
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
        completion_ids = full_ids[prompt_len:]
        tokens = prompt_ids + completion_ids
        weights = [0.0] * prompt_len + [1.0] * len(completion_ids)
        return tinker.Datum(
            model_input=tinker.ModelInput.from_ints(tokens[:-1]),
            loss_fn_inputs={
                "target_tokens": tokens[1:],
                "weights": weights[1:],
            },
        )

    data = [make_datum(row.prompt(), row.answer()) for row in rows]
    for start in range(0, len(data), BATCH_SIZE):
        batch = data[start : start + BATCH_SIZE]
        forward = training.forward_backward(batch, "cross_entropy")
        optim = training.optim_step(tinker.AdamParams(learning_rate=LEARNING_RATE))
        result = forward.result()
        logprobs = [
            lp for output in result.loss_fn_outputs for lp in output["logprobs"].tolist()
        ]
        weights = [w for datum in batch for w in datum.loss_fn_inputs["weights"].tolist()]
        loss = -sum(lp * w for lp, w in zip(logprobs, weights)) / sum(weights)
        print(f"step {start // BATCH_SIZE:3d}: loss {loss:.4f}", flush=True)

    saved = training.save_weights_for_sampler(name=CHECKPOINT_NAME).result()
    sampling = service.create_sampling_client(model_path=saved.path)
    print(f"checkpoint: {saved.path}")

    def sampler(prompt: str) -> str:
        prompt_ids, _ = render(prompt)
        result = sampling.sample(
            prompt=tinker.ModelInput.from_ints(prompt_ids),
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                max_tokens=MAX_ANSWER_TOKENS, temperature=TEMPERATURE
            ),
        )
        return tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=True).strip()

    return sampler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="stub model, no Tinker")
    parser.add_argument(
        "--fake-accuracy",
        type=float,
        default=1.0,
        help="dry runs only: fraction of answers the stub gets right",
    )
    parser.add_argument("--note", default="", help="one line recorded in results.tsv")
    parser.add_argument("--run-id", default="", help="label for this run")
    args = parser.parse_args(argv)

    rows = prepare.load_rows()
    train_rows, held_out = prepare.split(rows)
    run_id = args.run_id or os.environ.get("RUN_ID") or f"run-{len(train_rows)}-r{RANK}-s{STEPS}"
    config = {
        "run_id": run_id,
        "model": "stub" if args.dry_run else BASE_MODEL,
        "rank": RANK,
        "lr": LEARNING_RATE,
        "steps": STEPS,
        "train_rows": len(train_rows),
    }
    print(f"train rows {len(train_rows)}  held out {len(held_out)}  config {config}")

    sampler = stub_sampler(args.fake_accuracy) if args.dry_run else tinker_sampler(train_rows, False)
    predictions = build_predictions(held_out, sampler)

    summary = prepare.evaluate(predictions, held_out)
    RUNS_DIR.mkdir(exist_ok=True)
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(exist_ok=True)
    (run_dir / "predictions.jsonl").write_text(
        "".join(
            json.dumps({"clip_key": key, "answer": value}, ensure_ascii=False) + "\n"
            for key, value in predictions.items()
        ),
        encoding="utf-8",
    )

    print(
        f"\nheld-out score {summary['mean_score']:.4f}  "
        f"schema {summary['schema_valid_rate']:.2f}  "
        f"file {summary['file_match_rate']:.2f}  iou {summary['mean_iou']:.2f}"
    )
    for detail in summary["details"]:
        mark = "ok " if detail["schema_valid"] else "BAD"
        print(f"  {mark} {detail['clip_key']:<48} score {detail['score']:.2f}")

    if args.dry_run:
        # A dry run is not an experiment: it must not touch the research record.
        verdict = "KEEP" if prepare.decide(summary) else "DISCARD"
        print(f"\ndry run: ledger not written (would be {verdict})")
    else:
        keep = prepare.record_result(summary, config, args.note)
        print(f"\n{'KEEP' if keep else 'DISCARD'} — ledger updated at results.tsv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
