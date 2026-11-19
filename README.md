# Inkling Video Cutting LoRA

A LoRA fine-tune of [Thinking Machines](https://thinkingmachines.ai) **Inkling 2 Small**
that turns a transcript excerpt plus keyframes into **one** shareable clip pick, in a
shape a cutter script can consume without a human fixing the JSON.

Built on the Cafe Cursor Shanghai footage. The long-term shape of the project — a Drive
link in, a styled video out — is in [`docs/ROADMAP.md`](docs/ROADMAP.md).

Start with [`SPEC.md`](SPEC.md): my current contracts, winner, acceptance gates,
CPU/GPU compute plan and next steps. The one-clip harness below is separate from
the winner's letter / multi-pick / None template contract.

## The three files

This repo follows the same architecture as [`autoresearch-course`](https://github.com/lilaclilac09/autoresearch-course):
a fixed instrument, an editable target, and a written protocol.

| file | role | editable? |
|---|---|---|
| `prepare.py` | contract, data, deterministic split, eval, ledger | **never** |
| `train.py` | hyperparameters and the training body | the research loop edits this |
| `program.md` | the research protocol: loop rules, gate, budget | with approval |
| `results.tsv` | the experiment ledger, written by `train.py` | never by hand |

The eval is pinned because it is the measuring instrument. If a run could edit it, a score
could go up without the model getting better.

## Quick start

```bash
uv sync                                    # harness only, no Tinker needed
uv run pytest                              # tests: contract, scoring, split, ratchet
uv run python prepare.py --audit           # data report
uv run python prepare.py --split           # the 8 held-out rows
uv run python prepare.py --self-check      # score the references; must print 1.0
uv run python train.py --dry-run           # whole loop, stub model, no spend
```

Real training:

```bash
uv sync --extra train
tinker auth login          # or export TINKER_API_KEY
uv run python train.py --run-id exp1 --note "rank 16, 40 steps, lr 2e-4"
```

## The contract

One clip per answer. JSON only.

```json
{
  "file": "DJI_20260719103816_0051_D.MP4",
  "start": "0:03",
  "end": "0:15",
  "title": "The room erupts — open on the laugh",
  "reason": "The warmest twelve seconds of the morning..."
}
```

- `file` must be one of the source files named in the prompt's keyframes.
- `start` / `end` are `M:SS`, `end` after `start`, clip length 3–20 s.
- `title` under 60 characters, `reason` one sentence.

v1 asked for "3 clips of 30–90 s" while every training row was one clip of 3–12 s. That
contradiction is why v1's answers came back as `{"label": ...}` with a timestamp in
`file`. The contract now matches the data.

## Scoring

Each held-out answer scores:

| outcome | score |
|---|---:|
| not valid JSON, or breaks the contract | 0 |
| valid, wrong source file | 0.25 |
| valid, right file, temporal IoU *i* | 0.25 + 0.75 × *i* |

A run is **kept** only if `schema_valid_rate ≥ 0.9` **and** `mean_score` beats the best
kept run in `results.tsv`. Below the gate the model cannot be shipped, so a high score on
the rows it did get right is noise.

## Current state

- **Winner: v3-grpo-20**, 39/52 on eval v2: mc 11/11, style 14/16,
  stability 6/6 (pairs 3/3), multi 4/4, traps 4/4, open-window sft exact 0/11.
  See [results and checkpoint](RESULTS-2026-11-18.md) and [template](STYLE_TEMPLATE.md).
- **Real baselines are scored**: base 15/52, v1 26/52, v2 32/52.
  These template scores are separate from the fixed one-clip harness ledger.
- **Remaining gap:** exact open-generation windows (file match 1.00, mean IoU 0.36).
  Training and sampling use remote Tinker GPUs; no local GPU is needed.
- **Data** is now 31 rows, the union of the old 27 plus the 4 afternoon moments that
  existed only in the sibling repo. See [`DATA_CARD.md`](DATA_CARD.md).

## Layout

```
prepare.py              fixed: contract, data, split, eval, ledger
train.py                experiment target: hyperparameters + Tinker SFT
program.md              research protocol
results.tsv             run ledger
data/picks.jsonl        31 approved picks
SPEC.md                 current owner spec, contracts, compute and next work
STYLE_TEMPLATE.md       style-parameterized winner prompt and output shapes
RESULTS-2026-11-18.md   winner, leaderboard, historical cost/storage
eval/                   fixed v2 set, shared scoring, runners, saved results
eval/tinker_env.py      shared portable API-key/proxy setup
run_sft_v2.py           portable SFT reproduction launcher
run_sft_v3.py           portable refusal-SFT launcher
tests/                  harness, scoring, eval integrity, reproducibility, dry-run
docs/ROADMAP.md         Drive link in, styled video out
lora-training-run.log   the v1 run, kept as history
```
