# Inkling Video Cutting LoRA

A LoRA fine-tune of [Thinking Machines](https://thinkingmachines.ai) **Inkling 2 Small**
that turns a transcript excerpt plus keyframes into **one** shareable clip pick, in a
shape a cutter script can consume without a human fixing the JSON.

Built on the Cafe Cursor Shanghai footage. The long-term shape of the project — a Drive
link in, a styled video out — is in [`docs/ROADMAP.md`](docs/ROADMAP.md).

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

- **v1** (`lora-training-run.log`, checkpoint `inkling-video-cutting-v1`) trained 10 steps
  on 27 rows and dropped train loss 2.2364 → 0.1749. It has **no comparable score**: it
  predates this harness, and its own log shows 0 of 2 answers matching the schema.
- **v2 harness** is in place and verified with a stub model. No real Inkling run has been
  scored yet — that is the next step, and it needs a Tinker key on the machine doing it.
- **Data** is now 31 rows, the union of the old 27 plus the 4 afternoon moments that
  existed only in the sibling repo. See [`DATA_CARD.md`](DATA_CARD.md).

## Layout

```
prepare.py              fixed: contract, data, split, eval, ledger
train.py                experiment target: hyperparameters + Tinker SFT
program.md              research protocol
results.tsv             run ledger
data/picks.jsonl        31 approved picks
tests/test_prepare.py   contract, scoring, split, ratchet tests
docs/ROADMAP.md         Drive link in, styled video out
lora-training-run.log   the v1 run, kept as history
```
