# Inkling Video Cutting LoRA

Cut your event footage into shareable clips, in the style you pick.

<p>
  <a href="RESULTS-2026-11-18.md"><img alt="Winner: v3-grpo-20" src="https://img.shields.io/badge/WINNER-v3--grpo--20-0a0a0a.svg?style=for-the-badge&labelColor=000000" height="28"></a>
  <a href="RESULTS-2026-11-18.md#leaderboard-eval-set-v2-52-items-temp-0-1-sample"><img alt="Eval v2: 39/52, clean 36/47" src="https://img.shields.io/badge/EVAL%20V2-39%2F52%20%C2%B7%20CLEAN%2036%2F47-0a0a0a.svg?style=for-the-badge&labelColor=000000" height="28"></a>
  <a href="https://tinker.thinkingmachines.ai/playground?mode=checkpoint&checkpoint=tinker%3A%2F%2F422c642a-f8cf-57e0-b94e-575e21fdb8a5%3Atrain%3A0%2Fsampler_weights%2Finkling-video-cutting-v3-grpo-20&reasoning_effort=none"><img alt="Try it in the Tinker Playground" src="https://img.shields.io/badge/TRY%20IT-Tinker%20Playground-0a0a0a.svg?style=for-the-badge&labelColor=000000" height="28"></a>
  <a href="docs/INTEGRATION.md"><img alt="Integration guide" src="https://img.shields.io/badge/DOCS-Integration%20Guide-0a0a0a.svg?style=for-the-badge&labelColor=000000" height="28"></a>
</p>

A LoRA fine-tune of [Thinking Machines](https://thinkingmachines.ai) **Inkling 2 Small**.
You give it the moments you have (file, start, end, what is on screen) and the style
you want; it picks the right ones, and `cutter.py` cuts them with ffmpeg. It selects from
your candidates and never invents a window. Built on the Cafe Cursor Shanghai footage;
the long-term shape (a Drive link in, a styled video out) is in [`docs/ROADMAP.md`](docs/ROADMAP.md).

## Highlights

- Five trained styles as one prompt parameter: vlog, fresh, silicon-valley, glamour, hollywood. Or describe your own.
- One clip, or up to 3 beats in order from 8 candidates in one call.
- Refuses instead of inventing: a moment that is not in your footage gets `None`, and nothing is cut.
- One-step install with a short intro, and a guided builder that writes your first request.
- CPU-only harness: tests, scoring and a dry run need no GPU and no Tinker account.

## Models

Start with **v3-grpo-20**; it is the only checkpoint to use for new work. The others are kept for comparison.

| Checkpoint | Lineage | Eval v2: overall | Style | Multi-pick | Refusal | Clean subset | On Tinker |
|---|---|---:|---:|---:|---:|---:|---|
| **v3-grpo-20** | v3 + GRPO, 20 steps | **39/52** | **0.88** | **1.00** | **1.00** | **36/47** | kept ([path](#use-the-winner)) |
| v3-grpo-r2-60 | v3 + GRPO rerun, 60 steps | 38/52 | 0.81 | 1.00 | 1.00 | 34/47 | deleted |
| v2-grpo-40 | v2 + GRPO, 40 steps | 34/52 | 0.81 | 0.75 | 0.00 | 30/47 | kept |
| v2 | SFT, 27 picks, rank 16 | 32/52 | 0.62 | 0.75 | 0.00 | 28/47 | kept |
| v1 | SFT round 1 | 26/52 | 0.56 | 0.00 | 0.00 | 24/47 | kept |
| v3 | v2 + refusal SFT | 23/52 | 0.06 | 1.00 | 1.00 | 22/47 | kept |
| base | Inkling 2 Small | 15/52 | 0.50 | 0.50 | 0.00 | 12/47 | – |

"Eval v2" is the pinned 52-item set (`eval/eval-set-v2.jsonl`), temperature 0, one sample.
"Clean subset" drops the 5 items whose moments appeared in any training artifact.
Every row, with the GRPO runs not shown here, is in [the results](RESULTS-2026-11-18.md).
The remaining gap is exact windows in open generation (right file, mean IoU 0.36).

## Quick Start

### Install

You'll need [uv](https://docs.astral.sh/uv/) and ffmpeg.

```bash
git clone https://github.com/lilaclilac09/inkling-video-cutting-lora.git && cd inkling-video-cutting-lora
./install.sh
```

It checks the tools, installs, runs the tests and ends with four intro slides
(`uv run python cutter.py welcome` shows them again).

### Make Your First Cut

```bash
uv run python cutter.py init --out my-request.json          # style, beats, your candidate windows
uv run python cutter.py prompt my-request.json               # the prompt to send to the winner
uv run python cutter.py cut my-request.json "<answer>" --footage footage/ --out clips
```

### Use the Winner

Sampling runs on Tinker and is paid (`uv sync --extra train`, then `TINKER_API_KEY`). Sampler path:

```text
tinker://422c642a-f8cf-57e0-b94e-575e21fdb8a5:train:0/sampler_weights/inkling-video-cutting-v3-grpo-20
```

Or paste a prompt into the [Tinker Playground](https://tinker.thinkingmachines.ai/playground?mode=checkpoint&checkpoint=tinker%3A%2F%2F422c642a-f8cf-57e0-b94e-575e21fdb8a5%3Atrain%3A0%2Fsampler_weights%2Finkling-video-cutting-v3-grpo-20&reasoning_effort=none).
The [integration guide](docs/INTEGRATION.md) has every step, and `docs/walkthroughs/` has the scripts behind the walkthrough videos.

## Commands

| Command | What it does |
|---|---|
| `./install.sh` | Check uv and ffmpeg, install, verify, show the intro |
| `cutter.py welcome` | Banner and four intro slides |
| `cutter.py init [--out FILE]` | Guided builder for a request JSON, trained or custom style |
| `cutter.py prompt REQUEST` | Print the exact STYLE_TEMPLATE prompt |
| `cutter.py cut REQUEST ANSWER [--footage DIR] [--out DIR] [--dry-run]` | Check the model's answer against your candidates and cut with ffmpeg |
| `prepare.py --self-check` | Score the references; must print `self-check: pass` |
| `train.py --dry-run` | The whole training loop with a stub model, no spend |

## Research

Start with [`SPEC.md`](SPEC.md): my current contracts, winner, acceptance gates,
CPU/GPU compute plan and next steps. The one-clip harness below is separate from
the winner's letter / multi-pick / None template contract.

## The Three Files

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

## Harness Commands

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

## The Contract

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

## Current State

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

## Authors

- Aileen ([@lilaclilac09](https://github.com/lilaclilac09))
