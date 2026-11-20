# My video-cutting spec

## Goal

I want a refined, unified selection policy for my videos: a reusable template,
not a one-off Shanghai recap. I change the style, supply real candidates, and
get stable picks across many files in one call. The LoRA selects moments; it
does not encode video. This is the source of truth for scope and acceptance;
[results](RESULTS-2026-11-18.md) preserve measurements and
[the roadmap](docs/ROADMAP.md) expands the pipeline.

## Two output contracts

**Harness: `prepare.py` → `train.py`.** Transcript plus keyframes in, exactly
one JSON object out: `file, start, end, title, reason`. The file must be offered
in the keyframes; timestamps are M:SS, end after start, duration 3–20 seconds.
Title is under 60 characters; reason is one sentence. No prose or fences.
The fixed instrument owns the 31-row dataset, every-fourth-row held-out split
(8 rows), contract validation, scoring and ledger. I do not edit it during runs.

**Winner: [STYLE_TEMPLATE v2](STYLE_TEMPLATE.md).** Style, beats and real
candidate windows in; single pick is `B: <file> <start>-<end>`, multi-pick is
a JSON list of three `{file,start,end}` objects in beat order, refusal is
`None`. Never invent a moment. Four candidates for single picks, eight for
multi-picks. Change event, style, beats and candidates, not the prompt structure.

These are separate interfaces and separate scoreboards. The winner's letter
answer is not valid harness JSON. A future cutter adapter must resolve the
selected candidate and validate it before assembly; benchmark success alone
does not establish that adapter or harness compliance.

## Style parameter and catalog

| style | My selection intent |
|---|---|
| `vlog` | Warm, human, flowing story; standalone moments |
| `fresh` | Bright, airy, outdoors, calm breathing room |
| `silicon-valley` | Demos, code, substance, builders explaining work |
| `glamour` | Polished close-ups, warm light, poster-worthy frames |
| `hollywood` | Drama, scale, reactions, peak emotion |

Style changes selection, not just color grading. Future cut-length, order,
audio, caption and aspect constraints are in the roadmap, not implemented here.

## Eval sets and scoring

v1 is documented as 26 items: 11 mc, 11 sft, 4 traps. v2 contains those
formats plus 16 style-mc, 6 stability and 4 multi: **52 items**, unique IDs.
I restored `eval/eval-set-v1.jsonl` from the first 26 lines of the frozen v2
file, byte-for-byte, and test both the prefix and 11/11/4 composition.
I restored `eval/build_eval_set.py` as shared helpers inferred from callers:
integer M:SS/H:MM:SS parsing matches prepare.py, formatting rounds to seconds,
clip keys are (file,start,end), and overlap requires positive shared duration.
This is not a recovered historical v1 generator.

I searched the repo, tracked history, logs and docs: no original 20-row v5
picks file or 1411-row shot list is present. Saved prompts contain selected
pick details and shot descriptions, not the complete ordered source data.
The data card points to a sibling video-editing repo for other picks; that is
not evidence those are the v5 inputs. I do not fabricate missing sources.
Builder defaults are repo-root `sources/v5-picks.jsonl`,
`inkling-video-picks-27.jsonl` and `sources/shot-list.jsonl`; I can override
them with `INKLING_V5_PICKS_PATH`, `INKLING_TRAIN_PATH` and
`INKLING_SHOT_PATH`. Missing-file errors name the input. Picks need
`clip.file/start/end/title/reason`; shot rows need
`clip, timestamp_sec, description`. Full rebuild reproducibility still needs
the original ordered external sources; only the frozen v1 prefix is recovered.

For mc/style-mc/stability, either a correct letter or a parsed exact clip wins.
Exact means same file and each boundary within **0.5 seconds inclusive**.
Stability pairs compare resolved (file,start,end), ignoring title/reason and
shuffled letters. Traps require None or E and no JSON clip. SFT requires parsed
JSON and exact boundaries; file match, IoU and near-miss (same file, IoU > 0.5)
are diagnostics, not exact credit. Multi requires exactly three clips, all exact,
in order; partial slots are diagnostic only. Overall is correct/items; always
report each format. Sampling protocol: temperature 0, one sample per item.

Gold windows are not exact training rows in either the original 27 or current
31 rows. No gold overlaps the original 27, but the stronger no-overlap claim
does **not** extend to the expanded 31: mc-03, sft-03, multi-02 and
style-3-fresh overlap current training moments. This is not a file-held-out test.
Never train on eval questions. Keep full raw outputs for reproducibility:
the historical base raws lost late multi lists to truncation, so saved rescoring
gives multi 0/4 while live scoring recorded 2/4. Other saved per-format
accuracies are reproducible from their raws.

### My temporal-overlap audit and clean scores

I run `uv run python eval/check_overlap.py <training.jsonl> ...` on CPU.
I count positive-duration intersection on the same file, including any gold
slot in multi and any style gold. Training exposure includes clip targets,
GRPO golds **and candidates shown in training prompts**, including refusal
distractors. Touching endpoints do not count.

| Training artifact | Rows | Eval IDs with exposed gold windows |
|---|---:|---|
| Original 27 | 27 | none |
| Current data/picks | 31 | mc-03, sft-03, multi-02, style-3-fresh |
| SFT v3 data | 35 | style-3-silicon-valley |
| GRPO sweep 1 prompts | 27 | style-3-silicon-valley |
| GRPO sweep 2 prompts | 39 | none |

The additional exposure is DJI_20260719103001_0048_D.MP4: SFT v3 row 28
(zero-based, fireworks refusal) shows 0:17–0:23; GRPO sweep 1 rows
grpo-mc-09 and grpo-style-silicon-valley-1 show 0:17–0:23 / 0:18–0:24.
These overlap the style gold, even though they are not training targets.
My earlier blanket no-overlap wording did not account for that exposure.

For a common conservative comparison I exclude the **union of five IDs**
above from every run, leaving **47 items**. This does not assert all five
were training targets for each checkpoint. The table re-scores saved raws
with today's scorer; paired -rescored artifacts give identical clean scores.
Cells are correct counts over the denominator in the header.

| Saved run | Overall /47 | mc /10 | style /14 | stability /6 | multi /3 | sft /10 | trap /4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 12 | 2 | 7 | 3 | 0 | 0 | 0 |
| v1 | 24 | 8 | 8 | 6 | 0 | 2 | 0 |
| v2 | 28 | 10 | 8 | 6 | 2 | 2 | 0 |
| v2-grpo-20 | 29 | 10 | 10 | 6 | 2 | 1 | 0 |
| v2-grpo-40 | 30 | 10 | 11 | 6 | 2 | 1 | 0 |
| v2-grpo-60 | 30 | 10 | 10 | 6 | 2 | 2 | 0 |
| v3 | 22 | 9 | 1 | 5 | 3 | 0 | 4 |
| **v3-grpo-20** | **36** | **10** | **13** | **6** | **3** | **0** | **4** |
| v3-grpo-r2-20 | 34 | 10 | 11 | 6 | 3 | 0 | 4 |
| v3-grpo-r2-40 | 33 | 10 | 10 | 6 | 3 | 0 | 4 |
| v3-grpo-r2-60 | 34 | 10 | 11 | 6 | 3 | 0 | 4 |

My winner stays **v3-grpo-20: 36/47 (0.766)**; next best is 34/47.
Base remains affected by unrecoverable raw truncation, not a new model result.
This is only a temporal-exposure-clean score: GRPO trained on guitar, piano,
rain and group-photo refusal moments also used in eval. It is **not** proof of
unseen-question or file-held-out generalization. I need fresh refusal moments
and a file-held-out split for that claim.

To reproduce without changing saved artifacts:
`uv run python eval/rescore_v2.py --no-write eval/v2-*.json`.
It prints full and clean per-format/overall scores; by default I exclude the
five-file union above. Repeated `--clean-training <path>` flags select a
different union explicitly.

## Keep / ship gate

Harness KEEP requires schema-valid rate ≥ 0.9 **and** mean score strictly above
the best kept ledger run. Valid wrong-file answers score 0.25; right-file answers
score 0.25 + 0.75 × temporal IoU; invalid answers score zero.
Dry runs never write the ledger. Follow [program.md](program.md) for the loop.

For the template track I select checkpoints by fixed eval performance, not step
count; stop if pick accuracy fails to improve for two sweeps. Respect the recorded
$200 experiment cap. The template winner is not a blanket unattended-ship
approval: exact open windows remain unsolved, and a renderer must validate
offered files, boundaries and refusals before producing a cut.

## Current winner

`inkling-video-cutting-v3-grpo-20`, sweep 2 attempt 1, step 20:

```text
tinker://422c642a-f8cf-57e0-b94e-575e21fdb8a5:train:0/sampler_weights/inkling-video-cutting-v3-grpo-20
```

From [RESULTS-2026-11-18.md](RESULTS-2026-11-18.md): **39/52 (0.75)**;
mc 11/11, style 14/16, stability 6/6 with 3/3 agreeing pairs, multi 4/4,
trap 4/4, sft exact 0/11. SFT file match is 1.00 but mean IoU only 0.36.
Style: fresh 4/4, glamour 3/4, hollywood 4/4, silicon-valley 3/4.
Lineage: base → SFT v2 (27 picks, rank 16, 10 steps) → v3 refusal SFT
(6 steps) → GRPO (temperature 1.0, hard prompts, lr 1e-5).

## Compute plan

I need **no local GPU when using Tinker**. My CPU orchestrates requests;
training and model sampling happen on Tinker's GPUs. Run from the repo root.
For these shared scripts I explicitly export `TINKER_API_KEY`; they fail clearly
without it and never print it. `train.py` also supports SDK login
(`uv run tinker auth login`). Checkpoint access depends on my Tinker account.

| Work | Where | Command / prerequisites | Size or cost stated here |
|---|---|---|---|
| Tests | Local CPU | `uv sync && uv run pytest` | No model spend |
| Audit / contract self-check | Local CPU | `uv run python prepare.py --audit`; `uv run python prepare.py --self-check` | 31 picks, 8 held out |
| Full stub loop | Local CPU | `uv run python train.py --dry-run` | No key, no spend, no ledger write |
| Rescore saved raws | Local CPU | `uv run python eval/rescore_v2.py --no-write eval/v2-v3grpo-20.json` | Full 52 + common clean 47; no artifact writes |
| Build eval set | Local CPU | `uv run python eval/build_eval_set_v2.py` | Helpers/v1 restored; blocked on external v5 picks and shot list; seed 42 |
| SFT | Remote Tinker GPU | `uv sync --extra train`, export key; `uv run python run_sft_v2.py` or `run_sft_v3.py` | Recorded SFT spend negligible; not a price quote |
| GRPO | Remote Tinker GPU | `uv run python eval/run_grpo.py --state-path <my-resumable-state> --steps 60` | Prior 60-step sweeps ~900k tokens; $200 cap, no exact bill |
| Eval / sampling | Remote Tinker GPU | `uv run python eval/run_eval.py --model custom --model-path <checkpoint> --out <new-results.json>` | 52 items, max 512 output tokens each |
| Checkpoint inventory | Local CPU + remote API | `uv run python eval/list_checkpoints.py` | Recorded inventory 66.91 GB / 100 GB; final training state ~25 GB; recheck live |
| ffmpeg assembly | Local CPU / render host | `ffmpeg -i <source> -ss <start> -t <duration> <output>` | Full assembler is in sibling video-editing repo, not this one; no cost estimate |
| Whisper transcription | Local/rented GPU recommended | Roadmap stage 2; choose/install model and transcription tool | No implemented command, VRAM or cost budget in repo; CPU possible but slower |
| Shot detection, keyframes, visual tags | CPU extraction; local/rented GPU for learned visual analysis | Roadmap stages 2–3; ffmpeg can extract frames on CPU | Pipeline/model choices unimplemented; no GPU size estimate |
| LoRA inference without Tinker | Local/rented GPU infrastructure | Need compatible base model, weights access and serving stack | Roadmap describes 594B/17B MoE; no supported local command or VRAM budget; a LoRA alone is not a standalone model |

Token counts and storage above are historical measurements, not guaranteed
future charges. No new paid run is needed for this spec or the tests.

## Open work / next steps

| Next step | What I need |
|---|---|
| Window-IoU GRPO reward | Add open-window training prompts and graded reward (exact 1, IoU > 0.5 partial); CPU reward tests then a budgeted remote sweep. Current candidate-letter reward does not teach boundary generation. |
| Exact-window SFT | Existing 27 exact ranges, more targeted supervised steps on remote GPUs; check refusal/style regressions and all 11 open-window evals. |
| File-level held-out split | More footage and labels; design a new versioned instrument with approval, not edits to the pinned prepare.py. Current row split can share files. |
| P2 ingest + index | Drive access, footage inventory, content hashing, transcripts/shots/tags; CPU ingest and GPU-assisted understanding as above. |
| P3 brief + labels | Machine-readable cut grammar, a second style's ~20 approved picks and rejects; owner review, CPU tooling. |
| P4 style-conditioned LoRA | P2 + P3, remote GPU budget, per-style and file-held-out evaluation. Current template evidence is a start, not the whole pipeline. |
| P5 assembly + review | Render host, sibling ffmpeg assembler, contact sheet, approve/re-roll UI, output/audio/music decisions; CPU rendering initially. |

Before expanding training, recover the external builder inputs and keep future
train/eval sources disjoint. P1's real-model evaluation is done; the old roadmap
language calling it the next step is historical.
