# Run log — Inkling video-cutting LoRA (eval + tuning)

Chronological. Baselines, sweeps, costs, decisions. Times in Asia/Shanghai.

## 2026-11-18 (evening) — eval v2 + SFT v2 reproduction + GRPO setup

- **Eval set v2 built** (`eval-set-v2.jsonl`, 52 items = v1's 26 copied
  byte-identical + 16 style-mc + 6 stability + 4 multi). Builder
  `build_eval_set_v2.py` (seed 42). Style = a parameter: 4 candidate sets x
  4 styles (vlog/fresh/silicon-valley/glamour/hollywood); same candidates,
  only the Style line changes, gold differs by style. Stability = 3 Tier A
  golds x 2 distractor sets (pair must agree). Multi = 3 beats in one call,
  8 candidates. Template: `../STYLE_TEMPLATE.md`.
- **Fix (same night)**: stability reps and multi items first shared the
  top-of-pool distractors (v1 builder restarts the pool per item — known
  limitation, v1 items frozen for comparability). Rebuilt with a rotating
  draw; reps now share only the gold. Verified: v1/style-mc/trap items
  byte-identical after rebuild.
- **SFT v2 reproduction** (`sft-v2-run.log`, exit 0, ~2.7 min): same recipe
  as v1 (rank 16, 10 steps, lr 2e-4, 27 examples). Loss 2.2030 -> 0.1669
  (v1 was 2.2364 -> 0.1749 — reproduction confirmed). Saved BOTH:
  sampler `inkling-video-cutting-v2`
  (`tinker://0059270a-eeab-5add-bf67-b9f39ee0a5cb:train:0/sampler_weights/inkling-video-cutting-v2`)
  and resumable state `inkling-video-cutting-v2-state`
  (`tinker://0059270a-eeab-5add-bf67-b9f39ee0a5cb:train:0/weights/inkling-video-cutting-v2-state`).
  v1 only saved sampler weights, which `create_training_client_from_state`
  rejects — the state is what GRPO resumes from.
- **Baseline v2 running** (`v2-baseline-run.log`): base vs v1 vs v2 on the
  52-item set, temp 0, 1 sample. Runner `run_eval.py`, scorer `scoring.py`.
- **GRPO prompts built** (`grpo-prompts.jsonl`, 27 items: 11 mc + 12
  style-mc + 4 trap). Built ONLY from the 27 training examples; every
  distractor checked against all eval-v2 windows (57 unique options + 19
  golds excluded) — zero overlap with eval. Style golds are hand-picked
  training rows (a tag-only first pass mislabeled 3 golds: a laughter clip
  as "fresh", coffee cups as "silicon-valley"/"glamour", a demo talk as
  "hollywood" — all fixed, see `build_grpo_prompts.py` STYLE_ROWS).
- **Logprobs probe** (`probe_logprobs.py`): `sample()` DOES return
  per-token logprobs (`seq.logprobs`, len == len(tokens)) — GRPO uses them
  directly, no extra `compute_logprobs` calls.
- **GRPO loop** (`run_grpo.py`): resume from v2 state; per step refresh
  on-policy sampler, sample G=8 @ temp 0.7 for 4 prompts, reward 0/1
  (right clip / refused missing moment), advantages R - mean(R_group),
  `forward_backward(..., "importance_sampling")`, Adam lr 1e-5, no KL.
  Sampler checkpoints at steps 20/40/60; each scored on eval v2; best wins.
  Hard cap $200 (Suiche reference: 60 GRPO steps, $22).

- **Baseline v2 results** (`v2-baseline-run.log`, exit 0, ~12 min for 3x52
  items; `v2-base.json`, `v2-v1.json`, `v2-v2.json` + re-scored copies):
  base 15/52 = 0.29, v1 26/52 = 0.50, **v2 32/52 = 0.61**. Per format
  (base / v1 / v2): mc 0.18 / 0.82 / 1.00; style-mc 0.50 / 0.56 / 0.62;
  stability 0.50 / 1.00 / 1.00 (pairs agree 3/3 all); multi 0.50 / 0.00 /
  0.75; sft exact 0.00 / 0.18 / 0.18; trap 0.00 / 0.00 / 0.00.
  v2 beats v1 on mc (11/11, 0 empty vs 2) and on multi (3/4 exact — it
  follows the "JSON list of 3" instruction; v1 falls back to one clip).
  Same recipe, different init: run-to-run variance is real. Trap refusal
  is 0.00 everywhere — the main GRPO target. Style headroom 0.62 -> 1.00.
- **Scorer fix (same night)**: stability agreement compared the whole
  clip dict including reason text, so v1 scored agree=0 with acc=1.0.
  Now compares (file, start, end) only. Second fix: saved raw was
  truncated to 1500 chars, which lost base's late multi answers on
  re-score (live 0.50 vs re-scored 0.00); raw is now stored whole.
  Live numbers above are authoritative; the saved base re-score
  undercounts multi by 2.
- **Resume probe PASSED** (`probe_resume_v2.py`): training client resumed
  from the v2 state, on-policy sampler created, sample works with
  logprobs. GRPO can start. (v1's sampler path cannot resume — verified
  earlier; v2's state path is the fix.)
- **GRPO sweep 1 running** (`grpo-run.log`): 60 steps, G=8, temp 0.7, 4
  prompts/step, lr 1e-5, from the v2 state. Step rewards started 0.50 /
  0.66 / 0.31 and drifted up to 0.75-1.00 — but the training prompts
  saturated: by step 19 whole batches score 1.000 and most groups go
  degenerate (unanimous, 0 datums). Random-window distractors are too easy;
  the model already masters the training moments.
- **Logprobs verified** (`probe_logprobs_tempered.py`): sample() logprobs
  match compute_logprobs() exactly (max|diff| = 0.0000) — untempered, so
  the GRPO datums are correct. (Step-0 e_max_violation ~46 has another
  cause; it does not block learning.)
- **GRPO step-20 eval: 33/52 = 0.64** (`v2-grpo-20.json`), vs v2's 32/52.
  Style-mc 0.625 -> **0.75** (silicon-valley 4/4) — GRPO teaches style.
  But trap 0.00 got worse (hallucinated 0.75 -> 1.00): the 23 pick-prompts
  constantly reward picking, while trap groups are degenerate (the model
  never refuses, so no trap gradient) — refusal gets trained away. sft
  exact 0.18 -> 0.09 (window precision not protected by a clip-level
  reward). mc/stability/multi unchanged (1.00 / 1.00 / 0.75).
- **Sweep 2 planned** (`grpo-prompts-v2.jsonl`, 39 prompts): hard
  distractors (other training clips, one per other style for style items)
  + 16 style-traps (4 moments x 4 styles). Temp 1.0 for rollout diversity.
  `run_grpo.py` now takes --prompts / --temperature / --name-prefix.
- **GRPO sweep 1 DONE** (60 steps, ~32 min, exit 0): 792,773 sampled +
  107,056 trained tokens (~900k total — Suiche-scale, far under the $200
  cap). Rewards drifted 0.5 -> 0.75-1.00; last steps fully degenerate
  (saturated prompts). Checkpoints: step 20 = 33/52 (style 0.75), step 40
  = **34/52 (style 0.812)** — best of sweep 1, step 60 = 34/52 (style
  0.75, sft exact back to 0.18). Trap stayed 0.00 throughout (hallucinated
  1.00): GRPO cannot invent refusal.
- **STORAGE LIMIT HIT (2026-11-18 ~01:00)**: SFT v3's 6 steps ran fine
  (loss 0.1267 -> 0.0337) and its STATE saved, but the sampler save failed:
  "Free checkpoint storage limit reached". Inventory
  (`list_checkpoints.py`): 9 checkpoints, **116.98 GB** over the 100 GB
  free tier — training states are 25.04 GB each, samplers 8.35 GB each.
  Deleted the strictly dominated ones (`delete_checkpoints.py`): grpo-20
  sampler (33/52, dominated by grpo-40), grpo-60 sampler (34/52, tied
  overall but worse style 0.75 vs 0.812), v2-grpo-state (finished sweep,
  never resumed). Kept: v1, v2 + v2-state, grpo-40 (best), v3-state,
  first-sft. Now 6 checkpoints, 75.26 GB. The eval result files preserve
  the deleted checkpoints' measurements. v1 (published round-1) was NOT
  touched.
- **v3 sampler recovered** (`make_sampler_from_state.py`): resumed a
  training client from the v3 state and saved sampler
  `inkling-video-cutting-v3`
  (`tinker://055871af-7471-5462-bb94-432fa1cecbca:train:0/sampler_weights/inkling-video-cutting-v3`).
  Evaluating on eval v2 now — the test is whether refusal teaching worked
  (trap 0.00 -> ?) without breaking picks.
- **SFT v3 eval: 23/52 = 0.44** (`v2-v3.json`) — a sharp trade-off. Trap
  **4/4 (hallucinated 0.0)**: refusal teaching worked. Multi **4/4
  (3.0/3 slots)**: perfect. But style-mc collapsed to **1/16 (0.062)** and
  sft exact to 0.00. Failure mode (read the raws): v3 answers "None" to
  EVERY style item — refusal over-generalised; when the style line makes
  the choice less obvious, it refuses instead of picking. mc still 0.818
  (by letter now, not clip JSON — the refusal rows shifted the answer
  format). v3 is not a good base overall, but it holds the two perfect
  axes (trap, multi) that grpo-40 lacks, and the style knowledge is
  likely intact underneath (6 gentle steps cannot erase it) — it needs
  re-eliciting, which is exactly GRPO's job.
- **v2-state deleted** (25.04 GB): v3-state supersedes it as the training
  lineage (v3 = v2 + 6 refusal steps). Storage now 58.56 GB, 6 checkpoints.
  The v2 SAMPLER is kept (inference + baseline); only the training state
  went. If sweep 2 fails, the fallback is v3-state / grpo-40 / v2 samplers.
- **GRPO sweep 2 RUNNING** (`grpo2-run.log`): from the v3 state, temp 1.0,
  39 hard prompts, 60 steps, lr 1e-5, --no-final-state (storage). Early
  steps show signal again (16 datums/step, rewards 0.06-0.50 — the hard
  prompts + temp 1.0 restored contrast after sweep 1 saturated). The
  design: style-mc rewards picking right (cures over-refusal), style-traps
  reward refusing (preserves refusal) — the same style lines, opposite
  correct answers. Checkpoints at 20/40/60, each evaluated; delete each
  unless it is the new best.
