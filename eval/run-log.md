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
