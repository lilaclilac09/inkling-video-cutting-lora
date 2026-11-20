# Eval set v1 — Inkling video-cutting LoRA

Fixed benchmark. Same 26 questions every run, never trained on. Built 2026-11-18.

## Composition (26 items)

| Format | n | What it measures |
|---|---|---|
| `mc` | 11 | Pick the right segment among 4 candidates (gold = Aileen-approved v5 pick). |
| `sft` | 11 | Round-1 SFT prompt shape; exact (file, start, end) match. |
| `trap` | 4 | Moment verified missing from ALL footage (guitar / piano / rain / group photo). Correct = refuse ("None"). Any pick = hallucination. |

## Held-out proof

- Gold items are the 11 Tier A clean v5 picks: exact (file, start, end) is NOT in
  the 27 training examples AND the moment does not overlap any training example.
- Excluded v5 picks: #16 is an exact training example; #3, #4, #5, #6, #9, #14,
  #17, #19 overlap training moments (same clip, overlapping time range).
- Distractors: 6s windows from the 73-clip shot list, never overlapping any v5
  pick or training example; 566 of 803 candidate windows come from files never
  in the training set. Fixed seed (42) — the set is stable across runs.

## Scoring (per run, temp 0, 1 sample)

- `mc` accuracy: letter matches gold letter. File-match also recorded.
- `sft` accuracy: parsed JSON clip with exact file AND start/end (0.5s tolerance).
  File-match and parsed-JSON rates reported separately.
- `trap` accuracy: refused. Hallucinated-clip rate reported separately.
- Overall = correct / 26. Report per-format numbers, never only the overall.

## Files

- `eval-set-v1.jsonl` — the 26 questions (prompt, gold, source, held-out evidence).
- `build_eval_set.py` — my restored shared helpers, inferred from the callers,
  not the historical v1 generator. I restored v1 as v2's byte-identical 26-line
  prefix. I still need the original v5 picks and shot list to rebuild v2;
  see [my provenance and overlap audit](../SPEC.md#eval-sets-and-scoring).
- `run_baseline.py` — sampling-only baseline runner (base + v1 checkpoint).
- `baseline-base.json` / `baseline-v1.json` — per-item results.
- `run-log.md` — chronological run log (baselines, sweeps, costs, decisions).

## Rules (from the methodology note)

- Eval questions never enter training. If a question leaks into training, rebuild
  the set and say so in the log.
- Select checkpoints by eval score, never by step count.
- If pick accuracy does not rise for two consecutive sweeps, stop and report.

## Eval set v2 (52 items) — style is a parameter

v2 = v1 (26 items, copied byte-identical) + 26 new items. Built 2026-11-18.

| Format | n | What it measures |
|---|---|---|
| `mc` | 11 | (v1) Pick the right segment among 4 candidates. |
| `sft` | 11 | (v1) Round-1 SFT prompt shape; exact (file, start, end) match. |
| `trap` | 4 | (v1) Missing moment — correct = refuse ("None"). |
| `style-mc` | 16 | 4 candidate sets x 4 styles. Same candidates, same slot, only the Style line changes; the gold differs by style. Tests whether the model conditions on the style parameter at all. |
| `stability` | 6 | 3 Tier A golds x 2 distractor sets. The pair must agree — a template that flips its pick when distractors change is not stable. |
| `multi` | 4 | 3 beats in one call, 8 candidates. One call, many files: the "faster" axis. |

Styles: `vlog` (reference), `fresh`, `silicon-valley`, `glamour`, `hollywood`.
Style golds are Tier A held-out picks (sets 1-2) or shot-list windows asserted
non-overlapping with every pick and training example (sets 3-4); each gold's
evidence is quoted in the item's `style_evidence` field. Builder:
`build_eval_set_v2.py` (seed 42). Runner: `run_eval.py` (generic, any eval
file, any model). Scorer: `scoring.py` (shared) + `rescore_v2.py`.
Results: `v2-base.json`, `v2-v1.json`, `v2-v2.json` (re-scored copies alongside).

Template: `../STYLE_TEMPLATE.md` — the prompt shape with `style` as a parameter.
