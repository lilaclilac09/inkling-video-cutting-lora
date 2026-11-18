# program.md — the research protocol

You are the research agent on this repo. Your job is not to run experiments; it is to
write good experiment instructions and then work the loop. Read this file in full before
touching anything.

The goal is a LoRA on **Inkling 2 Small** that turns a transcript excerpt plus keyframes
into one shareable clip pick, in the shape `prepare.py` enforces. It ships when its
held-out score is good enough that the ffmpeg pipeline can cut the picks without a human
fixing the JSON.

## The three files

| file | role | may you edit it? |
|---|---|---|
| `prepare.py` | fixed infrastructure: contract, data, split, eval, ledger | **never** |
| `train.py` | the experiment target: hyperparameters and the training body | yes, one change per experiment |
| `program.md` | this protocol | only with Aileen's approval |

`prepare.py` is pinned because the eval is the measuring instrument. If you could edit
it, a run could score better without being better. If a score looks wrong, report it —
do not fix it by changing the instrument.

## The contract (owned by prepare.py)

- One clip per answer. JSON only, no prose, no code fences.
- Keys: `file`, `start`, `end`, `title`, `reason`. Nothing else.
- `file` must be one of the source files named in that prompt's keyframes.
- `start` and `end` are `M:SS`; `end` after `start`; clip length 3–20 s.
- `title` under 60 characters, `reason` one sentence.

The v1 run asked for "3 clips of 30–90 s" while every training row was one clip of
3–12 s. That contradiction is why v1's answers were unusable. Do not reintroduce it.

## The data

`data/picks.jsonl` — 32 approved picks from the Cafe Cursor Shanghai footage. Provenance
and gaps are in `DATA_CARD.md`. Rules:

- Never train on a held-out row.
- Never change the split to move a score. It is `sorted(clip_key)[::4]`, in `prepare.py`.
- Never add a moment "from memory". A row without verified footage is not a row.
- Rows marked `synthetic` are teaching examples derived from a real clip. They may be
  used, but report how many of each a run used.

## The loop

1. `git status` must be clean. If it is not, stop and report.
2. Read `results.tsv` and the last run's `runs/<run_id>/predictions.jsonl`.
3. Form one hypothesis. Write it down before you run anything, in one sentence.
4. Make **one** change in `train.py`. Commit it: `exp<N>: <what changed, old -> new>`.
5. Run it: `python train.py --run-id exp<N> --note "<hypothesis>"`.
6. Read the score. Compare `mean_score` against the best in `results.tsv`.
7. `keep` = yes → leave the commit. `keep` = no → `git revert` or `git reset --hard` back
   to the last kept commit. Do not leave a discarded experiment in the branch.
8. Append nothing to `results.tsv` by hand. `train.py` writes it.

One change per experiment. Two changes and a better score tells you nothing about which
one did it.

## What counts as an improvement

A run only counts if `schema_valid_rate >= 0.9`. Below that the model cannot be trusted to
produce parseable output, and a high `mean_score` on the remaining rows is noise.

Then `mean_score` must beat the best in `results.tsv`. `mean_score` is
`0` for an invalid answer, otherwise `0.25 + 0.75 × temporal IoU` against Aileen's pick.
So a valid answer with the wrong file is 0.25, and a perfect answer is 1.0.

Report both numbers. A run that raises `mean_score` while dropping `schema_valid_rate`
under the gate is a regression, not a win.

## Things worth trying, in rough order of expected value

1. **Steps and epochs.** 40 steps over 28 rows is still very few. Try 100, 200.
2. **Rank.** 8, 16, 32. Higher rank memorizes faster on small data.
3. **Learning rate.** 1e-4, 2e-4, 4e-4. Watch for loss collapsing then diverging.
4. **Batching.** `BATCH_SIZE` 1, 4, 8. Full-batch on 28 rows is one gradient step per
   "step", which is why v1's loss fell so fast.
5. **Reasoning effort.** `REASONING` low vs medium. This is the single biggest lever on
   output length and latency.
6. **Prompt template.** This is in `prepare.py`, so it is not yours to edit — propose the
   change to Aileen and explain the expected effect.
7. **Answer format pressure.** Nothing in the training body changes what the model must
   emit; the contract does. Do not add a regex "fixer" that repairs bad JSON before
   scoring. That would be gaming the instrument.

## Budget

Tinker charges per million training tokens. Inkling 2 Small is $1.50/M train. A 32-row run
of ~2000 tokens is a fraction of a cent; 200 steps is still small. The real cost is your
time per experiment. Cap a sweep at **10 experiments**, then report.

## Report format

After each sweep, write to `runs/SWEEP-<n>.md`:

- hypothesis for each experiment, in one line
- the ledger rows it produced
- what improved, what did not, and the best config so far
- what you would try next, and what you need from Aileen to try it

## Known failure modes

- **Tokenizer.** `training.get_tokenizer()` is the proven path on this model (it worked in
  the v1 run). It logs a `inkling2` type warning; that is expected. If prompt and answer
  boundaries drift, check `prompt_len` before blaming the data.
- **`file` leakage.** The model can copy a filename from the keyframe paths. That is
  intended: the keyframes are the only place the filename appears, and the contract
  requires the answer to name one of them. If `file_match_rate` hits 1.0 while `mean_iou`
  stays low, the model has learned to copy the filename but not to find the moment.
- **Prose.** v1's base model answered with a paragraph. If a run regresses to paragraphs,
  `schema_valid_rate` drops and the gate rejects it. That is working as intended.
