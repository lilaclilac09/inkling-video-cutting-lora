# Integrating the video-cutting LoRA

How to go from your own footage to cut clips with the winner checkpoint
(`v3-grpo-20`, [results](../RESULTS-2026-11-18.md), [template](../STYLE_TEMPLATE.md)).
The LoRA **selects** moments from candidates you supply; it does not watch video.
`cutter.py` builds the exact prompt, resolves the answer back to a real candidate,
and cuts it with ffmpeg.

## 1. Install and verify (CPU only, no spend)

```bash
./install.sh        # checks uv and ffmpeg, installs, verifies, shows the intro slides
```

Or step by step:

```bash
uv sync
uv run pytest
uv run python prepare.py --self-check   # must print self-check: pass
uv run python train.py --dry-run        # stub model, writes nothing
```

## 2. Describe the cut

A request is JSON: event, date, style, 0–3 beats and **real** candidate windows
(4 for a single pick, 8 for a 2–3 beat multi-pick). See
[`examples/request-single.json`](../examples/request-single.json) and
[`examples/request-multi.json`](../examples/request-multi.json).
Styles: `vlog`, `fresh`, `silicon-valley`, `glamour`, `hollywood`.

The quickest way to write one is the guided builder, which asks for each field
and saves a request the cutter accepts:

```bash
uv run python cutter.py init --out my-request.json
```

For your own look, pick "Your own style" (or set `"style": "<name>"` plus a
one-sentence `"style_description"` in the JSON). The LoRA was trained on the
five styles above, so a custom style is best effort: check its picks before
relying on it. `uv run python cutter.py welcome` replays the intro slides.

```bash
uv run python cutter.py prompt examples/request-single.json
```

The prompt is byte-identical to the eval prompts the winner was scored on
(`tests/test_cutter.py` pins this), so keep its shape and change only the request.

## 3. Sample the winner

Sampling runs on Tinker and is paid. Install the extras, export your key, and
send the prompt to the checkpoint in `RESULTS-2026-11-18.md`:

```bash
uv sync --extra train
export TINKER_API_KEY=...     # never commit it
```

Or paste the prompt into the Tinker playground link in the results file.

## 4. Resolve and cut

```bash
uv run python cutter.py cut examples/request-single.json \
  "C: DJI_20260719101336_0037_D.MP4 0:01-0:07" --footage /path/to/footage --out clips
uv run python cutter.py cut examples/request-multi.json @answer.txt --footage /path/to/footage --dry-run
```

- `None` is a refusal: nothing is cut, exit 0.
- An answer that is not exactly one of the offered candidates (letter and window
  disagree, invented window, wrong clip count) is rejected with exit 2.
- In Python: `prompt = cutter.build_prompt(request)`, `clips = cutter.resolve(answer, request)`.

## Train your own style

1. I save my approved picks as JSONL: one object with `request` and `answer`
   per line. The request uses the same event, date, candidates and optional
   beats as the examples above. My CLI style and brief replace any existing
   request style. Single answers are a letter (`"B"`) or the full
   `"B: file.MP4 0:01-0:07"`; multi answers are JSON lists of
   `{"file": "...", "start": "0:01", "end": "0:07"}` objects in beat order.
   JSON `null` or the string `"None"` records an approved refusal.
   [my-style-picks.jsonl](../examples/my-style-picks.jsonl) demonstrates all
   three; it is **synthetic, not footage evidence or data for paid training**.
   Candidate files must be .MP4 basenames without spaces; timestamps use
   whole-second M:SS (minutes may exceed 59). Every offered clip must be
   3–20 seconds. I supply 4 candidates for 0–1 beats, 8 for 2–3 beats.

2. I prepare data locally, with no Tinker import, network call or GPU:

   ```bash
   uv run python prepare_style.py examples/my-style-picks.jsonl \
     --style my-calm-recap --brief "Unhurried human moments and quiet transitions." \
     --out ~/my-calm-style --seed 42
   ```

   I replace the example with my own picks before training. The default warning
   threshold is 20 picks; `--min-picks N` changes only that warning, not validation.
   Errors name the physical input row. Duplicate requests are rejected.
   `--held-out-fraction` defaults to 0.2. The seeded split keeps shared candidate
   source files (even distractors, and transitively connected requests) together,
   so I need at least two independent file groups; actual split sizes may differ
   from the requested fraction. Output must be a new or empty directory.

3. I review `train.jsonl`, `held-out.jsonl` and `manifest.json`. Training rows
   have `kind: "prompt"`, `prompt` and `answer`: the SFT v3 runner consumes these
   verbatim, preserving the cutter prompt shape with only the style line changed.
   The legacy clip/refusal formats and default v3 run are unchanged. The manifest
   records counts, seed, validation, warnings and the existing overlap check's
   result (held-out golds versus training targets **and candidate exposure**).
   Held-out rows include `id`, `format`, `prompt`, `gold`, the request and
   approved answer for scoring; they are never included in training.

4. Only when I am ready to pay, I install `uv sync --extra train`, set my
   `TINKER_API_KEY`, and run the exact command printed in the manifest **from
   the repo root**. Preparation only prints it; it does not execute it.
   That command uses `run_sft_v3.py --data ... --from-base --name inkling-my-calm-recap`:
   a fresh rank-16 LoRA, six SFT steps at 1e-4, with uniquely named sampler/state
   checkpoints on Tinker's remote GPUs (no local GPU needed). It does not require
   access to the historical v2 state and does **not** resume the winner.
   To continue my own prior training instead, I replace `--from-base` with
   `--state-path 'tinker://.../weights/my-training-state'`, never a sampler path.
   I choose a new `--name` for each paid run.

5. I score only after training, using the new sampler path printed by the runner
   (this sampling command is also **paid**):

   ```bash
   uv run python eval/run_eval.py --eval-path ~/my-calm-style/held-out.jsonl \
     --model custom --model-path 'tinker://.../sampler_weights/inkling-my-calm-recap' \
     --out ~/my-calm-style/scores.json
   ```

   For new footage I use my style slug plus `"style_brief": "..."` in the cutter
   request. The guided builder's `style_description` is also accepted for custom
   styles; an explicit `style_brief` takes precedence. The same
   `cutter.build_prompt` and `cutter.resolve` still build
   prompts and validate selections; training does not make the model watch video.

## Walkthrough videos

[`docs/walkthroughs/`](walkthroughs/) holds the scripts the walkthrough videos
were recorded from (`asciinema rec -c "bash docs/walkthroughs/01-setup.sh"`):
`01-setup.sh`, `02-cut-clips.sh` (synthetic test-pattern footage named like the
real files), `03-real-model.sh` (Tinker setup, shown but not run) and
`04-install-and-init.py` (`./install.sh`, the intro slides and `cutter.py init`;
record it with `asciinema rec -c "python3 docs/walkthroughs/04-install-and-init.py"`).
