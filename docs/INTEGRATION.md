# Integrating the video-cutting LoRA

How to go from your own footage to cut clips with the winner checkpoint
(`v3-grpo-20`, [results](../RESULTS-2026-11-18.md), [template](../STYLE_TEMPLATE.md)).
The LoRA **selects** moments from candidates you supply; it does not watch video.
`cutter.py` builds the exact prompt, resolves the answer back to a real candidate,
and cuts it with ffmpeg.

## 1. Install and verify (CPU only, no spend)

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

## Walkthrough videos

[`docs/walkthroughs/`](walkthroughs/) holds the scripts the walkthrough videos
were recorded from (`asciinema rec -c "bash docs/walkthroughs/01-setup.sh"`):
`01-setup.sh`, `02-cut-clips.sh` (synthetic test-pattern footage named like the
real files) and `03-real-model.sh` (Tinker setup, shown but not run).
