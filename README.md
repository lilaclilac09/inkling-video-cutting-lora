# Inkling Video Cutting LoRA

LoRA fine-tune of [Thinking Machines](https://thinkingmachines.ai) **Inkling 2 Small**
for cutting tech-meetup videos into shareable clips. Built for the
[Cafe Cursor Shanghai](https://luma.com/nthhqbbg) recap (July 19, 2026).

The base model rambles when asked to pick clips. After this fine-tune it answers
with a clean JSON clip pick: `file`, `start`, `end`, `title`, `reason`.

## Checkpoint

- **Name:** `inkling-video-cutting-v1` (Tinker)
- **Base model:** `thinkingmachines/Inkling-2-Small`
- **Try it:** [Tinker Playground](https://tinker.thinkingmachines.ai/playground?mode=checkpoint&checkpoint=tinker%3A%2F%2Fd26e013b-64d9-5c5f-9b7d-713a2fc54286%3Atrain%3A0%2Fsampler_weights%2Finkling-video-cutting-v1&reasoning_effort=none)

## Training

- **Method:** SFT with LoRA, rank 16, 10 steps, lr 2e-4. Loss only on answer tokens.
- **Data:** `inkling-video-picks-27.jsonl` — 27 approved picks (17 real + 10 synthetic),
  each with transcript excerpt, keyframes, and Inkling's clip pick.
- **Result (2026-11-17):** loss 2.2364 → 0.1749 over 10 steps.

```bash
cd ~/tinker-first-session
INKLING_PICKS_PATH=inkling-video-picks-27.jsonl uv run inkling-video-sft.py
```

Requires `tinker auth login` (a real `tml-` API key). Runs on the machine that
holds the key — the Tinker Python SDK validates the key prefix, so a stored
surrogate credential cannot drive training.

## Before / after

Prompt: *"[14:02:11] someone sits at the piano and plays a short tune.
[14:02:40] the room goes quiet, then applauds. Pick the clip."*

- **Before:** a long paragraph hedging about missing options, no clip picked.
- **After:** `{"file": "0:02", "start": "0:02", "end": "0:06", "label": "a person plays piano and the room applauds"}`

## Known gaps

Five afternoon moments are **not** in this run — their source clips had not
reached Drive yet, so their rows are placeholders and were excluded on purpose:
guitar, piano, a piece written for his girlfriend, rain, group photo. They go
into a second run once the afternoon footage is verified. Do not re-add them
from memory.

## Files

- `inkling-video-sft.py` — training script (adapted from Tinker's `first-training.py`)
- `inkling-video-picks-27.jsonl` — training data (27 rows, zero placeholders)
- `lora-training-run.log` — full log of the 2026-11-17 run (loss per step + before/after)

## Related

- Video project (recap v10, picks, captions): [inkling-2-small-video-editing](https://github.com/lilaclilac09/inkling-2-small-video-editing)
- Base model write-up: [Tinker end-to-end 4-bit training](https://thinkingmachines.ai/blog/end-to-end-4-bit-training/)
