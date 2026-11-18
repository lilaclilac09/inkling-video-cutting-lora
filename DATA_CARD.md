# Data card — `data/picks.jsonl`

One row per approved clip pick from the Cafe Cursor Shanghai footage (July 19, 2026).
Each row is one training example: a transcript excerpt plus keyframes in, one clip out.

## Shape

```json
{
  "node": "morning | afternoon | evening",
  "tags": ["guitar", "showcase monologue"],
  "transcript_excerpt": "[10:38:05] ...and then the whole room just bursts out laughing",
  "keyframes": ["DJI_20260719103816_0051_D.MP4/frames/frame-0005.jpg"],
  "clip": {
    "file": "DJI_20260719103816_0051_D.MP4",
    "start": "0:03",
    "end": "0:15",
    "title": "The room erupts — open on the laugh",
    "reason": "The warmest twelve seconds of the morning..."
  },
  "synthetic": true,
  "synthetic_note": "Teaches hook-first ordering...",
  "covers": "hook-at-front; laughter as opening"
}
```

`synthetic` rows are teaching examples derived from a real clip: Aileen's rule expressed
against footage that exists. They are not invented footage. A row with no verified
footage does not belong in this file, at any `synthetic` value.

## Provenance

This file is the 31-row union of two earlier files:

- `inkling-video-picks-27.jsonl` — the v1 training set, 27 rows
- `inkling-video-picks-full.jsonl` from the sibling repo `inkling-2-small-video-editing` —
  the same 27 rows plus 4 afternoon rows: guitar, piano, the rain arriving, and the group
  photo

Those 4 were recorded in the v1 README as "known gaps, not in this run". They were not
missing from the project, only from the 27-row file. The union is now the single source of
truth; the two older files are gone so there is no second version to train on by mistake.

## Coverage

| | |
|---|---:|
| rows | 31 |
| synthetic | 14 |
| rows with a transcript | 10 |
| distinct source files | 25 |
| duplicate clip keys | 0 |
| clip length, min / mean / max | 3 s / 7.26 s / 12 s |
| tags | 49 |
| time nodes | afternoon, morning |
| train / held out | 23 / 8 |

Twenty-one rows have an empty transcript because the footage has no speech: drone
establishing shots, timelapses, close-ups of hands and screens. For those the model has
only the keyframe paths, which is why the prompt names them explicitly.

## What is still missing

- **Transcripts for the real rows.** Only the synthetic rows carry a transcript. If the
  model is meant to work from speech, the 21 silent rows teach it that the transcript is
  often empty — which is true, but it means the model has never seen a long transcript
  paired with a *real* (non-synthetic) pick.
- **Evening footage.** The `evening` node is empty.
- **A second event.** Every row is one meetup. Style-conditioned cutting (see
  `docs/ROADMAP.md`) needs more than one room.
- **Verified durations.** Clip bounds come from Aileen's review, not from probing the
  source files. A later pass should confirm each `end` is inside the file.

## Rules

- Never train on a held-out row. The split is in `prepare.py`.
- Never add a moment from memory. If the footage is not verified, the row does not exist.
- Never edit a `clip` to make a score go up. Fix the model, not the label.
