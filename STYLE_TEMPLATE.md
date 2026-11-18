# Video-cutting template (v2) — style is a parameter

The reusable cutting template behind the Inkling video-cutting LoRA.
One template, many styles: the same prompt shape produces different cuts
when only the `style` field changes. Built 2026-11-18 for the Cafe Cursor
Shanghai recap; designed to carry to any later video.

## Parameters

| Parameter | Values | Meaning |
|---|---|---|
| `style` | `vlog` / `fresh` / `silicon-valley` / `glamour` / `hollywood` | The cut's look and feel. The ONLY thing that changes between styles. |
| `beats` | 1–3 beat descriptions | Structural slots (opening, demo, breather, peak, closing). Content-neutral: a beat never names the clip. |
| `candidates` | 4 (single pick) / 8 (multi-pick) | Real segments from the footage: file + start/end + one-line shot description. |

## Style catalog

- **vlog** — the reference recap style: warm, human, one flowing story; a moment that works standalone.
- **fresh** — bright, airy, breathing room: outdoors, daylight, uncluttered frames, calm pacing.
- **silicon-valley** — substance first: product demos, live code, big-topic takeaways, a builder explaining their own work.
- **glamour** — polished and beautiful: close-ups, warm light, people at their best, a frame worth a poster.
- **hollywood** — drama and peak emotion: the room reacting, scale, the biggest beat of the day, a cinematic sweep.

## Prompt shape (single pick)

```
You are cutting the <EVENT> recap (<DATE>) into short clips.

Style: <style> — <style brief>

The recap needs one more clip.
Which candidate segment should be used? Answer with just the letter and the clip file + start/end, e.g. "B: DJI_...MP4 0:05-0:15". If none of them fits, answer "None".

A. <file> <start>-<end>: <one-line shot description>
B. ...
```

## Prompt shape (multi-pick — one call, many files)

```
You are cutting the <EVENT> recap (<DATE>) into short clips.

Style: <style> — <style brief>

The recap needs 3 clips in one pass, one per beat, in order:
Beat 1: "<title>" — <why it matters>
Beat 2: ...
Beat 3: ...

Candidates (pick one per beat, in order):
A. <file> <start>-<end>: <description>
... (8 candidates: 3 golds + 5 distractors)

Answer as a JSON list of 3 clips in beat order, e.g. [{"file": "DJI_...MP4", "start": "0:05", "end": "0:15"}, ...]. No other text.
```

## Output contract

- Single pick: `"<letter>: <file> <start>-<end>"`, or `"None"`.
- Multi-pick: a JSON list of clips, one per beat, in order.
- Refusal: if the requested moment does not exist anywhere in the footage,
  answer `"None"` — never invent a clip. (Trap rule: guitar / piano / rain /
  group photo were verified missing from all 73 clips + 9 Drive clips.)

## Rules for reuse on a new video

1. Keep the prompt shape byte-identical; change only `style`, `beats`,
   `candidates`, and the event line.
2. Candidates must be real segments. Distractors must not overlap any
   approved pick or training example.
3. Never train on the eval set. New eval items use held-out picks only.
4. Select checkpoints by eval score, never by step count.

## Scores (eval set v2, 52 items — final, 2026-11-19)

| Format | Base | v1 (SFT) | v2 (SFT repro) | v3-grpo-20 (winner) |
|---|---|---|---|---|
| mc (11) | 0.18 | 0.82 | 1.00 | **1.00** |
| style-mc (16) | 0.50 | 0.56 | 0.62 | **0.88** |
| stability (6) | 0.50 | 1.00 | 1.00 | **1.00** |
| stability pairs agree | 1/3 | 3/3 | 3/3 | **3/3** |
| multi (4) | 0.50 | 0.00 | 0.75 | **1.00** |
| sft exact (11) | 0.00 | 0.18 | 0.18 | 0.00 |
| trap (4) | 0.00 | 0.00 | 0.00 | **1.00** |
| **Overall** | **0.29 (15/52)** | **0.50 (26/52)** | **0.61 (32/52)** | **0.75 (39/52)** |

Per-style at the winner: fresh 4/4, glamour 3/4, hollywood 4/4,
silicon-valley 3/4. Full leaderboard + cost + what to do next:
`RESULTS-2026-11-18.md`. Results: `eval/v2-*.json` (re-scored copies
alongside; the saved base re-score undercounts multi by 2 — a
raw-truncation artifact in the old scorer, fixed in `scoring.py`
afterwards; live numbers above are authoritative).
