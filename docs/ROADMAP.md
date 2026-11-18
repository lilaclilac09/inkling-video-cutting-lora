# Roadmap — a Drive link in, a styled video out

The destination: anyone pastes a Google Drive link, says what kind of video they want
("Hollywood trailer", "quiet documentary", "TikTok energy"), and gets a cut back. Not a
demo — a pipeline where the only human step is approving or rejecting the picks.

This document is the plan. `program.md` is the loop. `prepare.py` is the contract that
keeps the two honest.

## What the model's job is — and is not

The LoRA picks **which moments to use, in what order, and says why in one sentence.** It
does not cut, grade, caption or mix. Those are deterministic ffmpeg steps with
inspectable parameters.

That boundary is deliberate:

- A 594B/17B MoE can learn taste from 31 examples. It cannot learn to encode video.
- Selection is the part that needs judgment. Cutting is the part that needs to be exact.
- If selection is wrong, the render is wrong in a way you can see in the picks list. If
  cutting were learned too, a bad render would be un-diagnosable.

So the fine-tune is a **selection policy**, conditioned on a style brief.

## The pipeline

| stage | what happens | today |
|---|---|---|
| 1. Ingest | Drive link → file list → download → `ffprobe` → dedupe by content hash | not built |
| 2. Understand | Whisper transcript, shot boundaries, keyframes, visual tags, audio events (laughter, applause, music), quality scores (blur, exposure, shake) | partial: transcripts and keyframes exist for the Shanghai footage |
| 3. Index | one row per second: time, transcript, scene, tags, quality | not built |
| 4. Brief | the user's style choice → structured constraints | not built |
| 5. Select | **the LoRA**: brief + index → ranked clip picks with reasons | the thing being trained |
| 6. Assemble | ffmpeg: cut, transitions, music bed, captions, grade, loudness | exists in `inkling-2-small-video-editing` |
| 7. Review | contact sheet + picks + reasons; approve or re-roll | not built |

Stages 1–3 and 7 are the real work. The model is one box in the middle.

## Style is cut grammar, not a filter

The mistake to avoid is treating "style" as a color grade. Two videos of the same room
feel completely different because of **shot length, shot order, and audio policy**. That
is learnable; a LUT is not.

| style | cut length | shot order | audio | captions | frame |
|---|---|---|---|---|---|
| Documentary calm | 8–15 s | wide → close, slow build | natural sound, quiet bed | none | 16:9 |
| Trailer | 1–2 s | hook → escalation → title card | music-driven, hard cuts on beats | punch lines only | 2.39:1 |
| Vlog / TikTok | 1–3 s | hook in first 1.5 s, then chronology | voice-forward, music under | always on | 9:16 |
| Fashion film | 3–6 s | texture → body → environment | ambient, no dialogue | none | 2.39:1 |
| News package | 2–3 s | talking head → b-roll → reaction | voice + room tone | lower thirds | 16:9 |
| Arthouse | one shot | single take, no coverage | room only | none | 4:3 |

Each row is a set of numbers a model can be trained against, and a set of ffmpeg
parameters a renderer can execute. That is why this table is the core of the plan.

**The brief** is the machine-readable version of a row:

```json
{
  "style": "trailer",
  "cut_seconds": [1, 2],
  "order": "hook_first",
  "audio": "music_driven",
  "captions": "punch_lines",
  "aspect": "2.39:1",
  "mood": "restless, escalating"
}
```

Same schema, different values, different video. The model is trained on (brief, index) →
picks.

## The data chain

```
Drive folder
   │
   ├─ raw/*.MP4                      source, immutable
   ├─ derived/<file>/transcript.json  Whisper, word timestamps
   ├─ derived/<file>/shots.json       shot boundaries + keyframes
   ├─ derived/<file>/tags.json        scene, people, screen, hands, stage, laughter
   ├─ derived/<file>/quality.json     blur, exposure, shake per second
   │
   ├─ labels/picks.<style>.jsonl      approved picks for one style
   └─ labels/rejects.<style>.jsonl    rejected picks and why — the negative signal
```

Three rules that matter more than the schema:

1. **The index is content-addressed.** Derived files are keyed by a hash of the source, so
   re-downloading or re-running Whisper never silently changes training data under a run.
2. **Rejects are data.** A rejected pick with a reason ("cuts mid-sentence", "logo
   unreadable", "too dark") is the most valuable label you have, and the current 31-row
   set has none. Even 20 rejects would let a future run learn a boundary instead of only a
   preference.
3. **Split by source file, not by row.** See the next section.

## The split problem — a known weakness

`prepare.py` holds out every fourth **clip**. With 25 source files and 31 rows, the same
file can appear in both train and held-out. A model can then memorize a file's look and
score well without generalizing.

The fix is to hold out **source files**, not clips. It is not done yet because 25 files
across 31 rows would leave about 6 files, and the held-out set would be too small to
score meaningfully. This is the strongest argument for more footage, ahead of more labels.

## Phases

| phase | deliverable | needs |
|---|---|---|
| **P0** | contract, eval, ledger, dry-run harness | done |
| **P1** | first real held-out score on Inkling 2 Small | Tinker key on the training machine |
| **P2** | Drive ingest + derived index, no model | Drive access |
| **P3** | brief schema + a second style's labels | 20 approved picks in that style |
| **P4** | style-conditioned LoRA, scored per style | P2 + P3 |
| **P5** | assembly + review surface (contact sheet, approve/re-roll) | render host |

P1 is small and should happen first: it turns the harness from "tested with a stub" into
"tested against the real model", and it costs cents.

## What I need from you

1. **Drive access.** The cleanest is a Google Cloud service account with the Drive API
   enabled, `drive.readonly` scope, and the footage folder shared with its email. The
   alternative is `rclone` with your OAuth token on the machine that runs the ingest. Tell
   me which, and for the service account, the JSON key goes into a secret, never into the
   repo.
2. **Footage shape.** How many files, how many hours, what camera, and whether it is raw
   or already cut. This decides whether ingest is a folder walk or a real media probe.
3. **Which two styles to start with.** I would start with **documentary calm** and
   **trailer**, because they are opposite on every axis in the table above — if one model
   can do both from the same index, the brief is working.
4. **Labels.** For each style, 20 approved picks. If you do not want to hand-label, I can
   bootstrap: a strong model proposes picks from the index, you approve or reject, and the
   decisions become the labels. Your rejects are worth more than the approvals.
5. **Output target.** One vertical short, or a set of clips with captions? Duration?
6. **Budget and host.** A cap on Tinker spend per sweep, and whether renders run in an orb
   or on your machine.

## Open questions

- **Music.** Trailer and vlog styles need a bed. Licensed or generated? This changes the
  assembly stage, not the model.
- **Who approves.** If the review step is a person, the product is a co-pilot. If it has
  to be unattended, the score gate in `prepare.py` has to be much stricter, and there
  needs to be a hard fail ("no clip met the bar") rather than a bad cut.
- **How many styles before it generalizes.** My guess is three. Below that the brief is a
  lookup table; above it, a model starts sharing cut grammar across styles.
