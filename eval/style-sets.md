# Style sets — worked examples for the template

The four candidate sets behind the `style-mc` eval items. Each set is asked
four times (once per style) with the SAME four candidates; only the Style
line changes, and the gold differs by style. This is the proof that style is
a parameter: if the model ignores the Style line, it cannot score above
chance (25%) across the four asks of one set.

Sets 1-2: Tier A v5 picks (Aileen-approved, held out from training).
Sets 3-4: 6-second shot-list windows (asserted non-overlapping with every
pick and every training example).

## Set 1 — one more clip for the recap

| Style | Gold | Why it fits the style |
|---|---|---|
| fresh | DJI_20260719101336_0037_D.MP4 0:01-0:07 | v5 pick 0 'Arrival on Yanqing Road': whole storefront, awning and plane trees, people outside |
| silicon-valley | DJI_20260719101632_0047_D.MP4 0:05-0:11 | v5 pick 7 'The Agent era, on title': the most legible title slide of the day |
| glamour | DJI_20260719092043_0007_D.MP4 0:00-0:04 | v5 pick 10 'Audience with a gimbal': a quick audience close-up, energy in the seats |
| hollywood | 延时.MP4 0:06-0:14 | v5 pick 1 'The room fills up': timelapse of the audience arriving under the paper lanterns |

## Set 2 — one more clip for the recap

| Style | Gold | Why it fits the style |
|---|---|---|
| fresh | DJI_20260719104432_0054_D.MP4 0:01-0:06 | v5 pick 18 'Chatting under the plane trees': a breather outside, four people talking over drinks |
| silicon-valley | DJI_20260719103442_0049_D.MP4 0:25-0:32 | v5 pick 15 'Information Radar': a builder shows his AI-news dashboard, GitHub trending and tool rankings |
| glamour | DJI_20260719084815_0042_D.MP4 0:29-0:36 | v5 pick 12 'A woman demos her phone app': a woman presents her own phone app with the springboard screen behind her |
| hollywood | DJI_20260719091529_0043_D.MP4 0:18-0:24 | v5 pick 13 'Your agents work': 'Your brand knows more than it remembers' on screen — the punchy line |

## Set 3 — one more clip for the recap

| Style | Gold | Why it fits the style |
|---|---|---|
| fresh | DJI_20260719095456_0027_D.MP4 0:08-0:14 | Shot list: EXTERIOR: storefront entrance, glass doors, awning, people at the door |
| silicon-valley | DJI_20260719103001_0048_D.MP4 0:15-0:21 | Shot list: Carl Yang presents detailed Harness slides; the screen shows dense tables and architecture |
| glamour | DJI_20260719104626_0056_D.MP4 0:00-0:06 | Shot list: Close-up of people at the bar (woman with a phone, men); a woman presenter (blonde, pink top) at the screen |
| hollywood | DJI_20260719084815_0042_D.MP4 0:18-0:24 | Shot list: Crowd/audience close-up; people standing and watching near the glass |

## Set 4 — one more clip for the recap

| Style | Gold | Why it fits the style |
|---|---|---|
| fresh | DJI_20260719094910_0025_D.MP4 0:00-0:06 | Shot list: EXTERIOR street: close-up of people outside (woman with a phone, men chatting) under the plane trees |
| silicon-valley | DJI_20260719072352_0033_D.MP4 0:13-0:19 | Shot list: man in striped shirt presents at the big screen; screen shows Chinese AI-native tools slides |
| glamour | DJI_20260719073634_0037_D.MP4 0:50-0:56 | Shot list: camera pans to a close-up of audience members watching |
| hollywood | 延时.MP4 0:14-0:20 | Shot list: timelapse continues — the room keeps filling, same sweeping scale as the opening beat |

## How to add a fifth set (for a new video)

1. Pick 4 real segments, one per style, from the shot list (never from
   training examples, never overlapping an approved pick).
2. Write one evidence line per segment, quoting the shot description.
3. Ask all 4 styles against the same 4 candidates.
4. Rebuild with `build_eval_set_v2.py` (append to STYLE_SETS) and re-run.
