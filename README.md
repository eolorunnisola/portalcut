# PortalCut

Local app: basketball game video → highlight clips for one selected player.

## Setup

```bash
brew install ffmpeg
uv sync
```

To find plays (milestone 4), put your Claude API key in a `.env` file in this folder. It is git-ignored and never committed:

```
ANTHROPIC_API_KEY=your-key
```

How finding plays works and what it costs:

- Your player's on-screen time is cut into 5 s windows. Each window is sent to Claude Haiku 5.5 as 10 frames, with your player outlined in magenta.
- Borderline calls, and every made shot and assist, get a second opinion from Claude Sonnet 5.5 (16 frames).
- The page shows an estimate before anything is sent. Each run stops at a $1 budget, which you can change in `BUDGET_USD` in `portalcut/events.py`.
- Answers are cached in `data/<job>/vlm_cache/`, so a re-run of the same moments is free.

## Measure (answer keys)

Hand-labelled plays live in `eval/`. Score a job's detected plays against one:

```bash
uv run python -m portalcut.eval <job_id> eval/uconn_arizona_az13.json
```

It prints precision and recall per play type (a match is the same type within ±2 s) and lists misses and false plays. Detections inside the key's `ignore` ranges (replays, dead time) are not counted.

## Run

```bash
uv run portalcut
```

Open http://localhost:8000 and drop in a game video. Everything stays in `data/` on this machine.

## Test

```bash
uv run pytest -q
```

## Status

| Milestone | State |
|---|---|
| 1. Upload + ingest (metadata, 720p proxy, audio loudness curve) | done |
| 2. Player and ball tracking (RF-DETR Medium + ByteTrack, 60 s debug video) | done |
| 3. Player selection (click your player, answer "Is this you?", 2 min player-only video) | done |
| 4. Event detection (Claude Haiku 5.5, Sonnet 5.5 second opinions, cost cap) | done; tested on the 1-on-1 sample only |
| 5. Review + export | next |
| 6. Eval | |

## Measured so far

| Step | Input | Time | Cost |
|---|---|---|---|
| Ingest (Apple Silicon, hardware encoder) | 60 s of 1080p60 | 9.2 s (≈0.15× real time) | $0 |
| Tracking, 5 frames/s on MPS | 2 min of 1080p60 | 47–50 s (≈0.4× real time) | $0 |
| Tracking, 3 frames/s on MPS | 2 min of 1080p60 | 28 s (≈0.23× real time), but more broken tracks | $0 |
| Appearance features (DINOv2-small + jersey colour) | 45 tracks | 6–14 s | $0 |
| Player selection on the 1-on-1 sample | 1 click | found in all 15 loop segments, 66% of frames, no wrong player | $0 |
| Find plays, first run (every borderline call re-checked) | 46 windows, 2 min | 43 s; 17 defensive stops + 1 possession | $0.71 (Sonnet $0.68, Haiku $0.04) |
| Find plays, current rules (only highlight plays re-checked) | same | 19 plays, 0 second opinions needed | ≈$0.04 fresh; $0 re-run from cache |
| UConn–Arizona broadcast, Arizona #13, full frames at 2 fps | 56 windows | 0 of 2 key plays, 0 false | $0.056 |
| Same, close-ups at 4 fps + before/after scoreboard | 56 windows | 1 of 2 key plays (the dunk), 0 false | $0.087 |
| Jersey numbers (Haiku, 10 torso strips per call) | 996 track pieces | 472 read; #13 coverage 3% → 12% | $0.032 |
| SAM 2 gap fill on Done (20 edges × 2 s) | #13 | +91 frames, incl. the dunk | 169 s, $0 |

A 2-hour game should take about 18 minutes to ingest and 50 minutes to track. Those figures are extrapolated from a 1-on-1 sample clip, not yet measured on a real game.

Re-run tracking on an existing job: `uv run python -m portalcut.track <job_id>`.

Models download once: RF-DETR Medium (~350 MB) to `~/.roboflow/models/`, DINOv2-small (~90 MB) to the Hugging Face cache.

## Known limits (v1)

- Local only, no login. The server listens on 127.0.0.1.
- A job interrupted by a server restart is marked failed; upload it again.
- Ball detection uses the generic COCO "sports ball" class and finds the ball in only ~28% of sample frames. It is used only as a ranking signal later, never as a filter.
- Tracks break when players cross or leave the frame. Player selection re-links them by appearance.
- Selection has only been tested on a 1-on-1 clip with two very different jerseys. Teammates in the same jersey are the hard case; the `BAND` threshold in `portalcut/reid.py` will need tuning on a real game.
- Jersey-number reading (with Claude Haiku) is not built; add it if teammates get confused.
- Clicking a different player switches the selection (with Undo). A click on someone who only *might* be a different person (similarity within `BAND`) is treated as another sighting of you instead; if that was wrong, click them again in a frame where your player is also boxed to switch.
