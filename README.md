# PortalCut

Local app: basketball game video → highlight clips for one selected player.

## Setup

```bash
brew install ffmpeg
uv sync
```

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
| 4. Event detection | next |
| 5. Review + export | |
| 6. Eval | |

## Measured so far

| Step | Input | Time | Cost |
|---|---|---|---|
| Ingest (Apple Silicon, hardware encoder) | 60 s of 1080p60 | 9.2 s (≈0.15× real time) | $0 |
| Tracking, 5 frames/s on MPS | 2 min of 1080p60 | 47–50 s (≈0.4× real time) | $0 |
| Tracking, 3 frames/s on MPS | 2 min of 1080p60 | 28 s (≈0.23× real time), but more broken tracks | $0 |
| Appearance features (DINOv2-small + jersey colour) | 45 tracks | 6–14 s | $0 |
| Player selection on the 1-on-1 sample | 1 click | found in all 15 loop segments, 66% of frames, no wrong player | $0 |

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
