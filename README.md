# PortalCut

Local sports player-tracking system: upload a game video, and every visible player is detected,
tracked, linked into named players (team + jersey number), drawn on an annotated video and turned
into movement statistics. On top of that: highlight plays for one selected player, and optional
Gemini Flash interpretation of the game. Architecture and design decisions: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Setup

```bash
brew install ffmpeg
uv sync
```

API keys go in a `.env` file in this folder (git-ignored, never committed). Both are optional:

```
ANTHROPIC_API_KEY=your-key   # jersey numbers (~$0.01 per 3 min) and highlight plays
GEMINI_API_KEY=your-key      # optional Gemini Flash game analysis; GEMINI_MODEL overrides the model
```

Models download once on first use: RF-DETR Medium (~350 MB), DINOv2-small (~90 MB), SAM 2.1 tiny (~150 MB).

## Run

```bash
uv run portalcut
```

Open http://localhost:8000, drop in a video, then open the dashboard link on the job
(`/dashboard?job=<id>`). Everything stays in `data/` on this machine.

## Pipeline

`ingest → tracking → features → numbers → identities → annotating → analytics → analysis (optional)`

| Stage | Output in `data/<job>/` |
|---|---|
| ingest: probe, 720p proxy, audio loudness | `proxy.mp4`, `audio_rms.json` |
| tracking: RF-DETR (person, ball) + BoT-SORT | `tracks.npz` (frame, time, track ID, class, box, confidence) |
| features: appearance + jersey-colour per track piece | `features.npz`, crops |
| numbers: jersey numbers read by Claude Haiku, majority vote | `numbers.json` |
| identities: team + number links track pieces into players | `identities.json` (`T1-13` = team 1, #13) |
| annotating: boxes, IDs, team colours, confidence, trails, ball | `annotated.mp4` |
| analytics: positions, distance, speed, heatmaps, zones | `analytics.json` |
| analysis: Gemini Flash on sampled frames + tracking summary | `analysis.json` |

### Settings

Pass any of these as query parameters on upload (`POST /api/jobs?filename=game.mp4&tracker=bytetrack`)
or in `config` when re-running (`POST /api/jobs/{id}/track`):

| Setting | Default | Meaning |
|---|---|---|
| `proxy_height` | 720 | processing resolution |
| `detect_fps` | 10 | detections per second |
| `person_conf`, `ball_conf` | 0.4, 0.25 | minimum detection confidence |
| `tracker` | `botsort` | `botsort`, `bytetrack` or `ocsort` |
| `lost_seconds` | 1.0 | how long a lost track can be re-found |
| `tracker_params` | empty | extra tracker settings, `name=value,...` (e.g. `high_conf_det_threshold=0.4`) |
| `annotate_fps` | 10 | annotated video frame rate |
| `jersey_numbers` | true | read numbers with Claude Haiku (capped at $0.25) |
| `gemini`, `gemini_interval_s` | false, 30 | Gemini analysis and its frame sampling |

### API

| Method and path | Purpose |
|---|---|
| `POST /api/jobs?filename=…` | upload (raw body) and start processing |
| `GET /api/jobs`, `GET /api/jobs/{id}` | list jobs, status and progress |
| `POST /api/jobs/{id}/track` | re-run from a stage: `{"start_stage": "identities", "config": {…}}` |
| `GET /api/jobs/{id}/tracks?start=&end=&track_id=` | raw tracking data |
| `GET /api/jobs/{id}/players?min_seconds=` | per-player statistics |
| `GET /api/jobs/{id}/players/{player_id}` | one player, with trajectory and heatmap |
| `GET /api/jobs/{id}/annotated?download=true` | annotated video |
| `PUT /api/jobs/{id}/calibration` | 4 image points + their court positions → statistics in metres |
| `POST`, `GET /api/jobs/{id}/analysis` | run and read the Gemini analysis |
| `/api/jobs/{id}/select/…`, `/plays` | pick one player and find their highlight plays |

Every distance and speed is labelled with its units. Without calibration they are **estimates in image
pixels** (screen positions, distorted by camera pans), not real-world measurements.

## Run on Google Colab

Heavy processing (detection, tracking, tracker sweeps) can run on a free or paid Colab GPU instead of this Mac.

1. Open `colab/portalcut_worker.ipynb` in Colab (*File → Open notebook → GitHub*), with a GPU runtime.
2. Follow its first cell: Drive folders `portalcut/videos`, `portalcut/rosters`, `portalcut/jobs`, and Colab Secrets
   `GITHUB_TOKEN` (read-only) and optionally `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`.
3. Set the options cell and *Run all*. Each video becomes `portalcut/jobs/<job_id>/` on Drive.
4. Paste the printed **SUMMARY** lines back to Claude. Do not paste videos, `.npz` files or full logs.
5. To open a job in the local app (Drive for desktop syncing the jobs folder):

```bash
uv run python -m portalcut.import_job "<Google Drive>/My Drive/portalcut/jobs/<job_id>"
```

The same tools run anywhere without the web server:

```bash
uv run python -m portalcut.batch game.mp4 --out jobs/ --set tracker=bytetrack --roster eval/rosters/uconn_arizona.json
```

```bash
uv run python scripts/compare_trackers.py jobs/<job_id> eval/tracking/uconn_arizona_clip/key.json
```

```bash
uv run python -m portalcut.track_eval jobs/<job_id> eval/tracking/uconn_arizona_clip/key.json
```

## Test

```bash
uv run pytest -q
```

Score detected highlight plays against a hand-labelled answer key:

```bash
uv run python -m portalcut.eval <job_id> eval/uconn_arizona_az13.json
```

## Measured

Tracker comparison on 3 minutes of a UConn–Arizona broadcast, identical detections for every tracker.
"Swaps" are track pieces whose jersey colour flips team (about 19% is measurement noise):

| Detection fps | Tracker | Lost buffer | People kept per frame | Swaps |
|---|---|---|---|---|
| 5 | ByteTrack (old setting) | 0.5 s | 6.52 | 37.4% |
| 5 | ByteTrack | 3 s | 6.73 | 45.8% |
| 10 | ByteTrack | 0.5 s | 7.17 | 31.7% |
| 10 | **BoT-SORT (default)** | 1 s | 5.80 | 26.9% |
| 10 | BoT-SORT | 0.5 s | 5.72 | 24.4% |
| 10 | OC-SORT | 3 s | 4.55 | 28.8% |

Full pipeline on that 3-minute clip (Apple Silicon): tracking 152 s (0.84× real time), 374 track pieces,
jersey numbers $0.0098, 94 pieces linked into 22 named players, annotated video 11 s, analytics instant.

| Earlier measurements | Result | Cost |
|---|---|---|
| Ingest, 60 s of 1080p60 | 9.2 s | $0 |
| Highlight plays, Arizona #13, close-ups at 4 fps + scoreboard | 1 of 2 key plays, 0 false | $0.087 per 56 windows |
| SAM 2 gap fill when selecting a player | follows the player through a dunk the tracker lost | 169 s |

## Known limits

- Local only, no login; the server listens on 127.0.0.1.
- BoT-SORT keeps about 20% fewer person detections than ByteTrack in exchange for fewer identity swaps.
- Identity linking needs a jersey number read at least twice; ~60% of visible person-time stays as
  unnamed track pieces (referees, bench, crowd, short pieces). Seen errors: a navy #24 clustered into the
  white team (fake "T1-24"), a tracker swap inside one piece, and #18 misread as 13.
- Referees' striped shirts often cluster with the dark-jersey team.
- Close-ups, bench shots and replays are tracked like live play (no live-play filter yet).
- The ball is found in ~36% of frames (generic COCO "sports ball" class).
- Gemini analysis is built and tested with a fake client only; the default model id `gemini-flash-latest`
  is unconfirmed until a real call runs with `GEMINI_API_KEY` set.
- Validated on basketball broadcast footage only; soccer and amateur single-camera video not yet tested.
