# PortalCut v1 — Claude Code build prompt (basketball only)

Paste everything below the line into Claude Code, opened in this `athlete_clipper` folder.

---

You are building **v1 of PortalCut**, a local app that turns a full basketball game video into highlight clips for one selected player.

## Working rules (read first)

1. Read and follow the three skill files in this folder:
   - `ponytail-skill.md` at level **full**: the smallest complete change. No features I did not ask for.
   - `overseerskill.md`: you (Opus) plan, own the data model, the tracking/re-ID logic and the event pipeline, and review everything. Delegate well-specified pieces (UI page, ffmpeg helpers, tests) to Sonnet and mechanical checks (lint, tests, inventories) to Haiku.
   - `cavemen-skill.md`: use the cavecrew agents only if they are installed. Otherwise ignore it.
2. Before writing any Claude API code, load the `claude-api` skill and follow it for model IDs, structured outputs and error handling.
3. Work in milestones (below). After each one, run its acceptance check, show me the result, and stop for my OK before starting the next.
4. Ask me before adding any dependency not listed here.
5. Track every run's cost (GPU/CPU seconds and API tokens, converted to dollars) and print it at the end of each job.

## What v1 does (the whole scope)

1. I drop in a basketball game video (MP4/MOV, up to ~2 hours, 720p–4K).
2. The app processes it: detects and tracks players and the ball.
3. I **select my player** by clicking them in a frame. The app then asks me to confirm a few uncertain matches ("Is this you?").
4. The app finds every moment where that player **has the ball or makes a play**: made shot, missed shot attempt, assist, steal, block, rebound, drive or dribble possession, and a good defensive stop (contest).
5. I review the clip list (play, include/exclude, nudge start/end, see the event label and confidence) and export:
   - individual MP4 clips, and
   - one stitched highlight reel MP4.

**Out of scope for v1:** accounts, payments, cloud hosting, other sports, YouTube links, coach analytics, overlay rings or graphics, mobile UI.

## Stack (keep it this small)

- **Language:** Python 3.11, managed with `uv`.
- **Backend:** FastAPI serving the API and one static HTML page (vanilla JS, no build step).
- **Storage:** SQLite (via `sqlite3` or SQLModel) plus a local `data/` folder for videos, frames and clips.
- **Jobs:** a background thread or process. No Redis, Celery or queue service.
- **Video:** `ffmpeg`/`ffprobe` through subprocess.
- **Hardware:** run on Apple Silicon (MPS), CUDA if present, and CPU as a fallback. Detect the device automatically.

## Models: cheapest options that still give high quality

| Job | Model | License | Why |
|---|---|---|---|
| Person and ball detection | **RF-DETR** (`rfdetr`, smallest variant that holds accuracy; COCO classes `person` and `sports ball`) | Apache-2.0 | Free, runs locally, commercial-safe. Do **not** use Ultralytics YOLO (AGPL). |
| Tracking | **ByteTrack** via `supervision` | MIT | Free and fast. |
| Player re-identification | **DINOv2-small** embeddings (`transformers`) on torso crops, plus a jersey color histogram | Apache-2.0 | Links broken tracklets across the game without training. |
| Jersey number check | **Claude Haiku 5.5** (`claude-haiku-5-5`), 2–3 best crops per uncertain tracklet | API | Far cheaper to build than an OCR model, and reads numbers well. |
| Event classification (main) | **Claude Haiku 5.5** (`claude-haiku-5-5`) on short frame sequences | API, $0.10/$0.50 per M tokens | Cheap enough to run on every candidate window. |
| Event re-check (escalation) | **Claude Sonnet 5.5** (`claude-sonnet-5-5`) | API, $2/$10 per M tokens | Only for borderline-confidence windows and every scoring or assist event. |

Claude API rules:
- Use the `anthropic` Python SDK and `client.messages.parse()` with Pydantic schemas for structured outputs.
- Don't use forced `tool_choice`, `temperature`, or assistant prefill. These models reject them.
- Check `stop_reason` (including `refusal`) before reading content.
- Run calls concurrently with a semaphore (default 8). Retry on rate limits.
- Cache results on disk by a hash of the input, so a re-run costs nothing.
- Read the API key from `ANTHROPIC_API_KEY` in `.env`. Never hard-code it.

## Pipeline

### Pass 0: Ingest
- Run `ffprobe` to get metadata.
- Make a **720p analysis proxy** at the original frame rate. Keep the original file for the final cuts.
- Extract mono 16 kHz audio and compute a per-second loudness (RMS) curve. It's used later as a weak excitement signal.

### Pass 1: Detection and tracking
- Run RF-DETR on the proxy at **5 fps** (configurable) for `person` and `sports ball`.
- Run ByteTrack on the person detections to get tracklets.
- Save per-frame boxes, track IDs and ball boxes to `data/<job>/tracks.parquet` (or SQLite).
- Measure throughput first on 2 minutes of video and report it. The target is processing a full game in ≤ 1× real time on an M-series Mac. If it's slower, try the smaller RF-DETR variant or 3 fps, and tell me the tradeoff.

### Pass 2: Player selection and re-ID
- UI shows 8 frames spread across the game, with person boxes drawn. I click my player in one or more frames.
- Build a gallery from those clicks: DINOv2 embeddings of torso crops (upper 20–60% of the box) plus a jersey color histogram.
- Score every tracklet against the gallery, using the mean of its best crops.
  - High similarity: auto-accept.
  - Low similarity or the wrong jersey color: reject.
  - **Middle band:** show up to 12 "Is this you?" cards with crops. Each answer updates the gallery, then rescore. This human-in-the-loop step is the main quality lever. Keep it under a minute of my time.
  - Optionally, have Haiku read the jersey number on middle-band tracklets and use it as an extra vote.
- Output: a timeline of where the selected player is visible, with their box per frame.

### Pass 3: Candidate windows (keep API cost low)
- From the player's timeline, create **5-second windows with a 2.5-second stride** wherever the player is visible.
- Score each window by:
  - the ball near the player (ball center inside the player box expanded by 50%),
  - player speed,
  - the audio loudness spike.
- Send the top windows to the VLM, all of them if the total cost is under $1, else the top N under a configurable budget. Basketball ball detection from COCO is unreliable, so ball proximity is **only a ranking signal, never a hard filter**.

### Pass 4: Event classification
- For each window, send Haiku **10 frames (2 fps)**. Each frame is the full scene downscaled to ~768 px wide, with the selected player outlined by a **thick colored box**, plus one higher-resolution crop around the player for the middle frame.
- Pydantic output schema:
  - `player_has_ball: bool`
  - `events: list[{type, frame_start, frame_end, confidence (0–1), short_reason}]`
  - The event `type` is one of: `made_shot`, `missed_shot`, `assist`, `steal`, `block`, `rebound`, `drive`, `possession`, `defensive_stop`, `none`.
- The system prompt must:
  - define each event in basketball terms,
  - stress that events must be performed **by the outlined player**,
  - for an assist, require a pass from the outlined player followed by a teammate's made shot inside the window (extend the window to 8 seconds when the model returns `assist_possible`),
  - say to return `none` when unsure.
- **Escalate to Sonnet 5.5** with 16 frames when confidence is between 0.4 and 0.75, and for every `made_shot` and `assist`.

### Pass 5: Clips and export
- Merge overlapping or nearby detections (gaps under 1.5 s) into one clip. Label it with its strongest event.
- Default clip bounds: event start − 3 s to event end + 2 s. Then snap the cut points to keyframes, or re-encode if needed for frame accuracy.
- The review UI lists the clips (with thumbnail, label, confidence and timestamp). Each clip has a play button, a keep toggle, and nudge buttons (±1 s on start and end). Sort by time, with a filter by event type.
- Export:
  - individual MP4s cut from the **original** video (H.264, AAC),
  - one stitched reel (ordered by time or by confidence, my choice), with an optional 2-second title card showing name, number and position (rendered with Pillow).

## Quality: build the evaluation in from day one
- Add `eval/`. I will label 1–2 games by hand in `eval/<game>.json` (`[{t, type}]`).
- Add `uv run python -m portalcut.eval <job_id> eval/<game>.json`. It reports **precision and recall per event type** with a ±2 s tolerance, the player-ID accuracy on a sample, and the cost per game.
- Targets for v1:
  - recall ≥ 80% on `made_shot`, `steal`, `block` and `rebound`,
  - precision ≥ 80% overall,
  - API cost under $1 per game,
  - my manual review takes under 10 minutes.
- Report where it fails (camera angle, crowding, the player off-screen), not just the averages.

## Milestones (stop after each for my OK)

1. **Skeleton:** uv project, FastAPI app, upload page, ingest (Pass 0), job status. *Check:* upload a video and see its metadata and the proxy file.
2. **Tracking:** Pass 1 with the throughput report. *Check:* a debug MP4 of 60 seconds with boxes and track IDs drawn.
3. **Player selection:** Pass 2 with the click UI and "Is this you?" cards. *Check:* a debug MP4 of 2 minutes with only my player boxed, and the share of the game where they're found.
4. **Events:** Passes 3–4 with cost logging. *Check:* a JSON of events for a 10-minute segment and its cost.
5. **Review and export:** Pass 5. *Check:* exported clips and a reel I can play.
6. **Eval:** the eval script plus a written summary of results and the top 3 failure causes, each with a proposed fix.

## Deliverables
- Working app: `uv run portalcut` opens on `localhost:8000`.
- A `README.md` with setup (install ffmpeg, set the API key, run), a cost and throughput table, and known limits.
- Small tests for the non-trivial logic: window merging, clip bounds, re-ID scoring thresholds and the eval matcher.
- No secrets in the repo. Add `.env` and `data/` to `.gitignore`.
