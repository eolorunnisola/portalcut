# PortalCut tracking pipeline: architecture and plan

Goal: track every player through a game with consistent identities, record positions, produce an
annotated video and movement analytics, and optionally ask Gemini Flash to interpret the result.
Priority: tracking accuracy and identity consistency over model cost.

## Pipeline

```
upload ─► ingest ─► detect ─► track ─► identify ─► annotate ─► analytics ─► (analysis)
          ffprobe   RF-DETR   ByteTrack/  teams +     OpenCV      positions,   Gemini Flash,
          720p      person,   BoT-SORT    jersey #s   boxes, IDs  distance,    sampled frames +
          proxy     ball      tracklets   + ReID      trails      heatmaps     tracking summary
```

Each stage reads the previous stage's files in `data/<job>/` and writes its own, so any stage can be
re-run or replaced alone. The core pipeline never calls an LLM per frame.

| Module | Role | Replaceable by |
|---|---|---|
| `ingest.py` | validate upload, probe, 720p proxy, audio loudness | — |
| `detect.py` | `Detector` interface; `RFDETRDetector` (person, ball) | any YOLO-style detector with the same `detect()` |
| `track.py` | tracker choice and settings, writes `tracks.npz`, debug video | other `trackers` package trackers |
| `reid.py` | appearance features, team colour clusters, player selection | stronger ReID model |
| `jersey.py` | jersey numbers from torso crops (Claude Haiku, majority vote) | PARSeq or another OCR |
| `identity.py` | links tracklets into players: team + jersey number → `identities.json` | appearance-based linking |
| `annotate.py` | annotated MP4: boxes, IDs, team colours, confidence, trails, ball | — |
| `analytics.py` | per-player statistics, heatmaps, zones; optional calibration | — |
| `analysis.py` | Gemini Flash interpretation of sampled frames + summaries | another LLM |
| `events.py`, `gapfill.py` | highlight plays for one selected player (kept from milestone 4) | — |
| `app.py` | FastAPI endpoints, background jobs | — |
| `static/` | upload page and analytics dashboard (separate from processing) | — |

## Data contracts (files in `data/<job>/`)

- `tracks.npz`: one row per detection. `frame` int32 (sample index), `t` float32 seconds,
  `track_id` int32 (-1 for the ball), `cls` int8 (0 person, 1 ball), `xyxy` float32 N×4 in proxy
  pixels, `conf` float32.
- `identities.json`: `{"<track_id>": {"team": 0|1|2|null, "number": "13"|null, "player_id": "T0-13" | "trk-512"}}`.
  Teams 0 and 1 are the two largest jersey-colour clusters; 2 is everyone else (referees, staff).
  Tracklets sharing a team and a jersey number read at least twice get the same `player_id`.
- `calibration.json` (optional): `{"image_points": [[x,y]×4], "world_points": [[X,Y]×4], "units": "m", "surface": [W, H]}`.
- `annotated.mp4`, `analytics.json`, `analysis.json`.

Job settings live in `meta["config"]` (see `config.py`): proxy height, detection fps, person and ball
confidence thresholds, tracker, lost-track buffer, annotated-video fps, Gemini on/off and sampling.

## Significant decisions

1. **Detector: RF-DETR, not Ultralytics YOLO.** Ultralytics is AGPL-3.0, which would force
   open-sourcing a sold or hosted product. RF-DETR (Apache-2.0) is the same class of real-time
   detector and already runs on Apple Silicon. `detect.Detector` keeps a YOLO adapter possible.
2. **Frames are decoded with ffmpeg, drawn with OpenCV.** ffmpeg handles phone rotation and HEVC
   reliably; OpenCV is used for annotation and image processing.
3. **Positions stay in per-job `.npz` files; SQLite holds jobs and settings.** Tens of thousands of
   detections per game are cheaper to store and read as columnar arrays. Moving them into
   PostgreSQL later only touches the loaders.
4. **Identity = tracklet linking, not tracking alone.** ByteTrack keeps IDs within continuous
   footage. It cannot recognise a player after a camera cut. Cross-cut identity comes from
   team colour + jersey number (+ appearance), and is reported separately from tracker IDs.
5. **Distances are pixels until calibrated.** Without `calibration.json` every distance and speed is
   labelled `estimated` in pixel units. With 4 known court points a homography converts foot
   positions to metres; broadcast camera motion still makes that approximate (one homography per
   clip), and the output says so.
6. **Tracker defaults were chosen by measurement** against a hand-labelled answer key
   (`eval/tracking/`, scored by `track_eval.py`): BoT-SORT with `track.TUNED` settings, 10 detections/s,
   1 s lost buffer. It names 51% of labelled players correctly at 0.92 precision; library defaults 45%,
   ByteTrack 18% (it finds more players but half its tracks swap people, so jersey votes never agree).
   Fragments are fine (`identity.py` links them by number); swaps are what break naming. Note: the `trackers`
   library counts `lost_track_buffer` in 30 FPS frames whatever the real rate (`track.make_tracker`).
7. **Gemini is optional and separate.** It sees sampled frames and tracking summaries only, with
   sampling, retries, structured JSON and cost logging.

## Stages and checks

| Stage | Work | Validated by |
|---|---|---|
| 1–2 | Detector interface; tracker comparison (ByteTrack, BoT-SORT, OC-SORT × fps) | fragmentation and identity-swap metrics on the UConn–Arizona recording |
| 3 | Annotated video | rendered frames checked by eye |
| 4 | Analytics + calibration | unit tests with known geometry; real job output |
| 5 | Gemini layer | tests with a fake client; real call once `GEMINI_API_KEY` is set |
| 6 | API endpoints, background jobs, dashboard | API tests; end-to-end run |
| 7 | Tests, docs, visual example | full suite |
