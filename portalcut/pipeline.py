"""Run a job's processing stages in order, each reading the previous stage's files in data/<job>/.

tracking -> features -> numbers -> identities -> annotating -> analytics -> analysis (optional, Gemini)

Any stage can be re-run on its own with `run(job_id, job_dir, start_stage=...)`. The optional analysis
stage never fails the job: tracking output stays usable if Gemini is down or not configured.
"""

import json
import logging
from pathlib import Path

from portalcut import db, track
from portalcut.config import Config, parse_params

log = logging.getLogger("portalcut.pipeline")
STAGES = ("tracking", "live", "features", "numbers", "identities", "annotating", "analytics", "analysis")
INVALIDATED_BY_TRACKING = ("live.json", "numbers.json", "identities.json", "selection.json", "player.npz", "events.json",
                           "annotated.mp4", "analytics.json", "analysis.json")


def run(job_id: str, job_dir: Path, start_stage: str = "tracking") -> None:
    with track._gpu:  # one job at a time: the detector and GPU are shared
        _run(job_id, Path(job_dir), start_stage)


def _identities(job_dir: Path) -> dict[int, dict]:
    f = job_dir / "identities.json"
    return {int(k): v for k, v in json.loads(f.read_text()).items()} if f.exists() else {}


def _run(job_id: str, job_dir: Path, start_stage: str) -> None:
    meta = db.get(job_id)["meta"]
    cfg = Config.from_dict(meta.get("config"))
    p = meta["proxy"]
    w, h = p["width"], p["height"]
    proxy = job_dir / "proxy.mp4"

    def progress(x: float) -> None:
        db.update(job_id, progress=round(x, 3))

    stage = start_stage
    try:
        for stage in STAGES[STAGES.index(start_stage):]:
            db.update(job_id, status=stage, progress=0, error=None)
            log.info("job %s: %s", job_id, stage)
            if stage == "tracking":
                from portalcut.detect import RFDETRDetector
                res = track.track(proxy, w, h, p["duration_sec"], cfg.detect_fps, progress,
                                  detector=RFDETRDetector(cfg.person_conf, cfg.ball_conf),
                                  tracker=cfg.tracker, lost_seconds=cfg.lost_seconds,
                                  tracker_params=parse_params(cfg.tracker_params))
                track.save(res["rows"], job_dir / "tracks.npz")
                for name in INVALIDATED_BY_TRACKING:  # new track IDs make all of these stale
                    (job_dir / name).unlink(missing_ok=True)
                for key in ("selection", "plays", "numbers", "identities", "annotated", "analytics", "analysis"):
                    meta.pop(key, None)
                meta["tracking"] = res["stats"]
            elif stage == "live":
                from portalcut import live
                tr = track.load(job_dir / "tracks.npz")
                frames = live.live_frames(tr, h, cfg.detect_fps)
                meta["live"] = live.summary(tr, h, cfg.detect_fps)
                (job_dir / "live.json").write_text(json.dumps({"frames": sorted(frames), "summary": meta["live"]}))
            elif stage == "features":
                from portalcut import reid
                meta["features"] = reid.build_features(job_dir, w, h, cfg.detect_fps, progress)
            elif stage == "numbers":
                from portalcut import events, jersey, reid
                if cfg.jersey_numbers and events.api_key_present():
                    sel = reid.Selection(job_dir)
                    tids = [int(t) for t, n in zip(sel.tids, sel.n) if n >= reid.MIN_TRACK_LEN]
                    meta["numbers"] = jersey.read_numbers(job_dir, tids)
                else:
                    meta["numbers"] = {"skipped": "jersey numbers off or ANTHROPIC_API_KEY not set"}
            elif stage == "identities":
                from portalcut import identity
                meta["identities"] = identity.build(job_dir)
            elif stage == "annotating":
                from portalcut import annotate
                lf = job_dir / "live.json"
                meta["annotated"] = annotate.render(proxy, w, h, track.load(job_dir / "tracks.npz"),
                                                    _identities(job_dir), cfg.detect_fps, job_dir / "annotated.mp4",
                                                    out_fps=cfg.annotate_fps, on_progress=progress,
                                                    live_frames=set(json.loads(lf.read_text())["frames"]) if lf.exists() else None)
            elif stage == "analytics":
                from portalcut import analytics
                a = analytics.run(job_dir, meta)
                meta["analytics"] = {"players": len(a["players"]), "units": a["units"], "calibrated": a["calibrated"]}
            elif stage == "analysis" and cfg.gemini:
                from portalcut import analysis
                try:
                    r = analysis.analyze(job_dir, meta, interval_s=cfg.gemini_interval_s)
                    meta["analysis"] = {k: r.get(k) for k in ("model", "frames_sent", "requests", "input_tokens",
                                                              "output_tokens", "cost_usd")}
                except Exception as e:  # optional layer: record the problem, keep the job usable
                    log.warning("job %s: Gemini analysis failed: %s", job_id, e)
                    meta["analysis"] = {"error": f"{type(e).__name__}: {e}"[:500]}
            db.update(job_id, meta=meta)
        db.update(job_id, status="ready", progress=1.0, meta=meta)
    except Exception as e:  # surface any failure in the UI instead of a stuck job
        log.exception("job %s failed in %s", job_id, stage)
        db.update(job_id, status="failed", error=f"{stage}: {type(e).__name__}: {e}"[:1000], meta=meta)
