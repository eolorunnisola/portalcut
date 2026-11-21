"""Process videos without the web server, e.g. on a Google Colab GPU.

    python -m portalcut.batch game.mp4 [more.mp4 ...] --out /content/drive/MyDrive/portalcut/jobs \\
        --work /content/work --set tracker=bytetrack --set lost_seconds=1 --roster rosters/game.json

Each video becomes a job folder (same layout as the app's data/<job>/) plus job.json with the job's
status and every stage's statistics. Copy or sync the folder to your Mac and register it in the app with
`python -m portalcut.import_job <folder>`. The summary printed at the end is what to paste back to Claude.
"""

import argparse
import json
import logging
import os
import shutil
import sys
import threading
import time
import uuid
from pathlib import Path

from portalcut import db, ingest
from portalcut.config import Config

SUMMARY_KEYS = ("tracking", "live", "features", "numbers", "identities", "annotated", "analytics", "analysis")


def summary(job: dict) -> dict:
    """The compact part of a job worth reading: status, error and each stage's stats (no bulky fields)."""
    m = job.get("meta") or {}
    return {"id": job["id"], "file": job["filename"], "status": job["status"], "error": job["error"],
            "duration_s": m.get("duration_sec"), "config": m.get("config"),
            **{k: m[k] for k in SUMMARY_KEYS if k in m}}


def _report_progress(job_id: str, stop: threading.Event, every_s: float = 30.0) -> None:
    """Print status every so often: a 2-hour game is otherwise silent for an hour."""
    t0 = time.time()
    while not stop.wait(every_s):
        j = db.get(job_id)
        print(f"  [{time.time() - t0:6.0f}s] {j['status']} {round(100 * j['progress'])}%", flush=True)


def run(video: Path, out_dir: Path, config: dict, roster: Path | None = None, work_dir: Path | None = None,
        keep_source: bool = False) -> dict:
    video, out_dir = Path(video).resolve(), Path(out_dir)
    work = Path(work_dir or out_dir)
    cfg = Config.from_dict(config)  # validate before spending any time
    db.DATA_DIR, db.DB_PATH = work, work / "portalcut.db"
    db.init()
    job_id = uuid.uuid4().hex[:12]
    job_dir = work / job_id
    job_dir.mkdir(parents=True)
    src = job_dir / f"source{video.suffix.lower()}"
    os.symlink(video, src)  # no multi-GB copy; replaced by a real file only if --keep-source
    if roster:
        shutil.copy(roster, job_dir / "roster.json")
    db.create(job_id, video.name)
    db.update(job_id, meta={"config": cfg.to_dict()})
    stop = threading.Event()
    threading.Thread(target=_report_progress, args=(job_id, stop), daemon=True).start()
    try:
        ingest.run(job_id, job_dir, src)  # ingest, then every pipeline stage, synchronously
    finally:
        stop.set()
    job = db.get(job_id)
    src.unlink()
    if keep_source:
        shutil.copy(video, src)
    (job_dir / "job.json").write_text(json.dumps(job, indent=1))
    if work.resolve() != out_dir.resolve():
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(job_dir, out_dir / job_id)
    return summary(job)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("videos", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True, help="where finished job folders go (e.g. Google Drive)")
    ap.add_argument("--work", type=Path, help="fast scratch folder for processing (e.g. /content/work)")
    ap.add_argument("--roster", type=Path, help="rosters JSON: {\"teams\": [{\"name\", \"players\": {num: name}}, ...]}")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="processing setting, repeatable")
    ap.add_argument("--keep-source", action="store_true", help="also copy the original video into the job folder")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    config = dict(kv.split("=", 1) for kv in a.set)
    results = []
    for v in a.videos:
        print(f"== {v.name}", flush=True)
        results.append(run(v, a.out, config, a.roster, a.work, a.keep_source))
        print(json.dumps(results[-1], indent=1), flush=True)
    print("\n=== SUMMARY (paste this back to Claude) ===\n" + json.dumps(results, separators=(",", ":")))
    return 0 if all(r["status"] == "ready" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
