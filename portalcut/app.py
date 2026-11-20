import json
import os
import shutil
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse

from pydantic import BaseModel

from portalcut import db, events, ingest, jersey, reid, track

ALLOWED_EXT = {".mp4", ".mov", ".m4v", ".mkv"}
STATIC = Path(__file__).parent / "static"



def _load_env(path: Path = Path(__file__).resolve().parent.parent / ".env") -> None:
    """KEY=VALUE lines from .env into the environment; real environment variables win."""
    if path.exists():
        for line in path.read_text().splitlines():
            k, sep, v = line.strip().partition("=")
            if sep and k and not k.startswith("#"):
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env()
app = FastAPI(title="PortalCut")
db.init()


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})  # always pick up app updates


@app.post("/api/jobs")
async def upload(request: Request, filename: str):
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"unsupported file type {ext or '(none)'}; use {sorted(ALLOWED_EXT)}")
    job_id = uuid.uuid4().hex[:12]
    job_dir = db.DATA_DIR / job_id
    job_dir.mkdir(parents=True)
    src = job_dir / f"source{ext}"  # never use the client's filename as a path

    # Stream the raw body to disk: game files are 8-20 GB, too big to buffer in memory.
    size = 0
    with src.open("wb") as f:
        async for chunk in request.stream():
            f.write(chunk)
            size += len(chunk)
    if size == 0:
        src.unlink()
        job_dir.rmdir()
        raise HTTPException(400, "empty upload")

    db.create(job_id, Path(filename).name)
    threading.Thread(target=ingest.run, args=(job_id, job_dir, src), daemon=True).start()
    return db.get(job_id)


@app.get("/api/jobs")
def list_jobs():
    return db.all_jobs()


def _job(job_id: str) -> dict:
    job = db.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    return _job(job_id)


@app.delete("/api/jobs/{job_id}")
def remove(job_id: str):
    # shortcut: no cancel for a running ingest (ffmpeg would keep writing); add one if people need it.
    if _job(job_id)["status"] not in db.FINAL:
        raise HTTPException(409, "still processing; remove it once it finishes")
    db.delete(job_id)
    shutil.rmtree(db.DATA_DIR / job_id, ignore_errors=True)
    return {"deleted": job_id}


@app.post("/api/jobs/{job_id}/track")
def retrack(job_id: str):
    """Run (or re-run) tracking on a job whose ingest already finished, e.g. one uploaded before tracking existed."""
    job = _job(job_id)
    if job["status"] not in db.FINAL or not (job["meta"] or {}).get("proxy"):
        raise HTTPException(409, "job is busy or was never ingested")
    db.update(job_id, status="ingested", progress=0, error=None)
    threading.Thread(target=track.run, args=(job_id, db.DATA_DIR / job_id), daemon=True).start()
    return db.get(job_id)


def _ready_for_selection(job_id: str) -> Path:
    job = _job(job_id)
    if job["status"] != "ready" or not (job["meta"] or {}).get("features"):
        raise HTTPException(409, "tracking has not finished for this video")
    return db.DATA_DIR / job_id


NUMBERS_BUDGET = 0.25


def _readable_tracks(sel: reid.Selection) -> list[int]:
    return [int(t) for t, n in zip(sel.tids, sel.n) if n >= reid.MIN_TRACK_LEN]


def _selection_payload(sel: reid.Selection) -> dict:
    n = len(_readable_tracks(sel))  # measured: ~$0.00026 per batch of 10 torso strips
    return {"summary": sel.summary(), "numbers": {"tracks": n, "estimate_usd": round(n / jersey.BATCH * 0.0003, 3),
                                                   "api_key": events.api_key_present()}}


def _read_numbers(job_id: str, job_dir: Path) -> None:
    try:
        stats = jersey.read_numbers(job_dir, _readable_tracks(reid.Selection(job_dir)), budget=NUMBERS_BUDGET)
        meta = db.get(job_id)["meta"]
        meta["numbers"] = stats
        db.update(job_id, status="ready", progress=1.0, meta=meta)
    except Exception as e:  # selection still works without numbers
        db.update(job_id, status="ready", progress=1.0, error=f"reading numbers: {type(e).__name__}: {e}"[:1000])


@app.post("/api/jobs/{job_id}/numbers")
def read_numbers(job_id: str):
    job_dir = _ready_for_selection(job_id)
    if not events.api_key_present():
        raise HTTPException(400, "Add ANTHROPIC_API_KEY=... to the .env file in the project folder, then restart the app")
    db.update(job_id, status="reading_numbers", progress=0, error=None)
    threading.Thread(target=_read_numbers, args=(job_id, job_dir), daemon=True).start()
    return db.get(job_id)


@app.get("/api/jobs/{job_id}/select")
def selection(job_id: str):
    sel = reid.Selection(_ready_for_selection(job_id))
    return _selection_payload(sel) | {"frames": sel.frames_payload()}


class Answer(BaseModel):
    track_id: int
    yes: bool


@app.post("/api/jobs/{job_id}/select/answer")
def answer(job_id: str, a: Answer):
    sel = reid.Selection(_ready_for_selection(job_id))
    try:
        switched = sel.answer(a.track_id, a.yes)
    except KeyError:
        raise HTTPException(404, "unknown track")
    sel.save()
    return _selection_payload(sel) | {"switched": switched}


@app.post("/api/jobs/{job_id}/select/undo")
def undo_switch(job_id: str):
    sel = reid.Selection(_ready_for_selection(job_id))
    if not sel.undo():
        raise HTTPException(409, "nothing to undo")
    sel.save()
    return _selection_payload(sel)


@app.post("/api/jobs/{job_id}/select/reset")
def reset_selection(job_id: str):
    job_dir = _ready_for_selection(job_id)
    (job_dir / "selection.json").unlink(missing_ok=True)
    meta = _job(job_id)["meta"]
    if meta.pop("selection", None) is not None:
        db.update(job_id, meta=meta)
    return _selection_payload(reid.Selection(job_dir))


def _finish(job_id: str, job_dir: Path) -> None:
    try:
        meta = db.get(job_id)["meta"]
        if meta.pop("selection", None) is not None:  # never show an old result beside half-rebuilt files
            db.update(job_id, meta=meta)
        p = meta["proxy"]
        meta["selection"] = reid.finish(job_dir, p["width"], p["height"],
                                        on_progress=lambda x: db.update(job_id, progress=round(x, 3)))
        db.update(job_id, status="ready", progress=1.0, meta=meta)
    except Exception as e:
        db.update(job_id, status="failed", error=f"selection: {type(e).__name__}: {e}"[:1000])


@app.post("/api/jobs/{job_id}/select/done")
def finish_selection(job_id: str):
    job_dir = _ready_for_selection(job_id)
    if not reid.Selection(job_dir).pos:
        raise HTTPException(400, "click your player in at least one frame first")
    db.update(job_id, status="selecting", progress=0)
    threading.Thread(target=_finish, args=(job_id, job_dir), daemon=True).start()
    return db.get(job_id)


def _selected(job_id: str) -> tuple[dict, Path]:
    job = _job(job_id)
    if job["status"] != "ready" or not (job["meta"] or {}).get("selection"):
        raise HTTPException(409, "pick your player first")
    return job, db.DATA_DIR / job_id


def _windows(job: dict, job_dir: Path, start: float, seconds: float | None) -> list[dict]:
    import numpy as np
    tr = track.load(job_dir / "tracks.npz")
    bm = tr["cls"] == track.BALL
    rms_file = job_dir / "audio_rms.json"
    return events.candidate_windows(dict(np.load(job_dir / "player.npz")), {"t": tr["t"][bm], "xyxy": tr["xyxy"][bm]},
                                    json.loads(rms_file.read_text()) if rms_file.exists() else [],
                                    job["meta"]["proxy"]["duration_sec"], start=start, seconds=seconds)


@app.get("/api/jobs/{job_id}/plays/estimate")
def plays_estimate(job_id: str, start: float = 0.0, seconds: float | None = None):
    job, job_dir = _selected(job_id)
    n = len(_windows(job, job_dir, start, seconds))
    return {"windows": n, "estimate_usd": round(events.estimate_cost(n), 4), "budget_usd": events.BUDGET_USD,
            "api_key": events.api_key_present()}


class PlaysRequest(BaseModel):
    start: float = 0.0
    seconds: float | None = None
    budget: float = events.BUDGET_USD


def _find_plays(job_id: str, job_dir: Path, req: PlaysRequest) -> None:
    try:
        meta = db.get(job_id)["meta"]
        meta["plays"] = events.detect(job_dir, meta, start=req.start, seconds=req.seconds, budget=req.budget,
                                      on_progress=lambda x: db.update(job_id, progress=round(x, 3)))
        db.update(job_id, status="ready", progress=1.0, meta=meta)
    except Exception as e:
        # the player selection is intact and paid answers are cached, so leave the job usable
        db.update(job_id, status="ready", progress=1.0, error=f"finding plays: {type(e).__name__}: {e}"[:1000])


@app.post("/api/jobs/{job_id}/plays")
def find_plays(job_id: str, req: PlaysRequest):
    _, job_dir = _selected(job_id)
    if not events.api_key_present():
        raise HTTPException(400, "Add ANTHROPIC_API_KEY=... to the .env file in the project folder, then restart the app")
    if not 0 < req.budget <= 5:
        raise HTTPException(400, "budget must be between $0 and $5")
    db.update(job_id, status="finding_plays", progress=0, error=None)
    threading.Thread(target=_find_plays, args=(job_id, job_dir, req), daemon=True).start()
    return db.get(job_id)


@app.get("/api/jobs/{job_id}/plays")
def plays(job_id: str):
    f = db.DATA_DIR / _job(job_id)["id"] / "events.json"
    return json.loads(f.read_text()) if f.exists() else []


@app.get("/api/jobs/{job_id}/frames/{frame}")
def frame_image(job_id: str, frame: int):
    return _image(db.DATA_DIR / _job(job_id)["id"] / "frames" / f"{frame}.jpg")


@app.get("/api/jobs/{job_id}/crops/{track_id}")
def crop_image(job_id: str, track_id: int):
    return _image(db.DATA_DIR / _job(job_id)["id"] / "crops" / f"{track_id}.jpg")


NO_STORE = {"Cache-Control": "no-store"}  # these files are rebuilt in place; a cached copy would be stale


def _image(path: Path):
    if not path.exists():  # path parts are ints or a job id from the database: no traversal possible
        raise HTTPException(404, "image not found")
    return FileResponse(path, media_type="image/jpeg", headers=NO_STORE)


@app.get("/api/jobs/{job_id}/debug_player")
def debug_player(job_id: str):
    path = db.DATA_DIR / job_id / "debug_player.mp4"
    if _job(job_id)["status"] != "ready" or not path.exists():
        raise HTTPException(409, "player video not ready")
    return FileResponse(path, media_type="video/mp4", headers=NO_STORE)


@app.get("/api/jobs/{job_id}/debug")
def debug_video(job_id: str):
    path = db.DATA_DIR / job_id / "debug_tracks.mp4"
    if _job(job_id)["status"] != "ready" or not path.exists():
        raise HTTPException(409, "debug video not ready")
    return FileResponse(path, media_type="video/mp4", headers=NO_STORE)


@app.get("/api/jobs/{job_id}/proxy")
def proxy(job_id: str):
    if _job(job_id)["status"] != "ready":
        raise HTTPException(409, "proxy not ready")
    return FileResponse(db.DATA_DIR / job_id / "proxy.mp4", media_type="video/mp4")


def main() -> None:
    import uvicorn

    ingest.check_tools()
    print("PortalCut running at http://localhost:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)  # local only: no auth in v1
