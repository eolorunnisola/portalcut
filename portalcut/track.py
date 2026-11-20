"""Pass 1: detect people and the ball on the 720p proxy, track people across frames.

Output: data/<job>/tracks.npz with one row per detection:
  frame (sample index), t (seconds), track_id (-1 for the ball), cls (0 person, 1 ball), xyxy, conf.
"""

import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from portalcut import db

SAMPLE_FPS = 5          # detections per second of video; tracks are re-linked later by re-ID
BATCH = 8
PERSON_CONF = 0.4
BALL_CONF = 0.25        # COCO's ball class is weak on basketball; keep low, it's only a ranking signal later
DEBUG_SECONDS = 60
PERSON, BALL = 0, 1
PALETTE = [(234, 88, 12), (37, 99, 235), (22, 163, 74), (219, 39, 119), (202, 138, 4), (124, 58, 237), (8, 145, 178)]

_model = None
_gpu = threading.Lock()  # one tracking job at a time: the model and GPU are shared


def _load_model():
    global _model
    if _model is None:
        from rfdetr import RFDETRMedium  # heavy import; only when tracking runs
        _model = RFDETRMedium()  # picks MPS on Apple Silicon, CUDA if present, else CPU
    return _model


def read_frames(video: Path, w: int, h: int, fps: float, seconds: float | None = None):
    """Yield RGB frames sampled at `fps`, decoded by ffmpeg (no OpenCV needed)."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(video)]
    if seconds:
        cmd += ["-t", str(seconds)]
    cmd += ["-vf", f"fps={fps}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    size = w * h * 3
    try:
        while (buf := proc.stdout.read(size)) and len(buf) == size:
            yield np.frombuffer(bytearray(buf), np.uint8).reshape(h, w, 3)  # writable, as torch expects
    finally:
        if proc.poll() is None:
            proc.kill()  # caller stopped early; without this ffmpeg reports a broken pipe
        proc.stdout.close()
        proc.wait()


def _batches(it, n):
    batch = []
    for x in it:
        batch.append(x)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def track(proxy: Path, w: int, h: int, duration: float, fps: float = SAMPLE_FPS, on_progress=lambda p: None) -> dict:
    import supervision as sv
    from rfdetr import ClassThresholds
    from trackers import ByteTrackTracker

    model = _load_model()
    tracker = ByteTrackTracker(frame_rate=fps, lost_track_buffer=int(fps * 3))
    thresholds = ClassThresholds({"person": PERSON_CONF, "sports ball": BALL_CONF}, default=float("inf"))
    rows = []  # (frame, t, track_id, cls, x1, y1, x2, y2, conf)
    total = max(1, int(duration * fps))
    i = 0
    t0 = time.time()
    for batch in _batches(read_frames(proxy, w, h, fps), BATCH):
        for det in model.predict(batch, threshold=thresholds, include_source_image=False):
            names = det.data["class_name"]
            people = det[names == "person"]
            tracked = tracker.update(sv.Detections(xyxy=people.xyxy, confidence=people.confidence,
                                                   class_id=np.zeros(len(people), int)))
            t = i / fps
            for box, tid, c in zip(tracked.xyxy, tracked.tracker_id, tracked.confidence):
                if tid >= 0:  # -1 = not yet confirmed
                    rows.append((i, t, int(tid), PERSON, *box, c))
            balls = det[names == "sports ball"]
            if len(balls):  # at most one ball on court: keep the most confident
                b = int(np.argmax(balls.confidence))
                rows.append((i, t, -1, BALL, *balls.xyxy[b], balls.confidence[b]))
            i += 1
        on_progress(min(1.0, i / total))
    elapsed = time.time() - t0

    arr = np.array(rows, dtype=np.float32).reshape(-1, 9)
    people = arr[arr[:, 3] == PERSON]
    stats = {
        "model": "rf-detr-medium",
        "device": str(model.model.device),
        "sample_fps": fps,
        "frames": i,
        "seconds": round(elapsed, 1),
        "x_realtime": round(elapsed / max(duration, 1e-6), 3),  # <1 means faster than the video plays
        "tracks": int(len(np.unique(people[:, 2]))),
        "people_per_frame": round(len(people) / max(i, 1), 2),
        "ball_found_pct": round(100 * len(np.unique(arr[arr[:, 3] == BALL][:, 0])) / max(i, 1), 1),
    }
    return {"rows": arr, "stats": stats}


def save(rows: np.ndarray, path: Path) -> None:
    np.savez_compressed(path, frame=rows[:, 0].astype(np.int32), t=rows[:, 1], track_id=rows[:, 2].astype(np.int32),
                        cls=rows[:, 3].astype(np.int8), xyxy=rows[:, 4:8], conf=rows[:, 8])


def load(path: Path) -> dict:
    return dict(np.load(path))


def debug_video(proxy: Path, tracks: dict, w: int, h: int, fps: float, out: Path, seconds: float = DEBUG_SECONDS,
                label: str | None = None) -> None:
    """First `seconds` of the proxy at the sample rate with boxes drawn, labelled by track ID or `label`."""
    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
         "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(out)],
        stdin=subprocess.PIPE,
    )
    for i, frame in enumerate(read_frames(proxy, w, h, fps, seconds)):
        img = Image.fromarray(frame)
        d = ImageDraw.Draw(img)
        m = tracks["frame"] == i
        cls_m = tracks["cls"][m] if "cls" in tracks else np.full(m.sum(), PERSON)
        for (x1, y1, x2, y2), tid, cls in zip(tracks["xyxy"][m], tracks["track_id"][m], cls_m):
            if cls == BALL:
                d.ellipse([x1 - 3, y1 - 3, x2 + 3, y2 + 3], outline=(255, 255, 0), width=3)
                continue
            col = PALETTE[0 if label else tid % len(PALETTE)]
            text = label or f"#{tid}"
            d.rectangle([x1, y1, x2, y2], outline=col, width=3)
            d.rectangle([x1, y1 - 18, x1 + 8 + 7 * len(text), y1], fill=col)
            d.text((x1 + 4, y1 - 16), text, fill=(255, 255, 255))
        d.text((8, 8), f"t={i / fps:.1f}s", fill=(255, 255, 255))
        enc.stdin.write(np.asarray(img).tobytes())
    enc.stdin.close()
    if enc.wait() != 0:
        raise RuntimeError("debug video encode failed")


def run(job_id: str, job_dir: Path) -> None:
    with _gpu:
        _run(job_id, job_dir)


def _run(job_id: str, job_dir: Path) -> None:
    try:
        meta = db.get(job_id)["meta"]
        p = meta["proxy"]
        db.update(job_id, status="tracking", progress=0, error=None)
        res = track(job_dir / "proxy.mp4", p["width"], p["height"], p["duration_sec"],
                    on_progress=lambda x: db.update(job_id, progress=round(x, 3)))
        save(res["rows"], job_dir / "tracks.npz")
        db.update(job_id, status="debug_video", progress=0)
        debug_video(job_dir / "proxy.mp4", load(job_dir / "tracks.npz"), p["width"], p["height"],
                    SAMPLE_FPS, job_dir / "debug_tracks.mp4")
        meta["tracking"] = res["stats"]
        db.update(job_id, status="features", progress=0, meta=meta)
        from portalcut import reid  # reid imports this module
        meta["features"] = reid.build_features(job_dir, p["width"], p["height"],
                                               on_progress=lambda x: db.update(job_id, progress=round(x, 3)))
        meta.pop("selection", None)  # new tracks invalidate any earlier selection
        (job_dir / "selection.json").unlink(missing_ok=True)
        db.update(job_id, status="ready", progress=1.0, meta=meta)
    except Exception as e:  # surface any failure in the UI instead of a stuck job
        db.update(job_id, status="failed", error=f"tracking: {type(e).__name__}: {e}"[:1000])


if __name__ == "__main__":
    # Re-run tracking on an already-ingested job: uv run python -m portalcut.track <job_id>
    jid = sys.argv[1]
    run(jid, db.DATA_DIR / jid)
    job = db.get(jid)
    print(job["status"], job["error"] or job["meta"]["tracking"])
