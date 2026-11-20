"""Pass 1: detect people and the ball on the 720p proxy, track people across frames.

Output: data/<job>/tracks.npz with one row per detection:
  frame (sample index), t (seconds), track_id (-1 for the ball), cls (0 person, 1 ball), xyxy, conf.
"""

import subprocess
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SAMPLE_FPS = 5          # detections per second of video; tracks are re-linked later by re-ID
BATCH = 8
PERSON_CONF = 0.4
BALL_CONF = 0.25        # COCO's ball class is weak on basketball; keep low, it's only a ranking signal later
DEBUG_SECONDS = 60
PERSON, BALL = 0, 1
PALETTE = [(234, 88, 12), (37, 99, 235), (22, 163, 74), (219, 39, 119), (202, 138, 4), (124, 58, 237), (8, 145, 178)]

_gpu = threading.Lock()  # one tracking job at a time: the model and GPU are shared


def read_frames(video: Path, w: int, h: int, fps: float, seconds: float | None = None, start: float = 0.0):
    """Yield RGB frames sampled at `fps`, decoded by ffmpeg (handles phone rotation and HEVC)."""
    cmd = ["ffmpeg", "-v", "error"] + (["-ss", f"{start:.3f}"] if start else []) + ["-i", str(video)]
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


def make_tracker(name: str, fps: float, lost_seconds: float):
    """ByteTrack, BoT-SORT (adds camera-motion compensation from the frames) or OC-SORT."""
    from trackers import BoTSORTTracker, ByteTrackTracker, OCSORTTracker
    # the trackers library counts lost_track_buffer in 30 FPS frames whatever the real rate:
    # passing fps*3 (=15 at 5 fps) meant 0.5 s, so every longer occlusion started a new ID
    buf = round(30 * lost_seconds)
    cls = {"bytetrack": ByteTrackTracker, "botsort": BoTSORTTracker, "ocsort": OCSORTTracker}[name]
    return cls(frame_rate=fps, lost_track_buffer=buf)


def detect_all(proxy: Path, w: int, h: int, fps: float, detector, start: float = 0.0, seconds: float | None = None,
               on_progress=lambda p: None) -> list:
    """Detections for every sampled frame (cacheable, so trackers can be compared on identical input)."""
    total = max(1, int((seconds or 1e9) * fps))
    out = []
    for batch in _batches(read_frames(proxy, w, h, fps, seconds, start), BATCH):
        out += detector.detect(batch)
        on_progress(min(1.0, len(out) / total))
    return out


def track(proxy: Path, w: int, h: int, duration: float, fps: float = SAMPLE_FPS, on_progress=lambda p: None,
          detector=None, tracker: str = "bytetrack", lost_seconds: float = 3.0, detections: list | None = None,
          start: float = 0.0, seconds: float | None = None) -> dict:
    """Detect (unless `detections` are given) and track. Frames are always decoded: BoT-SORT needs them."""
    import supervision as sv
    from portalcut.detect import RFDETRDetector

    detector = detector or RFDETRDetector(PERSON_CONF, BALL_CONF)
    trk = make_tracker(tracker, fps, lost_seconds)
    span = seconds or (duration - start)
    total = max(1, int(span * fps))
    rows = []  # (frame, t, track_id, cls, x1, y1, x2, y2, conf)
    i = 0
    t0 = time.time()
    for batch in _batches(read_frames(proxy, w, h, fps, seconds, start), BATCH):
        dets = detections[i:i + len(batch)] if detections is not None else detector.detect(batch)
        for frame, det in zip(batch, dets):
            tracked = trk.update(sv.Detections(xyxy=det.people_xyxy.reshape(-1, 4), confidence=det.people_conf,
                                               class_id=np.zeros(len(det.people_conf), int)), frame)
            t = start + i / fps
            fr = int(round(t * fps))
            for box, tid, c in zip(tracked.xyxy, tracked.tracker_id, tracked.confidence):
                if tid >= 0:  # -1 = not yet confirmed
                    rows.append((fr, t, int(tid), PERSON, *box, c))
            if len(det.ball_conf):  # at most one ball on court: keep the most confident
                b = int(np.argmax(det.ball_conf))
                rows.append((fr, t, -1, BALL, *det.ball_xyxy[b], det.ball_conf[b]))
            i += 1
        on_progress(min(1.0, i / total))
    elapsed = time.time() - t0

    arr = np.array(rows, dtype=np.float32).reshape(-1, 9)
    people = arr[arr[:, 3] == PERSON]
    stats = {
        "model": detector.name,
        "device": detector.device if hasattr(detector, "device") else "",
        "tracker": tracker,
        "lost_seconds": lost_seconds,
        "sample_fps": fps,
        "frames": i,
        "seconds": round(elapsed, 1),
        "x_realtime": round(elapsed / max(span, 1e-6), 3),  # <1 means faster than the video plays
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
