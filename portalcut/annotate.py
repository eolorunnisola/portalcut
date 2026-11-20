"""Annotated video: player boxes, persistent IDs, team colours, confidence, trails and the ball.

Tracks are sampled at ``sample_fps``; the output runs at a higher ``out_fps``, so boxes are
linearly interpolated between samples (never across gaps longer than two samples).
"""
from __future__ import annotations

import subprocess
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

TEAM_COLORS = {0: (0, 140, 255), 1: (255, 120, 30), 2: (160, 160, 160)}
UNKNOWN_COLOR = (80, 200, 80)
BALL_COLOR = (0, 255, 255)
MAX_GAP = 2  # samples


def interpolate(tracks: dict, t: float, sample_fps: float) -> list[dict]:
    s = t * sample_fps
    f_lo, f_hi = int(np.floor(s + 1e-6)), int(np.ceil(s - 1e-6))
    near = (tracks["cls"] == 0) & (tracks["frame"] >= f_lo - MAX_GAP) & (tracks["frame"] <= f_hi + MAX_GAP)
    out = []
    for tid in np.unique(tracks["track_id"][near]):
        m = near & (tracks["track_id"] == tid)
        fr, box, conf = tracks["frame"][m], tracks["xyxy"][m], tracks["conf"][m]
        exact = np.nonzero(np.abs(fr - s) < 1e-3)[0]
        if len(exact):
            i = exact[0]
            out.append({"track_id": int(tid), "xyxy": box[i], "conf": float(conf[i])})
            continue
        before, after = np.nonzero(fr < s)[0], np.nonzero(fr > s)[0]
        if not len(before) or not len(after):
            continue
        a, b = before[np.argmax(fr[before])], after[np.argmin(fr[after])]
        if fr[b] - fr[a] > MAX_GAP:
            continue
        w = (s - fr[a]) / (fr[b] - fr[a])
        out.append({"track_id": int(tid), "xyxy": (1 - w) * box[a] + w * box[b],
                    "conf": float((1 - w) * conf[a] + w * conf[b])})
    return out


def ball_at(tracks: dict, t: float, sample_fps: float) -> np.ndarray | None:
    m = tracks["cls"] == 1
    if not m.any():
        return None
    dt = np.abs(tracks["t"][m] - t)
    i = int(np.argmin(dt))
    return tracks["xyxy"][m][i] if dt[i] <= 1.0 / sample_fps + 1e-6 else None


def _color(ident: dict | None) -> tuple[int, int, int]:
    team = ident.get("team") if ident else None
    return TEAM_COLORS.get(team, UNKNOWN_COLOR)


def draw(frame: np.ndarray, people: list[dict], ball, identities: dict, trails: dict) -> np.ndarray:
    pid = {p["track_id"]: (identities.get(p["track_id"]) or {}).get("player_id", f"trk-{p['track_id']}")
           for p in people}
    for p in people:  # trails first so boxes stay on top
        pts = list(trails.get(pid[p["track_id"]], ()))
        if len(pts) > 1:
            col = _color(identities.get(p["track_id"]))
            for k in range(1, len(pts)):
                c = tuple(int(v * (0.35 + 0.65 * k / len(pts))) for v in col)
                cv2.line(frame, pts[k - 1], pts[k], c, 1 if k < len(pts) // 2 else 2, cv2.LINE_AA)
    for p in people:
        x1, y1, x2, y2 = (int(round(v)) for v in p["xyxy"])
        col = _color(identities.get(p["track_id"]))
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        ident = identities.get(p["track_id"]) or {}
        name = f"{ident['number']} {ident['name'].split()[-1]}" if ident.get("name") else pid[p["track_id"]]
        label = f"{name} {p['conf']:.2f}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(y1 - th - base - 4, 0)
        cv2.rectangle(frame, (x1, top), (x1 + tw + 6, top + th + base + 4), col, -1)
        cv2.putText(frame, label, (x1 + 3, top + th + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1, cv2.LINE_AA)
    if ball is not None:
        cx, cy = int((ball[0] + ball[2]) / 2), int((ball[1] + ball[3]) / 2)
        r = max(int(max(ball[2] - ball[0], ball[3] - ball[1]) / 2), 5)
        cv2.circle(frame, (cx, cy), r + 2, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), r + 2, BALL_COLOR, 2, cv2.LINE_AA)
    return frame


def render(proxy, w: int, h: int, tracks: dict, identities: dict, sample_fps: float, out_path,
           out_fps: int = 10, start: float = 0.0, seconds: float | None = None,
           trail_seconds: float = 2.0, on_progress: Callable[[float], None] = lambda p: None,
           live_frames: set[int] | None = None) -> dict:
    """live_frames: sample-frame indices of live play; elsewhere (close-ups, bench) nothing is drawn."""
    t0 = time.time()
    dec_cmd = ["ffmpeg", "-v", "error", "-ss", str(start), "-i", str(proxy)]
    if seconds is not None:
        dec_cmd += ["-t", str(seconds)]
    dec_cmd += ["-vf", f"fps={out_fps}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    enc_cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
               "-r", str(out_fps), "-i", "-", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
               "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(out_path)]
    total = seconds if seconds is not None else max(float(tracks["t"].max()) - start, 0.0) if len(tracks["t"]) else 0.0
    trails: dict[str, deque] = defaultdict(deque)
    dec = subprocess.Popen(dec_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    n, size = 0, w * h * 3
    try:
        while len(buf := dec.stdout.read(size)) == size:
            t = start + n / out_fps
            frame = np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()
            live = live_frames is None or int(round(t * sample_fps)) in live_frames
            people = interpolate(tracks, t, sample_fps) if live else []
            if not live:
                trails.clear()  # a cut away from live play: don't connect trails across it
                cv2.putText(frame, "not live play", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
            for p in people:
                pid = (identities.get(p["track_id"]) or {}).get("player_id", f"trk-{p['track_id']}")
                x1, _, x2, y2 = p["xyxy"]
                q = trails[pid]
                q.append((t, (int((x1 + x2) / 2), int(y2))))
                while q[0][0] < t - trail_seconds:
                    q.popleft()
            pts = {pid: [xy for _, xy in q] for pid, q in trails.items()}
            ball = ball_at(tracks, t, sample_fps) if live else None
            enc.stdin.write(draw(frame, people, ball, identities, pts).tobytes())
            n += 1
            if total:
                on_progress(min(n / out_fps / total, 1.0))
    finally:
        dec.stdout.close()
        dec.kill()
        dec.wait()
        enc.stdin.close()
        err = enc.stderr.read().decode(errors="replace")
        enc.stderr.close()
        rc = enc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg encoder failed ({rc}): {err}")
    return {"frames": n, "seconds": time.time() - t0, "out_fps": out_fps}
