"""Per-player movement statistics from tracking output.

Without calibration everything is in image pixels (screen positions, not court positions), so
distances and speeds are rough estimates. With a 4-point calibration, foot points are mapped
through one homography onto the playing surface, which is only approximate if the camera moves.
"""
import json
from pathlib import Path

import cv2
import numpy as np

from . import track

PERSON = 0
NOTE_PX = ("Distances and speeds are estimates in image pixels. They are not comparable across camera "
           "moves or to real-world units; calibrate the playing surface to get metres.")
NOTE_CAL = ("Positions are converted to surface units with a single homography, so values are "
            "approximate whenever the camera moves.")
COLS, ROWS = ("left", "centre", "right"), ("far", "middle", "near")


def homography(cal: dict) -> np.ndarray:
    src = np.asarray(cal["image_points"], dtype=np.float32)
    dst = np.asarray(cal["world_points"], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def to_world(points_px, H: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_px, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def _cell(p: np.ndarray, size: tuple[float, float], n: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Column and row index of each point on an n=(cols, rows) grid over size=(w, h), clipped to the edge."""
    ix = np.clip((p[:, 0] / size[0] * n[0]).astype(int), 0, n[0] - 1)
    iy = np.clip((p[:, 1] / size[1] * n[1]).astype(int), 0, n[1] - 1)
    return ix, iy


def _runs(t: np.ndarray, max_gap: float) -> list[slice]:
    cuts = [0, *(np.flatnonzero(np.diff(t) > max_gap) + 1), len(t)]
    return [slice(a, b) for a, b in zip(cuts[:-1], cuts[1:])]


def _smooth(p: np.ndarray) -> np.ndarray:
    out = p.copy()  # run ends stay raw so a straight path keeps its exact length
    if len(p) > 2:
        out[1:-1] = (p[:-2] + p[1:-1] + p[2:]) / 3
    return out


def _max_speed(t: np.ndarray, p: np.ndarray) -> float:
    steps = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    j = np.minimum(np.searchsorted(t, t + 1.0 - 1e-6), len(t) - 1)
    dt = t[j] - t
    ok = dt > 0
    if not ok.any():  # run shorter than one window: use the whole run
        return float(steps[-1] / (t[-1] - t[0])) if len(t) > 1 and t[-1] > t[0] else 0.0
    return float(((steps[j] - steps)[ok] / dt[ok]).max())


def _player(pid: str, ident: dict, rows: list, size, fps, H, surface, grid, max_gap) -> dict:
    best: dict[int, tuple] = {}  # frame -> (conf, t, x, y, track_id)
    for frame, t, tid, box, conf in rows:
        if frame not in best or conf > best[frame][0]:
            best[frame] = (conf, t, (box[0] + box[2]) / 2, box[3], tid)
    ordered = [best[f] for f in sorted(best)]
    t = np.array([r[1] for r in ordered], dtype=np.float64)
    raw = np.array([[r[2], r[3]] for r in ordered], dtype=np.float64)
    if H is not None:
        raw = to_world(raw, H)
    runs = _runs(t, max_gap)
    pos = np.vstack([_smooth(raw[s]) for s in runs])
    dist, vmax = 0.0, 0.0
    for s in runs:
        dist += float(np.hypot(*np.diff(pos[s], axis=0).T).sum())
        vmax = max(vmax, _max_speed(t[s], pos[s]))
    n = len(t)
    secs = n / fps
    ix, iy = _cell(pos, surface, grid)  # grid is (cols, rows)
    heat = np.zeros((grid[1], grid[0]))
    np.add.at(heat, (iy, ix), 1)
    zx, zy = _cell(pos, surface, (3, 3))
    zones = np.zeros((3, 3))
    np.add.at(zones, (zy, zx), 1 / fps)
    keep = np.unique(np.linspace(0, n - 1, min(n, 300)).round().astype(int))
    return {
        "player_id": pid, "team": ident.get("team"), "number": ident.get("number"),
        "name": ident.get("name"), "team_name": ident.get("team_name"),
        "track_ids": sorted({int(r[4]) for r in ordered}),
        "first_seen": round(float(t[0]), 2), "last_seen": round(float(t[-1]), 2),
        "visible_seconds": round(secs, 2), "samples": n, "runs": len(runs),
        "distance": round(dist, 2), "avg_speed": round(dist / secs, 3), "max_speed": round(vmax, 3),
        "avg_position": [round(float(v), 2) for v in pos.mean(axis=0)],
        "trajectory": [[round(float(t[i]), 2), round(float(pos[i, 0]), 2), round(float(pos[i, 1]), 2)] for i in keep],
        "heatmap": np.round(heat / heat.sum(), 4).tolist(),
        "zones": {f"{c}-{r}": round(float(zones[j, i]), 2) for j, r in enumerate(ROWS) for i, c in enumerate(COLS)},
        "mean_conf": round(float(np.mean([r[0] for r in ordered])), 3),
    }


def compute(tracks: dict, identities: dict, frame_size: tuple[int, int], sample_fps: float,
            calibration: dict | None = None, grid: tuple[int, int] = (10, 6), max_gap_s: float = 1.0) -> dict:
    H = homography(calibration) if calibration else None
    surface = tuple(calibration["surface"]) if calibration else tuple(frame_size)
    by_player: dict[str, list] = {}
    idents: dict[str, dict] = {}
    people = tracks["cls"] == PERSON
    for i in np.flatnonzero(people):
        tid = int(tracks["track_id"][i])
        ident = identities.get(tid) or identities.get(str(tid)) or {}
        pid = ident.get("player_id") or f"trk-{tid}"
        idents.setdefault(pid, ident)
        by_player.setdefault(pid, []).append(
            (int(tracks["frame"][i]), float(tracks["t"][i]), tid, tracks["xyxy"][i], float(tracks["conf"][i])))
    players = [_player(pid, idents[pid], rows, frame_size, sample_fps, H, surface, grid, max_gap_s)
               for pid, rows in by_player.items()]
    players.sort(key=lambda p: -p["visible_seconds"])
    ball = np.unique(tracks["frame"][tracks["cls"] == 1]) if len(tracks["cls"]) else []
    return {
        "units": calibration.get("units", "m") if calibration else "px",
        "coordinate_space": "surface" if calibration else "image", "calibrated": calibration is not None,
        "note": NOTE_CAL if calibration else NOTE_PX, "sample_fps": sample_fps, "players": players,
        "ball": {"samples": len(ball), "visible_seconds": round(len(ball) / sample_fps, 2)},
    }


def run(job_dir: Path, meta: dict) -> dict:
    job_dir = Path(job_dir)
    load = lambda name: json.loads((job_dir / name).read_text()) if (job_dir / name).exists() else None  # noqa: E731
    tracks = track.load(job_dir / "tracks.npz")
    live = load("live.json")
    if live is not None:  # close-ups and bench shots are not court positions: leave them out
        keep = np.isin(tracks["frame"], live["frames"])
        tracks = {k: v[keep] for k, v in tracks.items()}
    out = compute(tracks, load("identities.json") or {},
                  (meta["proxy"]["width"], meta["proxy"]["height"]),
                  meta.get("tracking", {}).get("sample_fps", 5), load("calibration.json"))
    if live is not None:
        out["live"] = live["summary"]
    (job_dir / "analytics.json").write_text(json.dumps(out))
    return out
