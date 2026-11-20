"""Live-action detection: which sampled frames show the wide court camera.

A frame is live when enough court-sized people are visible and nobody is close-up sized.
Close-ups, bench/crowd shots and graphics fail one of those tests. Replays are NOT detected:
they use the same wide camera, so they look live here.
"""
from __future__ import annotations

import numpy as np

PERSON = 0


def frame_scores(tracks: dict, h: int) -> dict[int, dict]:
    """Per sampled frame with a person: n_people, n_court_sized, max_height_frac."""
    m = tracks["cls"] == PERSON
    frames, box = tracks["frame"][m], tracks["xyxy"][m]
    hf = (box[:, 3] - box[:, 1]) / h
    out: dict[int, dict] = {}
    for f in np.unique(frames):
        x = hf[frames == f]
        out[int(f)] = {
            "n_people": int(x.size),
            "n_court_sized": int(((x >= 0.08) & (x <= 0.45)).sum()),
            "max_height_frac": float(x.max()),
        }
    return out


def live_frames(tracks: dict, h: int, sample_fps: float, min_players: int = 4,
                closeup_frac: float = 0.55, smooth_s: float = 1.0) -> set[int]:
    scores = frame_scores(tracks, h)
    if not scores:
        return set()
    lo, hi = min(scores), max(scores)
    raw = np.zeros(hi - lo + 1)
    for f, s in scores.items():
        raw[f - lo] = s["n_court_sized"] >= min_players and s["max_height_frac"] <= closeup_frac
    k = max(1, round(smooth_s * sample_fps)) | 1  # odd, so the window is centred
    # Zero-pad, since np.convolve "same" returns the longer input when the window exceeds the clip.
    c = np.concatenate([[0.0], np.cumsum(np.pad(raw, k // 2))])
    ones = np.concatenate([[0.0], np.cumsum(np.pad(np.ones_like(raw), k // 2))])
    # Divide by the in-range window size so the first and last frames are not penalised.
    votes = (c[k:] - c[:-k]) / (ones[k:] - ones[:-k])
    return {int(i) + lo for i in np.flatnonzero(votes > 0.5)}


def segments(live: set[int], sample_fps: float) -> list[tuple[float, float]]:
    """Contiguous live ranges as (start_s, end_s); end is exclusive."""
    out: list[tuple[float, float]] = []
    start = prev = None
    for f in sorted(live):
        if prev is None or f != prev + 1:
            if start is not None:
                out.append((start / sample_fps, (prev + 1) / sample_fps))
            start = f
        prev = f
    if start is not None:
        out.append((start / sample_fps, (prev + 1) / sample_fps))
    return out


def summary(tracks: dict, h: int, sample_fps: float) -> dict:
    live = live_frames(tracks, h, sample_fps)
    n = int(tracks["frame"].max() - tracks["frame"].min() + 1) if len(tracks["frame"]) else 0
    total = n / sample_fps
    live_s = len(live) / sample_fps
    return {"live_seconds": live_s, "total_seconds": total,
            "live_pct": 100 * live_s / total if total else 0.0,
            "segments": len(segments(live, sample_fps))}
