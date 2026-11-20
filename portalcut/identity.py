"""Link tracklets into players: the step from "tracked objects" to "the same person all game".

ByteTrack-style trackers keep an ID only through continuous footage; after an occlusion longer than
the lost-track buffer or a camera cut, the same player gets a new track ID. This module joins those
pieces using team colour (jersey-colour clusters) and jersey numbers read at least twice.

Output data/<job>/identities.json: {track_id: {"team": 0|1|2|None, "number": str|None, "player_id": str}}.
Teams 0 and 1 are the two largest colour clusters; 2 is everyone else (referees, staff, crowd).
Unnumbered tracklets keep their own ID ("trk-<id>"); appearance-only linking is a future step.
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np

from portalcut import reid, track


def team_labels(job_dir: Path) -> dict[int, int]:
    """Track id -> 0/1 (the two biggest jersey-colour clusters, biggest first) or 2 (everyone else)."""
    f = dict(np.load(job_dir / "features.npz"))
    clusters = reid.team_clusters(f.get("team_feat", f["hist"]), f["n"])
    sizes = Counter(clusters[f["n"] >= reid.MIN_TRACK_LEN].tolist() or clusters.tolist())
    order = [c for c, _ in sizes.most_common()]
    rank = {c: min(i, 2) for i, c in enumerate(order)}
    return {int(t): rank.get(int(c), 2) for t, c in zip(f["track_id"], clusters)}


def link(teams: dict[int, int], numbers: dict, frames_of: dict[int, set]) -> dict[int, dict]:
    """Same team + same number read >= 2 times -> one player_id. Two such tracklets on screen together
    can't both be that player: the one with fewer readings falls back to its own ID."""
    out = {t: {"team": team, "number": None, "player_id": f"trk-{t}"} for t, team in teams.items()}
    claims: dict[str, list[int]] = {}
    for t, team in teams.items():
        v = numbers.get(t) or numbers.get(str(t)) or {}
        if team in (0, 1) and v.get("number") is not None and v.get("votes", 0) >= 2:
            claims.setdefault(f"T{team}-{v['number']}", []).append(t)
    for pid, tids in claims.items():
        kept: list[int] = []
        votes = lambda t: (numbers.get(t) or numbers.get(str(t)))["votes"]  # noqa: E731
        for t in sorted(tids, key=lambda t: (-votes(t), -len(frames_of.get(t, ())))):
            if not any(frames_of.get(t, set()) & frames_of.get(k, set()) for k in kept):
                kept.append(t)
                out[t].update(number=pid.split("-", 1)[1], player_id=pid)
    return out


def build(job_dir: Path) -> dict:
    job_dir = Path(job_dir)
    tr = track.load(job_dir / "tracks.npz")
    frames_of: dict[int, set] = {}
    for fr, tid, c in zip(tr["frame"], tr["track_id"], tr["cls"]):
        if c == track.PERSON:
            frames_of.setdefault(int(tid), set()).add(int(fr))
    nf = job_dir / "numbers.json"
    ident = link(team_labels(job_dir), json.loads(nf.read_text()) if nf.exists() else {}, frames_of)
    (job_dir / "identities.json").write_text(json.dumps({str(k): v for k, v in ident.items()}))
    players = Counter(v["player_id"] for v in ident.values())
    linked = {p: n for p, n in players.items() if not p.startswith("trk-")}
    return {"tracklets": len(ident), "players_named": len(linked), "tracklets_linked": sum(linked.values()),
            "distinct_ids": len(players)}
