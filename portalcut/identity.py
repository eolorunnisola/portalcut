"""Link tracklets into players: the step from "tracked objects" to "the same person all game".

ByteTrack-style trackers keep an ID only through continuous footage; after an occlusion longer than
the lost-track buffer or a camera cut, the same player gets a new track ID. This module joins those
pieces using team colour and jersey numbers read at least twice, checked against team rosters.

- Team: majority vote of per-crop jersey-colour clusters. Teams 0 and 1 are the two biggest clusters;
  2 is everyone else, including referees (vertical stripes on most crops).
- Mixed pieces (crops disagree on team) are a tracker identity swap: they keep their own ID.
- Rosters (optional, data/<job>/roster.json): a number on neither roster is rejected as a misread; a number
  only on the other team's roster corrects the team (a navy #24 clustered as white is UConn's #24).

Output data/<job>/identities.json:
{track_id: {"team", "number", "player_id", "mixed", "referee", "name", "team_name", "team_fixed_by_number"}}.
Unnumbered tracklets keep their own ID ("trk-<id>"); appearance-only linking is a future step.
"""

import json
from collections import Counter
from itertools import permutations
from pathlib import Path

import numpy as np

from portalcut import reid, track

STRIPES = 8          # stripe_flips at or above this on a crop looks like a referee shirt
MIXED_SHARE = 0.75   # a piece is mixed when its majority team holds less than this share of 3+ crops


def team_votes(job_dir: Path) -> dict[int, dict]:
    """Track id -> {"team": 0|1|2, "mixed": bool, "referee": bool} from per-crop votes."""
    f = dict(np.load(job_dir / "features.npz"))
    if "crop_team" not in f or not len(f["crop_team"]):  # features from before per-crop data: track means
        lab = reid.team_clusters(f.get("team_feat", f["hist"]), f["n"])
        owners, flips = f["track_id"], np.zeros(len(lab), int)
    else:
        lab = reid.team_clusters(f["crop_team"], np.full(len(f["crop_team"]), reid.MIN_TRACK_LEN))
        owners, flips = f["crop_owner"], f["crop_flips"]
    order = [c for c, _ in Counter(lab.tolist()).most_common()]
    rank = {c: min(i, 2) for i, c in enumerate(order)}  # two biggest clusters are the teams
    out = {}
    for tid in np.unique(owners):
        m = owners == tid
        votes = Counter(rank[int(c)] for c in lab[m])
        team, n_top = votes.most_common(1)[0]
        referee = m.sum() >= 2 and (flips[m] >= STRIPES).mean() >= 0.5
        out[int(tid)] = {"team": 2 if referee else team, "colour_team": team, "referee": bool(referee),
                         "purity": round(n_top / m.sum(), 2),
                         "mixed": bool(m.sum() >= 3 and n_top / m.sum() < MIXED_SHARE)}
    for tid in f["track_id"]:
        out.setdefault(int(tid), {"team": None, "referee": False, "mixed": False, "purity": 0.0})
    return out


def match_rosters(teams: dict[int, dict], numbers: dict, roster: dict) -> dict[int, dict]:
    """Colour team (0/1) -> roster entry, choosing the pairing that explains the most numbers read."""
    rosters = roster["teams"][:2]
    seen = {0: Counter(), 1: Counter()}
    for t, v in teams.items():
        n = (numbers.get(t) or numbers.get(str(t)) or {})
        if v["team"] in (0, 1) and n.get("number") is not None and n.get("votes", 0) >= 2:
            seen[v["team"]][n["number"]] += 1
    best = max(permutations(range(len(rosters))), key=lambda p: sum(
        sum(c for num, c in seen[team].items() if num in rosters[r]["players"]) for team, r in zip((0, 1), p)))
    return {team: rosters[r] for team, r in zip((0, 1), best)}


def link(teams: dict[int, dict], numbers: dict, frames_of: dict[int, set], roster: dict | None = None) -> dict[int, dict]:
    """Same team + same number read >= 2 times -> one player_id. Two such tracklets on screen together
    can't both be that player: the one with fewer readings falls back to its own ID."""
    by_team = match_rosters(teams, numbers, roster) if roster else {}
    out = {t: v | {"number": None, "player_id": f"trk-{t}", "team_fixed_by_number": False} for t, v in teams.items()}
    claims: dict[str, list[int]] = {}
    for t, v in teams.items():
        n = numbers.get(t) or numbers.get(str(t)) or {}
        num, team = n.get("number"), v["team"]
        if v.get("referee") and by_team and num is not None and n.get("votes", 0) >= 2 and \
                any(num in r["players"] for r in by_team.values()):
            team = v.get("colour_team", team)  # a roster number read twice beats the stripe guess
            out[t].update(team=team, referee=False)
        if v["mixed"] or team not in (0, 1) or num is None or n.get("votes", 0) < 2:
            continue
        if by_team:
            mine, other = by_team[team]["players"], by_team[1 - team]["players"]
            if num not in mine and num in other:  # the number knows the team better than the colour does
                team = 1 - team
                out[t].update(team=team, team_fixed_by_number=True)
            elif num not in mine:  # on neither roster: a misread
                continue
        claims.setdefault(f"T{team}-{num}", []).append(t)
    votes = lambda t: (numbers.get(t) or numbers.get(str(t)))["votes"]  # noqa: E731
    for pid, tids in claims.items():
        kept: list[int] = []
        # overlapping claims mean one piece is a swap: the one whose crops agree on colour wins
        # (real case: two #3s, a piece that swapped from Arizona's #3 to UConn's beat the true one on length)
        for t in sorted(tids, key=lambda t: (-teams[t].get("purity", 1.0), -votes(t), -len(frames_of.get(t, ())))):
            if not any(frames_of.get(t, set()) & frames_of.get(k, set()) for k in kept):
                kept.append(t)
                team, num = int(pid[1]), pid.split("-", 1)[1]
                out[t].update(number=num, player_id=pid)
                if by_team:
                    out[t].update(name=by_team[team]["players"][num], team_name=by_team[team]["name"])
    return out


def build(job_dir: Path) -> dict:
    job_dir = Path(job_dir)
    tr = track.load(job_dir / "tracks.npz")
    frames_of: dict[int, set] = {}
    for fr, tid, c in zip(tr["frame"], tr["track_id"], tr["cls"]):
        if c == track.PERSON:
            frames_of.setdefault(int(tid), set()).add(int(fr))
    load = lambda n: json.loads((job_dir / n).read_text()) if (job_dir / n).exists() else None  # noqa: E731
    ident = link(team_votes(job_dir), load("numbers.json") or {}, frames_of, load("roster.json"))
    (job_dir / "identities.json").write_text(json.dumps({str(k): v for k, v in ident.items()}))
    players = Counter(v["player_id"] for v in ident.values())
    linked = {p: n for p, n in players.items() if not p.startswith("trk-")}
    return {"tracklets": len(ident), "players_named": len(linked), "tracklets_linked": sum(linked.values()),
            "distinct_ids": len(players), "referee_pieces": sum(v["referee"] for v in ident.values()),
            "mixed_pieces": sum(v["mixed"] for v in ident.values()),
            "teams_fixed_by_number": sum(v["team_fixed_by_number"] for v in ident.values())}
