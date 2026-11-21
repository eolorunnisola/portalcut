"""Score following ONE player against the tracking answer key, with the yes/no selection answered by the key.

    uv run python -m portalcut.follow_eval <job_folder_or_id> eval/tracking/uconn_arizona_clip/key.json \\
        [--player Arizona-13] [--clicks 12]

Without --player, every player the key names is followed in turn and the table ends with the averages.

The simulated user clicks the player once (in the click-to-select frames if they appear there), then answers the
app's questions, most useful first, up to --clicks answers. A question about a track the key can't identify
(it never overlaps a named key box) is skipped, as if the user scrolled past it.

Per keyframe where the key has a box for the player: found = the followed box overlaps it (IoU >= 0.5).
On any keyframe: wrong = the followed box is on someone else (a named player, unknown-number player, referee,
or nobody), except a box on a same-team unknown-number player ("Arizona-?"), which can't be judged.
- coverage = found / keyframes the player is visible: how much of the time we have him.
- precision = found / (found + wrong): how much of what we show is really him. Following the wrong person
  is the worse failure, so precision comes first.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from portalcut import reid, track
from portalcut.track_eval import TEAMS, match


def _nearest(t_all: np.ndarray, t: float, fps: float) -> np.ndarray:
    """Indices of rows on the sampled frame nearest key time t (within half a sample)."""
    near = np.abs(t_all - t) <= 0.5 / fps
    if near.any():
        near &= np.abs(t_all - t_all[near][np.argmin(np.abs(t_all[near] - t))]) < 1e-6
    return np.flatnonzero(near)


def track_labels(key: list, rows: dict, fps: float) -> dict[int, str]:
    """Track id -> the key label it overlaps most often (named players only; unknown otherwise)."""
    votes: dict[int, Counter] = defaultdict(Counter)
    for kf in key:
        idx = _nearest(rows["t"], float(kf["t"]), fps)
        for gi, pj in match(kf["boxes"], [rows["xyxy"][j] for j in idx]):
            votes[int(rows["track_id"][idx[pj]])][kf["labels"][gi]] += 1
    out = {}
    for tid, c in votes.items():
        lab, n = c.most_common(1)[0]
        if lab.startswith(TEAMS) and not lab.endswith("-?"):
            out[tid] = lab
    return out


def score_player(key: list, player: dict, target: str, fps: float) -> dict:
    """player = rows of the followed boxes (t, xyxy). See the module docstring for the definitions."""
    team = target.split("-")[0]
    visible = found = wrong = unsure = 0
    for kf in key:
        labels = kf["labels"]
        idx = _nearest(player["t"], float(kf["t"]), fps) if len(player["t"]) else []
        visible += target in labels
        if not len(idx):
            continue
        pairs = dict((pj, gi) for gi, pj in match(kf["boxes"], [player["xyxy"][idx[0]]]))
        lab = labels[pairs[0]] if 0 in pairs else None
        if lab == target:
            found += 1
        elif lab == f"{team}-?":
            unsure += 1
        else:
            wrong += 1
    r = lambda a, b: round(a / b, 3) if b else None
    return {"player": target, "visible": visible, "found": found, "wrong": wrong, "unsure": unsure,
            "coverage": r(found, visible), "precision": r(found, found + wrong)}


def simulate(job_dir: Path, key: list, target: str, fps: float, clicks: int = 12) -> dict:
    sel = reid.Selection(job_dir)
    sel.pos, sel.neg, sel.prev = set(), set(), None  # start fresh, never saved
    rows = sel.rows
    truth = track_labels(key, rows, fps)
    mine = [t for t, lab in truth.items() if lab == target and t in sel.index]
    if not mine:
        return {"player": target, "error": "no track of this player overlaps the key"}
    in_select = [t for t in mine if set(sel.select_frames) & sel._frames_of[t]]
    first = max(in_select or mine, key=lambda t: len(sel._frames_of[t]))
    sel.answer(first, True)
    asked = skipped = 0
    seen: set[int] = set()
    while asked < clicks:
        q = [t for t in sel.score()["uncertain"] if t not in seen]
        if not q:
            break
        t = q[0]
        seen.add(t)
        if t not in truth:
            skipped += 1
            continue
        sel.answer(t, truth[t] == target)
        asked += 1
    acc = sel.score()["accepted"]
    p = sel.player_rows(acc)
    order = np.lexsort((-p["conf"], p["frame"]))  # one box per frame, as reid.finish does
    first_of = np.unique(p["frame"][order], return_index=True)[1]
    p = {k: v[order[first_of]] for k, v in p.items()}
    wrong_tracks = sorted({t for t in acc if t in truth and truth[t] != target})
    return score_player(key, p, target, fps) | {
        "first_click_in_select_frames": bool(in_select), "answers": asked, "skipped_questions": skipped,
        "accepted_tracks": len(acc), "true_tracks": len(mine), "accepted_wrong_tracks": wrong_tracks}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("job")
    ap.add_argument("key", type=Path)
    ap.add_argument("--player", help='e.g. "Arizona-13"; default: every named player in the key')
    ap.add_argument("--clicks", type=int, default=12)
    a = ap.parse_args(argv)
    from portalcut import db
    job = Path(a.job) if Path(a.job).is_dir() else db.DATA_DIR / a.job
    jf = job / "job.json"
    meta = json.loads(jf.read_text()).get("meta", {}) if jf.exists() else (db.get(a.job) or {}).get("meta", {})
    fps = (meta.get("tracking") or {}).get("sample_fps", track.SAMPLE_FPS)
    key = json.loads(a.key.read_text())
    names = [a.player] if a.player else sorted({l for kf in key for l in kf["labels"]
                                                 if l.startswith(TEAMS) and not l.endswith("-?")})
    res = [simulate(job, key, n, fps, a.clicks) for n in names]
    for r in res:
        print(" ".join(f"{k}={v}" for k, v in r.items()), flush=True)
    ok = [r for r in res if "error" not in r]
    mean = {m: round(float(np.mean([r[m] or 0 for r in ok])), 3) for m in ("coverage", "precision")} if ok else {}
    print("\n=== SUMMARY (paste this back to Claude) ===\n" + json.dumps(
        {"players": len(res), "mean": mean, "clicks": a.clicks,
         "rows": [{k: r.get(k) for k in ("player", "coverage", "precision", "answers", "error")} for r in res]},
        separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
