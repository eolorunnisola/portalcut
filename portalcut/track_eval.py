"""Score a tracker against a hand-labelled answer key (`key`: one dict per keyframe, see `merge_labels`).

Per keyframe, predicted persons within 0.5/sample_fps of the key time are IoU-matched (>=0.5) to the key's player boxes.
- players / matched / detection_recall: player boxes + `missed` / boxes with a match / matched / players.
- named_correct / named_wrong / named_missing: over known-identity boxes ("UConn-24"); the prediction's name is
  "<team_name>-<number>". Wrong = a different name; missing = unmatched or the prediction has no name.
- identity_precision: correct / (correct + wrong), None without named predictions. identity_coverage: correct / known.
- wrong_team: matched player boxes whose predicted team_name differs from the label's team.
- id_switches: per GT identity, changes of raw track_id between consecutive matched keyframes.
- swapped_tracks / swap_rate: track_ids matched to 2+ GT identities / track_ids with at least one known match.
- idf1_tracker / idf1_named: 2*IDTP / (known GT boxes + predictions matched to known boxes), IDTP from a greedy one-to-one
  identity-to-id assignment, with id = raw track_id (tracker) or player_id (after identity linking).
- identities: per GT identity, observations, matched, correct, track_ids. errors: wrong names, wrong teams, swaps, with t.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

TEAMS = ("UConn", "Arizona")


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lt, rb = np.maximum(a[:, None, :2], b[None, :, :2]), np.minimum(a[:, None, 2:4], b[None, :, 2:4])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    area = lambda x: np.prod(np.clip(x[:, 2:4] - x[:, :2], 0, None), axis=1)
    return inter / np.maximum(area(a)[:, None] + area(b)[None, :] - inter, 1e-9)


def match(gt_boxes: list, pred_boxes: list, iou_min: float = 0.5) -> list[tuple[int, int]]:
    """One-to-one matching maximising total IoU; returns (gt_index, pred_index) pairs with IoU >= iou_min."""
    if not len(gt_boxes) or not len(pred_boxes):
        return []
    iou = _iou(np.asarray(gt_boxes, float)[:, :4], np.asarray(pred_boxes, float)[:, :4])
    rows, cols = linear_sum_assignment(-iou)
    return [(int(r), int(c)) for r, c in zip(rows, cols) if iou[r, c] >= iou_min]


def _idf1(obs: list[tuple[str, object]], n_gt: int) -> float | None:
    """obs = (gt identity, predicted id) for each matched known box; greedy one-to-one assignment by count."""
    used_g, used_p, idtp = set(), set(), 0
    for (g, p), n in Counter(obs).most_common():
        if g not in used_g and p not in used_p:
            used_g.add(g), used_p.add(p)
            idtp += n
    return 2 * idtp / (n_gt + len(obs)) if n_gt + len(obs) else None


def _r(x: float | None) -> float | None:
    return None if x is None else round(float(x), 4)


def _ratio(a: float, b: float) -> float | None:
    return _r(a / b) if b else None


def score(key: list, tracks: dict, identities: dict, sample_fps: float) -> dict:
    ids = {int(k): v for k, v in identities.items()}
    person = tracks["cls"] == 0
    t_all, xyxy, conf, tid_all = tracks["t"][person], tracks["xyxy"][person], tracks["conf"][person], tracks["track_id"][person]
    players = matched = correct = wrong = known = 0
    errors: dict[str, list] = {"wrong_name": [], "wrong_team": [], "swaps": []}
    obs: dict[str, list] = defaultdict(list)  # identity -> [(t, track_id, player_id)] for matched known boxes
    stats = defaultdict(lambda: Counter())
    for kf in key:
        t = float(kf["t"])
        near = np.abs(t_all - t) <= 0.5 / sample_fps
        if near.any():  # keep only the nearest sampled frame
            near &= np.abs(t_all - t_all[near][np.argmin(np.abs(t_all[near] - t))]) < 1e-6
        idx = np.flatnonzero(near)
        pidx = [i for i, lab in enumerate(kf["labels"]) if lab.startswith(TEAMS)]
        players += len(pidx) + int(kf.get("missed", 0))
        pairs = dict(match([kf["boxes"][i] for i in pidx], [[*xyxy[j], conf[j]] for j in idx]))
        for gi, i in enumerate(pidx):
            lab = kf["labels"][i]
            team, _, num = lab.partition("-")
            is_known = num != "?"
            known += is_known
            if is_known:
                stats[lab]["observations"] += 1
            if gi not in pairs:
                continue
            j = idx[pairs[gi]]
            tid, ident = int(tid_all[j]), ids.get(int(tid_all[j]), {})
            matched += 1
            if ident.get("team_name") not in (None, team):
                errors["wrong_team"].append({"t": t, "label": lab, "pred_team": ident["team_name"], "track_id": tid})
            if not is_known:
                continue
            name = f"{ident['team_name']}-{ident['number']}" if ident.get("team_name") and ident.get("number") else None
            stats[lab]["matched"] += 1
            if name == lab:
                correct += 1
                stats[lab]["correct"] += 1
            elif name:
                wrong += 1
                errors["wrong_name"].append({"t": t, "label": lab, "pred": name, "track_id": tid})
            obs[lab].append((t, tid, ident.get("player_id", f"trk-{tid}")))
    switches = sum(a[1] != b[1] for o in obs.values() for a, b in zip(sorted(o), sorted(o)[1:]))
    by_track: dict[int, dict[str, float]] = defaultdict(dict)
    for lab, o in obs.items():
        for t, tid, _ in o:
            by_track[tid].setdefault(lab, t)
    swapped = {tid: labs for tid, labs in by_track.items() if len(labs) > 1}
    errors["swaps"] = [{"track_id": tid, "identities": sorted(labs), "t": min(labs.values())} for tid, labs in swapped.items()]
    pairs_trk = [(lab, tid) for lab, o in obs.items() for _, tid, _ in o]
    pairs_named = [(lab, pid) for lab, o in obs.items() for _, _, pid in o]
    per_id = {lab: {**stats[lab], "track_ids": sorted({tid for _, tid, _ in obs.get(lab, [])})} for lab in sorted(stats)}
    return {
        "keyframes": len(key), "players": players, "matched": matched, "detection_recall": _ratio(matched, players),
        "known_boxes": known, "named_correct": correct, "named_wrong": wrong, "named_missing": known - correct - wrong,
        "identity_precision": _ratio(correct, correct + wrong), "identity_coverage": _ratio(correct, known),
        "wrong_team": len(errors["wrong_team"]), "id_switches": switches, "swapped_tracks": len(swapped),
        "swap_rate": _ratio(len(swapped), len(by_track)),
        "idf1_tracker": _r(_idf1(pairs_trk, known)), "idf1_named": _r(_idf1(pairs_named, known)),
        "identities": per_id, "errors": errors,
    }


def merge_labels(detections_path: str | Path, *label_paths: str | Path) -> list:
    """Combine detections.json ({t, frame, boxes}) with label files ({t, labels, missed, notes}) into the key format."""
    labels: dict[float, dict] = {}
    for p in label_paths:
        labels.update({round(float(e["t"]), 3): e for e in json.loads(Path(p).read_text())})
    key = []
    for d in json.loads(Path(detections_path).read_text()):
        e = labels.get(round(float(d["t"]), 3))
        if e is None:
            raise ValueError(f"no labels for t={d['t']}")
        if len(e["labels"]) != len(d["boxes"]):
            raise ValueError(f"t={d['t']}: {len(e['labels'])} labels for {len(d['boxes'])} boxes")
        key.append({"t": float(d["t"]), "boxes": d["boxes"], "labels": e["labels"], "missed": int(e.get("missed", 0))})
    return key


def main(job_id: str, key_path: str) -> None:
    from . import db, track
    if Path(job_id).is_dir():  # a job folder, e.g. from portalcut.batch on Colab
        job = Path(job_id)
        meta = json.loads((job / "job.json").read_text()).get("meta", {}) if (job / "job.json").exists() else {}
    else:
        job, meta = db.DATA_DIR / job_id, (db.get(job_id) or {}).get("meta", {})
    fps = (meta.get("tracking") or {}).get("sample_fps", 10)
    ident_path = job / "identities.json"
    res = score(json.loads(Path(key_path).read_text()), track.load(job / "tracks.npz"),
                json.loads(ident_path.read_text()) if ident_path.exists() else {}, fps)
    out = job / f"track_eval_{Path(key_path).stem}.json"
    out.write_text(json.dumps(res, indent=1))
    print({k: v for k, v in res.items() if k not in ("identities", "errors")}, f"\nwrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
