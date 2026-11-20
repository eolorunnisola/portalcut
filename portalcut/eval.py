"""Score detected plays against a hand-made answer key."""
import argparse
import json
from pathlib import Path

from portalcut import db


def _gap(t, p):
    return max(p["t_start"] - t, t - p["t_end"], 0.0)


def _stats(truth, pred, tp):
    return {"truth": truth, "pred": pred, "tp": tp,
            "precision": round(tp / pred, 3) if pred else None,
            "recall": round(tp / truth if truth else 0, 3) if truth else None}


def score(truth_plays, preds, tol=2.0, segment=None, min_conf=0.0, ignore=()):
    preds = [p for p in preds if p.get("confidence", 1.0) >= min_conf]
    # replays and dead time: a detection there is neither right nor wrong
    preds = [p for p in preds if not any(a <= p["t_start"] <= b for a, b in ignore)]
    if segment:
        lo, hi = segment
        truth_plays = [t for t in truth_plays if lo <= t["t"] <= hi]
        preds = [p for p in preds if lo <= p["t_start"] <= hi]
    cands = sorted((_gap(t["t"], p), i, j)
                   for i, t in enumerate(truth_plays) for j, p in enumerate(preds)
                   if t["type"] == p["type"] and _gap(t["t"], p) <= tol)
    used_t, used_p = set(), set()
    for _, i, j in cands:  # greedy closest-first, one-to-one
        if i not in used_t and j not in used_p:
            used_t.add(i)
            used_p.add(j)
    types = {x["type"] for x in truth_plays} | {x["type"] for x in preds}
    per_type = {}
    for ty in sorted(types):
        tp = sum(truth_plays[i]["type"] == ty for i in used_t)
        per_type[ty] = _stats(sum(t["type"] == ty for t in truth_plays),
                              sum(p["type"] == ty for p in preds), tp)
    return {"per_type": per_type,
            "overall": _stats(len(truth_plays), len(preds), len(used_t)),
            "misses": [t for i, t in enumerate(truth_plays) if i not in used_t],
            "false_positives": [p for j, p in enumerate(preds) if j not in used_p]}


def _mmss(s):
    return f"{int(s) // 60:02d}:{int(s) % 60:02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job_id")
    ap.add_argument("key", type=Path)
    ap.add_argument("--tol", type=float, default=2.0)
    ap.add_argument("--min-conf", type=float, default=0.0)
    a = ap.parse_args()
    key = json.loads(a.key.read_text())
    out_dir = db.DATA_DIR / a.job_id
    preds = json.loads((out_dir / "events.json").read_text())
    res = score(key["plays"], preds, a.tol, key.get("segment"), a.min_conf, key.get("ignore", ()))
    meta = ((db.get(a.job_id) or {}).get("meta") or {}).get("plays", {})
    for k in ("cost_usd", "api_calls", "escalations"):
        if k in meta:
            res[k] = meta[k]
    f = lambda v: "-" if v is None else f"{v:.3f}"
    print(f"{'type':<16}{'truth':>6}{'pred':>6}{'tp':>5}{'prec':>7}{'rec':>7}")
    for name, s in [*res["per_type"].items(), ("OVERALL", res["overall"])]:
        print(f"{name:<16}{s['truth']:>6}{s['pred']:>6}{s['tp']:>5}"
              f"{f(s['precision']):>7}{f(s['recall']):>7}")
    print("cost_usd={} api_calls={} escalations={}".format(
        *(res.get(k, "-") for k in ("cost_usd", "api_calls", "escalations"))))
    print("misses:")
    for t in res["misses"]:
        print(f"  {_mmss(t['t'])} {t['type']}")
    print("false positives:")
    for p in res["false_positives"]:
        print(f"  {_mmss(p['t_start'])} {p['type']}")
    (out_dir / f"eval_{a.key.stem}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
