"""Compare trackers on identical detections against a tracking answer key (ground truth, not a proxy).

    python scripts/compare_trackers.py <job_folder> eval/tracking/uconn_arizona_clip/key.json \\
        --configs bytetrack:0.5 bytetrack:1 botsort:1 ocsort:1 --fps 10

<job_folder> is a processed job (has proxy.mp4 and job.json), e.g. from portalcut.batch. Detection runs once
per fps and is cached in the job folder. Prints one table and a JSON line to paste back to Claude.
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

from portalcut import track
from portalcut.detect import RFDETRDetector
from portalcut.track_eval import score

METRICS = ("detection_recall", "id_switches", "swapped_tracks", "swap_rate", "idf1_tracker")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("job", type=Path)
    ap.add_argument("key", type=Path)
    ap.add_argument("--configs", nargs="+", default=["bytetrack:0.5", "bytetrack:1", "botsort:1", "ocsort:1"],
                    help="tracker:lost_seconds")
    ap.add_argument("--fps", type=float, default=10.0)
    a = ap.parse_args(argv)
    meta = json.loads((a.job / "job.json").read_text())["meta"]
    p = meta["proxy"]
    w, h, dur, proxy = p["width"], p["height"], p["duration_sec"], a.job / "proxy.mp4"
    key = json.loads(a.key.read_text())
    det = RFDETRDetector()
    cache = a.job / f"detections_{a.fps:g}fps.pkl"
    t0 = time.time()
    dets = pickle.loads(cache.read_bytes()) if cache.exists() else track.detect_all(proxy, w, h, a.fps, det)
    cache.write_bytes(pickle.dumps(dets))
    print(f"detections: {len(dets)} frames in {time.time() - t0:.0f}s on {det.device}", flush=True)
    rows = []
    for c in a.configs:
        name, lost = c.split(":")
        t0 = time.time()
        r = track.track(proxy, w, h, dur, a.fps, detector=det, tracker=name, lost_seconds=float(lost), detections=dets)
        s = score(key, _cols(r["rows"]), {}, a.fps)
        rows.append({"tracker": name, "lost_s": float(lost), "fps": a.fps, "tracks": r["stats"]["tracks"],
                     **{m: s[m] for m in METRICS}, "seconds": round(time.time() - t0, 1)})
        print(" ".join(f"{k}={v}" for k, v in rows[-1].items()), flush=True)
    print("\n=== SUMMARY (paste this back to Claude) ===\n" + json.dumps(rows, separators=(",", ":")))
    return 0


def _cols(rows):
    """track.track returns an N×9 array; the scorer wants the tracks.npz column dict."""
    return {"frame": rows[:, 0].astype(int), "t": rows[:, 1], "track_id": rows[:, 2].astype(int),
            "cls": rows[:, 3].astype(int), "xyxy": rows[:, 4:8], "conf": rows[:, 8]}


if __name__ == "__main__":
    sys.exit(main())
