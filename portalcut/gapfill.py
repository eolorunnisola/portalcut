"""Fill gaps in the selected player's boxes with SAM 2, only at track edges."""
import subprocess
import time

import numpy as np

CKPT = "facebook/sam2.1-hiera-tiny"
_M = {}


def _ctr(b):
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])


def seeds(player, ball, n_frames_total, fps=5, seconds=3.0, max_runs=60):
    fr = np.asarray(player["frame"]); xy = np.asarray(player["xyxy"])
    if len(fr) == 0:
        return []
    cap = int(seconds * fps)
    cuts = np.flatnonzero(np.diff(fr) > 1) + 1
    starts = np.r_[0, cuts]; ends = np.r_[cuts, len(fr)] - 1
    runs = [(int(fr[a]), int(fr[b]), int(a), int(b)) for a, b in zip(starts, ends)]
    bt, bb = np.asarray(ball["t"]), np.asarray(ball["xyxy"])
    out = []
    for i, (f0, f1, a, b) in enumerate(runs):
        nxt = runs[i + 1][0] if i + 1 < len(runs) else n_frames_total
        prv = runs[i - 1][1] if i else -1
        for f, idx, d, room in ((f1, b, "forward", nxt - f1 - 1), (f0, a, "backward", f0 - prv - 1)):
            lim = min(cap, int(room))  # clipped at the video end / neighbouring run
            if lim <= 0:
                continue
            box = xy[idx].astype(np.float32)
            near = np.abs(bt - f / fps) <= 1.0
            h = max(box[3] - box[1], 1e-6)
            score = min((np.linalg.norm(_ctr(q) - _ctr(box)) / h for q in bb[near]), default=np.inf)
            out.append((score, {"frame": int(f), "box": box, "direction": d, "limit": lim}))
    out.sort(key=lambda s: s[0])  # stable: ties keep time order
    return [s for _, s in out[:max_runs]]


def mask_to_box(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float32)


def keep(box, prev_box, seed_box):
    area = lambda b: max(b[2] - b[0], 0) * max(b[3] - b[1], 0)
    r = area(box) / max(area(seed_box), 1e-6)
    jump = np.linalg.norm(_ctr(box) - _ctr(prev_box)) / max(seed_box[3] - seed_box[1], 1e-6)
    return 0.3 <= r <= 3 and jump <= 1.5


def _decode(proxy, w, h, start, n, fps):
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start / fps:.3f}", "-i", str(proxy), "-t", f"{n / fps + 0.1:.3f}",
           "-vf", f"fps={fps}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    k = min(len(raw) // (w * h * 3), n)
    return list(np.frombuffer(raw, np.uint8)[:k * w * h * 3].reshape(k, h, w, 3))


def _device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def sam2_track(frames, seed_index, box, direction):
    """Masks for the frames after the seed, in travel order (boolean HxW arrays)."""
    import torch
    from transformers import Sam2VideoModel, Sam2VideoProcessor
    if not _M:
        dev = _device()
        _M.update(dev=dev, proc=Sam2VideoProcessor.from_pretrained(CKPT),
                  model=Sam2VideoModel.from_pretrained(CKPT).to(dev).eval())
    proc, model, dev = _M["proc"], _M["model"], _M["dev"]
    sess = proc.init_video_session(video=frames, inference_device=dev)
    proc.add_inputs_to_inference_session(sess, frame_idx=seed_index, obj_ids=1,
                                         input_boxes=[[[float(v) for v in box]]])
    h, w = frames[0].shape[:2]
    masks = []
    with torch.no_grad():
        for o in model.propagate_in_video_iterator(sess, start_frame_idx=seed_index,
                                                   reverse=direction == "backward"):
            m = proc.post_process_masks([o.pred_masks], [[h, w]], binarize=False)[0]
            masks.append((m[0, 0] > 0).cpu().numpy())
    return masks[1:]  # first output is the seed frame itself


def fill(proxy, w, h, player, ball, n_frames_total, fps=5, seconds=3.0, max_runs=60,
         track_fn=None, on_progress=lambda p: None):
    t0 = time.time()
    track_fn = track_fn or sam2_track
    have = set(int(f) for f in player["frame"])
    got = {}  # frame -> box; first seed to reach a frame wins
    ss = seeds(player, ball, n_frames_total, fps, seconds, max_runs)
    for n, s in enumerate(ss):
        step = 1 if s["direction"] == "forward" else -1
        start = s["frame"] if step == 1 else max(0, s["frame"] - s["limit"])
        frames = _decode(proxy, w, h, start, s["limit"] + 1, fps)
        si = s["frame"] - start
        if si < len(frames):
            prev = s["box"]
            for k, m in enumerate(track_fn(frames, si, s["box"], s["direction"])[:s["limit"]], 1):
                b = None if m is None else mask_to_box(m)
                if b is None or not keep(b, prev, s["box"]):
                    break
                f = s["frame"] + step * k
                if f not in have and f not in got:
                    got[f] = b
                prev = b
        on_progress((n + 1) / len(ss))
    fs = sorted(got)
    rows = {"frame": np.array(fs, np.int32), "t": np.array([f / fps for f in fs], np.float32),
            "track_id": np.full(len(fs), -2, np.int32),
            "xyxy": np.array([got[f] for f in fs], np.float32).reshape(-1, 4),
            "conf": np.full(len(fs), 0.5, np.float32)}
    return rows, {"seeds": len(ss), "frames_added": len(fs), "seconds": time.time() - t0}
