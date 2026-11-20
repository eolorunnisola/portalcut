"""Pass 2: pick out one player across the whole game.

build_features(): per track, an appearance embedding (DINOv2-small on torso crops) plus a jersey-colour
histogram, crop images for the "Is this you?" cards, and the frames shown for clicking.

Selection works on tracks (ByteTrack's short fragments of one person):
  pos  = tracks you confirmed (clicked or answered yes)
  neg  = tracks you rejected + any track on screen at the same time as a confirmed one (two people can't be one)
  every other track is scored by margin = best similarity to pos - best similarity to neg,
  accepted if margin > BAND, rejected if < -BAND, and the near-boundary ones become questions.
"""

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from portalcut import track

EMBED_MODEL = "facebook/dinov2-small"  # Apache-2.0, ~90 MB, downloaded once from Hugging Face
CROPS_PER_TRACK = 4
SELECT_FRAMES = 8
BAND = 0.04            # margin band treated as uncertain; tune on real games in the eval milestone
MAX_QUESTIONS = 12
MIN_TRACK_LEN = 3      # shorter fragments are never asked about; decided by margin sign
W_EMB, W_HIST = 0.75, 0.25

_embedder = None


def _load_embedder():
    global _embedder
    if _embedder is None:
        import torch
        from transformers import AutoImageProcessor, AutoModel
        dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        _embedder = (AutoImageProcessor.from_pretrained(EMBED_MODEL), AutoModel.from_pretrained(EMBED_MODEL).to(dev).eval(), dev)
    return _embedder


def embed(images: list[Image.Image], batch: int = 32) -> np.ndarray:
    import torch
    proc, model, dev = _load_embedder()
    out = []
    for i in range(0, len(images), batch):
        inputs = proc(images=images[i:i + batch], return_tensors="pt").to(dev)
        with torch.inference_mode():
            cls = model(**inputs).last_hidden_state[:, 0]
        out.append(torch.nn.functional.normalize(cls, dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 384), np.float32)


def color_hist(img: Image.Image) -> np.ndarray:
    """Hue x saturation histogram of the jersey; ignores near-black pixels whose hue is noise."""
    hsv = np.asarray(img.convert("HSV")).reshape(-1, 3)
    hsv = hsv[hsv[:, 2] > 40]
    h, _, _ = np.histogram2d(hsv[:, 0], hsv[:, 1], bins=(12, 4), range=((0, 256), (0, 256)))
    h = h.ravel()
    return (h / h.sum()) if h.sum() else h


def _torso(frame: np.ndarray, box) -> Image.Image:
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    return Image.fromarray(frame).crop((x1 + 0.15 * bw, y1 + 0.15 * bh, x2 - 0.15 * bw, y1 + 0.6 * bh))


def _pick_select_frames(t: dict) -> list[int]:
    """Frames with many players visible, spread across the game, for the click-to-select view."""
    frames, counts = np.unique(t["frame"][t["cls"] == track.PERSON], return_counts=True)
    if len(frames) == 0:
        return []
    good = frames[counts >= max(1, 0.6 * counts.max())]
    idx = np.linspace(0, len(good) - 1, min(SELECT_FRAMES, len(good))).round().astype(int)
    return sorted(set(good[idx].tolist()))


def build_features(job_dir: Path, w: int, h: int, fps: float = track.SAMPLE_FPS, on_progress=lambda p: None) -> dict:
    t0 = time.time()
    t = track.load(job_dir / "tracks.npz")
    person = np.where(t["cls"] == track.PERSON)[0]
    tids = np.unique(t["track_id"][person])

    # choose the clearest detections of each track (confident and large)
    wanted: dict[int, list[tuple[int, np.ndarray]]] = {}  # frame -> [(tid, box)]
    meta_rows = []
    for tid in tids:
        rows = person[t["track_id"][person] == tid]
        area = (t["xyxy"][rows, 2] - t["xyxy"][rows, 0]) * (t["xyxy"][rows, 3] - t["xyxy"][rows, 1])
        best = rows[np.argsort(-(t["conf"][rows] * area))][:CROPS_PER_TRACK]
        for r in best:
            wanted.setdefault(int(t["frame"][r]), []).append((int(tid), t["xyxy"][r]))
        meta_rows.append((int(tid), int(t["frame"][rows].min()), int(t["frame"][rows].max()), len(rows)))

    select_frames = set(_pick_select_frames(t))
    (job_dir / "frames").mkdir(exist_ok=True)
    (job_dir / "crops").mkdir(exist_ok=True)
    torsos, owners, shows = [], [], {}
    last = max(wanted.keys() | select_frames, default=-1)
    for i, frame in enumerate(track.read_frames(job_dir / "proxy.mp4", w, h, fps)):
        if i > last:
            break
        if i in select_frames:
            Image.fromarray(frame).save(job_dir / "frames" / f"{i}.jpg", quality=85)
        for tid, box in wanted.get(i, []):
            torsos.append(_torso(frame, box))
            owners.append(tid)
            if len(shows.setdefault(tid, [])) < 3:
                shows[tid].append(Image.fromarray(frame).crop(tuple(box)))
        if i % 50 == 0:
            on_progress(0.6 * i / max(last, 1))

    for tid, crops in shows.items():  # one strip of full-body crops per track for the question cards
        crops = [c.resize((max(1, round(c.width * 200 / c.height)), 200)) for c in crops if c.height > 0]
        strip = Image.new("RGB", (sum(c.width for c in crops) + 6 * (len(crops) - 1), 200), (24, 24, 27))
        x = 0
        for c in crops:
            strip.paste(c, (x, 0))
            x += c.width + 6
        strip.save(job_dir / "crops" / f"{tid}.jpg", quality=85)

    embs = embed(torsos)
    on_progress(0.95)
    owners_arr = np.array(owners)
    emb = np.zeros((len(tids), embs.shape[1] if len(embs) else 384), np.float32)
    hist = np.zeros((len(tids), 48), np.float32)
    for k, tid in enumerate(tids):
        m = owners_arr == tid
        if m.any():
            e = embs[m].mean(0)
            emb[k] = e / (np.linalg.norm(e) or 1)
            hist[k] = np.mean([color_hist(torsos[j]) for j in np.where(m)[0]], axis=0)
    mr = np.array(meta_rows, dtype=np.int32).reshape(-1, 4)
    np.savez_compressed(job_dir / "features.npz", track_id=mr[:, 0], first=mr[:, 1], last=mr[:, 2], n=mr[:, 3],
                        emb=emb, hist=hist, select_frames=np.array(sorted(select_frames), dtype=np.int32))
    return {"tracks": int(len(tids)), "crops": len(torsos), "model": EMBED_MODEL, "seconds": round(time.time() - t0, 1)}


# ---------- selection ----------

class Selection:
    """Selection state for one job, persisted to selection.json."""

    def __init__(self, job_dir: Path):
        self.dir = job_dir
        f = dict(np.load(job_dir / "features.npz"))
        self.tids = f["track_id"]
        self.n = f["n"]
        self.select_frames = f["select_frames"].tolist()
        self.index = {int(t): k for k, t in enumerate(self.tids)}
        hist = np.sqrt(f["hist"])  # Bhattacharyya coefficient as a matmul
        self.sim = W_EMB * (f["emb"] @ f["emb"].T) + W_HIST * (hist @ hist.T)

        tr = track.load(job_dir / "tracks.npz")
        m = tr["cls"] == track.PERSON
        self.rows = {k: tr[k][m] for k in ("frame", "t", "track_id", "xyxy", "conf")}
        frames_of = {int(t): set() for t in self.tids}
        for fr, tid in zip(self.rows["frame"], self.rows["track_id"]):
            frames_of[int(tid)].add(int(fr))
        self._frames_of = frames_of
        self.total_frames = int(tr["frame"].max()) + 1 if len(tr["frame"]) else 0

        p = job_dir / "selection.json"
        s = json.loads(p.read_text()) if p.exists() else {}
        self.pos: set[int] = set(s.get("pos", []))
        self.neg: set[int] = set(s.get("neg", []))
        self.prev: dict | None = s.get("prev")  # state before the last switch, for undo

    def save(self) -> None:
        (self.dir / "selection.json").write_text(
            json.dumps({"pos": sorted(self.pos), "neg": sorted(self.neg), "prev": self.prev}))

    def undo(self) -> bool:
        """Go back to the selection before the last switch. Returns False if there is none."""
        if not self.prev:
            return False
        self.pos, self.neg, self.prev = set(self.prev["pos"]), set(self.prev["neg"]), None
        return True

    def _margin(self, tid: int) -> float:
        """How much more `tid` looks like your confirmed tracks than like anyone known not to be you."""
        k = self.index[tid]
        neg = (set(self.neg) | {t for p in self.pos for t in self._frames_of if t not in self.pos and self.overlaps(p, t)}) - {tid}
        s_pos = self.sim[k, [self.index[t] for t in self.pos]].max()
        s_neg = self.sim[k, [self.index[t] for t in neg]].max() if neg else 0.7
        return float(s_pos - s_neg)

    def answer(self, tid: int, yes: bool) -> bool:
        """Record a yes/no for one track. Returns True when a yes switched to a different player."""
        if tid not in self.index:
            raise KeyError(tid)
        if yes and self.pos and tid not in self.pos:
            clashes = {a for a in self.score()["accepted"] if a != tid and self.overlaps(tid, a)}
            # On screen with a track you confirmed, or clearly unlike you: you mean a different player.
            # Start over from this click (undo-able); whoever it replaced on screen is not you.
            if any(self.overlaps(tid, p) for p in self.pos) or self._margin(tid) < -BAND:
                self.prev = {"pos": sorted(self.pos), "neg": sorted(self.neg)}
                self.pos, self.neg = {tid}, clashes
                return True
            # Otherwise it's another sighting of you: any automatic match in the same frames was wrong.
            self.neg = (self.neg | clashes) - {tid}
        (self.pos if yes else self.neg).add(tid)
        (self.neg if yes else self.pos).discard(tid)
        self.prev = None  # undo only reverses a switch you just made
        return False

    def overlaps(self, a: int, b: int) -> bool:
        return bool(self._frames_of[a] & self._frames_of[b])

    def score(self) -> dict:
        """Classify every track. Returns accepted (incl. pos), rejected, uncertain sorted most-useful-first."""
        neg = set(self.neg)
        for p in self.pos:  # anyone on screen with a confirmed track is someone else
            neg |= {t for t in self._frames_of if t != p and t not in self.pos and self.overlaps(p, t)}
        if not self.pos:
            return {"accepted": [], "neg": sorted(neg), "uncertain": [], "margin": {}}
        P = [self.index[t] for t in self.pos]
        N = [self.index[t] for t in neg]
        margin = {}
        for t in self.tids:
            t = int(t)
            if t in self.pos or t in neg:
                continue
            k = self.index[t]
            s_pos = self.sim[k, P].max()
            s_neg = self.sim[k, N].max() if N else 0.7  # no negatives yet: compare to a fixed bar
            margin[t] = float(s_pos - s_neg)

        # accept by margin, never two overlapping tracks: the player is in one place at a time
        accepted = list(self.pos)
        for t in sorted((t for t, m in margin.items() if m > BAND), key=lambda t: -margin[t]):
            if not any(self.overlaps(t, a) for a in accepted):
                accepted.append(t)
        uncertain = [t for t, m in margin.items() if abs(m) <= BAND and self.n[self.index[t]] >= MIN_TRACK_LEN
                     and not any(self.overlaps(t, p) for p in self.pos)]
        uncertain.sort(key=lambda t: (abs(margin[t]), -self.n[self.index[t]]))
        # short fragments are never asked about: decided by which side of zero they fall
        for t in (t for t, m in margin.items() if 0 < m <= BAND and self.n[self.index[t]] < MIN_TRACK_LEN):
            if not any(self.overlaps(t, a) for a in accepted):
                accepted.append(t)
        return {"accepted": accepted, "neg": sorted(neg), "uncertain": uncertain, "margin": margin}

    def player_rows(self, accepted: list[int]) -> dict:
        m = np.isin(self.rows["track_id"], accepted)
        return {k: v[m] for k, v in self.rows.items()}

    def summary(self, fps: float = track.SAMPLE_FPS) -> dict:
        s = self.score()
        frames = np.unique(self.player_rows(s["accepted"])["frame"]) if s["accepted"] else np.array([])
        return {
            "confirmed": sorted(self.pos),
            "rejected": sorted(self.neg),
            "accepted": sorted(s["accepted"]),
            "auto_rejected": len(s["neg"]) - len(self.neg),
            "questions": s["uncertain"][:MAX_QUESTIONS],
            "uncertain_total": len(s["uncertain"]),
            "coverage_pct": round(100 * len(frames) / max(self.total_frames, 1), 1),
            "minutes_visible": round(len(frames) / fps / 60, 1),
        }

    def frames_payload(self) -> list[dict]:
        out = []
        for fr in self.select_frames:
            m = self.rows["frame"] == fr
            out.append({"frame": fr, "boxes": [{"track_id": int(t), "xyxy": [round(float(v), 1) for v in b]}
                                               for t, b in zip(self.rows["track_id"][m], self.rows["xyxy"][m])]})
        return out


PLAYER_DEBUG_SECONDS = 120


def finish(job_dir: Path, w: int, h: int, fps: float = track.SAMPLE_FPS) -> dict:
    """Freeze the selection: write the player's per-frame boxes and a player-only debug video."""
    t0 = time.time()
    sel = Selection(job_dir)
    summary = sel.summary(fps)
    rows = sel.player_rows(summary["accepted"])
    # one box per frame: if two accepted tracks ever share a frame, keep the more confident
    order = np.lexsort((-rows["conf"], rows["frame"]))
    first = np.unique(rows["frame"][order], return_index=True)[1]
    keep = order[first]
    np.savez_compressed(job_dir / "player.npz", **{k: v[keep] for k, v in rows.items()})
    track.debug_video(job_dir / "proxy.mp4", {k: v[keep] for k, v in rows.items()}, w, h, fps,
                      job_dir / "debug_player.mp4", seconds=PLAYER_DEBUG_SECONDS, label="you")
    summary.pop("questions")
    # version changes on every rebuild so the page asks for the new video instead of a cached old one
    return summary | {"seconds": round(time.time() - t0, 1), "version": time.time_ns()}
