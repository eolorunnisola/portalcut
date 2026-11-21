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
CROPS_PER_TRACK = 6
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


def spread_picks(rows: np.ndarray, frames: np.ndarray, quality: np.ndarray, k: int) -> np.ndarray:
    """Best row (by quality) from each of k equal time slices of a track. Picking the k best overall took
    them all from one end, so a piece that swapped players mid-way looked clean (answer key, 84 s)."""
    order = np.argsort(frames, kind="stable")
    return np.array([c[np.argmax(q)] for c, q in zip(np.array_split(rows[order], k), np.array_split(quality[order], k))
                     if len(c)], dtype=rows.dtype)


def stripe_flips(img: Image.Image) -> int:
    """How often column brightness crosses its mean across the jersey centre. Referees' vertical stripes
    measured 8+ on 5 of 7 crops; players 8+ on 6.8% of 278 crops (UConn-Arizona broadcast)."""
    v = np.asarray(img.convert("HSV"))[..., 2].astype(np.float32)
    c = v[:, v.shape[1] // 5: -max(1, v.shape[1] // 5)]
    prof = c.mean(0) if c.size else np.zeros(1)
    return int((np.diff(np.sign(prof - prof.mean())) != 0).sum())


def team_color(img: Image.Image) -> np.ndarray:
    """Brightness x saturation of the jersey centre: what separates white from navy (hue alone can't)."""
    a = np.asarray(img.convert("HSV"))
    hsv = a[:, a.shape[1] // 5: -max(1, a.shape[1] // 5)].reshape(-1, 3)  # centre columns: less court and crowd
    h, _, _ = np.histogram2d(hsv[:, 2], hsv[:, 1], bins=(4, 3), range=((0, 256), (0, 256)))
    h = h.ravel()
    return (h / h.sum()) if h.sum() else h


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
        best = spread_picks(rows, t["frame"][rows], t["conf"][rows] * area, CROPS_PER_TRACK)
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

    by_tid = {}
    for tid, im in zip(owners, torsos):
        by_tid.setdefault(tid, []).append(im)
    for tid, ims in by_tid.items():  # torso strip per track for the jersey-number reader
        ims = [i.resize((max(1, round(i.width * 128 / i.height)), 128)) for i in ims if i.height > 0 and i.width > 0]
        if ims:
            strip = Image.new("RGB", (sum(i.width for i in ims) + 6 * (len(ims) - 1), 128), (24, 24, 27))
            x = 0
            for i in ims:
                strip.paste(i, (x, 0))
                x += i.width + 6
            strip.save(job_dir / "crops" / f"{tid}_torso.jpg", quality=90)

    embs = embed(torsos)
    on_progress(0.95)
    owners_arr = np.array(owners)
    emb = np.zeros((len(tids), embs.shape[1] if len(embs) else 384), np.float32)
    hist = np.zeros((len(tids), 48), np.float32)
    team_feat = np.zeros((len(tids), 12), np.float32)
    for k, tid in enumerate(tids):
        m = owners_arr == tid
        if m.any():
            e = embs[m].mean(0)
            emb[k] = e / (np.linalg.norm(e) or 1)
            hist[k] = np.mean([color_hist(torsos[j]) for j in np.where(m)[0]], axis=0)
            team_feat[k] = np.mean([team_color(torsos[j]) for j in np.where(m)[0]], axis=0)
    mr = np.array(meta_rows, dtype=np.int32).reshape(-1, 4)
    # per-crop colour and stripes too: identity votes per crop, and a piece whose crops disagree is a tracker swap
    crop_team = np.array([team_color(im) for im in torsos], np.float32).reshape(-1, 12)
    crop_flips = np.array([stripe_flips(im) for im in torsos], np.int16)
    np.savez_compressed(job_dir / "features.npz", track_id=mr[:, 0], first=mr[:, 1], last=mr[:, 2], n=mr[:, 3],
                        emb=emb, hist=hist, team_feat=team_feat, crop_owner=owners_arr.astype(np.int32),
                        crop_team=crop_team, crop_flips=crop_flips,
                        select_frames=np.array(sorted(select_frames), dtype=np.int32))
    return {"tracks": int(len(tids)), "crops": len(torsos), "model": EMBED_MODEL, "seconds": round(time.time() - t0, 1)}


# ---------- selection ----------

def team_clusters(hist: np.ndarray, n: np.ndarray, k: int = 3, iters: int = 20) -> np.ndarray:
    """Group tracks by jersey colour: two teams plus one cluster for referees and everything else."""
    x = np.sqrt(np.clip(hist, 0, None))  # Hellinger space: euclidean distance compares colour histograms sensibly
    fit = x[n >= MIN_TRACK_LEN] if (n >= MIN_TRACK_LEN).sum() >= k else x
    if len(fit) < k:
        return np.zeros(len(x), int)
    c = [fit[np.argmax(fit.sum(1))]]  # deterministic farthest-point start
    for _ in range(k - 1):
        c.append(fit[np.argmax(np.min([((fit - ci) ** 2).sum(1) for ci in c], axis=0))])
    c = np.array(c)
    for _ in range(iters):
        lab = np.argmin(((fit[:, None] - c[None]) ** 2).sum(-1), axis=1)
        c = np.array([fit[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(k)])
    return np.argmin(((x[:, None] - c[None]) ** 2).sum(-1), axis=1)


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
        self.team = team_clusters(f.get("team_feat", f["hist"]), f["n"])  # older jobs: fall back to hue
        nf = job_dir / "numbers.json"  # written by portalcut.jersey; None = not readable
        nums = {int(t): v for t, v in (json.loads(nf.read_text()) if nf.exists() else {}).items()
                if v.get("number") is not None and int(t) in self.index}
        self.number = {t: v["number"] for t, v in nums.items()}
        # one reading can be a misread (18 -> 13) or a tracker identity swap: accepting by number needs two
        self.strong = {t for t, v in nums.items() if v.get("votes", 0) >= 2}

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

    def target(self) -> tuple[int | None, str | None]:
        """Your team (jersey-colour cluster) and jersey number, by majority over your confirmed tracks."""
        mode = lambda xs: max(set(xs), key=xs.count) if xs else None
        return (mode([int(self.team[self.index[p]]) for p in self.pos]),
                mode([self.number[p] for p in self.pos if p in self.number]))

    def _other_player(self, t: int, team: int | None, num: str | None) -> bool:
        """Other team's colours, or a readable number that isn't yours (an unread number never counts).
        Colour wins over a matching number: on real footage a navy #18 was misread as 13. Precision first."""
        return (team is not None and int(self.team[self.index[t]]) != team) or \
            (num is not None and self.number.get(t, num) != num)

    def _auto_neg(self) -> set[int]:
        """Everyone known not to be you: your No answers, anyone on screen with you, other team, other number."""
        team, num = self.target()
        neg = set(self.neg)
        for p in self.pos:
            neg |= {t for t in self._frames_of if t not in self.pos and self.overlaps(p, t)}
        return neg | {int(t) for t in self.tids if int(t) not in self.pos and self._other_player(int(t), team, num)}

    def _margin(self, tid: int) -> float:
        """How much more `tid` looks like your confirmed tracks than like anyone known not to be you."""
        k = self.index[tid]
        neg = self._auto_neg() - {tid}
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
            if any(self.overlaps(tid, p) for p in self.pos) or self._other_player(tid, *self.target()) \
                    or self._margin(tid) < -BAND:
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
        neg = self._auto_neg()
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

        # your number on your team is the strongest evidence: accept those first, longest first
        _, num = self.target()
        same_number = [t for t in margin if num is not None and self.number.get(t) == num and t in self.strong]
        # then accept by margin; never two overlapping tracks: the player is in one place at a time
        accepted = list(self.pos)
        for t in sorted(same_number, key=lambda t: -self.n[self.index[t]]) + \
                sorted((t for t, m in margin.items() if m > BAND), key=lambda t: -margin[t]):
            if t not in accepted and not any(self.overlaps(t, a) for a in accepted):
                accepted.append(t)
        uncertain = [t for t, m in margin.items() if abs(m) <= BAND and self.n[self.index[t]] >= MIN_TRACK_LEN
                     and t not in same_number
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
        team, num = self.target()
        return {
            "team": team, "number": num, "numbers_read": len(self.number),
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


GAP_SEEDS, GAP_SECONDS = 20, 2.0  # SAM 2 runs ~1.1 s/frame on a Mac: 20 edges x 2 s is about 4 minutes at most


def finish(job_dir: Path, w: int, h: int, fps: float = track.SAMPLE_FPS, gap_fill: bool = True,
           on_progress=lambda p: None) -> dict:
    """Freeze the selection: write the player's per-frame boxes and a player-only debug video.
    gap_fill follows the player with SAM 2 for a few seconds where tracking lost him (clumps, jumps)."""
    t0 = time.time()
    sel = Selection(job_dir)
    summary = sel.summary(fps)
    rows = sel.player_rows(summary["accepted"])
    # one box per frame: if two accepted tracks ever share a frame, keep the more confident
    order = np.lexsort((-rows["conf"], rows["frame"]))
    first = np.unique(rows["frame"][order], return_index=True)[1]
    rows = {k: v[order[first]] for k, v in rows.items()}
    added = 0
    if gap_fill and len(rows["frame"]):
        from portalcut import gapfill
        tr = track.load(job_dir / "tracks.npz")
        bm = tr["cls"] == track.BALL
        extra, _ = gapfill.fill(job_dir / "proxy.mp4", w, h, rows, {"t": tr["t"][bm], "xyxy": tr["xyxy"][bm]},
                                sel.total_frames, fps, GAP_SECONDS, GAP_SEEDS, on_progress=on_progress)
        added = len(extra["frame"])
        order = np.argsort(np.r_[rows["frame"], extra["frame"]], kind="stable")
        rows = {k: np.concatenate([rows[k], extra[k]])[order] for k in rows}
    np.savez_compressed(job_dir / "player.npz", **rows)
    track.debug_video(job_dir / "proxy.mp4", rows, w, h, fps,
                      job_dir / "debug_player.mp4", seconds=PLAYER_DEBUG_SECONDS, label="you")
    summary.pop("questions")
    # version changes on every rebuild so the page asks for the new video instead of a cached old one
    return summary | {"gap_filled_frames": added, "seconds": round(time.time() - t0, 1), "version": time.time_ns()}
