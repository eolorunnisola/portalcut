"""Passes 3-4: find the moments where your player has the ball or makes a play.

Pass 3 (free, local): cut the player's on-screen time into 5 s windows (2.5 s stride) and rank them by
  ball near the player, player speed and crowd noise. COCO's ball detector is weak, so the ball is only
  a ranking signal, never a filter.
Pass 4 (paid): each window becomes 10 frames (2 fps) with the player outlined in magenta, plus one close-up.
  Claude Haiku 5.5 labels the events; borderline calls and every made shot / assist are re-checked by
  Claude Sonnet 5.5 with 16 frames. Every answer is cached on disk, so a re-run costs nothing.
"""

import base64
import hashlib
import io
import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw
from pydantic import BaseModel

from portalcut import track

HAIKU, SONNET = "claude-haiku-5-5", "claude-sonnet-5-5"
PRICE = {HAIKU: (0.10, 0.50), SONNET: (2.00, 10.00)}  # $ per million input / output tokens
WINDOW_S, STRIDE_S, FPS = 5.0, 2.5, 4.0
ASSIST_WINDOW_S = 8.0
ESCALATE_FPS = 6.0                                    # 30 close-ups over 5 s
MIN_VISIBLE = 0.4                                     # player must be on screen in 40% of a window's frames
CONF_LOW, CONF_HIGH = 0.4, 0.75                       # escalate confidences in this band...
ALWAYS_ESCALATE = {"made_shot", "assist"}
ESCALATE_IF_UNSURE = {"steal", "block", "rebound", "missed_shot"}  # ...but only for these highlight plays.
# Never escalated: defensive_stop, possession, drive. On the first real run, borderline defensive stops sent
# 37 of 46 windows to Sonnet (95% of a $0.71 bill) and Sonnet came back no more certain.
HAIKU_PER_WINDOW = (6700, 1080)                       # measured median in / out tokens, close-ups (out includes thinking)
SONNET_PER_CALL = (8700, 240)                         # measured on 30 close-ups
EXPECTED_ESCALATION_RATE = 0.10                       # share of windows that get a second opinion; for the estimate only
BUDGET_USD = 1.00                                     # per run; windows are sent best-first until it's spent
CLOSEUP_PX, CONTEXT_W = 384, 640                     # close-up crops; full-court frames for context
SCORE_LAG_S = 3.0                                     # broadcast scoreboards update a few seconds after a basket
CONCURRENCY = 8
PROMPT_VERSION = 3                                    # bump to invalidate the cache when the prompt changes
TARGET = (255, 0, 255)

EventType = Literal["made_shot", "missed_shot", "assist", "steal", "block", "rebound", "drive", "possession",
                    "defensive_stop"]


class Event(BaseModel):
    type: EventType
    frame_start: int
    frame_end: int
    confidence: float
    short_reason: str


class WindowResult(BaseModel):
    player_has_ball: bool
    assist_possible: bool
    events: list[Event]


SYSTEM = """You review short basketball clips for a player's highlight reel.

You get frames in time order (numbered from 0) and one close-up. ONE player is outlined with a thick MAGENTA \
box labelled TARGET. Only report what that TARGET player does. If the box is missing in a frame, the player \
may be off-screen or hidden there.

Event types (the TARGET player must be the one doing it):
- made_shot: shoots and the ball goes in.
- missed_shot: shoots and misses.
- assist: passes to a teammate who scores right after the pass.
- steal: takes the ball from an opponent (pick-off or poke-away recovered by their team).
- block: deflects an opponent's shot attempt.
- rebound: secures the ball after a missed shot.
- drive: dribbles hard toward the basket.
- possession: has the ball (dribbling, holding, setting up) without one of the plays above.
- defensive_stop: clearly contests or forces a bad shot or turnover without a steal or block.

Rules:
- frame_start and frame_end are the frame numbers where the event begins and ends.
- confidence is 0 to 1: how sure you are that the TARGET player did it, not just that it happened.
- If the TARGET player passes and you cannot see whether a teammate scored, set assist_possible true \
and do not report an assist.
- When unsure whether anything happened, return no events. Missing a play is better than inventing one.
- short_reason: under 15 words."""


# ---------- prompt v2: ask for visible facts, decide the play in code ----------
# shortcut: PROMPT_MODE keeps the v1 "labels" prompt runnable only to measure it against v2; delete the loser.
PROMPT_MODE = "facts"


class Facts(BaseModel):
    live_action: bool
    target_visible: bool
    target_shot: bool
    shot_frame: int
    shot_result: Literal["made", "missed", "unclear", "no_shot"]
    target_passed: bool
    pass_frame: int
    pass_result: Literal["receiver_scored", "receiver_did_not_score", "unclear", "no_pass"]
    target_blocked_shot: bool
    block_frame: int
    target_took_ball_from_opponent: bool
    steal_frame: int
    target_team_has_ball_after_steal: Literal["yes", "no", "unclear", "no_steal"]
    target_rebounded_missed_shot: bool
    rebound_frame: int
    target_team_score_went_up: Literal["yes", "no", "unclear"]
    certainty: Literal["high", "medium", "low"]
    what_happened: str


SYSTEM_FACTS = """You check short basketball clips for one player's highlight reel. You answer factual questions; \
you do not judge how good the play was.

You get close-up crops that follow ONE player, in time order (numbered from 0, 4 per second), then one \
full-court view. That player is outlined with a MAGENTA box labelled TARGET. Every question is about the TARGET \
player only. If a crop says "TARGET box missing", the camera stays where he was last seen.

Answer only from what is visible. If you cannot see it, answer false, "unclear" or -1. Never guess.

- live_action: false if these frames are a replay (slow motion, a repeated play, replay graphics), a close-up \
during a stoppage, the crowd, the bench, or an ad. If false, answer false or "no_..." for everything else.
- target_visible: the TARGET box appears in at least one frame.
- target_shot: the TARGET releases a shot toward the rim (including free throws and layups). shot_frame: the \
release frame, else -1. shot_result: "made" only if you see the ball go through the net or the score change; \
"missed" only if you see it miss; otherwise "unclear"; "no_shot" if no shot.
- target_passed: the TARGET passes the ball. pass_frame: the frame of the pass, else -1. pass_result: \
"receiver_scored" only if the teammate who caught it scores with at most a couple of dribbles and you see the make.
- target_blocked_shot: an OPPONENT released a shot and the TARGET's hand clearly deflected the ball. Contesting \
without touching the ball is false. block_frame, else -1.
- target_took_ball_from_opponent: the TARGET intercepts a pass or knocks the ball away from an opponent. \
steal_frame, else -1. target_team_has_ball_after_steal: does the TARGET's team end up with the ball.
- target_team_score_went_up: compare the scoreboard in the two full-court views (before, and a few seconds \
after). "yes" only if the TARGET's team's score is higher in the second; "unclear" if there is no readable \
scoreboard.
- target_rebounded_missed_shot: a shot visibly missed and then the TARGET clearly secures the ball. rebound_frame, \
else -1.
- certainty: how sure you are about whichever of the above you answered true.
- what_happened: under 15 words, what the TARGET did."""

CERTAINTY = {"high": 0.9, "medium": 0.7, "low": 0.45}


def facts_to_result(f: dict) -> dict:
    """The rules: a play exists only when the facts that define it were seen. Same shape as WindowResult."""
    conf, why = CERTAINTY[f["certainty"]], f["what_happened"]
    if not (f["live_action"] and f["target_visible"]):
        return {"player_has_ball": False, "assist_possible": False, "events": [], "facts": f}
    ev = []

    def add(kind, frame):
        if frame >= 0:
            ev.append({"type": kind, "frame_start": frame, "frame_end": frame, "confidence": conf, "short_reason": why})

    if f["target_shot"] and f["shot_result"] == "unclear" and f.get("target_team_score_went_up") == "yes":
        f = f | {"shot_result": "made", "certainty": {"high": "medium"}.get(f["certainty"], "low")}  # scoreboard confirms
        conf = CERTAINTY[f["certainty"]]
    if f["target_shot"] and f["shot_result"] in ("made", "missed"):
        add("made_shot" if f["shot_result"] == "made" else "missed_shot", f["shot_frame"])
    if f["target_passed"] and f["pass_result"] == "receiver_scored":
        add("assist", f["pass_frame"])
    if f["target_blocked_shot"]:
        add("block", f["block_frame"])
    if f["target_took_ball_from_opponent"] and f["target_team_has_ball_after_steal"] == "yes":
        add("steal", f["steal_frame"])
    if f["target_rebounded_missed_shot"]:
        add("rebound", f["rebound_frame"])
    return {"player_has_ball": f["target_shot"] or f["target_passed"], "events": ev, "facts": f,
            "assist_possible": f["target_passed"] and f["pass_result"] == "unclear"}


# ---------- Pass 3: candidate windows ----------

def candidate_windows(player: dict, ball: dict, rms: list[float], duration: float, sample_fps: float = track.SAMPLE_FPS,
                      start: float = 0.0, seconds: float | None = None) -> list[dict]:
    """Windows where the player is on screen, best first. `start`/`seconds` limit to one segment of the game."""
    end = min(duration, start + seconds) if seconds else duration
    frames_per_window = int(WINDOW_S * sample_fps)
    p_t, p_box = player["t"], player["xyxy"]
    centers = (p_box[:, :2] + p_box[:, 2:]) / 2
    heights = np.maximum(p_box[:, 3] - p_box[:, 1], 1)
    ball_at = {round(float(t), 2): b for t, b in zip(ball["t"], ball["xyxy"])}
    out = []
    t0 = start
    while t0 + 1.0 <= end:
        m = (p_t >= t0) & (p_t < t0 + WINDOW_S)
        n = int(m.sum())
        if n >= MIN_VISIBLE * frames_per_window:
            idx = np.where(m)[0]
            near = 0
            for i in idx:  # ball centre inside the player box grown by 50%
                b = ball_at.get(round(float(p_t[i]), 2))
                if b is not None:
                    x1, y1, x2, y2 = p_box[i]
                    gx, gy = 0.25 * (x2 - x1), 0.25 * (y2 - y1)
                    bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                    near += (x1 - gx <= bx <= x2 + gx) and (y1 - gy <= by <= y2 + gy)
            # speed in player-heights per second, so it doesn't depend on how far the camera is
            speed = float(np.median(np.linalg.norm(np.diff(centers[idx], axis=0), axis=1) / heights[idx][1:])) * sample_fps \
                if len(idx) > 1 else 0.0
            loud = max(rms[int(t0):int(t0 + WINDOW_S) + 1], default=0.0)
            out.append({"start": round(t0, 2), "end": round(min(t0 + WINDOW_S, duration), 2),
                        "visible": round(n / frames_per_window, 3), "ball_near": round(float(near) / n, 3),
                        "speed": round(speed, 3), "loud": round(float(loud), 4)})
        t0 += STRIDE_S
    if out:  # rank: ball near counts most; speed and noise are normalised within this game
        top_speed = max(w["speed"] for w in out) or 1
        top_loud = max(w["loud"] for w in out) or 1
        for w in out:
            w["score"] = round(float(0.5 * w["ball_near"] + 0.3 * w["speed"] / top_speed + 0.2 * w["loud"] / top_loud), 4)
        out.sort(key=lambda w: -w["score"])
    return out


# ---------- frame packs ----------

def _player_box_at(player: dict, t: float):
    if not len(player["t"]):
        return None
    i = int(np.argmin(np.abs(player["t"] - t)))
    return player["xyxy"][i] if abs(player["t"][i] - t) <= 0.21 else None  # within one 5 fps sample


def _crop_around(box, w: int, h: int):
    """Square region around the player, tall enough to keep the rim in view on a drive or dunk."""
    x1, y1, x2, y2 = (float(v) for v in box)
    side = min(2.6 * (y2 - y1), w, h)
    cx, top = (x1 + x2) / 2, y1 - 1.2 * (y2 - y1)
    left = min(max(0.0, cx - side / 2), w - side)
    top = min(max(0.0, top), h - side)
    return left, top, left + side, top + side


def frame_pack(proxy: Path, w: int, h: int, player: dict, start: float, seconds: float, fps: float) -> dict:
    """JPEGs: close-up crops that follow the player (TARGET outlined), plus one full-court frame for context.
    Full frames at 768 px made players tiny: a real dunk fell between two frames and both models missed it."""
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-i", str(proxy), "-t", f"{seconds:.3f}",
           "-vf", f"fps={fps}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    n = len(raw) // (w * h * 3)
    boxes = [_player_box_at(player, start + i / fps) for i in range(n)]
    seen = [b for b in boxes if b is not None]
    held, out, context = (seen[0] if seen else None), [], None
    for i in range(n):
        img = Image.fromarray(np.frombuffer(raw, np.uint8, w * h * 3, i * w * h * 3).reshape(h, w, 3))
        box = boxes[i]
        held = box if box is not None else held  # box missing for a moment: keep the camera where he was
        if i == 0:
            ctx = img.resize((CONTEXT_W, round(h * CONTEXT_W / w)))
            if held is not None:
                ImageDraw.Draw(ctx).rectangle([float(v) * CONTEXT_W / w for v in held], outline=TARGET, width=4)
            context = [_jpeg(ctx)]
        if held is None:
            continue
        region = _crop_around(held, w, h)
        crop = img.crop(region).resize((CLOSEUP_PX, CLOSEUP_PX))
        d = ImageDraw.Draw(crop)
        k = CLOSEUP_PX / (region[2] - region[0])
        if box is not None:
            x1, y1, x2, y2 = ((float(v) - region[j % 2]) * k for j, v in enumerate(box))  # x minus left, y minus top
            d.rectangle([x1, y1, x2, y2], outline=TARGET, width=4)
            d.text((x1 + 3, max(0, y1 - 14)), "TARGET", fill=TARGET)
        else:
            d.text((6, 22), "TARGET box missing", fill=TARGET)
        d.text((6, 6), f"frame {i}", fill=(255, 255, 0))
        out.append(_jpeg(crop))
    after = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{start + seconds + SCORE_LAG_S:.3f}", "-i", str(proxy),
                            "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    if context and len(after) >= w * h * 3:  # past the end of the video there is no "after" view
        img = Image.fromarray(np.frombuffer(after, np.uint8, w * h * 3).reshape(h, w, 3))
        context.append(_jpeg(img.resize((CONTEXT_W, round(h * CONTEXT_W / w)))))
    return {"frames": out, "context": context or []}


def _jpeg(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=80)
    return buf.getvalue()


# ---------- Pass 4: Claude ----------

def _client():
    import anthropic
    return anthropic.Anthropic(max_retries=5)


PLACEHOLDER_KEY = "paste-your-key-here"  # what the starter .env contains


def api_key_present() -> bool:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    return bool((key and key != PLACEHOLDER_KEY) or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def _image(b: bytes) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.standard_b64encode(b).decode()}}


def _content(pack: dict, fps: float) -> list[dict]:
    blocks = []
    for i, b in enumerate(pack["frames"]):
        blocks += [{"type": "text", "text": f"Frame {i}"}, _image(b)]
    for label, b in zip(("Full-court view at the start (TARGET boxed):",
                         f"Full-court view {SCORE_LAG_S:g} s after the end:"), pack["context"]):
        blocks += [{"type": "text", "text": label}, _image(b)]
    blocks.append({"type": "text", "text": f"{len(pack['frames'])} frames, {fps:g} per second. What does the TARGET player do?"})
    return blocks


def classify(pack: dict, fps: float, model: str, cache_dir: Path, client=None) -> dict:
    """One Claude call, cached by model + prompt + images. Returns {"result", "usage", "cost", "cached"}."""
    images = pack["frames"] + pack["context"]
    facts = PROMPT_MODE == "facts"
    system = SYSTEM_FACTS if facts else SYSTEM
    key = hashlib.sha256(json.dumps([model, PROMPT_VERSION, system, fps]).encode() + b"".join(images)).hexdigest()
    f = cache_dir / f"{key}.json"
    if f.exists():
        return json.loads(f.read_text()) | {"cost": 0.0, "cached": True}
    client = client or _client()
    kwargs = dict(model=model, max_tokens=4096, system=system, output_format=Facts if facts else WindowResult,
                  output_config={"effort": "medium"},
                  messages=[{"role": "user", "content": _content(pack, fps)}])
    if model == SONNET:  # on a refusal, the API retries on a fallback model inside the same call
        resp = client.beta.messages.parse(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
    else:
        resp = client.messages.parse(**kwargs)
    usage = {"input": resp.usage.input_tokens, "output": resp.usage.output_tokens}
    if resp.stop_reason == "refusal" or resp.parsed_output is None:
        result = {"player_has_ball": False, "assist_possible": False, "events": [], "refused": True}
    else:
        result = resp.parsed_output.model_dump()
        result = facts_to_result(result) if facts else result
    pin, pout = PRICE[model]
    out = {"result": result, "usage": usage, "model": model, "cost": (usage["input"] * pin + usage["output"] * pout) / 1e6}
    cache_dir.mkdir(exist_ok=True)
    f.write_text(json.dumps(out))
    return out | {"cached": False}


def _cost(model: str, tokens: tuple[int, int]) -> float:
    pin, pout = PRICE[model]
    return (tokens[0] * pin + tokens[1] * pout) / 1e6


def haiku_cost(n_windows: int) -> float:
    """First-pass cost only: used to decide how many windows to queue under the budget."""
    return n_windows * _cost(HAIKU, HAIKU_PER_WINDOW)


def estimate_cost(n_windows: int) -> float:
    """What a run will likely cost: first pass plus the expected share of Sonnet second opinions."""
    return haiku_cost(n_windows) + n_windows * EXPECTED_ESCALATION_RATE * _cost(SONNET, SONNET_PER_CALL)


def _events_from(window: dict, res: dict, model: str, fps: float) -> list[dict]:
    return [{"type": e["type"], "t_start": round(window["start"] + e["frame_start"] / fps, 2),
             "t_end": round(window["start"] + max(e["frame_end"], e["frame_start"]) / fps, 2),
             "confidence": round(float(e["confidence"]), 3), "model": model, "reason": e["short_reason"],
             "window": window["start"]}
            for e in res["result"]["events"]]


def merge(events: list[dict], gap: float = 1.0) -> list[dict]:
    """Overlapping windows see the same play twice: merge same-type events that touch, keep the surest."""
    out = []
    for e in sorted(events, key=lambda e: (e["type"], e["t_start"])):
        last = out[-1] if out and out[-1]["type"] == e["type"] else None
        if last and e["t_start"] <= last["t_end"] + gap:
            best = max(last, e, key=lambda x: x["confidence"])
            out[-1] = best | {"t_start": min(last["t_start"], e["t_start"]), "t_end": max(last["t_end"], e["t_end"])}
        else:
            out.append(dict(e))
    return sorted(out, key=lambda e: e["t_start"])


def detect(job_dir: Path, meta: dict, start: float = 0.0, seconds: float | None = None, budget: float = BUDGET_USD,
           classify_fn=classify, on_progress=lambda p: None) -> dict:
    t0 = time.time()
    p = meta["proxy"]
    w, h, duration = p["width"], p["height"], p["duration_sec"]
    player = dict(np.load(job_dir / "player.npz"))
    tr = track.load(job_dir / "tracks.npz")
    bm = tr["cls"] == track.BALL
    ball = {"t": tr["t"][bm], "xyxy": tr["xyxy"][bm]}
    rms = json.loads((job_dir / "audio_rms.json").read_text()) if (job_dir / "audio_rms.json").exists() else []
    windows = candidate_windows(player, ball, rms, duration, meta.get("tracking", {}).get("sample_fps", track.SAMPLE_FPS),
                                start=start, seconds=seconds)
    cache = job_dir / "vlm_cache"
    lock = threading.Lock()
    spent = {"usd": 0.0, "calls": 0, "cached": 0, "escalations": 0, "assist_rechecks": 0, "refusals": 0, "skipped": 0}
    per_window = haiku_cost(1)  # queue by first-pass cost; the hard stop below covers second opinions
    chosen = windows[: max(0, int(budget / per_window))] if per_window else windows  # best-first within budget
    events, done = [], [0]

    def call(pack, fps, model):
        if spent["usd"] >= budget:  # stop once the budget is spent (calls already in flight may finish)
            with lock:
                spent["skipped"] += 1
            return None
        r = classify_fn(pack, fps, model, cache)
        with lock:
            spent["usd"] += r["cost"]
            spent["calls"] += 1
            spent["cached"] += r["cached"]
            spent["refusals"] += bool(r["result"].get("refused"))
        return r

    def run(win):
        found = []
        secs = win["end"] - win["start"]
        r = call(frame_pack(job_dir / "proxy.mp4", w, h, player, win["start"], secs, FPS), FPS, HAIKU)
        if r is None:
            return found
        found += _events_from(win, r, HAIKU, FPS)
        if r["result"]["assist_possible"]:  # look longer to see whether the pass led to a basket
            long = min(ASSIST_WINDOW_S, duration - win["start"])
            r2 = call(frame_pack(job_dir / "proxy.mp4", w, h, player, win["start"], long, FPS), FPS, HAIKU)
            if r2:
                with lock:
                    spent["assist_rechecks"] += 1
                found += [e for e in _events_from(win, r2, HAIKU, FPS) if e["type"] == "assist"]
        unsure = [e for e in found if e["type"] in ALWAYS_ESCALATE
                  or (e["type"] in ESCALATE_IF_UNSURE and CONF_LOW <= e["confidence"] < CONF_HIGH)]
        if unsure:  # second opinion from a stronger model on more frames; it replaces Haiku's answer
            r3 = call(frame_pack(job_dir / "proxy.mp4", w, h, player, win["start"], secs, ESCALATE_FPS), ESCALATE_FPS, SONNET)
            if r3:
                with lock:
                    spent["escalations"] += 1
                found = [e for e in found if e not in unsure] + _events_from(win, r3, SONNET, ESCALATE_FPS)
        return found

    with ThreadPoolExecutor(CONCURRENCY) as pool:
        for found in pool.map(run, chosen):
            events += found
            done[0] += 1
            on_progress(done[0] / max(len(chosen), 1))

    merged = merge(events)
    (job_dir / "events.json").write_text(json.dumps(merged, indent=1))
    by_type = {}
    for e in merged:
        by_type[e["type"]] = by_type.get(e["type"], 0) + 1
    return {
        "events": len(merged), "by_type": by_type, "segment": [start, start + seconds if seconds else duration],
        "windows_found": len(windows), "windows_sent": len(chosen), "budget_usd": budget,
        "cost_usd": round(spent["usd"], 4), "api_calls": spent["calls"], "cached_calls": spent["cached"],
        "escalations": spent["escalations"], "assist_rechecks": spent["assist_rechecks"],
        "refusals": spent["refusals"], "skipped_over_budget": spent["skipped"], "seconds": round(time.time() - t0, 1),
    }
