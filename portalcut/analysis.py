"""Optional Gemini Flash interpretation of a processed game.

Sends a few sampled frames (annotated video preferred) plus a compact tracking summary, never
per-frame detection. Retries transient errors, returns structured JSON, logs tokens and cost.
Configured only through env vars: GEMINI_API_KEY (or GOOGLE_API_KEY), GEMINI_MODEL,
GEMINI_PRICE_IN, GEMINI_PRICE_OUT ($ per million tokens, used for cost logging only).
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Literal

from google import genai
from google.genai import errors, types
from pydantic import BaseModel

log = logging.getLogger("portalcut.analysis")

SYSTEM = (
    "You analyse a team-sport game from a few sampled video frames and a structured tracking summary. "
    "Player positions come from automatic tracking and may be wrong. Distances may be uncalibrated "
    "pixel estimates, so do not state them as metres. Refer to players by the player_id labels drawn "
    "in the frames. Say 'unclear' rather than guess. Use the given timestamps (seconds) for all times."
)


class Segment(BaseModel):
    t_start: float
    t_end: float
    phase: Literal["attack", "defense", "transition", "stoppage", "unclear"]
    team: Literal["0", "1", "unknown"]
    description: str


class Event(BaseModel):
    t: float
    type: str
    player_id: str | None
    description: str
    confidence: float


class PlayerNote(BaseModel):
    player_id: str
    note: str


class GameAnalysis(BaseModel):
    summary: str
    segments: list[Segment]
    notable_events: list[Event]
    player_notes: list[PlayerNote]


def api_key() -> str | None:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or None


def api_key_present() -> bool:
    return api_key() is not None


def model_name() -> str:
    return os.environ.get("GEMINI_MODEL") or "gemini-flash-latest"


def _price(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def sample_frames(video: Path | str, duration_s: float, interval_s: float = 30.0,
                  max_frames: int = 24, width: int = 768) -> list[tuple[float, bytes]]:
    """JPEG frames every interval_s (at most max_frames), grabbed with ffmpeg."""
    n = min(max_frames, int(max(duration_s, 0) // interval_s) + 1)
    out: list[tuple[float, bytes]] = []
    for i in range(n):
        t = i * interval_s
        cmd = ["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1",
               "-vf", f"scale={width}:-2", "-f", "image2", "-c:v", "mjpeg", "-"]
        data = subprocess.run(cmd, capture_output=True).stdout
        if data:
            out.append((t, data))
    return out


def summarize_tracking(analytics: dict, top_n: int = 12) -> dict:
    """Compact summary for the prompt: units, calibration, note, and the top players by time on screen."""
    players = analytics.get("players") or []
    if isinstance(players, dict):
        players = [{"player_id": k, **v} for k, v in players.items()]
    top = sorted(players, key=lambda p: p.get("visible_seconds") or 0, reverse=True)[:top_n]
    keys = ("player_id", "team", "number", "visible_seconds", "distance", "avg_position", "zones")
    return {
        "units": analytics.get("units"),
        "calibrated": analytics.get("calibrated"),
        "note": analytics.get("note"),
        "players": [{k: p.get(k) for k in keys} for p in top],
    }


def _transient(e: Exception) -> bool:
    code = getattr(e, "code", None)
    return isinstance(e, errors.APIError) and isinstance(code, int) and (code == 429 or 500 <= code < 600)


def call_gemini(contents: list[Any], client: Any = None, retries: int = 4,
                sleep: Callable[[float], None] = time.sleep) -> tuple[dict, dict]:
    """generate_content with a JSON schema. Retries 429/5xx with 1,2,4,8 s backoff; other errors raise."""
    client = client or genai.Client(api_key=api_key())
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM, response_mime_type="application/json",
        response_schema=GameAnalysis, temperature=0.2)
    for attempt in range(retries + 1):
        try:
            resp = client.models.generate_content(model=model_name(), contents=contents, config=config)
            break
        except Exception as e:
            if not _transient(e) or attempt == retries:
                raise
            delay = 2.0 ** attempt
            log.warning("gemini transient error %s, retry %d in %.0fs", getattr(e, "code", "?"), attempt + 1, delay)
            sleep(delay)
    parsed = resp.parsed
    result = parsed.model_dump() if isinstance(parsed, BaseModel) else parsed or json.loads(resp.text)
    um = resp.usage_metadata
    tin, tout = (getattr(um, "prompt_token_count", 0) or 0), (getattr(um, "candidates_token_count", 0) or 0)
    cost = (tin * _price("GEMINI_PRICE_IN", 0.30) + tout * _price("GEMINI_PRICE_OUT", 2.50)) / 1e6
    log.info("gemini %s: %d in, %d out tokens, ~$%.4f", model_name(), tin, tout, cost)
    return result, {"input_tokens": tin, "output_tokens": tout, "cost_usd": cost}


def analyze(job_dir: Path | str, meta: dict, interval_s: float = 30.0, max_frames: int = 24,
            batch: int = 8, client: Any = None) -> dict:
    """Run the analysis for one job and write analysis.json."""
    if client is None and not api_key_present():
        raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set; Gemini analysis is unavailable")
    job_dir = Path(job_dir)
    video = job_dir / "annotated.mp4"
    if not video.exists():
        video = job_dir / (meta.get("proxy") or {}).get("path", "proxy.mp4")
    duration = float(meta.get("duration_sec") or 0)
    frames = sample_frames(video, duration, interval_s, max_frames)
    ap = job_dir / "analytics.json"
    summary = summarize_tracking(json.loads(ap.read_text())) if ap.exists() else {}
    merged: dict[str, Any] = {"summary": [], "segments": [], "notable_events": [], "player_notes": []}
    usage = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
    requests = 0
    for i in range(0, len(frames), batch):
        chunk = frames[i:i + batch]
        text = (f"Tracking summary (JSON): {json.dumps(summary)}\n"
                f"Game duration: {duration:.0f}s. The {len(chunk)} frames below are at t = "
                f"{', '.join(f'{t:.0f}s' for t, _ in chunk)}, in order.")
        parts: list[Any] = [text] + [types.Part.from_bytes(data=b, mime_type="image/jpeg") for _, b in chunk]
        res, u = call_gemini(parts, client=client)
        requests += 1
        for k in usage:
            usage[k] += u[k]
        merged["summary"].append(res.get("summary", ""))
        for k in ("segments", "notable_events", "player_notes"):
            merged[k] += res.get(k, [])
    merged["segments"].sort(key=lambda s: s["t_start"])
    merged["notable_events"].sort(key=lambda e: e["t"])
    merged["summary"] = " ".join(s for s in merged["summary"] if s)
    out = {"model": model_name(), "frames_sent": len(frames), "requests": requests, **usage, "result": merged}
    (job_dir / "analysis.json").write_text(json.dumps(out, indent=2))
    return out
