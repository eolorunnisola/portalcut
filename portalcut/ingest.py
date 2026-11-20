"""Pass 0: probe the upload, make a 720p analysis proxy, and a per-second loudness curve."""

import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from portalcut import db, pipeline
from portalcut.config import Config

PROXY_MAX_HEIGHT = 720
AUDIO_RATE = 16000


def probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    info = json.loads(out)
    video = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    if video is None:
        raise ValueError("file has no video stream")
    num, den = (int(x) for x in video["avg_frame_rate"].split("/"))
    return {
        "duration_sec": float(info["format"]["duration"]),
        "width": video["width"],
        "height": video["height"],
        "fps": round(num / den, 3) if den else None,
        "video_codec": video["codec_name"],
        "has_audio": any(s["codec_type"] == "audio" for s in info["streams"]),
        "size_bytes": int(info["format"]["size"]),
    }


def _hw_encoder() -> bool:
    out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    return "h264_videotoolbox" in out


def make_proxy(src: Path, dst: Path, duration: float, on_progress=lambda p: None,
               max_height: int = PROXY_MAX_HEIGHT) -> None:
    # Hardware encoding on Apple Silicon is several times faster for 2-hour games; libx264 elsewhere.
    codec = ["-c:v", "h264_videotoolbox", "-b:v", "4M"] if _hw_encoder() else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23"]
    cmd = [
        "ffmpeg", "-y", "-v", "error", "-i", str(src),
        "-vf", f"scale=-2:'min({max_height},ih)'",
        *codec, "-pix_fmt", "yuv420p", "-an",
        "-progress", "pipe:1", "-nostats", str(dst),
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in proc.stdout:
        if line.startswith("out_time_us=") and duration > 0:
            us = line.split("=", 1)[1].strip()
            if us.isdigit():
                on_progress(min(1.0, int(us) / 1e6 / duration))
    if proc.wait() != 0:
        raise RuntimeError(f"proxy encode failed: {proc.stderr.read()[-500:]}")


def audio_rms(src: Path, wav: Path) -> list[float]:
    """Mono 16 kHz WAV plus RMS loudness per second (0..1). A weak excitement signal for later passes."""
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", str(AUDIO_RATE), str(wav)],
        check=True,
    )
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(wav), "-f", "s16le", "-"], capture_output=True, check=True
    ).stdout
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    n = len(samples) // AUDIO_RATE
    if n == 0:
        return []
    per_sec = samples[: n * AUDIO_RATE].reshape(n, AUDIO_RATE)
    return np.sqrt((per_sec**2).mean(axis=1)).round(5).tolist()


def run(job_id: str, job_dir: Path, src: Path) -> None:
    t0 = time.time()
    try:
        db.update(job_id, status="probing", progress=0)
        cfg = Config.from_dict((db.get(job_id)["meta"] or {}).get("config"))  # set at upload
        meta = probe(src) | {"config": cfg.to_dict()}
        db.update(job_id, status="making_proxy", meta=meta)

        make_proxy(src, job_dir / "proxy.mp4", meta["duration_sec"],
                   lambda p: db.update(job_id, progress=round(p, 3)), cfg.proxy_height)

        db.update(job_id, status="audio", progress=0)
        rms = audio_rms(src, job_dir / "audio.wav") if meta["has_audio"] else []
        (job_dir / "audio_rms.json").write_text(json.dumps(rms))

        meta["proxy"] = probe(job_dir / "proxy.mp4") | {"path": "proxy.mp4"}
        meta["audio_seconds"] = len(rms)
        meta["ingest_cpu_seconds"] = round(time.time() - t0, 1)
        meta["cost_usd"] = 0.0  # all local in Pass 0
        db.update(job_id, status="ingested", progress=1.0, meta=meta)
    except Exception as e:  # surface any failure in the UI instead of a stuck job
        db.update(job_id, status="failed", error=f"{type(e).__name__}: {e}"[:1000])
        return
    pipeline.run(job_id, job_dir)


def check_tools() -> None:
    missing = [t for t in ("ffmpeg", "ffprobe") if shutil.which(t) is None]
    if missing:
        raise SystemExit(f"Missing {', '.join(missing)}. Install with: brew install ffmpeg")
