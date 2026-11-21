"""Read jersey numbers off torso-crop strips with Claude Haiku; a majority vote in code decides the number."""

import hashlib
import json
from collections import Counter
from pathlib import Path

from pydantic import BaseModel

from portalcut.events import HAIKU, PRICE, _client, _image

BATCH = 10
PROMPT_VERSION = 1  # bump to invalidate the cache when the prompt changes


class TrackReading(BaseModel):
    track_id: int
    readings: list[str]


class Batch(BaseModel):
    tracks: list[TrackReading]


SYSTEM = """These are torso crops of basketball players. Each track is one strip of 1 to 4 crops of the same \
player, left to right, separated by dark gaps.

For each track, read the jersey number on each crop, left to right. Give digits only (for example "13", "0", \
"00"). Give "" for a crop where the number is not clearly readable. Never guess from context."""


def read_batch(job_dir: Path, tids: list[int], client=None) -> dict:
    """One Haiku call for up to BATCH tracks. Returns {"readings": {tid: [str]}, "cost", "cached"}."""
    job_dir = Path(job_dir)
    content, images = [], []
    for tid in tids:
        img = (job_dir / "crops" / f"{tid}_torso.jpg").read_bytes()
        images.append(img)
        content += [{"type": "text", "text": f"Track {tid}:"}, _image(img)]
    key = hashlib.sha256(json.dumps([HAIKU, PROMPT_VERSION, SYSTEM, tids]).encode() + b"".join(images)).hexdigest()
    cache = job_dir / "vlm_cache"
    f = cache / f"jersey_{key}.json"
    if f.exists():
        return {"readings": {int(k): v for k, v in json.loads(f.read_text()).items()}, "cost": 0.0, "cached": True}
    resp = (client or _client()).messages.parse(
        model=HAIKU, max_tokens=2048, system=SYSTEM, output_format=Batch, output_config={"effort": "low"},
        messages=[{"role": "user", "content": content}])
    refused = resp.stop_reason == "refusal" or resp.parsed_output is None
    readings = {} if refused else {t.track_id: t.readings for t in resp.parsed_output.tracks if t.track_id in tids}
    pin, pout = PRICE[HAIKU]
    cost = (resp.usage.input_tokens * pin + resp.usage.output_tokens * pout) / 1e6
    cache.mkdir(exist_ok=True)
    f.write_text(json.dumps(readings))
    return {"readings": readings, "cost": cost, "cached": False}


def vote(readings: list[str]) -> dict:
    counts = Counter(r for r in readings if r)
    if not counts:
        return {"number": None, "votes": 0, "conflict": False}
    number, votes = counts.most_common(1)[0]
    conflict = len(counts) > 1
    # a lone reading counts only if nothing disagrees with it; ties like 13/18 stay undecided
    ok = votes >= 2 or not conflict
    return {"number": number if ok else None, "votes": votes, "conflict": conflict}


def load_numbers(job_dir: Path) -> dict[int, dict]:
    f = Path(job_dir) / "numbers.json"
    return {int(k): v for k, v in json.loads(f.read_text()).items()} if f.exists() else {}


def read_numbers(job_dir: Path, tids: list[int], budget: float = 1.0, client=None) -> dict:
    job_dir = Path(job_dir)
    numbers = {str(k): v for k, v in load_numbers(job_dir).items()}
    todo = [t for t in tids if str(t) not in numbers and (job_dir / "crops" / f"{t}_torso.jpg").exists()]
    spent, calls, cached, n_read = 0.0, 0, 0, 0
    for i in range(0, len(todo), BATCH):
        if spent >= budget:
            break
        batch = todo[i:i + BATCH]
        r = read_batch(job_dir, batch, client)
        spent += r["cost"]
        calls += not r["cached"]
        cached += r["cached"]
        for tid in batch:
            rd = r["readings"].get(tid, [])
            numbers[str(tid)] = vote(rd) | {"readings": rd}
        n_read += len(batch)
    (job_dir / "numbers.json").write_text(json.dumps(numbers, indent=1))
    return {"read": n_read, "skipped_over_budget": len(todo) - n_read, "cost_usd": round(spent, 4),
            "api_calls": calls, "cached_calls": cached}
