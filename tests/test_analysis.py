import json
import subprocess
from types import SimpleNamespace

import pytest

from portalcut import analysis
from portalcut.analysis import GameAnalysis, call_gemini


class Err(analysis.errors.APIError):
    def __init__(self, code):
        super().__init__(code, {"error": {"message": "x", "status": "S"}})


class FakeClient:
    def __init__(self, items=None):
        self.items = list(items or [])
        self.calls = []
        self.models = SimpleNamespace(generate_content=self._gen)

    def _gen(self, model, contents, config):
        self.calls.append(contents)
        if self.items:
            it = self.items.pop(0)
            if isinstance(it, Exception):
                raise it
            return it
        n = len(self.calls)
        data = {"summary": f"s{n}", "segments": [], "player_notes": [],
                "notable_events": [{"t": 100.0 - n, "type": "shot", "player_id": None,
                                    "description": "d", "confidence": 0.5}]}
        return ok(data)


def ok(data, tin=1000, tout=100):
    return SimpleNamespace(parsed=GameAnalysis(**data), text=json.dumps(data),
                           usage_metadata=SimpleNamespace(prompt_token_count=tin, candidates_token_count=tout))


EMPTY = {"summary": "s", "segments": [], "notable_events": [], "player_notes": []}


def test_analyze_batches_merges_and_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "sample_frames", lambda *a, **k: [(i * 30.0, b"jpg") for i in range(20)])
    (tmp_path / "analytics.json").write_text(json.dumps({"units": "px", "players": []}))
    c = FakeClient()
    out = analysis.analyze(tmp_path, {"duration_sec": 600}, batch=8, client=c)
    assert len(c.calls) == 3 and out["requests"] == 3 and out["frames_sent"] == 20
    assert [len(x) for x in c.calls] == [9, 9, 5]
    ts = [e["t"] for e in out["result"]["notable_events"]]
    assert ts == sorted(ts) and len(ts) == 3
    assert out["result"]["summary"] == "s1 s2 s3"
    assert out["input_tokens"] == 3000 and out["output_tokens"] == 300
    assert out["cost_usd"] == pytest.approx(3 * (1000 * 0.30 + 100 * 2.50) / 1e6)
    assert json.loads((tmp_path / "analysis.json").read_text())["requests"] == 3


def test_retry_transient_then_success():
    sleeps = []
    c = FakeClient([Err(429), Err(503), ok(EMPTY)])
    res, usage = call_gemini(["x"], client=c, sleep=sleeps.append)
    assert res["summary"] == "s" and sleeps == [1.0, 2.0] and usage["input_tokens"] == 1000


def test_no_retry_on_400():
    sleeps = []
    c = FakeClient([Err(400), ok(EMPTY)])
    with pytest.raises(Err):
        call_gemini(["x"], client=c, sleep=sleeps.append)
    assert len(c.calls) == 1 and sleeps == []


def test_missing_key_raises(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert not analysis.api_key_present()
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        analysis.analyze(tmp_path, {})


def test_summarize_tracking_top_n():
    players = [{"player_id": f"p{i}", "team": 0, "number": None, "visible_seconds": i,
                "distance": 1.0, "avg_position": [0, 0], "zones": {}, "extra": "drop"} for i in range(30)]
    s = analysis.summarize_tracking({"units": "px", "calibrated": False, "note": "n", "players": players}, top_n=5)
    assert [p["player_id"] for p in s["players"]] == ["p29", "p28", "p27", "p26", "p25"]
    assert s["units"] == "px" and s["note"] == "n" and "extra" not in s["players"][0]


def test_sample_frames_real_ffmpeg(tmp_path):
    v = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10:duration=5",
                    "-c:v", "libx264", str(v)], check=True)
    f = analysis.sample_frames(v, 5.0, interval_s=1.0, max_frames=10, width=160)
    assert [t for t, _ in f] == [0.0, 1.0, 2.0, 3.0, 4.0] and f[0][1][:2] == b"\xff\xd8"
    assert len(analysis.sample_frames(v, 5.0, interval_s=1.0, max_frames=3)) == 3
