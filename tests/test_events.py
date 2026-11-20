"""Event detection, with Claude replaced by fakes: these tests never call the API or spend money."""
import json
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from portalcut import events, track


def _player(t_from, t_to, fps=5, box=(100, 50, 140, 150)):
    t = np.arange(t_from * fps, t_to * fps) / fps
    return {"frame": (t * fps).astype(np.int32), "t": t.astype(np.float32), "track_id": np.zeros(len(t), np.int32),
            "xyxy": np.tile(np.array(box, np.float32), (len(t), 1)), "conf": np.full(len(t), 0.9, np.float32)}


NO_BALL = {"t": np.zeros(0, np.float32), "xyxy": np.zeros((0, 4), np.float32)}


def test_windows_only_where_player_is_visible_and_ranked_by_ball():
    p = _player(0, 20)
    ball_t = np.arange(50, 75) / 5  # ball near the player from 10 s to 15 s
    ball = {"t": ball_t.astype(np.float32), "xyxy": np.tile(np.array([115, 90, 125, 100], np.float32), (len(ball_t), 1))}
    ws = events.candidate_windows(p, ball, [0.1] * 40, duration=40)
    assert all(w["start"] < 20 for w in ws)              # nothing after the player leaves at 20 s
    assert ws[0]["start"] == 10.0 and ws[0]["ball_near"] == 1.0
    json.dumps(ws)                                       # plain JSON, safe to store in meta


def test_windows_respect_segment():
    ws = events.candidate_windows(_player(0, 60), NO_BALL, [], duration=60, start=20, seconds=10)
    assert ws and all(20 <= w["start"] and w["start"] < 30 for w in ws)


def test_merge_same_play_seen_by_two_windows():
    a = {"type": "steal", "t_start": 10.0, "t_end": 11.0, "confidence": 0.6, "model": "h", "reason": "", "window": 7.5}
    b = a | {"t_start": 10.5, "t_end": 11.5, "confidence": 0.9, "window": 10.0}
    c = a | {"type": "rebound", "t_start": 10.2}
    out = events.merge([a, b, c])
    steals = [e for e in out if e["type"] == "steal"]
    assert len(out) == 2 and steals[0]["confidence"] == 0.9 and (steals[0]["t_start"], steals[0]["t_end"]) == (10.0, 11.5)


# ---- the Claude call: caching, cost, refusals ----

class FakeClient:
    def __init__(self, stop="end_turn", parsed=None):
        self.calls = []
        resp = SimpleNamespace(usage=SimpleNamespace(input_tokens=5000, output_tokens=200), stop_reason=stop,
                               parsed_output=parsed)
        rec = lambda **kw: (self.calls.append(kw), resp)[1]
        self.messages = SimpleNamespace(parse=rec)
        self.beta = SimpleNamespace(messages=SimpleNamespace(parse=rec))


PACK = {"frames": [b"\xff\xd8a", b"\xff\xd8b"], "context": [b"\xff\xd8c"]}


def test_classify_costs_once_then_reads_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(events, "PROMPT_MODE", "labels")  # this fake answers in the v1 shape
    out = events.WindowResult(player_has_ball=True, assist_possible=False,
                              events=[events.Event(type="steal", frame_start=1, frame_end=2, confidence=0.8, short_reason="x")])
    c = FakeClient(parsed=out)
    r1 = events.classify(PACK, 2.0, events.HAIKU, tmp_path, client=c)
    assert r1["cost"] == pytest.approx((5000 * 0.10 + 200 * 0.50) / 1e6) and not r1["cached"]
    r2 = events.classify(PACK, 2.0, events.HAIKU, tmp_path, client=c)
    assert r2["cached"] and r2["cost"] == 0 and len(c.calls) == 1
    assert r2["result"]["events"][0]["type"] == "steal"
    sent = c.calls[0]
    assert sent["model"] == "claude-haiku-5-5" and "temperature" not in sent and "tool_choice" not in sent
    assert sum(b["type"] == "image" for b in sent["messages"][0]["content"]) == 3


def test_sonnet_uses_refusal_fallback_and_refusal_means_no_events(tmp_path):
    c = FakeClient(stop="refusal")
    r = events.classify(PACK, 2.0, events.SONNET, tmp_path, client=c)
    assert c.calls[0]["fallbacks"] == "default" and "server-side-fallback-2026-07-01" in c.calls[0]["betas"]
    assert r["result"]["events"] == [] and r["result"]["refused"]


# ---- the whole pass with a fake classifier ----

@pytest.fixture
def job(tmp_path):
    tmp_path = tmp_path / "job"  # kept apart from tmp_db's data folder
    tmp_path.mkdir()
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=30:duration=12",
                    "-c:v", "libx264", "-preset", "ultrafast", str(tmp_path / "proxy.mp4")], check=True)
    np.savez_compressed(tmp_path / "player.npz", **_player(0, 12, box=(100, 40, 140, 160)))
    track.save(np.zeros((0, 9), np.float32), tmp_path / "tracks.npz")
    (tmp_path / "audio_rms.json").write_text(json.dumps([0.1] * 12))
    return tmp_path, {"proxy": {"width": 320, "height": 180, "duration_sec": 12.0}}


def fake(answers):
    """classify_fn that answers by model and records what it was asked."""
    calls = []

    def fn(pack, fps, model, cache_dir):
        calls.append((model, len(pack["frames"]), fps))
        res = answers(model, len(pack["frames"]))
        return {"result": res, "cost": 0.001, "cached": False, "model": model}
    fn.calls = calls
    return fn


def ev(type_, conf, f0=2, f1=3):
    return {"type": type_, "frame_start": f0, "frame_end": f1, "confidence": conf, "short_reason": "r"}


def test_confident_play_from_haiku_is_kept_without_escalation(job):
    d, meta = job
    fn = fake(lambda m, n: {"player_has_ball": True, "assist_possible": False, "events": [ev("steal", 0.9)]})
    out = events.detect(d, meta, classify_fn=fn)
    assert {c[0] for c in fn.calls} == {events.HAIKU}
    full = int(events.WINDOW_S * events.FPS)                       # 5 s of close-ups (end windows are shorter)
    assert all(c[2] == events.FPS for c in fn.calls) and max(c[1] for c in fn.calls) == full
    saved = json.loads((d / "events.json").read_text())
    # the fake reports a steal 1 s into every window, so each lands at a different time and stays separate
    assert out["by_type"] == {"steal": out["windows_sent"]} and {e["model"] for e in saved} == {events.HAIKU}
    assert out["cost_usd"] == pytest.approx(0.001 * out["api_calls"])


def test_made_shot_gets_a_sonnet_second_opinion(job):
    d, meta = job
    fn = fake(lambda m, n: {"player_has_ball": True, "assist_possible": False,
                            "events": [ev("made_shot", 0.95 if m == events.SONNET else 0.9)]})
    out = events.detect(d, meta, classify_fn=fn)
    assert out["escalations"] == out["windows_sent"]
    assert any(c == (events.SONNET, int(events.WINDOW_S * events.ESCALATE_FPS), events.ESCALATE_FPS) for c in fn.calls)
    saved = json.loads((d / "events.json").read_text())
    assert {e["model"] for e in saved} == {events.SONNET}


def test_sonnet_can_overrule_a_borderline_call(job):
    d, meta = job
    fn = fake(lambda m, n: {"player_has_ball": False, "assist_possible": False,
                            "events": [] if m == events.SONNET else [ev("block", 0.5)]})
    out = events.detect(d, meta, classify_fn=fn)
    assert out["events"] == 0 and out["escalations"] > 0


def test_possible_assist_gets_a_longer_look(job):
    d, meta = job
    short, long = int(events.WINDOW_S * events.FPS), int(events.ASSIST_WINDOW_S * events.FPS)
    fn = fake(lambda m, n: {"player_has_ball": True, "assist_possible": n == short,
                            "events": [ev("assist", 0.9, 4, 12)] if n == long and m == events.HAIKU else []})
    out = events.detect(d, meta, classify_fn=fn)
    assert out["assist_rechecks"] > 0
    assert any(c == (events.HAIKU, long, events.FPS) for c in fn.calls)  # 8 s window


def test_budget_limits_windows_and_segment_limits_time(job):
    d, meta = job
    fn = fake(lambda m, n: {"player_has_ball": False, "assist_possible": False, "events": []})
    out = events.detect(d, meta, budget=events.haiku_cost(2), classify_fn=fn)
    assert out["windows_sent"] == 2 < out["windows_found"]
    out = events.detect(d, meta, start=0, seconds=5, classify_fn=fn)
    assert out["segment"] == [0, 5] and out["windows_found"] <= 2


# ---- API ----

@pytest.fixture
def api_job(job, tmp_db, monkeypatch):
    import shutil
    from fastapi.testclient import TestClient
    from portalcut import app as app_mod, db
    d, meta = job
    shutil.copytree(d, db.DATA_DIR / "pl")
    db.create("pl", "g.mp4")
    db.update("pl", status="ready", meta=meta | {"selection": {"confirmed": [0]}})
    return TestClient(app_mod.app), db


from tests.test_ingest import tmp_db  # noqa: E402,F401  (pytest fixture)


def test_estimate_and_missing_key(api_job, monkeypatch):
    c, db = api_job
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    est = c.get("/api/jobs/pl/plays/estimate").json()
    assert est["windows"] == 5 and est["estimate_usd"] > 0 and est["api_key"] is False
    r = c.post("/api/jobs/pl/plays", json={})
    assert r.status_code == 400 and ".env" in r.json()["detail"]
    assert db.get("pl")["status"] == "ready"                 # nothing started


def test_budget_must_be_sane_and_player_picked_first(api_job, monkeypatch):
    c, db = api_job
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-not-used")
    assert c.post("/api/jobs/pl/plays", json={"budget": 50}).status_code == 400
    meta = db.get("pl")["meta"]
    meta.pop("selection")
    db.update("pl", meta=meta)
    assert c.post("/api/jobs/pl/plays", json={}).status_code == 409
    assert c.get("/api/jobs/pl/plays/estimate").status_code == 409
    assert c.get("/api/jobs/pl/plays").json() == []


def test_run_with_fake_claude_stores_results(api_job, monkeypatch):
    import time
    c, db = api_job
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-not-used")
    monkeypatch.setattr(events, "classify", fake(lambda m, n: {"player_has_ball": True, "assist_possible": False,
                                                               "events": [ev("rebound", 0.9)]}))
    real = events.detect
    monkeypatch.setattr(events, "detect", lambda *a, **k: real(*a, **k | {"classify_fn": events.classify}))
    assert c.post("/api/jobs/pl/plays", json={"seconds": 6}).status_code == 200
    for _ in range(100):
        if db.get("pl")["status"] == "ready":
            break
        time.sleep(0.1)
    meta = db.get("pl")["meta"]
    assert meta["plays"]["by_type"] == {"rebound": meta["plays"]["windows_sent"]} and meta["plays"]["segment"] == [0, 6]
    assert c.get("/api/jobs/pl/plays").json()[0]["type"] == "rebound"


def test_restart_while_finding_plays_keeps_job_usable(tmp_db):
    from portalcut import db
    db.create("f", "g.mp4")
    db.update("f", status="finding_plays")
    db.init()
    assert db.get("f")["status"] == "ready"


def test_env_file_is_loaded_but_real_env_wins(tmp_path, monkeypatch):
    from portalcut import app as app_mod
    (tmp_path / ".env").write_text("# comment\nPC_TEST_A='from-file'\nPC_TEST_B=from-file\n")
    monkeypatch.delenv("PC_TEST_A", raising=False)
    monkeypatch.setenv("PC_TEST_B", "real")
    app_mod._load_env(tmp_path / ".env")
    import os
    assert os.environ["PC_TEST_A"] == "from-file" and os.environ["PC_TEST_B"] == "real"
    monkeypatch.delenv("PC_TEST_A")


def test_placeholder_key_counts_as_missing(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", events.PLACEHOLDER_KEY)
    assert not events.api_key_present()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-real-looking")
    assert events.api_key_present()


def test_borderline_low_value_plays_get_no_second_opinion(job):
    d, meta = job
    fn = fake(lambda m, n: {"player_has_ball": False, "assist_possible": False,
                            "events": [ev("defensive_stop", 0.5), ev("possession", 0.5, 4, 5)]})
    out = events.detect(d, meta, classify_fn=fn)
    assert out["escalations"] == 0 and {c[0] for c in fn.calls} == {events.HAIKU}
    assert out["by_type"].get("defensive_stop")              # Haiku's answer stands


def test_borderline_steal_still_gets_a_second_opinion(job):
    d, meta = job
    fn = fake(lambda m, n: {"player_has_ball": False, "assist_possible": False, "events": [ev("steal", 0.5)]})
    assert events.detect(d, meta, classify_fn=fn)["escalations"] > 0


def test_estimate_includes_expected_second_opinions():
    assert events.estimate_cost(100) > events.haiku_cost(100) > 0
    # measured first-pass cost with close-ups: about $0.0012 per window
    assert events.haiku_cost(1) == pytest.approx(0.0012, abs=0.0001)


# ---- prompt v2: facts in, plays decided by rules ----

def facts(**kw):
    base = dict(live_action=True, target_visible=True, target_shot=False, shot_frame=-1, shot_result="no_shot",
                target_passed=False, pass_frame=-1, pass_result="no_pass", target_blocked_shot=False, block_frame=-1,
                target_took_ball_from_opponent=False, steal_frame=-1, target_team_has_ball_after_steal="no_steal",
                target_rebounded_missed_shot=False, rebound_frame=-1, target_team_score_went_up="unclear",
                certainty="high", what_happened="x")
    return base | kw


def kinds(f):
    return [e["type"] for e in events.facts_to_result(f)["events"]]


def test_rules_need_the_result_not_just_the_effort():
    assert kinds(facts(target_shot=True, shot_frame=3, shot_result="made")) == ["made_shot"]
    assert kinds(facts(target_shot=True, shot_frame=3, shot_result="unclear")) == []      # no visible result, no play
    assert kinds(facts(target_passed=True, pass_frame=2, pass_result="receiver_did_not_score")) == []
    assert kinds(facts(target_took_ball_from_opponent=True, steal_frame=4,
                       target_team_has_ball_after_steal="no")) == []                          # tipped but lost it
    assert kinds(facts(target_took_ball_from_opponent=True, steal_frame=4,
                       target_team_has_ball_after_steal="yes")) == ["steal"]
    assert kinds(facts()) == []                                                               # just defending


def test_rules_ignore_replays_and_unplaced_plays():
    assert kinds(facts(live_action=False, target_shot=True, shot_frame=3, shot_result="made")) == []
    assert kinds(facts(target_blocked_shot=True, block_frame=-1)) == []                     # no frame given
    r = events.facts_to_result(facts(target_passed=True, pass_frame=1, pass_result="unclear", certainty="low"))
    assert r["assist_possible"] and r["events"] == []


def test_assist_and_make_in_one_window_and_confidence():
    r = events.facts_to_result(facts(target_shot=True, shot_frame=8, shot_result="made", target_passed=True,
                                     pass_frame=2, pass_result="receiver_scored", certainty="medium"))
    assert sorted(e["type"] for e in r["events"]) == ["assist", "made_shot"]
    assert all(e["confidence"] == 0.7 for e in r["events"])


def test_classify_in_facts_mode_converts_and_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(events, "PROMPT_MODE", "facts")
    c = FakeClient(parsed=events.Facts(**facts(target_shot=True, shot_frame=3, shot_result="made")))
    r = events.classify(PACK, 2.0, events.HAIKU, tmp_path, client=c)
    assert [e["type"] for e in r["result"]["events"]] == ["made_shot"] and r["result"]["facts"]["shot_result"] == "made"
    assert c.calls[0]["output_format"] is events.Facts and c.calls[0]["system"] is events.SYSTEM_FACTS
    assert events.classify(PACK, 2.0, events.HAIKU, tmp_path, client=c)["cached"]


def test_frame_pack_is_closeups_that_follow_the_player(job):
    import io
    from PIL import Image
    d, _ = job
    pack = events.frame_pack(d / "proxy.mp4", 320, 180, dict(np.load(d / "player.npz")), 0.0, 2.0, events.FPS)
    assert len(pack["frames"]) == int(2.0 * events.FPS)
    assert Image.open(io.BytesIO(pack["frames"][0])).size == (events.CLOSEUP_PX, events.CLOSEUP_PX)
    assert len(pack["context"]) == 2 and Image.open(io.BytesIO(pack["context"][0])).width == events.CONTEXT_W


def test_crop_stays_inside_the_frame_and_keeps_room_above():
    left, top, right, bottom = events._crop_around((5, 10, 25, 70), 320, 180)   # player near the top-left corner
    assert left >= 0 and top >= 0 and right <= 320 and bottom <= 180
    assert abs((right - left) - (bottom - top)) < 1e-6 and (right - left) == pytest.approx(2.6 * 60)


def test_scoreboard_confirms_an_unclear_shot_one_level_lower():
    r = events.facts_to_result(facts(target_shot=True, shot_frame=15, shot_result="unclear",
                                     target_team_score_went_up="yes", certainty="high"))
    assert [(e["type"], e["confidence"]) for e in r["events"]] == [("made_shot", events.CERTAINTY["medium"])]
    assert kinds(facts(target_shot=True, shot_frame=15, shot_result="unclear", target_team_score_went_up="no")) == []
    assert kinds(facts(target_team_score_went_up="yes")) == []          # score went up but he didn't shoot
