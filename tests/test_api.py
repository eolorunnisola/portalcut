import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from portalcut import app as app_mod, db, track
from portalcut.config import Config
from tests.test_ingest import tmp_db  # noqa: F401  (pytest fixture)


def test_config_parses_query_strings_and_rejects_bad_values():
    c = Config.from_dict({"detect_fps": "10", "gemini": "true", "tracker": "bytetrack", "unknown": 1})
    assert c.detect_fps == 10.0 and c.gemini is True and c.tracker == "bytetrack"
    for bad in ({"tracker": "magic"}, {"detect_fps": 0}, {"person_conf": 1.5}, {"proxy_height": 100}):
        with pytest.raises(ValueError):
            Config.from_dict(bad)


@pytest.fixture
def done_job(tmp_db):
    jid = "done"
    d = db.DATA_DIR / jid
    d.mkdir(parents=True)
    rows = np.array([(0, 0.0, 3, 0, 10, 10, 50, 110, 0.9), (1, 0.2, 3, 0, 20, 10, 60, 110, 0.8),
                     (1, 0.2, -1, 1, 70, 70, 80, 80, 0.4)], np.float32)
    track.save(rows, d / "tracks.npz")
    (d / "analytics.json").write_text(json.dumps({"units": "px", "calibrated": False, "note": "n", "players": [
        {"player_id": "T0-13", "visible_seconds": 4.0, "trajectory": [[0, 1, 2]], "heatmap": [[1]]},
        {"player_id": "trk-9", "visible_seconds": 0.4, "trajectory": [], "heatmap": [[1]]}]}))
    db.create(jid, "game.mp4")
    db.update(jid, status="ready", meta={"proxy": {"width": 320, "height": 180, "duration_sec": 1.0},
                                         "tracking": {"sample_fps": 5}, "config": Config().to_dict()})
    return TestClient(app_mod.app), jid


def test_tracking_data_filters_by_time_and_track(done_job):
    c, jid = done_job
    r = c.get(f"/api/jobs/{jid}/tracks", params={"start": 0.1, "track_id": 3}).json()
    assert r["columns"][2] == "track_id" and [row[0] for row in r["rows"]] == [1]


def test_players_are_slim_and_one_player_is_full(done_job):
    c, jid = done_job
    ps = c.get(f"/api/jobs/{jid}/players", params={"min_seconds": 1}).json()
    assert [p["player_id"] for p in ps["players"]] == ["T0-13"] and "trajectory" not in ps["players"][0]
    assert ps["units"] == "px" and ps["calibrated"] is False
    assert c.get(f"/api/jobs/{jid}/players/T0-13").json()["player"]["trajectory"] == [[0, 1, 2]]
    assert c.get(f"/api/jobs/{jid}/players/nobody").status_code == 404


def test_missing_outputs_are_404_and_bad_runs_rejected(done_job):
    c, jid = done_job
    assert c.get(f"/api/jobs/{jid}/annotated").status_code == 404
    assert c.get(f"/api/jobs/{jid}/analysis").status_code == 404
    assert c.post(f"/api/jobs/{jid}/track", json={"start_stage": "nope"}).status_code == 400
    assert c.post(f"/api/jobs/{jid}/track", json={"config": {"tracker": "magic"}}).status_code == 400
    assert c.put(f"/api/jobs/{jid}/calibration",
                 json={"image_points": [[0, 0]], "world_points": [[0, 0]], "surface": [28, 15]}).status_code == 400


def test_upload_rejects_bad_settings(tmp_db):
    r = TestClient(app_mod.app).post("/api/jobs", params={"filename": "g.mp4", "detect_fps": "500"}, content=b"x")
    assert r.status_code == 400 and "detect_fps" in r.json()["detail"]
