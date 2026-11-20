import subprocess

import numpy as np
from fastapi.testclient import TestClient

from portalcut import app as app_mod, db, track
from tests.test_ingest import sample, tmp_db  # noqa: F401  (pytest fixtures)


def _fake_rows():
    # frame, t, track_id, cls, x1, y1, x2, y2, conf
    return np.array([
        (0, 0.0, 3, track.PERSON, 10, 10, 60, 120, 0.9),
        (0, 0.0, -1, track.BALL, 70, 70, 80, 80, 0.4),
        (1, 0.2, 3, track.PERSON, 12, 10, 62, 120, 0.9),
    ], dtype=np.float32)


def test_save_load_roundtrip(tmp_path):
    track.save(_fake_rows(), tmp_path / "t.npz")
    t = track.load(tmp_path / "t.npz")
    assert t["track_id"].tolist() == [3, -1, 3]
    assert t["cls"].tolist() == [track.PERSON, track.BALL, track.PERSON]
    assert t["xyxy"].shape == (3, 4) and t["frame"].dtype == np.int32


def test_read_frames_and_debug_video(sample, tmp_path):
    proxy = tmp_path / "p.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(sample), "-vf", "scale=320:180", str(proxy)], check=True)
    frames = list(track.read_frames(proxy, 320, 180, 5))
    assert len(frames) == 20 and frames[0].shape == (180, 320, 3) and frames[0].flags.writeable

    track.save(_fake_rows(), tmp_path / "t.npz")
    out = tmp_path / "debug.mp4"
    track.debug_video(proxy, track.load(tmp_path / "t.npz"), 320, 180, 5, out, seconds=2)
    dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(out)],
                         capture_output=True, text=True).stdout
    assert abs(float(dur) - 2.0) < 0.3


def test_upload_runs_tracking_end_to_end(sample, tmp_db):
    """Real model on a synthetic video with no people: checks the wiring, not accuracy."""
    import time
    client = TestClient(app_mod.app)
    job_id = client.post("/api/jobs", params={"filename": "g.mp4"}, content=sample.read_bytes()).json()["id"]
    for _ in range(600):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in db.FINAL:
            break
        time.sleep(0.5)
    assert job["status"] == "ready", job["error"]
    t = job["meta"]["tracking"]
    assert t["frames"] == 20 and t["sample_fps"] == track.SAMPLE_FPS
    assert client.get(f"/api/jobs/{job_id}/debug").status_code == 200


def test_retrack_rules(tmp_db):
    client = TestClient(app_mod.app)
    db.create("busy", "g.mp4")
    db.update("busy", status="tracking", meta={"proxy": {}})
    assert client.post("/api/jobs/busy/track").status_code == 409  # still running
    db.create("raw", "g.mp4")
    db.update("raw", status="failed")
    assert client.post("/api/jobs/raw/track").status_code == 409  # never ingested
    assert client.post("/api/jobs/nope/track").status_code == 404
