import subprocess
import time

import pytest
from fastapi.testclient import TestClient

from portalcut import app as app_mod, db, ingest


@pytest.fixture
def sample(tmp_path):
    """4-second 1080p clip with a tone, made by ffmpeg so the test needs no fixtures on disk."""
    p = tmp_path / "game.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", "testsrc=size=1920x1080:rate=30:duration=4",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
         "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(p)],
        check=True,
    )
    return p


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "data" / "portalcut.db")
    db.init()


def test_probe_and_rms(sample, tmp_path):
    meta = ingest.probe(sample)
    assert (meta["width"], meta["height"], meta["fps"]) == (1920, 1080, 30)
    assert meta["has_audio"] and 3.9 < meta["duration_sec"] < 4.2
    rms = ingest.audio_rms(sample, tmp_path / "a.wav")
    # ffmpeg's sine source plays at 1/8 amplitude, so RMS = 0.125 / sqrt(2) ≈ 0.088
    assert len(rms) == 4 and all(abs(r - 0.088) < 0.01 for r in rms)


def test_upload_to_ready(sample, tmp_db):
    client = TestClient(app_mod.app)
    r = client.post("/api/jobs", params={"filename": "My Game.MP4"}, content=sample.read_bytes())
    assert r.status_code == 200
    job_id = r.json()["id"]

    for _ in range(120):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in db.FINAL:
            break
        time.sleep(0.5)
    assert job["status"] == "ready", job["error"]
    assert job["meta"]["proxy"]["height"] == 720
    assert job["meta"]["proxy"]["fps"] == 30
    assert client.get(f"/api/jobs/{job_id}/proxy").status_code == 200

    assert client.delete(f"/api/jobs/{job_id}").status_code == 200
    assert client.get(f"/api/jobs/{job_id}").status_code == 404
    assert not (db.DATA_DIR / job_id).exists()
    assert client.delete(f"/api/jobs/{job_id}").status_code == 404


def test_cannot_remove_while_processing(tmp_db):
    db.create("busy", "g.mp4")
    db.update("busy", status="making_proxy")
    assert TestClient(app_mod.app).delete("/api/jobs/busy").status_code == 409


def test_rejects_bad_type_and_unknown_job(tmp_db):
    client = TestClient(app_mod.app)
    assert client.post("/api/jobs", params={"filename": "x.exe"}, content=b"1").status_code == 400
    assert client.get("/api/jobs/../../etc").status_code == 404
