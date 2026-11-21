import json
import subprocess

import pytest

from portalcut import batch, db, import_job, pipeline


@pytest.fixture
def clip(tmp_path):
    p = tmp_path / "Game.MP4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=640x360:rate=30:duration=2",
                    "-c:v", "libx264", "-preset", "ultrafast", str(p)], check=True)
    return p


def test_batch_then_import(clip, tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "run", lambda job_id, job_dir: db.update(job_id, status="ready"))  # ingest only
    roster = tmp_path / "roster.json"
    roster.write_text('{"teams": []}')
    out = tmp_path / "drive"
    s = batch.run(clip, out, {"tracker": "bytetrack", "lost_seconds": "1"}, roster, tmp_path / "work")
    assert s["status"] == "ready" and s["config"]["tracker"] == "bytetrack" and s["config"]["lost_seconds"] == 1.0
    folder = out / s["id"]
    assert (folder / "proxy.mp4").exists() and (folder / "roster.json").exists()
    assert not any(folder.glob("source*"))  # symlink to the video removed, not copied to Drive
    assert json.loads((folder / "job.json").read_text())["filename"] == "Game.MP4"

    monkeypatch.setattr(db, "DATA_DIR", tmp_path / "app")
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "app" / "portalcut.db")
    assert import_job.import_job(folder) == s["id"]
    assert db.get(s["id"])["status"] == "ready" and (tmp_path / "app" / s["id"] / "proxy.mp4").exists()
    with pytest.raises(FileExistsError):
        import_job.import_job(folder)
    import_job.import_job(folder, replace=True)
    assert folder.exists()  # copied, not moved


def test_batch_rejects_bad_setting(clip, tmp_path):
    with pytest.raises(Exception):
        batch.run(clip, tmp_path / "out", {"tracker": "nope"})
    assert not (tmp_path / "out").exists()
