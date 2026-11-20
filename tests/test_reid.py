import subprocess

import numpy as np
import pytest
from fastapi.testclient import TestClient

from portalcut import app as app_mod, db, reid, track
from tests.test_ingest import tmp_db  # noqa: F401  (pytest fixture)

# Six made-up tracks, 10 frames each:
#   0 = you (frames 0-9)      1 = someone else on screen with you (frames 0-9)
#   2 = looks like you (10-19)  3 = looks like the other player (10-19)
#   4 = halfway between the two (20-29): should be a question
#   5 = also looks like you but on screen at the same time as 2: only one can be you
SPANS = {0: (0, 10), 1: (0, 10), 2: (10, 20), 3: (10, 20), 4: (20, 30), 5: (10, 20)}


def _emb():
    e1, e2 = np.eye(384, dtype=np.float32)[:2]
    mid = (e1 + e2) / np.linalg.norm(e1 + e2)
    near1 = e1 * 0.9 + np.eye(384, dtype=np.float32)[2] * 0.1
    return np.stack([e1, e2, e1, e2, mid, near1 / np.linalg.norm(near1)])


@pytest.fixture
def job_dir(tmp_path):
    tmp_path = tmp_path / "job"  # kept apart from tmp_db's data folder
    tmp_path.mkdir()
    rows = []
    for tid, (a, b) in SPANS.items():
        x = 20 + 40 * (tid % 3)
        rows += [(f, f / 5, tid, track.PERSON, x, 20, x + 30, 120, 0.9) for f in range(a, b)]
    track.save(np.array(rows, np.float32), tmp_path / "tracks.npz")
    tids = np.array(list(SPANS), np.int32)
    np.savez_compressed(tmp_path / "features.npz", track_id=tids, first=np.array([a for a, _ in SPANS.values()]),
                        last=np.array([b - 1 for _, b in SPANS.values()]), n=np.full(6, 10),
                        emb=_emb(), hist=np.zeros((6, 48), np.float32), select_frames=np.array([0, 15], np.int32))
    return tmp_path


def test_one_click_finds_you_and_asks_about_the_unclear_one(job_dir):
    sel = reid.Selection(job_dir)
    sel.answer(0, True)
    s = sel.summary()
    assert 1 in reid.Selection.score(sel)["neg"]       # on screen with you, so not you
    assert 2 in s["accepted"] and 3 not in s["accepted"]
    assert not (2 in s["accepted"] and 5 in s["accepted"])  # never two of you at once
    assert s["questions"] == [4]

    sel.answer(4, True)
    sel.save()
    s = reid.Selection(job_dir).summary()               # reloads from selection.json
    assert s["confirmed"] == [0, 4] and 4 in s["accepted"] and s["questions"] == []
    assert s["coverage_pct"] == 100.0                   # frames 0-29 all covered


def test_no_answer_rejects(job_dir):
    sel = reid.Selection(job_dir)
    sel.answer(0, True)
    sel.answer(4, False)
    s = sel.summary()
    assert 4 not in s["accepted"] and s["rejected"] == [4]
    with pytest.raises(KeyError):
        sel.answer(99, True)


def test_frames_payload(job_dir):
    fr = reid.Selection(job_dir).frames_payload()
    assert [f["frame"] for f in fr] == [0, 15]
    assert sorted(b["track_id"] for b in fr[1]["boxes"]) == [2, 3, 5]


def test_finish_writes_one_box_per_frame(job_dir):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=30:duration=6",
                    "-c:v", "libx264", "-preset", "ultrafast", str(job_dir / "proxy.mp4")], check=True)
    sel = reid.Selection(job_dir)
    sel.answer(0, True)
    sel.answer(4, True)
    sel.save()
    out = reid.finish(job_dir, 320, 180)
    p = dict(np.load(job_dir / "player.npz"))
    assert sorted(p["frame"].tolist()) == list(range(30))  # one row per frame, no duplicates
    assert set(p["track_id"].tolist()) <= {0, 2, 4, 5}
    assert (job_dir / "debug_player.mp4").exists() and out["coverage_pct"] == 100.0


def test_selection_api(job_dir, tmp_db):
    import shutil
    jid = "seljob"
    shutil.copytree(job_dir, db.DATA_DIR / jid)
    db.create(jid, "g.mp4")
    db.update(jid, status="ready", meta={"proxy": {"width": 320, "height": 180}, "features": {"tracks": 6}})
    c = TestClient(app_mod.app)

    assert c.post(f"/api/jobs/{jid}/select/done").status_code == 400  # nothing selected yet
    assert len(c.get(f"/api/jobs/{jid}/select").json()["frames"]) == 2
    r = c.post(f"/api/jobs/{jid}/select/answer", json={"track_id": 0, "yes": True}).json()
    assert r["summary"]["questions"] == [4]
    assert c.post(f"/api/jobs/{jid}/select/answer", json={"track_id": 99, "yes": True}).status_code == 404
    assert c.post(f"/api/jobs/{jid}/select/reset").json()["summary"]["confirmed"] == []
    assert c.get(f"/api/jobs/{jid}/crops/1").status_code == 404     # no crop image written in this fixture
    assert c.get("/api/jobs/nope/select").status_code == 404

    db.update(jid, status="tracking")
    assert c.get(f"/api/jobs/{jid}/select").status_code == 409      # not ready yet
