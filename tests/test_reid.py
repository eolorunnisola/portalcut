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


# ---- switching to a different player (audit: clicks used to add a second "you" instead of switching) ----

def test_click_other_player_in_same_frame_switches(job_dir):
    sel = reid.Selection(job_dir)
    sel.answer(0, True)
    assert sel.answer(1, True) is True                   # 1 is on screen with 0: you can't be both
    s = sel.summary()
    assert s["confirmed"] == [1]
    assert 0 in s["rejected"]                            # the person you moved off is now "not you"
    assert 3 in s["accepted"]                            # 3 looks like 1
    assert not {0, 2, 5} & set(s["accepted"])            # nothing of the old player remains


def test_click_other_player_in_another_frame_switches(job_dir):
    sel = reid.Selection(job_dir)
    sel.answer(0, True)
    assert 2 in sel.summary()["accepted"]
    assert sel.answer(3, True) is True                   # 3 shares frames with auto-matched 2
    s = sel.summary()
    assert s["confirmed"] == [3] and 2 in s["rejected"]
    assert 1 in s["accepted"] and not {0, 2, 5} & set(s["accepted"])


def test_click_clearly_different_player_switches_even_without_overlap(job_dir):
    sel = reid.Selection(job_dir)
    sel.answer(0, True)
    sel.answer(2, False)
    sel.answer(5, False)                                 # frames 10-19 now have no match
    assert sel.answer(3, True) is True                   # looks like the other player, not like 0
    assert sel.summary()["confirmed"] == [3]


def test_second_click_on_same_player_adds(job_dir):
    sel = reid.Selection(job_dir)
    sel.answer(0, True)
    assert sel.answer(2, True) is False                  # same person in another frame: keep both
    assert sel.summary()["confirmed"] == [0, 2]


def test_api_reports_switch(job_dir, tmp_db):
    import shutil
    shutil.copytree(job_dir, db.DATA_DIR / "sw")
    db.create("sw", "g.mp4")
    db.update("sw", status="ready", meta={"proxy": {"width": 320, "height": 180}, "features": {"tracks": 6}})
    c = TestClient(app_mod.app)
    assert c.post("/api/jobs/sw/select/answer", json={"track_id": 0, "yes": True}).json()["switched"] is False
    r = c.post("/api/jobs/sw/select/answer", json={"track_id": 1, "yes": True}).json()
    assert r["switched"] is True and r["summary"]["confirmed"] == [1]


def test_restart_during_selection_keeps_job_usable(tmp_db):
    db.create("s", "g.mp4")
    db.update("s", status="selecting")
    db.create("t", "g.mp4")
    db.update("t", status="tracking")
    db.init()                                            # what a server restart does
    assert db.get("s")["status"] == "ready" and db.get("s")["error"] is None  # tracking is intact: just pick again
    assert db.get("t")["status"] == "failed"


# ---- independent review: switching must not throw away the user's own answers ----

def test_yes_on_other_piece_of_you_corrects_instead_of_wiping(job_dir):
    sel = reid.Selection(job_dir)
    sel.answer(0, True)                                  # auto-matches 2
    assert sel.answer(5, True) is False                  # 5 also looks like you and shares frames with 2
    s = sel.summary()
    assert s["confirmed"] == [0, 5]                      # earlier confirmation kept
    assert 2 in s["rejected"] and 2 not in s["accepted"]  # you said 5, so 2 (same frames) is someone else


def test_no_then_yes_flips_one_track(job_dir):
    sel = reid.Selection(job_dir)
    sel.answer(0, True)
    sel.answer(4, False)
    assert sel.answer(4, True) is False
    s = sel.summary()
    assert s["confirmed"] == [0, 4] and s["rejected"] == []


def test_switch_can_be_undone(job_dir):
    sel = reid.Selection(job_dir)
    sel.answer(0, True)
    sel.answer(2, True)
    assert sel.answer(1, True) is True                   # switch to the other player
    sel.save()
    sel = reid.Selection(job_dir)                        # survives a reload
    assert sel.undo() is True
    assert sel.summary()["confirmed"] == [0, 2]
    assert sel.undo() is False                           # nothing more to undo


def test_api_undo(job_dir, tmp_db):
    import shutil
    shutil.copytree(job_dir, db.DATA_DIR / "un")
    db.create("un", "g.mp4")
    db.update("un", status="ready", meta={"proxy": {"width": 320, "height": 180}, "features": {"tracks": 6}})
    c = TestClient(app_mod.app)
    c.post("/api/jobs/un/select/answer", json={"track_id": 0, "yes": True})
    c.post("/api/jobs/un/select/answer", json={"track_id": 1, "yes": True})
    r = c.post("/api/jobs/un/select/undo").json()
    assert r["summary"]["confirmed"] == [0]
    assert c.post("/api/jobs/un/select/undo").status_code == 409


def test_finish_clears_old_result_before_rebuilding(job_dir, tmp_db, monkeypatch):
    import shutil
    shutil.copytree(job_dir, db.DATA_DIR / "fx")
    db.create("fx", "g.mp4")
    db.update("fx", status="selecting", meta={"proxy": {"width": 320, "height": 180}, "selection": {"old": 1}})
    monkeypatch.setattr(reid, "finish", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):               # simulate the server dying mid-build
        app_mod._finish("fx", db.DATA_DIR / "fx")
    assert "selection" not in db.get("fx")["meta"]       # old result is not shown next to half-written files


def test_undo_only_right_after_a_switch(job_dir):
    sel = reid.Selection(job_dir)
    sel.answer(0, True)
    sel.answer(1, True)                                  # switch
    sel.answer(3, True)                                  # a later answer
    assert sel.undo() is False                           # undo would silently drop that answer
    assert sel.summary()["confirmed"] == [1, 3]


# ---- rebuilt player video must not be served from the browser's cache ----

def test_each_rebuild_gets_a_new_video_version(job_dir):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=30:duration=6",
                    "-c:v", "libx264", "-preset", "ultrafast", str(job_dir / "proxy.mp4")], check=True)
    sel = reid.Selection(job_dir)
    sel.answer(0, True)
    sel.save()
    first = reid.finish(job_dir, 320, 180)["version"]
    sel = reid.Selection(job_dir)
    sel.answer(1, True)                                  # switch players, rebuild
    sel.save()
    assert reid.finish(job_dir, 320, 180)["version"] != first


def test_generated_media_is_not_cached(job_dir, tmp_db):
    import shutil
    shutil.copytree(job_dir, db.DATA_DIR / "nc")
    (db.DATA_DIR / "nc" / "debug_player.mp4").write_bytes(b"x")
    (db.DATA_DIR / "nc" / "debug_tracks.mp4").write_bytes(b"x")
    (db.DATA_DIR / "nc" / "frames").mkdir()
    (db.DATA_DIR / "nc" / "frames" / "0.jpg").write_bytes(b"x")
    db.create("nc", "g.mp4")
    db.update("nc", status="ready", meta={"proxy": {"width": 320, "height": 180}, "features": {"tracks": 6}})
    c = TestClient(app_mod.app)
    for url in ("/api/jobs/nc/debug_player", "/api/jobs/nc/debug", "/api/jobs/nc/frames/0"):
        r = c.get(url)
        assert r.status_code == 200 and "no-store" in r.headers.get("cache-control", ""), url
