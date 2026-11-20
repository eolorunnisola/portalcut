import json

import numpy as np
import pytest

from portalcut import analytics, track

FPS = 5
SIZE = (1000, 600)


def _tracks(rows):
    # (frame, t, track_id, cls, x1, y1, x2, y2, conf); foot point = (x, y2)
    return track.load(_save(rows))


def _save(rows, path=None):
    import tempfile
    from pathlib import Path
    p = Path(path) if path else Path(tempfile.mkdtemp()) / "t.npz"
    track.save(np.array(rows, dtype=np.float32), p)
    return p


def _walk(n=11, speed=50.0, tid=1, start=0, conf=0.9):
    # foot x = 100 + speed*t, y = 300
    return [(start + i, (start + i) / FPS, tid, 0, 100 + speed * (start + i) / FPS - 10, 200,
             100 + speed * (start + i) / FPS + 10, 300, conf) for i in range(n)]


def test_straight_line():
    out = analytics.compute(_tracks(_walk()), {}, SIZE, FPS)
    p = out["players"][0]
    assert p["player_id"] == "trk-1" and out["units"] == "px" and not out["calibrated"]
    assert p["distance"] == pytest.approx(100, rel=0.02)
    assert p["avg_speed"] == pytest.approx(100 / (11 / FPS), rel=0.02)
    assert p["max_speed"] == pytest.approx(50, rel=0.02)
    assert (p["first_seen"], p["last_seen"], p["runs"]) == (0.0, 2.0, 1)
    assert "pixels" in out["note"]


def test_gap_splits_runs():
    rows = _walk(5) + _walk(5, start=20)  # 0.8 s then a 3.2 s gap
    p = analytics.compute(_tracks(rows), {}, SIZE, FPS)["players"][0]
    assert p["runs"] == 2
    assert p["distance"] == pytest.approx(2 * 4 * 10, rel=0.02)  # 4 steps of 10 px per run


def test_merge_tracks_and_duplicate_frame_keeps_higher_conf():
    a, b = _walk(4, tid=1), _walk(4, tid=2, start=4)
    dup = list(_walk(1, tid=2, start=0, conf=0.95))
    dup[0] = dup[0][:4] + (500, 200, 520, 300, 0.95)
    ident = {1: {"player_id": "T0-9", "team": 0, "number": "9"}, 2: {"player_id": "T0-9", "team": 0, "number": "9"}}
    p = analytics.compute(_tracks(a + b + dup), ident, SIZE, FPS)["players"][0]
    assert p["track_ids"] == [1, 2] and p["samples"] == 8 and p["number"] == "9"
    assert p["trajectory"][0][1] > 400  # frame 0 took the 0.95 detection


def test_calibration_scaling_exact():
    cal = {"image_points": [[0, 0], [1000, 0], [1000, 600], [0, 600]],
           "world_points": [[0, 0], [10, 0], [10, 6], [0, 6]], "units": "m", "surface": [10, 6]}
    out = analytics.compute(_tracks(_walk()), {}, SIZE, FPS, cal)
    assert (out["units"], out["calibrated"], out["coordinate_space"]) == ("m", True, "surface")
    assert out["players"][0]["distance"] == pytest.approx(1.0, abs=0.01)
    assert "homography" in out["note"]
    np.testing.assert_allclose(analytics.to_world([[500, 300]], analytics.homography(cal)), [[5, 3]], atol=1e-4)


def test_heatmap_and_zones():
    p = analytics.compute(_tracks(_walk()), {}, SIZE, FPS)["players"][0]
    assert sum(map(sum, p["heatmap"])) == pytest.approx(1, abs=1e-2)
    assert len(p["heatmap"]) == 6 and len(p["heatmap"][0]) == 10
    assert sum(p["zones"].values()) == pytest.approx(p["visible_seconds"], abs=0.05)


def test_empty_inputs_and_ball():
    empty = analytics.compute(_tracks(np.zeros((0, 9))), {}, SIZE, FPS)
    assert empty["players"] == [] and empty["ball"]["samples"] == 0
    out = analytics.compute(_tracks(_walk(3) + [(0, 0.0, -1, 1, 5, 5, 9, 9, 0.5)]), {}, SIZE, FPS)
    assert out["ball"] == {"samples": 1, "visible_seconds": 0.2}


def test_run_writes_json(tmp_path):
    _save(_walk(), tmp_path / "tracks.npz")
    meta = {"proxy": {"width": 1000, "height": 600}, "tracking": {"sample_fps": FPS}}
    out = analytics.run(tmp_path, meta)
    assert json.loads((tmp_path / "analytics.json").read_text())["players"][0]["player_id"] == "trk-1"
    assert out["sample_fps"] == FPS
