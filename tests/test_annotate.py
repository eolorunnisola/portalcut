import json
import subprocess

import numpy as np

from portalcut import annotate


def _tracks(rows):
    """rows: (frame, track_id, cls, x1, y1, x2, y2, conf) at 5 fps."""
    a = np.array(rows, dtype=np.float32)
    return {"frame": a[:, 0].astype(np.int32), "t": (a[:, 0] / 5).astype(np.float32),
            "track_id": a[:, 1].astype(np.int32), "cls": a[:, 2].astype(np.int8),
            "xyxy": a[:, 3:7], "conf": a[:, 7]}


def test_interpolate_midpoint_exact_and_gap():
    tr = _tracks([(0, 1, 0, 0, 0, 10, 20, 0.8), (1, 1, 0, 10, 0, 20, 20, 0.6),
                  (5, 2, 0, 0, 0, 10, 10, 0.9), (8, 2, 0, 30, 0, 40, 10, 0.9)])
    (p,) = annotate.interpolate(tr, 0.1, 5)
    assert np.allclose(p["xyxy"], [5, 0, 15, 20]) and abs(p["conf"] - 0.7) < 1e-5
    (p,) = annotate.interpolate(tr, 0.2, 5)
    assert p["track_id"] == 1 and np.allclose(p["xyxy"], [10, 0, 20, 20])
    assert annotate.interpolate(tr, 1.3, 5) == []  # 3-sample gap


def test_ball_at_tolerance():
    tr = _tracks([(5, -1, 1, 10, 10, 14, 14, 0.5)])
    assert annotate.ball_at(tr, 1.1, 5) is not None
    assert annotate.ball_at(tr, 1.5, 5) is None


def test_draw_box_colour():
    f = np.zeros((100, 100, 3), np.uint8)
    p = [{"track_id": 3, "xyxy": np.array([20, 30, 60, 90], np.float32), "conf": 0.91}]
    annotate.draw(f, p, None, {3: {"team": 0, "number": "13", "player_id": "T0-13"}}, {})
    assert tuple(f[60, 20]) == annotate.TEAM_COLORS[0]


def test_render_synthetic(tmp_path):
    src, out = tmp_path / "in.mp4", tmp_path / "out.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=10:duration=3",
                    "-pix_fmt", "yuv420p", str(src)], check=True)
    rows = [(f, 7, 0, 10 + 8 * f, 40, 40 + 8 * f, 120, 0.9) for f in range(15)]
    rows += [(f, -1, 1, 100, 100, 108, 108, 0.5) for f in range(15)]
    seen = []
    r = annotate.render(src, 320, 180, _tracks(rows), {}, 5, out, out_fps=10, on_progress=seen.append)
    info = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
         "stream=nb_read_frames:format=duration", "-of", "json", str(out)],
        capture_output=True, text=True, check=True).stdout)
    assert abs(float(info["format"]["duration"]) - 3) < 0.3
    assert abs(int(info["streams"][0]["nb_read_frames"]) - 30) <= 2
    assert r["frames"] >= 29 and seen
