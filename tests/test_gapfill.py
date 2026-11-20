import subprocess

import numpy as np
import pytest

from portalcut import gapfill as g


def rows(frames, box=(100, 50, 140, 130)):
    n = len(frames)
    return {"frame": np.array(frames, np.int32), "t": np.array(frames, np.float32) / 5,
            "track_id": np.ones(n, np.int32), "xyxy": np.tile(np.float32(box), (n, 1)),
            "conf": np.ones(n, np.float32)}


NOBALL = {"t": np.zeros(0, np.float32), "xyxy": np.zeros((0, 4), np.float32)}
RUNS = rows(list(range(10)) + list(range(15, 21)))


def test_seeds_two_runs():
    ss = g.seeds(RUNS, NOBALL, 60)
    got = {(s["direction"], s["frame"], s["limit"]) for s in ss}
    assert got == {("forward", 9, 5), ("backward", 15, 5), ("forward", 20, 15)}


def test_seeds_skip_video_end_and_clip():
    ss = g.seeds(RUNS, NOBALL, 21)  # last run ends at the final frame
    assert {(s["direction"], s["frame"]) for s in ss} == {("forward", 9), ("backward", 15)}


def test_seeds_ball_priority_and_cap():
    ball = {"t": np.array([20 / 5], np.float32), "xyxy": np.array([[100, 60, 130, 90]], np.float32)}
    ss = g.seeds(RUNS, ball, 60)
    assert (ss[0]["direction"], ss[0]["frame"]) == ("forward", 20)
    assert len(g.seeds(RUNS, ball, 60, max_runs=1)) == 1


def test_mask_to_box():
    m = np.zeros((20, 30), bool); m[5:10, 7:12] = True
    assert g.mask_to_box(m).tolist() == [7, 5, 12, 10]
    assert g.mask_to_box(np.zeros((4, 4), bool)) is None


def test_keep():
    seed = np.float32([0, 0, 10, 20])
    assert g.keep(seed, seed, seed)
    assert not g.keep(np.float32([0, 0, 3, 5]), seed, seed)      # area < 0.3x
    assert not g.keep(np.float32([0, 0, 30, 40]), seed, seed)    # area > 3x
    assert not g.keep(np.float32([0, 40, 10, 60]), seed, seed)   # jump > 1.5 heights
    assert g.keep(np.float32([0, 10, 10, 30]), seed, seed)


@pytest.fixture(scope="module")
def proxy(tmp_path_factory):
    p = tmp_path_factory.mktemp("px") / "p.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=25:duration=6",
                    "-pix_fmt", "yuv420p", str(p)], check=True)
    return p


def mover(stop_at=None):
    def fn(frames, si, box, direction):
        out = []
        for k in range(1, len(frames)):
            if stop_at is not None and k > stop_at:
                out.append(np.zeros(frames[0].shape[:2], bool)); continue
            m = np.zeros(frames[0].shape[:2], bool)
            d = k if direction == "forward" else -k
            x = int(box[0]) + 2 * d
            m[int(box[1]):int(box[3]), x:x + int(box[2] - box[0])] = True
            out.append(m)
        return out
    return fn


def test_fill_exact_gap_no_duplicates(proxy):
    p = rows(list(range(5, 10)) + list(range(15, 20)), box=(100, 40, 130, 100))
    new, st = g.fill(proxy, 320, 180, p, NOBALL, 30, track_fn=mover())
    assert new["frame"].tolist() == list(range(0, 5)) + list(range(10, 15)) + list(range(20, 30))
    assert not set(new["frame"].tolist()) & set(p["frame"].tolist())
    assert len(set(new["frame"].tolist())) == len(new["frame"])
    assert (new["track_id"] == -2).all() and (new["conf"] == 0.5).all()
    assert st["frames_added"] == len(new["frame"]) and st["seeds"] == 4


def test_fill_stops_on_empty_mask(proxy):
    p = rows(list(range(5, 10)) + list(range(20, 25)), box=(100, 40, 130, 100))
    new, _ = g.fill(proxy, 320, 180, p, NOBALL, 30, track_fn=mover(stop_at=2))
    assert set(new["frame"].tolist()) == {10, 11, 18, 19, 3, 4, 25, 26}
