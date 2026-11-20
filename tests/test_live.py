import numpy as np

from portalcut import live

H, FPS = 720, 10


def _tracks(frames: dict[int, list[float]]):
    """frames: sample index -> list of box heights in pixels."""
    rows = [(f, ht) for f, hs in frames.items() for ht in hs]
    n = len(rows)
    xyxy = np.zeros((n, 4), np.float32)
    for i, (_, ht) in enumerate(rows):
        xyxy[i] = (100 + 10 * i, 300 - ht, 140 + 10 * i, 300)
    return {"frame": np.array([r[0] for r in rows], np.int32),
            "t": np.array([r[0] / FPS for r in rows], np.float32),
            "track_id": np.arange(n, dtype=np.int32), "cls": np.zeros(n, np.int8),
            "xyxy": xyxy, "conf": np.ones(n, np.float32)}


WIDE = [150.0] * 6
CLOSE = [500.0, 100, 100, 100, 100]


def test_six_court_players_live():
    assert live.live_frames(_tracks({0: WIDE}), H, FPS) == {0}


def test_huge_box_not_live_even_with_small_boxes():
    assert live.live_frames(_tracks({0: CLOSE}), H, FPS) == set()


def test_two_players_not_live():
    assert live.live_frames(_tracks({0: [150.0, 150.0]}), H, FPS) == set()


def test_smoothing_fills_isolated_gap_and_keeps_closeup_block():
    fr = {i: WIDE for i in range(30)}
    fr[15] = CLOSE
    fr.update({i: CLOSE for i in range(30, 60)})
    got = live.live_frames(_tracks(fr), H, FPS)
    assert set(range(30)) <= got
    assert not got & set(range(35, 60))


def test_segments_and_summary():
    fr = {i: WIDE for i in range(20)}
    fr.update({i: CLOSE for i in range(20, 40)})
    fr.update({i: WIDE for i in range(40, 60)})
    t = _tracks(fr)
    got = live.live_frames(t, H, FPS)
    assert live.segments(got, FPS) == [(0.0, 2.0), (4.0, 6.0)]
    assert live.segments(set(), FPS) == []
    s = live.summary(t, H, FPS)
    assert s["total_seconds"] == 6.0 and s["segments"] == 2
    assert abs(s["live_seconds"] / s["total_seconds"] * 100 - s["live_pct"]) < 1e-9
    assert abs(s["live_seconds"] - 4.0) < 1e-9
