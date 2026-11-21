import numpy as np

from portalcut.follow_eval import score_player, track_labels

KEY = [  # two keyframes: target A-13 at left, another player at right; second keyframe target absent
    {"t": 1.0, "boxes": [[0, 0, 10, 20, 0.9], [50, 0, 60, 20, 0.9]], "labels": ["Arizona-13", "UConn-5"]},
    {"t": 2.0, "boxes": [[50, 0, 60, 20, 0.9], [80, 0, 90, 20, 0.9]], "labels": ["UConn-5", "Arizona-?"]},
]


def rows(ts, boxes, tids=None):
    return {"t": np.array(ts, float), "xyxy": np.array(boxes, float),
            "track_id": np.array(tids or [1] * len(ts)), "frame": (np.array(ts) * 10).astype(int)}


def test_found_wrong_unsure():
    s = score_player(KEY, rows([1.0, 2.0], [[0, 0, 10, 20], [50, 0, 60, 20]]), "Arizona-13", 10)
    assert (s["visible"], s["found"], s["wrong"], s["coverage"], s["precision"]) == (1, 1, 1, 1.0, 0.5)
    s = score_player(KEY, rows([2.0], [[80, 0, 90, 20]]), "Arizona-13", 10)  # same-team unknown: not judged
    assert (s["found"], s["wrong"], s["unsure"], s["precision"]) == (0, 0, 1, None)


def test_nothing_shown_is_not_wrong():
    s = score_player(KEY, rows([], np.zeros((0, 4))), "Arizona-13", 10)
    assert (s["found"], s["wrong"], s["coverage"]) == (0, 0, 0.0)


def test_track_labels_majority_named_only():
    r = rows([1.0, 2.0, 1.0, 2.0], [[0, 0, 10, 20], [50, 0, 60, 20], [50, 0, 60, 20], [80, 0, 90, 20]], [1, 1, 2, 2])
    labs = track_labels(KEY + [KEY[0]], r, 10)  # track 1: A-13 twice, U-5 once
    assert labs[1] == "Arizona-13" and labs[2] == "UConn-5"
