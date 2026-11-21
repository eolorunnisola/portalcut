import json

import numpy as np
import pytest

from portalcut.track_eval import match, merge_labels, score


def box(i: int) -> list:
    x = 100 * i
    return [x, 0, x + 50, 100, 1.0]


def make_key(labels: list[str], n: int = 10, missed: int = 0) -> list:
    return [{"t": k / 10, "boxes": [box(i) for i in range(len(labels))], "labels": labels, "missed": missed}
            for k in range(n)]


def make_tracks(per_frame: list[list[tuple[int, int]]]) -> dict:
    """per_frame[k] = [(box index, track_id)]."""
    rows = [(k, k / 10, tid, box(i)[:4]) for k, fr in enumerate(per_frame) for i, tid in fr]
    return {"frame": np.array([r[0] for r in rows]), "t": np.array([r[1] for r in rows], np.float32),
            "track_id": np.array([r[2] for r in rows]), "cls": np.zeros(len(rows), np.int8),
            "xyxy": np.array([r[3] for r in rows], np.float32), "conf": np.ones(len(rows), np.float32)}


def ident(team: str, number: str, pid: str) -> dict:
    return {"team_name": team, "number": number, "player_id": pid}


IDS = {1: ident("UConn", "24", "T0-24"), 2: ident("Arizona", "13", "T1-13")}
LABELS = ["UConn-24", "Arizona-13"]


def test_perfect():
    r = score(make_key(LABELS), make_tracks([[(0, 1), (1, 2)]] * 10), IDS, 10)
    assert (r["detection_recall"], r["identity_precision"], r["identity_coverage"]) == (1, 1, 1)
    assert (r["id_switches"], r["swapped_tracks"], r["wrong_team"]) == (0, 0, 0)
    assert r["idf1_tracker"] == r["idf1_named"] == 1


def test_split_identity_is_a_switch():
    frames = [[(0, 1 if k < 5 else 3), (1, 2)] for k in range(10)]
    ids = {**IDS, 3: ident("UConn", "24", "T0-24")}
    r = score(make_key(LABELS), make_tracks(frames), ids, 10)
    assert r["id_switches"] == 1
    assert r["idf1_tracker"] < 1
    assert r["idf1_named"] == 1  # identity linking repairs it


def test_track_covering_two_identities_is_swapped():
    frames = [[(0, 1), (1, 2)] for _ in range(5)] + [[(0, 2), (1, 1)] for _ in range(5)]
    r = score(make_key(LABELS), make_tracks(frames), IDS, 10)
    assert r["swapped_tracks"] == 2 and r["swap_rate"] == 1
    assert len(r["errors"]["swaps"]) == 2


def test_wrong_missing_and_missed():
    ids = {1: ident("UConn", "99", "T0-99"), 2: {"team_name": None, "number": None, "player_id": "trk-2"}}
    r = score(make_key(LABELS, missed=1), make_tracks([[(0, 1), (1, 2)]] * 10), ids, 10)
    assert r["named_wrong"] == 10 and r["named_missing"] == 10 and r["named_correct"] == 0
    assert r["identity_precision"] == 0
    assert r["detection_recall"] == pytest.approx(20 / 30, abs=1e-3)
    r = score(make_key(LABELS), make_tracks([[(0, 1)]] * 10), IDS, 10)  # second player unmatched
    assert r["named_missing"] == 10 and r["detection_recall"] == 0.5 and r["identity_precision"] == 1


def test_wrong_team_and_unknown_identity():
    ids = {1: ident("Arizona", "24", "T1-24"), 2: ident("UConn", "5", "T0-5")}
    r = score(make_key(["UConn-?", "Arizona-13"]), make_tracks([[(0, 1), (1, 2)]] * 10), ids, 10)
    assert r["wrong_team"] == 20 and r["known_boxes"] == 10 and r["named_wrong"] == 10


def test_non_players_are_not_errors():
    r = score(make_key(["ref", "other", "bad"]), make_tracks([[(0, 1), (1, 2), (2, 9)]] * 10), IDS, 10)
    assert r["players"] == 0 and r["detection_recall"] is None
    assert (r["named_wrong"], r["wrong_team"], r["id_switches"]) == (0, 0, 0)


def test_match_is_one_to_one():
    assert match([box(0)], [box(0), box(0)]) == [(0, 0)]
    assert match([box(0)], [box(3)]) == []


def test_merge_labels(tmp_path):
    det = tmp_path / "d.json"
    det.write_text(json.dumps([{"t": 1.0, "frame": 10, "boxes": [box(0), box(1)]}]))
    good = tmp_path / "g.json"
    good.write_text(json.dumps([{"t": 1.0, "labels": LABELS, "missed": 2, "notes": ""}]))
    assert merge_labels(det, good)[0]["missed"] == 2
    bad = tmp_path / "b.json"
    bad.write_text(json.dumps([{"t": 1.0, "labels": ["ref"], "missed": 0, "notes": ""}]))
    with pytest.raises(ValueError):
        merge_labels(det, bad)
    empty = tmp_path / "e.json"
    empty.write_text("[]")
    with pytest.raises(ValueError):
        merge_labels(det, empty)
