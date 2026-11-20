from portalcut.eval import score


def T(t, ty="dunk"):
    return {"t": t, "type": ty}


def P(a, b=None, ty="dunk", conf=0.9):
    return {"type": ty, "t_start": a, "t_end": a if b is None else b, "confidence": conf}


def test_exact_match():
    r = score([T(10)], [P(9, 11)])
    assert r["overall"] == {"truth": 1, "pred": 1, "tp": 1, "precision": 1.0, "recall": 1.0}
    assert not r["misses"] and not r["false_positives"]


def test_type_mismatch():
    r = score([T(10, "dunk")], [P(10, ty="three")])
    assert r["overall"]["tp"] == 0
    assert len(r["misses"]) == 1 and len(r["false_positives"]) == 1
    assert set(r["per_type"]) == {"dunk", "three"}


def test_tolerance_edge():
    assert score([T(10)], [P(12)], tol=2.0)["overall"]["tp"] == 1
    assert score([T(10)], [P(12.5)], tol=2.0)["overall"]["tp"] == 0


def test_one_to_one():
    r = score([T(10)], [P(9), P(10.5)])
    assert r["overall"]["tp"] == 1 and len(r["false_positives"]) == 1
    assert r["false_positives"][0]["t_start"] == 9


def test_min_conf():
    r = score([T(10)], [P(10, conf=0.3)], min_conf=0.5)
    assert r["overall"]["pred"] == 0 and len(r["misses"]) == 1


def test_segment():
    r = score([T(5), T(50)], [P(5), P(50)], segment=[0, 20])
    assert r["overall"]["truth"] == 1 and r["overall"]["pred"] == 1


def test_none_when_zero():
    r = score([], [P(1)])
    assert r["overall"]["recall"] is None and r["overall"]["precision"] == 0.0
    r = score([T(1)], [])
    assert r["overall"]["precision"] is None and r["overall"]["recall"] == 0.0


def test_ignore_ranges_drop_replay_predictions():
    from portalcut.eval import score
    preds = [{"type": "made_shot", "t_start": 50, "t_end": 51, "confidence": 0.9}]
    assert score([], preds, ignore=[[45, 60]])["overall"]["pred"] == 0
    assert score([], preds)["overall"]["pred"] == 1
