import numpy as np

from portalcut import identity

ROSTER = {"teams": [{"name": "Arizona", "players": {"13": "Motiejus Krivas", "3": "Caleb Holt", "7": "Derek Dixon"}},
                    {"name": "UConn", "players": {"24": "Braylon Mullins", "18": "Oskar Giltay", "13": "Jacob Ross",
                                                  "3": "Colben Landrew"}}]}


def team(t, mixed=False, referee=False):
    return {"team": t, "mixed": mixed, "referee": referee}


def num(n, v=2):
    return {"number": n, "votes": v}


def test_same_team_and_number_read_twice_link_into_one_player():
    teams = {1: team(0), 2: team(0), 3: team(1), 4: team(0)}
    numbers = {"1": num("13", 3), "2": num("13"), "3": num("13", 4), "4": num("13", 1)}
    out = identity.link(teams, numbers, {1: {0, 1}, 2: {10, 11}, 3: {0, 1}, 4: {20}})
    assert out[1]["player_id"] == out[2]["player_id"] == "T0-13"
    assert out[3]["player_id"] == "T1-13"                 # same number, other team: a different player
    assert out[4]["player_id"] == "trk-4"                 # one reading is not enough


def test_two_claims_on_screen_together_keep_the_stronger():
    out = identity.link({1: team(0), 2: team(0)}, {"1": num("5"), "2": num("5", 4)}, {1: {3, 4}, 2: {4, 5}})
    assert out[2]["player_id"] == "T0-5" and out[1]["player_id"] == "trk-1"


def test_referees_mixed_pieces_and_unread_tracks_keep_their_own_id():
    out = identity.link({7: team(2, referee=True), 8: team(0), 9: team(1, mixed=True)},
                        {"7": num("21", 3), "9": num("13", 3)}, {7: {1}, 8: {1}, 9: {1}})
    assert out[7]["player_id"] == "trk-7" and out[8]["player_id"] == "trk-8"
    assert out[9]["player_id"] == "trk-9"                 # crops disagree on team: a tracker swap, never named


def _roster_case():
    # colour team 0 shows Arizona's numbers, team 1 UConn's, so 0 -> Arizona and 1 -> UConn
    teams = {1: team(0), 2: team(0), 3: team(1), 4: team(1), 5: team(0), 6: team(0)}
    numbers = {"1": num("13"), "2": num("7"), "3": num("18"), "4": num("24"),
               "5": num("24"),   # a navy #24 clustered with the white team
               "6": num("99")}   # on no roster: a misread
    return identity.link(teams, numbers, {t: {t} for t in teams}, ROSTER)


def test_roster_names_players_and_matches_colour_teams():
    out = _roster_case()
    assert {"player_id": "T0-13", "name": "Motiejus Krivas", "team_name": "Arizona"}.items() <= out[1].items()
    assert out[3]["name"] == "Oskar Giltay" and out[3]["team_name"] == "UConn"


def test_roster_fixes_team_from_number_and_rejects_misreads():
    out = _roster_case()
    assert out[5]["player_id"] == "T1-24" and out[5]["team"] == 1 and out[5]["team_fixed_by_number"]
    assert out[5]["name"] == "Braylon Mullins"
    assert out[6]["player_id"] == "trk-6" and out[6]["number"] is None


def test_team_votes_flags_referees_and_mixed(tmp_path):
    rng = np.random.default_rng(0)
    navy, white, grey = np.eye(12)[2], np.eye(12)[10], np.eye(12)[6]  # real footage has a third group
    feats, owners, flips = [], [], []
    for tid, kinds in {1: "nnnn", 2: "wwww", 3: "nnww", 4: "gggg", 5: "wwww", 6: "nnnn"}.items():
        for k in kinds:
            feats.append({"n": navy, "w": white, "g": grey}[k] + rng.uniform(0, 0.02, 12)); owners.append(tid)
            flips.append(12 if tid == 4 else 2)       # track 4: striped shirt
    np.savez(tmp_path / "features.npz", track_id=np.array([1, 2, 3, 4, 5, 6]), n=np.full(6, 10),
             hist=np.zeros((6, 48)), crop_team=np.array(feats, np.float32), crop_owner=np.array(owners),
             crop_flips=np.array(flips))
    v = identity.team_votes(tmp_path)
    assert v[1]["team"] != v[2]["team"] and v[1]["team"] in (0, 1) and v[2]["team"] in (0, 1)
    assert v[3]["mixed"] and not v[1]["mixed"]
    assert v[4]["referee"] and v[4]["team"] == 2


def test_roster_number_beats_a_referee_stripe_guess():
    teams = {1: team(0), 2: team(1), 3: {"team": 2, "colour_team": 0, "referee": True, "mixed": False}}
    out = identity.link(teams, {"1": num("13"), "2": num("18"), "3": num("3", 4)}, {1: {1}, 2: {2}, 3: {3}}, ROSTER)
    assert out[3]["player_id"] == "T0-3" and out[3]["name"] == "Caleb Holt" and not out[3]["referee"]
    plain_ref = identity.link({7: {"team": 2, "colour_team": 0, "referee": True, "mixed": False}}, {}, {7: {1}}, ROSTER)
    assert plain_ref[7]["referee"] and plain_ref[7]["player_id"] == "trk-7"


def test_overlapping_claims_prefer_the_piece_whose_colour_is_unanimous():
    teams = {80: {"team": 1, "mixed": False, "referee": False, "purity": 0.75},   # swapped from one #3 to the other
             100: {"team": 1, "mixed": False, "referee": False, "purity": 1.0}}
    out = identity.link(teams, {"80": num("3", 3), "100": num("3", 3)}, {80: set(range(50)), 100: {10, 11}})
    assert out[100]["player_id"] == "T1-3" and out[80]["player_id"] == "trk-80"
