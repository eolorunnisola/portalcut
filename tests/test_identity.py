from portalcut import identity


def test_same_team_and_number_read_twice_link_into_one_player():
    teams = {1: 0, 2: 0, 3: 1, 4: 0}
    numbers = {"1": {"number": "13", "votes": 3}, "2": {"number": "13", "votes": 2},
               "3": {"number": "13", "votes": 4}, "4": {"number": "13", "votes": 1}}
    frames = {1: {0, 1}, 2: {10, 11}, 3: {0, 1}, 4: {20}}
    out = identity.link(teams, numbers, frames)
    assert out[1]["player_id"] == out[2]["player_id"] == "T0-13"
    assert out[3]["player_id"] == "T1-13"                 # same number, other team: a different player
    assert out[4]["player_id"] == "trk-4"                 # one reading is not enough


def test_two_claims_on_screen_together_keep_the_stronger():
    out = identity.link({1: 0, 2: 0}, {"1": {"number": "5", "votes": 2}, "2": {"number": "5", "votes": 4}},
                        {1: {3, 4}, 2: {4, 5}})
    assert out[2]["player_id"] == "T0-5" and out[1]["player_id"] == "trk-1"


def test_referees_and_unread_tracks_keep_their_own_id():
    out = identity.link({7: 2, 8: 0}, {"7": {"number": "21", "votes": 3}}, {7: {1}, 8: {1}})
    assert out[7]["player_id"] == "trk-7" and out[8] == {"team": 0, "number": None, "player_id": "trk-8"}
