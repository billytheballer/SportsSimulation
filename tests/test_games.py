import pytest

from conftest import make_record

from cfbsim.games import ALL_FIELDS, parse_games


def test_post_game_fields_are_removed():
    games, _ = parse_games([make_record()])
    for leaked in ("elo_post", "homePostgameElo", "homePostgameWinProbability", "excitementIndex",
                   "attendance", "homeLineScores"):
        assert leaked not in games.columns
    assert not any("post" in column.lower() for column in games.columns)


def test_unknown_new_api_fields_are_ignored():
    games, _ = parse_games([make_record(homeFinalRating=99)])
    assert "homeFinalRating" not in games.columns


def test_every_dropped_row_is_counted():
    records = [
        make_record(1),
        make_record(1),
        make_record(2, completed=False, homePoints=None, awayPoints=None),
        make_record(3, homePoints=None),
        make_record(4, homeClassification="fcs", awayClassification="fcs"),
        make_record(5, homePoints=14, awayPoints=14),
        make_record(6, startDate=None),
    ]
    games, report = parse_games(records)
    assert list(games["game_id"]) == [1]
    assert report.dropped == {
        "duplicate game id": 1,
        "not completed": 1,
        "completed but missing a score": 1,
        "no FBS team": 1,
        "tied final score": 1,
        "missing kickoff, site, or team id": 1,
    }
    assert report.rows_read == 7 and report.games_kept == 1


def test_missing_values_are_not_filled_in():
    games, _ = parse_games([make_record(homePregameElo=None)])
    assert games["cfbd_home_elo"].isna().all()


def test_changed_api_shape_fails_loudly():
    record = make_record()
    del record["neutralSite"]
    with pytest.raises(ValueError, match="neutralSite"):
        parse_games([record])


def test_games_are_sorted_by_kickoff():
    games, _ = parse_games(
        [make_record(1, startDate="2019-09-14T19:00:00Z"), make_record(2, startDate="2019-09-07T19:00:00Z")]
    )
    assert list(games["game_id"]) == [2, 1]


def test_empty_response_gives_empty_table():
    games, report = parse_games([])
    assert games.empty and report.games_kept == 0
    assert set(ALL_FIELDS.values()) <= set(games.columns)
