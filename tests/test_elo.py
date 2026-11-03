import pandas as pd
import pytest

from cfbsim.elo import EloParams, run_elo, win_probability


def test_first_meeting_uses_starting_ratings_and_home_edge(schedule):
    params = EloParams(home_advantage=65)
    predictions, _ = run_elo(schedule, params)
    assert predictions.loc[0, "elo_diff"] == 0
    assert predictions.loc[0, "p_elo"] == pytest.approx(win_probability(65))


def test_neutral_site_has_no_home_edge(schedule):
    neutral = schedule.assign(neutral_site=True)
    predictions, _ = run_elo(neutral, EloParams())
    assert predictions.loc[0, "p_elo"] == 0.5


def test_ratings_are_zero_sum_within_a_season(schedule):
    first_season = schedule[schedule["season"] == 2019]
    _, ratings = run_elo(first_season, EloParams())
    assert sum(ratings.values()) == pytest.approx(1500 * 3)


def test_winner_gains_and_loser_drops(schedule):
    _, ratings = run_elo(schedule.head(1), EloParams())
    assert ratings[1] > 1500 > ratings[2]


def test_offseason_pulls_ratings_toward_the_baseline(schedule):
    params = EloParams(season_carryover=0.5)
    first_two, ratings = run_elo(schedule.head(3), params)
    predictions, _ = run_elo(schedule, params)
    assert predictions.loc[3, "elo_home"] == pytest.approx(1500 + 0.5 * (ratings[1] - 1500))
    assert first_two.loc[2, "elo_home"] == predictions.loc[2, "elo_home"]


def test_lower_division_team_starts_at_its_own_baseline(schedule):
    games = schedule.copy()
    games.loc[0, "away_classification"] = "fcs"
    predictions, _ = run_elo(games, EloParams(lower_division_initial=1200))
    assert predictions.loc[0, "elo_away"] == 1200


@pytest.mark.parametrize("cutoff", range(1, 5))
def test_predictions_do_not_depend_on_future_results(schedule, cutoff):
    """The core leakage check: rewriting later results can't change earlier forecasts."""
    params = EloParams()
    original, _ = run_elo(schedule, params)

    altered = schedule.copy()
    altered.loc[cutoff:, "margin"] = -altered.loc[cutoff:, "margin"] * 3
    changed, _ = run_elo(altered, params)

    pd.testing.assert_frame_equal(original.iloc[: cutoff + 1], changed.iloc[: cutoff + 1])
    assert not original.equals(changed)


def test_a_game_cannot_see_its_own_result(schedule):
    params = EloParams()
    original, _ = run_elo(schedule, params)
    altered = schedule.copy()
    altered.loc[2, "margin"] = -50
    changed, _ = run_elo(altered, params)
    assert original.loc[2, "p_elo"] == changed.loc[2, "p_elo"]


def test_out_of_order_games_are_rejected(schedule):
    with pytest.raises(ValueError, match="sorted"):
        run_elo(schedule.iloc[::-1].reset_index(drop=True), EloParams())
