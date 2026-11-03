"""Stage 3 tests. All games and stats here are made up and never reach a report."""

import numpy as np
import pandas as pd
import pytest

from conftest import make_record
from test_stage2 import make_league

from cfbsim import backtest, stage2, stage3
from cfbsim.config import DEVELOPMENT_SEASONS, HOLDOUT_SEASONS, TUNE_SEASONS, VALIDATION_SEASONS
from cfbsim.elo import EloParams, run_elo
from cfbsim.features import FeatureParams, matchup_row, walk_forward
from cfbsim.games import parse_regulation
from cfbsim.matchup import factors_table, format_matchup
from cfbsim.metrics import crps, point_errors
from cfbsim.scoring import SCORE_FEATURES, ScoreModel, team_table, team_view
from cfbsim.simulate import PairSimulator, ScoreTable, SeparateDrawSimulator, SimulatedGames, summarize
from cfbsim.stage3 import fit_stack, run_stage3, scored_frame
from cfbsim.stats import parse_lines


def regulation_for(games):
    frame = games[["game_id", "home_points", "away_points"]].rename(
        columns={"home_points": "home_regulation", "away_points": "away_regulation"}
    )
    # Fake league games never tie, so mark a sample as overtime by rewinding them to a tie.
    frame = frame.assign(overtime=False)
    sample = frame.index[::25]
    frame.loc[sample, "home_regulation"] = frame.loc[sample, ["home_regulation", "away_regulation"]].min(axis=1)
    frame.loc[sample, "away_regulation"] = frame.loc[sample, "home_regulation"]
    frame.loc[sample, "overtime"] = True
    return frame


@pytest.fixture(scope="module")
def league():
    games, stats = make_league(seasons=DEVELOPMENT_SEASONS)
    return games, stats, regulation_for(games)


@pytest.fixture(scope="module")
def frame(league):
    games, stats, regulation = league
    elo, _ = run_elo(games, EloParams())
    return scored_frame(games, elo, stats, FeatureParams(), regulation)


@pytest.fixture(scope="module")
def stack(frame):
    return fit_stack(frame[frame["season"].isin(TUNE_SEASONS)], 100.0, 3.0)


def line_record(game_id=1, home=(7, 7, 7, 7), away=(0, 7, 0, 7), **extra):
    return make_record(
        game_id, homeLineScores=list(home), awayLineScores=list(away),
        homePoints=sum(home), awayPoints=sum(away), **extra,
    )


def test_regulation_and_overtime_are_split():
    rows, report = parse_regulation([
        line_record(1),
        line_record(2, home=(7, 7, 0, 7, 7), away=(7, 0, 7, 7, 3)),
    ])
    assert rows["overtime"].tolist() == [False, True]
    assert rows.loc[1, "home_regulation"] == 21 and rows.loc[1, "away_regulation"] == 21
    assert report.games_kept == 2


def test_inconsistent_line_scores_are_dropped_and_counted():
    records = [
        make_record(1, homeLineScores=[7, 7, 7, 7], awayLineScores=[0, 0, 0, 0], homePoints=99, awayPoints=0),
        make_record(2, homeLineScores=None, awayLineScores=None),
        make_record(3, homeLineScores=[7, 0, 0, 0, 3], awayLineScores=[0, 0, 0, 0, 0], homePoints=10, awayPoints=0),
    ]
    rows, report = parse_regulation(records)
    assert rows.empty
    assert report.dropped == {
        "line scores do not add up to the final": 1,
        "missing or partial line scores": 1,
        "overtime does not match a regulation tie": 1,
    }


def test_line_totals_parse_and_are_not_model_inputs():
    rows = parse_lines([{"id": 1, "lines": [{"provider": "consensus", "spread": -3, "overUnder": 49.5}]}])
    assert rows.loc[0, "over_under"] == 49.5
    for column in SCORE_FEATURES:
        assert "spread" not in column and "over_under" not in column and "line" not in column


def test_post_game_columns_are_not_model_inputs():
    for column in SCORE_FEATURES:
        for banned in ("regulation", "overtime", "home_points", "away_points", "margin_actual"):
            assert banned not in column


def test_stage3_refuses_holdout_seasons(league):
    games, stats, regulation = league
    holdout = games.head(3).assign(season=HOLDOUT_SEASONS[0])
    with pytest.raises(ValueError, match="holdout"):
        run_stage3(pd.concat([games, holdout], ignore_index=True), stats, regulation)


def test_score_model_refuses_later_seasons(frame):
    table = team_table(frame[frame["season"] == VALIDATION_SEASONS[0]])
    with pytest.raises(ValueError, match="tune seasons"):
        ScoreModel().fit(table)


def test_expected_points_ignore_the_games_own_stats_and_score(league):
    games, stats, regulation = league
    elo, _ = run_elo(games, EloParams())
    original = scored_frame(games, elo, stats, FeatureParams(), regulation)
    game_id = original.loc[300, "game_id"]
    box = stats["box"].copy()
    box.loc[box["game_id"] == game_id, ["points", "total_yards", "plays"]] = [99.0, 999.0, 200.0]
    altered_regulation = regulation.copy()
    altered_regulation.loc[altered_regulation["game_id"] == game_id, ["home_regulation", "away_regulation"]] = [0, 70]
    changed = scored_frame(games, elo, {**stats, "box": box}, FeatureParams(), altered_regulation)
    model = ScoreModel().fit(team_table(original[original["season"].isin(TUNE_SEASONS)]))
    row = original.index == 300
    assert model.predict(team_view(original[row], "home")) == model.predict(team_view(changed[row], "home"))
    pd.testing.assert_frame_equal(team_view(original[row], "away"), team_view(changed[row], "away"))


def test_later_seasons_cannot_change_the_stack(league):
    """Corrupting everything after 2020 must leave model coefficients and simulator untouched."""
    games, stats, regulation = league
    elo, _ = run_elo(games, EloParams())
    first = scored_frame(games, elo, stats, FeatureParams(), regulation)
    later_ids = set(games.loc[games["season"] > max(TUNE_SEASONS), "game_id"])
    box = stats["box"].copy()
    box.loc[box["game_id"].isin(later_ids), "total_yards"] = 1.0
    changed_regulation = regulation.copy()
    changed_regulation.loc[changed_regulation["game_id"].isin(later_ids), "home_regulation"] = 0
    second = scored_frame(games, elo, {**stats, "box": box}, FeatureParams(), changed_regulation)
    a = fit_stack(first[first["season"].isin(TUNE_SEASONS)], 100.0, 3.0)
    b = fit_stack(second[second["season"].isin(TUNE_SEASONS)], 100.0, 3.0)
    assert np.array_equal(a.model.pipeline[-1].coef_, b.model.pipeline[-1].coef_)
    assert np.array_equal(a.simulator.scores, b.simulator.scores)
    assert np.array_equal(a.simulator.expected, b.simulator.expected)


def test_score_table_matches_the_expected_mean_and_uses_real_scores():
    rng = np.random.default_rng(0)
    expected = rng.uniform(15, 45, 4000)
    real_values = np.array([0, 3, 7, 10, 14, 17, 21, 24, 28, 31, 35, 38, 42, 45, 49, 52])
    actual = np.array([real_values[np.argmin(np.abs(real_values - (m + rng.normal(0, 12))))] for m in expected])
    table = ScoreTable(expected, actual, 3.0)
    draws = table.draw(30.0, rng.random(50_000))
    assert draws.mean() == pytest.approx(30.0, abs=0.3)
    assert set(np.unique(draws)) <= set(real_values)


def test_simulated_games_never_tie_and_tied_regulation_means_overtime(stack):
    games = stack.simulator.simulate(46.0, 46.0, 20_000, np.random.default_rng(1))
    assert np.all(games.home != games.away)
    assert games.overtime.any()
    assert np.all(games.home >= 0) and np.all(games.away >= 0)


def _pair_games(pairs):
    """A small table of made-up training games with known score pairs."""
    rows = np.array(pairs, dtype=float)
    return pd.DataFrame({
        "home_expected": rows[:, 0], "away_expected": rows[:, 1],
        "home_regulation": rows[:, 2], "away_regulation": rows[:, 3],
        "home_points": rows[:, 2], "away_points": rows[:, 3],
        "overtime": rows[:, 2] == rows[:, 3],
    })


@pytest.fixture(scope="module")
def correlated_games():
    """Scores that always land exactly 3 apart or tied, so pair draws must preserve that."""
    rng = np.random.default_rng(4)
    rows = []
    for _ in range(3000):
        base = rng.integers(10, 40)
        gap = rng.choice([-3, 0, 3])
        rows.append((base + gap / 2, base - gap / 2, base + gap, base))
    games = _pair_games(rows)
    # Real overtime rows add points, so the overtime step has a winner to learn from.
    tied = games["overtime"].to_numpy()
    games.loc[tied, "home_points"] = games.loc[tied, "home_regulation"] + np.where(np.arange(tied.sum()) % 2, 7, 0)
    games.loc[tied, "away_points"] = games.loc[tied, "away_regulation"] + np.where(np.arange(tied.sum()) % 2, 0, 7)
    return games


def test_pair_draws_keep_how_real_scores_move_together(correlated_games):
    simulator = PairSimulator(correlated_games, 5.0)
    games = simulator.simulate(25.0, 25.0, 20_000, np.random.default_rng(2))
    regulation = ~games.overtime
    assert set(np.unique(np.abs(games.home[regulation] - games.away[regulation]))) == {3}
    assert games.overtime.mean() > 0.2


def test_separate_draws_lose_that_relationship(correlated_games):
    simulator = SeparateDrawSimulator(correlated_games, 5.0)
    games = simulator.simulate(25.0, 25.0, 20_000, np.random.default_rng(2))
    assert len(np.unique(np.abs(games.home[~games.overtime] - games.away[~games.overtime]))) > 5


def test_pair_draws_match_the_expected_points(correlated_games):
    simulator = PairSimulator(correlated_games, 5.0)
    # These made-up games are never more than 3 apart, so the target must respect that.
    weights = simulator.weights(26.0, 24.5)
    home_mean, away_mean = weights @ simulator.scores
    assert home_mean == pytest.approx(26.0, abs=0.05)
    assert away_mean == pytest.approx(24.5, abs=0.05)


def test_pair_draws_widen_to_avoid_resting_on_too_few_games(correlated_games):
    simulator = PairSimulator(correlated_games, 0.5)
    weights = simulator.weights(26.0, 24.0)
    assert 1.0 / np.sum(weights**2) >= 200


def test_extreme_expected_points_still_give_valid_probabilities(stack):
    weights = stack.simulator.weights(70.0, 3.0)
    assert np.isfinite(weights).all() and weights.sum() == pytest.approx(1.0)


def test_stronger_expected_team_wins_more_often(stack):
    games = stack.simulator.simulate(55.0, 44.0, 20_000, np.random.default_rng(3))
    assert np.mean(games.home > games.away) > 0.7


def test_same_seed_gives_same_result(stack):
    first = stack.simulator.simulate(30.0, 24.0, 1000, np.random.default_rng(9))
    second = stack.simulator.simulate(30.0, 24.0, 1000, np.random.default_rng(9))
    assert np.array_equal(first.home, second.home) and np.array_equal(first.away, second.away)


def test_summary_is_consistent():
    games = SimulatedGames(np.array([30, 20, 10, 7]), np.array([20, 10, 14, 0]), np.array([False] * 4))
    summary = summarize(games)
    assert summary["home_win_probability"] == 0.75
    assert summary["home_win_probability"] + summary["away_win_probability"] == 1.0
    assert summary["margin"]["mean"] == pytest.approx(5.75)
    assert summary["total"]["median"] == pytest.approx(27.0)
    low, high = summary["home"]["intervals"]["95%"]
    assert low >= 7 and high <= 30


def test_matchup_row_matches_walked_state(league, stack):
    games, stats, _ = league
    elo, ratings = run_elo(games, EloParams())
    _, state = walk_forward(games.merge(elo, on="game_id"), stats, FeatureParams())
    row = matchup_row(state, ratings, 2022, 1, 2, "T1", "T2", True)
    assert row.loc[0, "elo_diff"] == ratings[1] - ratings[2]
    assert row.loc[0, "true_home"] == 0
    home, away = stack.expected(row)
    assert 0 <= home[0] < 100 and 0 <= away[0] < 100
    # At a neutral site, swapping the teams must swap the projections.
    swapped = matchup_row(state, ratings, 2022, 2, 1, "T2", "T1", True)
    swapped_home, swapped_away = stack.expected(swapped)
    assert swapped_home[0] == pytest.approx(away[0]) and swapped_away[0] == pytest.approx(home[0])


def test_factors_add_up_to_the_expected_margin(league, stack, frame):
    row = frame.tail(1).reset_index(drop=True)
    home_view, away_view = team_view(row, "home"), team_view(row, "away")
    table = factors_table(stack.model.contributions(home_view).iloc[0], stack.model.contributions(away_view).iloc[0])
    home, away = stack.expected(row)
    assert sum(item["margin"] for item in table) == pytest.approx(home[0] - away[0], abs=1e-6)


def test_crps_prefers_a_distribution_centered_on_the_result():
    rng = np.random.default_rng(0)
    centered, off = rng.normal(0, 10, 5000), rng.normal(20, 10, 5000)
    assert crps(centered, 0) < crps(off, 0)


def test_point_errors_values():
    result = point_errors(np.array([10, 20]), np.array([12, 16]))
    assert result["mae"] == 3 and result["bias"] == -1
    assert result["rmse"] == pytest.approx(np.sqrt(10))


def test_stage3_runs_end_to_end(league, monkeypatch):
    games, stats, regulation = league
    monkeypatch.setattr(backtest, "ELO_GRID", {"k": (30.0,)})
    monkeypatch.setattr(stage2, "HALFLIVES", (8.0,))
    monkeypatch.setattr(stage2, "SEASON_WEIGHTS", (0.5,))
    monkeypatch.setattr(stage2, "REGULARIZATION", (0.1,))
    monkeypatch.setattr(stage3, "HALFLIVES", (8.0,))
    monkeypatch.setattr(stage3, "SEASON_WEIGHTS", (0.5,))
    monkeypatch.setattr(stage3, "ALPHAS", (100.0,))
    monkeypatch.setattr(stage3, "POISSON_ALPHAS", (0.01,))
    monkeypatch.setattr(stage3, "BANDWIDTHS", (5.0,))
    monkeypatch.setattr(stage3, "SIMULATIONS", 200)
    monkeypatch.setattr(stage3, "SELECTION_SIMULATIONS", 100)
    report = run_stage3(games, stats, regulation)
    assert report["diagnostic_seasons"] == list(VALIDATION_SEASONS)
    assert "simulation" in report["win_probability"]
    text = stage3.report_to_markdown(report)
    assert "closing" not in text.lower().replace("not described as closing lines", "")
    assert "## Point forecasts" in text and "## Interval coverage" in text


def test_matchup_text_includes_the_caveats():
    games = SimulatedGames(np.array([30, 20, 24]), np.array([20, 10, 28]), np.array([False] * 3))
    result = {
        "home_team": "A", "away_team": "B", "neutral_site": True, "as_of": "end of the 2022 season",
        "stats_model_home_win_probability": 0.6, "simulation": summarize(games),
        "margin_factors": [{"factor": "site", "margin": 1.0}],
    }
    text = format_matchup(result)
    assert "not a current forecast" in text and "Injuries" in text
