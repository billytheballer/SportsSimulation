"""Stage 2 tests. All games and stats here are made up and never reach a report."""

import numpy as np
import pandas as pd
import pytest

from conftest import make_record

from cfbsim import backtest, stage2
from cfbsim.config import HISTORICAL_SEASONS, HOLDOUT_SEASONS, TUNE_SEASONS, VALIDATION_SEASONS
from cfbsim.elo import EloParams, run_elo
from cfbsim.features import ALL_FEATURES, FeatureParams, build_features, recruiting_scores
from cfbsim.games import parse_games
from cfbsim.metrics import favorite_calibration
from cfbsim.model import WinModel
from cfbsim.stats import fetch_plan, parse_advanced, parse_box_scores, parse_lines

TEAMS = 12


def box_row(game_id, team_id, points, yards=400.0, turnovers=1.0):
    return {
        "game_id": game_id, "team_id": team_id, "points": float(points), "total_yards": yards,
        "plays": 70.0, "pass_yards": yards / 2, "pass_attempts": 30.0, "rush_yards": yards / 2,
        "rush_attempts": 40.0, "turnovers": turnovers,
    }


def advanced_row(game_id, team, value):
    row = {"game_id": game_id, "team": team}
    for side in ("off", "def"):
        for metric in ("ppa", "pass_ppa", "rush_ppa", "success", "explosiveness"):
            row[f"{side}_{metric}"] = value
    return row


def make_league(seed=1, seasons=HISTORICAL_SEASONS):
    rng = np.random.default_rng(seed)
    strength = rng.normal(0, 10, TEAMS)
    records, box, advanced, game_id = [], [], [], 0
    lines = []
    for season in seasons:
        for week in range(1, 11):
            for home, away in rng.permutation(TEAMS).reshape(-1, 2).tolist():
                game_id += 1
                margin = int(round(strength[home] - strength[away] + 3 + rng.normal(0, 12))) or 1
                home_points, away_points = 40 + max(margin, 0), 40 + max(-margin, 0)
                records.append(
                    make_record(
                        game_id, season=season, week=week,
                        startDate=f"{season}-09-{week:02d}T19:00:00.000Z",
                        homeId=home, awayId=away, homeTeam=f"T{home}", awayTeam=f"T{away}",
                        homePoints=home_points, awayPoints=away_points,
                    )
                )
                box += [box_row(game_id, home, home_points, 350 + 8 * strength[home]),
                        box_row(game_id, away, away_points, 350 + 8 * strength[away])]
                advanced += [advanced_row(game_id, f"T{home}", 0.01 * strength[home]),
                             advanced_row(game_id, f"T{away}", 0.01 * strength[away])]
                lines.append(
                    {"game_id": game_id, "home_spread": -margin / 2, "over_under": 85.0, "line_source": "consensus"}
                )
    games, _ = parse_games(records)
    recruiting = pd.DataFrame(
        [{"year": y, "team": f"T{t}", "points": 200 + strength[t]}
         for y in range(min(seasons) - 3, max(seasons) + 1) for t in range(TEAMS)]
    )
    returning = pd.DataFrame(
        [{"season": s, "team": f"T{t}", "usage": 0.6} for s in seasons for t in range(TEAMS)]
    )
    stats = {"box": pd.DataFrame(box), "advanced": pd.DataFrame(advanced),
             "lines": pd.DataFrame(lines), "recruiting": recruiting, "returning": returning}
    return games, stats


@pytest.fixture(scope="module")
def league():
    return make_league()


def featured(games, stats, params=FeatureParams()):
    elo, _ = run_elo(games, EloParams())
    return build_features(games.merge(elo, on="game_id"), stats, params)


def test_first_game_has_no_stat_history(league):
    frame = featured(*league)
    assert frame.loc[0, ALL_FEATURES[2:12]].isna().all()


def test_features_ignore_future_stats(league):
    """The core check: rewriting later stats can't change any earlier game's features."""
    games, stats = league
    original = featured(games, stats)
    cutoff = len(games) // 2
    later = set(games["game_id"].iloc[cutoff:])

    box = stats["box"].copy()
    box.loc[box["game_id"].isin(later), ["points", "total_yards", "turnovers"]] = [99.0, 999.0, 9.0]
    advanced = stats["advanced"].copy()
    advanced.loc[advanced["game_id"].isin(later), "off_ppa"] = 5.0
    changed = featured(games, {**stats, "box": box, "advanced": advanced})

    pd.testing.assert_frame_equal(
        original.loc[: cutoff, ALL_FEATURES], changed.loc[: cutoff, ALL_FEATURES]
    )
    assert not original[ALL_FEATURES].equals(changed[ALL_FEATURES])


def test_features_ignore_the_games_own_stats(league):
    games, stats = league
    target = 300
    game_id = games.loc[target, "game_id"]
    box = stats["box"].copy()
    box.loc[box["game_id"] == game_id, ["points", "total_yards"]] = [99.0, 999.0]
    original = featured(games, stats)
    changed = featured(games, {**stats, "box": box})
    pd.testing.assert_series_equal(
        original.loc[target, ALL_FEATURES], changed.loc[target, ALL_FEATURES]
    )


def test_features_ignore_the_games_own_score(league):
    games, stats = league
    altered = games.copy()
    altered.loc[300, ["home_points", "away_points", "margin", "home_win"]] = [0, 70, -70, 0]
    frame_a, frame_b = featured(games, stats), featured(altered, stats)
    pd.testing.assert_series_equal(frame_a.loc[300, ALL_FEATURES], frame_b.loc[300, ALL_FEATURES])


def test_out_of_order_games_are_rejected(league):
    games, stats = league
    elo, _ = run_elo(games, EloParams())
    shuffled = games.merge(elo, on="game_id").iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="sorted"):
        build_features(shuffled, stats, FeatureParams())


def test_missing_stats_stay_missing_rather_than_zero():
    rows = parse_box_scores(
        [{"id": 1, "teams": [{"teamId": 5, "points": 21, "stats": [{"category": "totalYards", "stat": "300"}]}]}]
    )
    assert rows.loc[0, "total_yards"] == 300
    assert np.isnan(rows.loc[0, "turnovers"]) and np.isnan(rows.loc[0, "plays"])


def test_box_score_attempts_are_parsed():
    rows = parse_box_scores(
        [{"id": 1, "teams": [{"teamId": 5, "points": 21, "stats": [
            {"category": "completionAttempts", "stat": "12-20"},
            {"category": "rushingAttempts", "stat": "30"},
        ]}]}]
    )
    assert rows.loc[0, "pass_attempts"] == 20 and rows.loc[0, "plays"] == 50


def test_advanced_stats_are_flattened():
    block = {"ppa": 0.2, "successRate": 0.4, "explosiveness": 1.2,
             "passingPlays": {"ppa": 0.3}, "rushingPlays": {"ppa": 0.1}}
    rows = parse_advanced([{"gameId": 1, "team": "A", "offense": block, "defense": block}])
    assert rows.loc[0, "off_pass_ppa"] == 0.3 and rows.loc[0, "def_success"] == 0.4


def test_lines_prefer_consensus_then_median():
    rows = parse_lines(
        [
            {"id": 1, "lines": [{"provider": "consensus", "spread": -3}, {"provider": "x", "spread": -7}]},
            {"id": 2, "lines": [{"provider": "x", "spread": 2}, {"provider": "y", "spread": 4}, {"provider": "z", "spread": 9}]},
            {"id": 3, "lines": []},
            {"id": 4, "lines": [{"provider": "x", "spread": None}]},
        ]
    )
    assert rows["home_spread"].tolist() == [-3, 4]
    assert rows["line_source"].tolist() == ["consensus", "median"]


def test_recruiting_needs_all_four_classes():
    table = pd.DataFrame(
        [{"year": y, "team": "A", "points": float(y)} for y in (2015, 2016, 2017, 2018)]
        + [{"year": 2018, "team": "B", "points": 100.0}]
    )
    scores = recruiting_scores(table)
    assert scores[(2018, "A")] == 2016.5
    assert (2018, "B") not in scores


def test_recruiting_uses_no_later_class():
    table = pd.DataFrame(
        [{"year": y, "team": "A", "points": 10.0} for y in (2015, 2016, 2017, 2018)]
        + [{"year": 2019, "team": "A", "points": 999.0}]
    )
    assert recruiting_scores(table)[(2018, "A")] == 10.0


def test_fetch_plan_uses_known_weeks_only(league):
    games, _ = league
    plan = fetch_plan(games, [2019])
    weeks = {params["week"] for endpoint, params in plan if endpoint == "games/teams"}
    assert weeks == set(range(1, 11))


def test_win_model_refuses_later_seasons(league):
    frame = featured(*league)
    with pytest.raises(ValueError, match="tune seasons"):
        WinModel(ALL_FEATURES).fit(frame[frame["season"] == VALIDATION_SEASONS[0]])


def test_win_model_handles_missing_values_with_indicators(league):
    frame = featured(*league)
    model = WinModel(ALL_FEATURES).fit(frame[frame["season"].isin(TUNE_SEASONS)])
    blank = frame.head(3).copy()
    blank[ALL_FEATURES[2:]] = np.nan
    assert np.isfinite(model.predict(blank)).all()


@pytest.fixture
def small_grids(monkeypatch):
    monkeypatch.setattr(backtest, "ELO_GRID", {"k": (20.0, 30.0)})
    monkeypatch.setattr(stage2, "HALFLIVES", (8.0,))
    monkeypatch.setattr(stage2, "SEASON_WEIGHTS", (0.5,))
    monkeypatch.setattr(stage2, "REGULARIZATION", (0.01, 0.1))


def test_selection_refuses_later_seasons(league, small_grids):
    games, stats = league
    elo, _ = run_elo(games, EloParams())
    with pytest.raises(ValueError, match="after the tune seasons"):
        stage2.select_settings(games, elo, stats)


def test_later_results_do_not_change_the_fit(league, small_grids):
    """Corrupting all stats and results after 2020 must leave the trained model untouched."""
    games, stats = league
    flipped = games.copy()
    later = flipped["season"] > max(TUNE_SEASONS)
    flipped.loc[later, "margin"] = -flipped.loc[later, "margin"]
    flipped.loc[later, "home_win"] = 1 - flipped.loc[later, "home_win"]
    later_ids = set(games.loc[later, "game_id"])
    box = stats["box"].copy()
    box.loc[box["game_id"].isin(later_ids), "total_yards"] = 1.0

    first = stage2.run_stage2(games, stats)
    second = stage2.run_stage2(flipped, {**stats, "box": box})
    for key in ("elo_params", "feature_params", "regularization_C", "coefficients"):
        assert first[key] == second[key]
    assert first["validation"]["models"] != second["validation"]["models"]


def test_holdout_is_scored_only_on_request(league, small_grids):
    games, stats = league
    assert "holdout" not in stage2.run_stage2(games, stats)
    report = stage2.run_stage2(games, stats, include_holdout=True)
    assert report["holdout"]["seasons"] == list(HOLDOUT_SEASONS)
    text = stage2.report_to_markdown(report)
    assert "## Validation" in text and "## Holdout" in text


def test_favorite_calibration_takes_the_favorites_side():
    table = favorite_calibration(np.array([0, 1]), np.array([0.1, 0.9]))
    row = table.iloc[0]
    assert row["games"] == 2 and row["forecast"] == pytest.approx(0.9) and row["actual"] == 1.0
