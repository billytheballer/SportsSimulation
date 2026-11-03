import numpy as np
import pandas as pd
import pytest

from conftest import make_record

from cfbsim import backtest
from cfbsim.backtest import EloCalibrator, report_to_markdown, run_backtest, tune_elo
from cfbsim.config import HISTORICAL_SEASONS, HOLDOUT_SEASONS, TUNE_SEASONS, VALIDATION_SEASONS
from cfbsim.games import parse_games
from cfbsim.metrics import calibration_table, log_loss_edge_interval, score

TEAMS = 12


@pytest.fixture
def league(monkeypatch) -> pd.DataFrame:
    """Random fake league where stronger teams win more, across all configured seasons."""
    monkeypatch.setattr(backtest, "ELO_GRID", {"k": (20.0, 30.0)})
    rng = np.random.default_rng(1)
    strength = rng.normal(0, 10, TEAMS)
    records, game_id = [], 0
    for season in HISTORICAL_SEASONS:
        for week in range(1, 11):
            for home, away in rng.permutation(TEAMS).reshape(-1, 2).tolist():
                game_id += 1
                margin = int(round(strength[home] - strength[away] + 3 + rng.normal(0, 12))) or 1
                records.append(
                    make_record(
                        game_id,
                        season=season,
                        week=week,
                        startDate=f"{season}-09-{week:02d}T19:00:00.000Z",
                        homeId=home,
                        awayId=away,
                        homePoints=40 + max(margin, 0),
                        awayPoints=40 + max(-margin, 0),
                    )
                )
    games, _ = parse_games(records)
    return games


def test_calibrator_refuses_evaluation_seasons(league):
    from cfbsim.elo import EloParams, run_elo

    predictions, _ = run_elo(league, EloParams())
    frame = league.merge(predictions, on="game_id")
    with pytest.raises(ValueError, match="tune seasons"):
        EloCalibrator().fit(frame[frame["season"] == VALIDATION_SEASONS[0]])


def test_tuning_refuses_later_seasons(league):
    with pytest.raises(ValueError, match="after the tune seasons"):
        tune_elo(league)


def test_results_after_the_tune_seasons_do_not_change_the_fit(league):
    """Flipping every later result must leave chosen settings and calibration untouched."""
    flipped = league.copy()
    later = flipped["season"] > max(TUNE_SEASONS)
    flipped.loc[later, "margin"] = -flipped.loc[later, "margin"]
    flipped.loc[later, "home_win"] = 1 - flipped.loc[later, "home_win"]

    first, second = run_backtest(league), run_backtest(flipped)
    assert first["elo_params"] == second["elo_params"]
    assert first["calibration_coefficients"] == second["calibration_coefficients"]
    assert first["validation"]["models"] != second["validation"]["models"]


def test_holdout_is_scored_only_on_request(league):
    assert "holdout" not in run_backtest(league)
    report = run_backtest(league, include_holdout=True)
    assert report["holdout"]["seasons"] == list(HOLDOUT_SEASONS)
    assert set(report["holdout"]["by_season"]) == set(HOLDOUT_SEASONS)


def test_model_beats_baseline_when_team_strength_is_real(league):
    models = run_backtest(league)["validation"]["models"]
    assert models["elo_calibrated"]["log_loss"] < models["baseline_home_rate"]["log_loss"]
    assert models["baseline_coin_flip"]["accuracy"] == 0.5


def test_report_renders(league):
    report = run_backtest(league, include_holdout=True)
    text = report_to_markdown(report)
    assert "## Validation" in text and "## Holdout" in text


def test_score_values():
    result = score(np.array([1, 0]), np.array([0.8, 0.4]))
    assert result["accuracy"] == 1.0
    assert result["brier"] == pytest.approx((0.04 + 0.16) / 2)
    assert result["log_loss"] == pytest.approx(-(np.log(0.8) + np.log(0.6)) / 2)


def test_calibration_table_counts_all_games():
    table = calibration_table(np.array([1, 0, 1, 1]), np.array([0.05, 0.15, 0.95, 1.0]))
    assert table["games"].sum() == 4


def test_edge_interval_detects_a_better_model():
    outcomes = np.array([1, 0] * 200)
    good = np.where(outcomes == 1, 0.8, 0.2)
    mean, low, high = log_loss_edge_interval(outcomes, good, np.full(400, 0.5))
    assert 0 < low <= mean <= high
