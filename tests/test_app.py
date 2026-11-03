"""Tests for the prediction service and interface. All teams and stats here are made up."""

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest
from test_stage2 import make_league
from test_stage3 import regulation_for

from cfbsim import bundle as bundle_module
from cfbsim.app import charts
from cfbsim.bundle import (
    COMPARISON_METRICS,
    HISTORICAL,
    PRODUCTION,
    PredictionBundle,
    latest_teams,
    load_bundle,
    save_bundle,
    team_profiles,
    team_records,
)
from cfbsim.config import DEVELOPMENT_SEASONS, TUNE_SEASONS
from cfbsim.elo import EloParams, run_elo
from cfbsim.features import ALL_FEATURES, FeatureParams, walk_forward
from cfbsim.model import WinModel
from cfbsim.predictor import Predictor, assess_freshness
from cfbsim.stage3 import fit_stack, scored_frame

APP_PATH = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")


def make_bundle(kind=HISTORICAL, data_through=date(2023, 1, 9), **overrides):
    games, stats = make_league(seasons=DEVELOPMENT_SEASONS)
    regulation = regulation_for(games)
    elo, ratings = run_elo(games, EloParams())
    with_elo = games.merge(elo, on="game_id")
    params = FeatureParams()
    frame = scored_frame(games, elo, stats, params, regulation)
    train = frame[frame["season"].isin(TUNE_SEASONS)]
    stack = fit_stack(train, 100.0, 5.0)
    _, state = walk_forward(with_elo, stats, params)
    teams, conferences = latest_teams(games, 2022)
    values = dict(
        kind=kind,
        title="Fake league",
        season=2022,
        training_seasons=list(TUNE_SEASONS),
        data_through=data_through,
        created_on=date(2026, 11, 3),
        teams=teams,
        conferences=conferences,
        records=team_records(games, 2022),
        ratings=ratings,
        score_state=state,
        win_state=state,
        score_model=stack.model,
        win_model=WinModel(ALL_FEATURES, 0.1).fit(train),
        simulator=stack.simulator,
        profiles=team_profiles(state, ratings, 2022, teams),
    )
    values.update(overrides)
    return PredictionBundle(**values)


@pytest.fixture(scope="module")
def demo():
    return make_bundle()


@pytest.fixture(scope="module")
def prediction(demo):
    return Predictor(demo).predict("T1", "T2", simulations=2000)


def test_probabilities_are_complete_and_never_tie(prediction):
    summary = prediction.summary
    assert summary["home_win_probability"] + summary["away_win_probability"] == pytest.approx(1.0)
    assert np.all(prediction.home_scores != prediction.away_scores)


def test_same_seed_repeats_and_neutral_site_changes_the_result(demo):
    predictor = Predictor(demo)
    first = predictor.predict("T1", "T2", simulations=500)
    second = predictor.predict("T1", "T2", simulations=500)
    neutral = predictor.predict("T1", "T2", neutral_site=True, simulations=500)
    assert np.array_equal(first.home_scores, second.home_scores)
    assert neutral.home_expected != first.home_expected
    assert next(f for f in neutral.factors if f["factor"] == "site")["margin"] == pytest.approx(0.0)


def test_rejects_same_or_unknown_teams(demo):
    predictor = Predictor(demo)
    with pytest.raises(ValueError, match="different"):
        predictor.predict("T1", "T1")
    with pytest.raises(ValueError, match="not available"):
        predictor.predict("T1", "Nowhere State")


def test_factors_have_readable_labels_and_add_up(prediction):
    assert all(factor["label"] for factor in prediction.factors)
    total = sum(factor["margin"] for factor in prediction.factors)
    assert total == pytest.approx(prediction.home_expected - prediction.away_expected, abs=1e-6)


def test_percentile_100_is_always_best(demo):
    metric = next(m for m in COMPARISON_METRICS if m.key == "def_points")
    profiles = demo.profiles
    best = profiles["def_points"].idxmin()
    assert profiles.at[best, f"{metric.key}_percentile"] == profiles[f"{metric.key}_percentile"].max()


def test_historical_bundle_is_never_a_current_forecast(demo):
    # Even on the day its data ends, a historical bundle isn't a current forecast.
    freshness = assess_freshness(demo, today=demo.data_through)
    assert not freshness.is_current_forecast
    assert "not a forecast" in freshness.headline


def test_production_bundle_reports_its_date_and_goes_stale():
    through = date(2026, 11, 1)
    production = make_bundle(kind=PRODUCTION, data_through=through)
    fresh = assess_freshness(production, today=through + timedelta(days=2))
    stale = assess_freshness(production, today=through + timedelta(days=30))
    assert fresh.is_current_forecast and "November 1, 2026" in fresh.headline
    assert not stale.is_current_forecast and "30 days old" in stale.headline


def test_notice_for_missing_returning_data(demo):
    profiles = demo.profiles.copy()
    profiles.loc["T1", "returning"] = np.nan
    altered = make_bundle(profiles=profiles)
    texts = [notice.text for notice in Predictor(altered).predict("T1", "T2", simulations=200).notices]
    assert any("returning-production" in text for text in texts)


def test_notice_when_teams_already_played(demo):
    altered = make_bundle(completed_matchups=[("T2", "T1", date(2026, 10, 3), "T1 28, T2 21")])
    texts = [notice.text for notice in Predictor(altered).predict("T1", "T2", simulations=200).notices]
    assert any("hypothetical rematch" in text for text in texts)


def test_bundle_round_trip(demo, tmp_path):
    path = save_bundle(demo, tmp_path / "bundle.joblib")
    restored = Predictor(load_bundle(path)).predict("T1", "T2", simulations=200)
    original = Predictor(demo).predict("T1", "T2", simulations=200)
    assert np.array_equal(restored.home_scores, original.home_scores)


@pytest.mark.parametrize(
    "builder",
    [
        charts.score_ranges,
        charts.margin_distribution,
        charts.total_distribution,
        charts.team_score_distribution,
        charts.final_score_heatmap,
        charts.factor_chart,
    ],
)
def test_every_chart_builds_and_has_a_table(prediction, builder):
    chart, table = builder(prediction)
    assert chart.to_dict()
    assert len(table) > 0


@pytest.mark.parametrize("group", ["Overall", "Offense", "Defense"])
def test_comparison_chart_handles_each_group(prediction, group):
    chart, table = charts.comparison_chart(prediction, group)
    assert chart.to_dict()
    assert "T1 percentile" in table.columns


def test_comparison_chart_handles_missing_values(demo):
    profiles = demo.profiles.copy()
    profiles.loc["T1", "elo_percentile"] = np.nan
    profiles.loc["T1", "elo"] = np.nan
    altered = make_bundle(profiles=profiles)
    chart, _ = charts.comparison_chart(Predictor(altered).predict("T1", "T2", simulations=200), "Overall")
    assert chart.to_dict()


@pytest.fixture
def app(tmp_path, demo, monkeypatch):
    path = save_bundle(demo, tmp_path / "demo.joblib")
    monkeypatch.setattr(bundle_module, "DEMO_BUNDLE", path)
    monkeypatch.setattr(bundle_module, "PRODUCTION_BUNDLE", tmp_path / "missing.joblib")
    return AppTest.from_file(APP_PATH, default_timeout=60)


def page_text(page) -> str:
    return " ".join(item.value for item in page.markdown)


def test_app_warns_about_old_data_before_any_prediction(app):
    page = app.run()
    assert not page.exception
    assert "Historical demonstration" in page_text(page)
    assert "not a forecast" in page_text(page)


def test_app_generates_a_prediction_with_the_warning_still_visible(app):
    page = app.run()
    page.button[0].click().run()
    assert not page.exception
    text = page_text(page)
    assert "Win probability" in text and "not a forecast" in text
    assert "10,000" in text


def test_app_neutral_site_option_changes_the_card(app):
    page = app.run()
    page.checkbox[0].check()
    page.button[0].click().run()
    assert not page.exception
    assert "Neutral site" in page_text(page)


def test_app_rejects_the_same_team_twice(app):
    page = app.run()
    team = page.selectbox[0].value
    page.selectbox[1].set_value(team)
    page.button[0].click().run()
    assert not page.exception
    assert any("different" in item.value for item in page.error)


def test_app_explains_when_no_bundle_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(bundle_module, "DEMO_BUNDLE", tmp_path / "none.joblib")
    monkeypatch.setattr(bundle_module, "PRODUCTION_BUNDLE", tmp_path / "none2.joblib")
    page = AppTest.from_file(APP_PATH, default_timeout=60).run()
    assert any("build-demo" in item.value for item in page.error)


@pytest.fixture
def two_bundle_app(tmp_path, demo, monkeypatch):
    upcoming = [
        {
            "kickoff": date(2026, 11, 7), "kickoff_time": None, "week": 11,
            "home": "T3", "away": "T4", "neutral_site": True,
        }
    ]
    production = make_bundle(kind=PRODUCTION, data_through=date.today(), upcoming=upcoming)
    monkeypatch.setattr(bundle_module, "DEMO_BUNDLE", save_bundle(demo, tmp_path / "demo.joblib"))
    monkeypatch.setattr(bundle_module, "PRODUCTION_BUNDLE", save_bundle(production, tmp_path / "production.joblib"))
    return AppTest.from_file(APP_PATH, default_timeout=60)


def test_current_season_shows_its_data_date_by_default(two_bundle_app):
    page = two_bundle_app.run()
    assert not page.exception
    assert page.radio[0].value == "Current season"
    assert "Team stats current through" in page_text(page)
    assert "not a forecast" not in page_text(page)


def test_switching_to_history_shows_the_warning_and_hides_the_old_prediction(two_bundle_app):
    page = two_bundle_app.run()
    page.button[0].click().run()
    assert "Win probability" in page_text(page)
    page.radio[0].set_value("Historical demonstration (2022)").run()
    assert not page.exception
    text = page_text(page)
    assert "not a forecast" in text and "Win probability" not in text


def test_picking_an_upcoming_game_fills_in_the_matchup(two_bundle_app):
    page = two_bundle_app.run()
    picker = page.selectbox[0]
    assert picker.label == "Upcoming games (optional)"
    picker.set_value("Nov 7 · T4 vs. T3").run()
    assert not page.exception
    assert page.selectbox[1].value == "T3" and page.selectbox[2].value == "T4"
    assert page.checkbox[0].value is True
