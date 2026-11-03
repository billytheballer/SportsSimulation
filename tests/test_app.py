"""Tests for the prediction service and interface. All teams and stats here are made up."""

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
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
from cfbsim.app import ui
from cfbsim.app.theme import team_accent
from cfbsim.predictor import Predictor, assess_freshness
from cfbsim.season_board import (
    build_board,
    confidence_label,
    projected_records,
    season_stat_table,
    team_details,
)
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


def test_app_warns_about_old_data_on_the_first_screen(app):
    page = app.run()
    assert not page.exception
    assert "Historical demonstration" in page_text(page)
    assert "not a forecast" in page_text(page)


def test_demo_view_builds_a_custom_matchup_with_the_warning_still_visible(app):
    page = app.run()
    page.button[0].click().run()
    assert not page.exception
    text = page_text(page)
    assert "Win chance" in text and "not a forecast" in text
    assert "10,000" in text


def test_neutral_site_appears_on_the_scoreboard(app):
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


def upcoming_game(home, away, week, offset=30, neutral=False):
    kickoff = date.today() + timedelta(days=offset)
    return {
        "kickoff": kickoff, "kickoff_time": None, "week": week, "time_tbd": True,
        "home": home, "away": away, "neutral_site": neutral,
    }


@pytest.fixture(scope="module")
def season_bundle():
    upcoming = [
        upcoming_game("T3", "T4", 11, neutral=True),
        upcoming_game("T1", "T2", 11),
        upcoming_game("T5", "T6", 12, offset=37),
    ]
    bundle = make_bundle(kind=PRODUCTION, data_through=date.today(), upcoming=upcoming)
    box = [
        {"id": 1, "teams": [
            {"team": "T1", "points": 30, "stats": [
                {"category": "totalYards", "stat": "400"}, {"category": "rushingAttempts", "stat": "30"},
                {"category": "completionAttempts", "stat": "20-30"}, {"category": "thirdDownEff", "stat": "6-12"},
                {"category": "turnovers", "stat": "1"}, {"category": "totalPenaltiesYards", "stat": "5-40"},
                {"category": "possessionTime", "stat": "30:30"},
            ]},
            {"team": "T2", "points": 20, "stats": [
                {"category": "totalYards", "stat": "300"}, {"category": "rushingAttempts", "stat": "25"},
                {"category": "completionAttempts", "stat": "15-25"}, {"category": "thirdDownEff", "stat": "3-12"},
                {"category": "turnovers", "stat": "2"}, {"category": "sacks", "stat": "3"},
            ]},
        ]}
    ]
    bundle.season_stats = season_stat_table(box, pd.DataFrame(), ["T1", "T2"])
    bundle.game_logs = {"T1": [{"date": date(2026, 9, 5), "week": 1, "opponent": "T2", "site": "vs", "neutral": False,
                                "result": "W", "points_for": 30, "points_against": 20}]}
    bundle.team_info = team_details([{"school": "T1", "abbreviation": "ONE", "mascot": "Ones", "color": "#000000",
                                      "alternateColor": "#ff0000", "location": {"name": "One Field", "city": "Town"}}])
    bundle.board = build_board(bundle)
    return bundle


@pytest.fixture
def season_app(tmp_path, season_bundle, monkeypatch):
    monkeypatch.setattr(bundle_module, "DEMO_BUNDLE", tmp_path / "missing.joblib")
    monkeypatch.setattr(bundle_module, "PRODUCTION_BUNDLE", save_bundle(season_bundle, tmp_path / "production.joblib"))
    return AppTest.from_file(APP_PATH, default_timeout=60)


def by_label(page, label):
    return next(item for item in page.selectbox if item.label.startswith(label))


def test_every_remaining_game_is_in_the_dropdown(season_app):
    page = season_app.run()
    assert not page.exception
    by_label(page, "Week").set_value("All").run()
    options = by_label(page, "Game").options
    assert len(options) == 3
    assert any("T4 vs. T3" in option for option in options)
    assert any("T2 at T1" in option for option in options)


def test_game_center_opens_with_the_next_week_and_a_full_scoreboard(season_app):
    page = season_app.run()
    assert not page.exception
    assert by_label(page, "Week").value == "Week 11"
    text = page_text(page)
    assert "Win chance" in text and "Projected score" in text
    assert "Stats through" in text


def test_picking_a_game_changes_the_scoreboard(season_app):
    page = season_app.run()
    choice = next(option for option in by_label(page, "Game").options if "T2 at T1" in option)
    by_label(page, "Game").set_value(choice).run()
    assert not page.exception
    text = page_text(page)
    assert ">T1<" in text and ">T2<" in text


def test_filters_narrow_the_game_list(season_app):
    page = season_app.run()
    by_label(page, "Week").set_value("All").run()
    by_label(page, "Team").set_value("T5").run()
    assert not page.exception
    assert len(by_label(page, "Game").options) == 1


def test_build_your_own_matchup_still_works(season_app):
    page = season_app.run()
    page.toggle[0].set_value(True).run()
    page.button[0].click().run()
    assert not page.exception
    assert "Hypothetical matchup" in page_text(page)


@pytest.mark.parametrize("view", ["Schedule", "Teams", "Injuries", "About the Model"])
def test_every_page_loads(season_app, view):
    page = season_app.run()
    page.radio[0].set_value(view).run()
    assert not page.exception


def test_schedule_shows_every_game_and_its_highlights(season_app):
    page = season_app.run()
    page.radio[0].set_value("Schedule").run()
    by_label(page, "Week").set_value("All").run()
    assert not page.exception
    assert "Closest game" in page_text(page)
    assert len(page.dataframe[0].value) == 3


def test_about_page_reports_the_final_evaluation_and_the_untested_add_on(season_app):
    page = season_app.run()
    page.radio[0].set_value("About the Model").run()
    text = page_text(page)
    assert "2023-25" in text and "untested" in text
    assert "closing line" not in text.lower()


def test_board_numbers_match_the_game_page(season_bundle):
    row = next(item for item in season_bundle.board if item["home"] == "T1")
    result = Predictor(season_bundle).predict("T1", "T2", False, game_date=row["kickoff"])
    assert result.summary["home_win_probability"] == row["home_win"]
    assert result.summary["margin"]["mean"] == row["margin"]


def test_games_that_have_kicked_off_leave_the_lists(season_bundle):
    past = dict(season_bundle.board[0], kickoff_time=pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=1))
    future = dict(season_bundle.board[1], kickoff_time=pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=1))
    altered = make_bundle(board=[past, future])
    assert ui.remaining_games(altered) == [future]


def test_projected_wins_add_win_chances_to_the_current_record():
    board = [{"home": "A", "away": "B", "home_win": 0.75, "away_win": 0.25}]
    projections = projected_records(board, {"A": "5-3", "B": "2-6"})
    assert projections["A"]["projected_wins"] == pytest.approx(5.75)
    assert projections["B"]["projected_losses"] == pytest.approx(6.75)


def test_season_stats_rank_one_is_always_best(season_bundle):
    table = season_bundle.season_stats
    assert table.at["T1", "points"] == 30 and table.at["T1", "points_allowed"] == 20
    assert table.at["T1", "yards_per_play"] == pytest.approx(400 / 60)
    assert table.at["T1", "third_down_pct"] == pytest.approx(50.0)
    assert table.at["T1", "points_rank"] == 1 and table.at["T1", "points_allowed_rank"] == 1
    assert table.at["T2", "turnovers_rank"] == 2 and table.at["T1", "takeaways"] == 2


def test_missing_stats_stay_missing():
    table = season_stat_table(
        [{"id": 1, "teams": [{"team": "A", "points": 7, "stats": []}, {"team": "B", "points": 3, "stats": []}]}],
        pd.DataFrame(), ["A", "B"],
    )
    assert pd.isna(table.at["A", "total_yards"]) and table.at["A", "points"] == 7


@pytest.mark.parametrize(
    "color, expected",
    [("#000000", "#ff0000"), ("#bb0000", "#bb0000"), ("nonsense", "#ff0000"), (None, "#ff0000")],
)
def test_team_accents_stay_visible_on_the_dark_page(color, expected):
    assert team_accent({"color": color, "alternate_color": "#ff0000"})[0] == expected


def test_confidence_words():
    assert [confidence_label(p) for p in (0.52, 0.6, 0.7, 0.9, 0.1)] == ["Toss-up", "Lean", "Solid", "Strong", "Strong"]
