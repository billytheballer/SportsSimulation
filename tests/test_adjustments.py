"""Tests for the FPI and availability layer. All teams, players, and reports here are made up."""

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from test_app import APP_PATH, make_bundle, page_text

from cfbsim import bundle as bundle_module
from cfbsim.adjustments import (
    AdjustmentInputs,
    AvailabilityError,
    PlayerValue,
    active_reports,
    add_report,
    apply_adjustments,
    current_player_values,
    fit_impact_coefficients,
    impact_design,
    latest_fpi_snapshot,
    load_reports,
    normalize_name,
    parse_player_box,
    save_fpi_snapshot,
    team_availability,
    validate_report,
    walk_absences,
    write_forecast_log,
)
from cfbsim.bundle import PRODUCTION, save_bundle
from cfbsim.config import HOLDOUT_SEASONS
from cfbsim.predictor import Predictor
from cfbsim.production import public_copy

TODAY = date(2026, 11, 3)


def report_row(**overrides):
    row = {
        "team": "T1",
        "player": "Star Passer",
        "status": "Out",
        "game_date": "2026-11-07",
        "reported_at": "2026-11-04T20:00:00-05:00",
        "source_url": "https://bigten.org/report",
    }
    return {**row, **overrides}


def make_inputs(**overrides):
    values = dict(
        fpi={"T1": 10.0, "T2": 0.0},
        fpi_as_of=TODAY,
        fpi_weight=0.25,
        reports={},
        player_values={
            "T1": {
                "star passer": PlayerValue("Star Passer", "qb", 300.0, 8, 0),
                "long gone": PlayerValue("Long Gone", "skill", 100.0, 3, 5),
            }
        },
        points_per_100_yards={"qb": 1.0, "skill": 1.0},
    )
    return AdjustmentInputs(**{**values, **overrides})


def listed(status="out", player="Star Passer"):
    report = validate_report(report_row(status=status, player=player))
    return {"T1": [report]}


@pytest.mark.parametrize(
    "url",
    [
        "https://www.espn.com/report",
        "https://fakebigten.org/report",
        "https://bigten.org.evil.com/report",
        "http://bigten.org/report",
        "not a link",
    ],
)
def test_unofficial_sources_are_refused(url):
    with pytest.raises(AvailabilityError, match="official"):
        validate_report(report_row(source_url=url))


@pytest.mark.parametrize("url", ["https://bigten.org/a", "https://www.bigten.org/a", "https://athletics.stateu.edu/a"])
def test_official_sources_are_accepted(url):
    assert validate_report(report_row(source_url=url))["source_url"] == url


@pytest.mark.parametrize("field", ["team", "player", "status", "game_date", "reported_at", "source_url"])
def test_incomplete_reports_are_refused(field):
    with pytest.raises(AvailabilityError, match=field):
        validate_report(report_row(**{field: ""}))


def test_unknown_status_and_missing_time_zone_are_refused():
    with pytest.raises(AvailabilityError, match="Unknown status"):
        validate_report(report_row(status="banged up"))
    with pytest.raises(AvailabilityError, match="time zone"):
        validate_report(report_row(reported_at="2026-11-04T20:00:00"))


def test_a_report_dated_after_its_game_is_refused():
    with pytest.raises(AvailabilityError, match="after its game"):
        validate_report(report_row(game_date="2026-11-01", reported_at="2026-11-04T20:00:00-05:00"))


def test_reports_save_and_reload(tmp_path):
    path = tmp_path / "reports.csv"
    add_report(report_row(), path)
    add_report(report_row(player="Other Guy", status="doubtful"), path)
    assert [report["status"] for report in load_reports(path)] == ["out", "doubtful"]


def test_refused_report_is_not_saved(tmp_path):
    path = tmp_path / "reports.csv"
    with pytest.raises(AvailabilityError):
        add_report(report_row(source_url="https://www.espn.com/x"), path)
    assert not path.exists()


def test_only_the_next_games_latest_report_applies():
    reports = [
        validate_report(report_row(status="questionable", reported_at="2026-11-04T20:00:00-05:00")),
        validate_report(report_row(status="out", reported_at="2026-11-06T20:00:00-05:00")),
        validate_report(report_row(player="Old Injury", game_date="2026-10-31", reported_at="2026-10-30T20:00:00-05:00")),
        validate_report(report_row(player="Far Future", game_date="2026-11-28", reported_at="2026-11-02T20:00:00-05:00")),
    ]
    active = active_reports(reports, data_through=date(2026, 10, 31), today=TODAY)
    assert [(r["player"], r["status"]) for r in active["T1"]] == [("Star Passer", "out")]


def test_a_report_for_a_game_already_in_the_data_is_ignored():
    reports = [validate_report(report_row(game_date="2026-11-07"))]
    assert active_reports(reports, data_through=date(2026, 11, 7), today=date(2026, 11, 7)) == {}


def test_out_costs_the_full_estimate_and_questionable_half():
    assert team_availability("T1", make_inputs(reports=listed("out")))[0] == pytest.approx(3.0)
    assert team_availability("T1", make_inputs(reports=listed("questionable")))[0] == pytest.approx(1.5)
    assert team_availability("T1", make_inputs(reports=listed("available")))[0] == 0.0


def test_unknown_players_are_listed_with_no_invented_effect():
    points, rows = team_availability("T1", make_inputs(reports=listed(player="Star Linebacker")))
    assert points == 0.0
    assert "No estimated effect" in rows[0]["note"]


def test_a_long_absence_is_not_counted_twice():
    points, rows = team_availability("T1", make_inputs(reports=listed(player="Long Gone")))
    assert points == 0.0
    assert "already in the team's recent stats" in rows[0]["note"]


def test_availability_lowers_only_that_team_and_the_margin_moves():
    inputs = make_inputs(reports=listed("out"), fpi={})
    result = apply_adjustments("T1", "T2", 30.0, 20.0, 3.0, inputs, TODAY)
    assert result["home_expected"] == pytest.approx(27.0) and result["away_expected"] == pytest.approx(20.0)
    assert result["availability_margin"] == pytest.approx(-3.0)
    assert not result["fpi_used"]


def test_fpi_moves_the_margin_by_its_weight_and_keeps_the_total():
    # Model margin 10. FPI margin 10 - 0 + 3 home field = 13. Weight 0.25 moves it 0.75.
    result = apply_adjustments("T1", "T2", 30.0, 20.0, 3.0, make_inputs(), TODAY)
    assert result["fpi_used"] and result["fpi_margin"] == pytest.approx(13.0)
    assert result["home_expected"] - result["away_expected"] == pytest.approx(10.75)
    assert result["home_expected"] + result["away_expected"] == pytest.approx(50.0)


def test_old_or_missing_fpi_is_not_used():
    old = make_inputs(fpi_as_of=TODAY - timedelta(days=30))
    assert not apply_adjustments("T1", "T2", 30.0, 20.0, 3.0, old, TODAY)["fpi_used"]
    absent = make_inputs(fpi={"T1": 5.0})
    result = apply_adjustments("T1", "T2", 30.0, 20.0, 3.0, absent, TODAY)
    assert not result["fpi_used"] and result["home_expected"] == 30.0


def test_expected_points_never_go_below_zero():
    inputs = make_inputs(reports=listed("out"), fpi={})
    assert apply_adjustments("T1", "T2", 1.0, 20.0, 0.0, inputs, TODAY)["home_expected"] == 0.0


def test_fpi_snapshots_keep_their_date(tmp_path):
    rows = [{"year": 2026, "team": "T1", "fpi": 12.5}, {"year": 2026, "team": "T2", "fpi": None}]
    save_fpi_snapshot(rows, tmp_path, datetime(2026, 11, 3, tzinfo=timezone.utc))
    ratings, as_of = latest_fpi_snapshot(2026, tmp_path)
    assert ratings == {"T1": 12.5} and as_of == TODAY


def test_no_fpi_snapshot_gives_nothing_rather_than_a_guess(tmp_path):
    assert latest_fpi_snapshot(2026, tmp_path) == ({}, None)


def test_names_match_despite_suffixes_and_punctuation():
    assert normalize_name("A.J. Smith Jr.") == normalize_name("AJ Smith")


def box_game(game_id, players):
    def athlete(key, name, value):
        return {"id": key, "name": name, "stat": value}

    categories = [
        {"name": "passing", "types": [
            {"name": "C/ATT", "athletes": [athlete(k, n, f"10/{a}") for k, n, a, _, _ in players if a]},
            {"name": "YDS", "athletes": [athlete(k, n, str(p)) for k, n, a, p, _ in players if a]},
        ]},
        {"name": "rushing", "types": [
            {"name": "YDS", "athletes": [athlete(k, n, str(o)) for k, n, _, _, o in players if o]},
        ]},
    ]
    return {"id": game_id, "teams": [{"team": "T1", "categories": categories}, {"team": "T2", "categories": []}]}


def absence_games(count):
    return pd.DataFrame(
        {
            "game_id": range(1, count + 1),
            "season": 2019,
            "home_team": "T1",
            "away_team": "T2",
        }
    )


QB = ("1", "Star Passer", 30, 300, 0)
BACKUP = ("2", "Backup Passer", 30, 150, 0)
BACK = ("3", "Top Back", 0, 0, 100)


def test_a_missing_starter_counts_only_in_the_game_missed():
    records = [box_game(game, [QB, BACK]) for game in (1, 2, 3)] + [box_game(4, [BACKUP, BACK]), box_game(5, [QB, BACK])]
    table = walk_absences(absence_games(5), parse_player_box(records))
    assert table["home_lost_qb"].tolist() == [0, 0, 0, 300, 0]


def test_absence_needs_enough_history():
    records = [box_game(1, [QB]), box_game(2, [BACKUP])]
    table = walk_absences(absence_games(2), parse_player_box(records))
    assert table["home_lost_qb"].tolist() == [0, 0]


def test_absence_stops_counting_once_stats_have_caught_up():
    records = [box_game(game, [QB]) for game in (1, 2, 3)] + [box_game(game, [BACKUP]) for game in (4, 5, 6, 7, 8)]
    table = walk_absences(absence_games(8), parse_player_box(records))
    assert (table["home_lost_qb"].iloc[3:6] > 0).all()
    assert (table["home_lost_qb"].iloc[6:] == 0).all()


def test_a_games_own_box_score_cannot_change_values_for_earlier_games():
    records = [box_game(game, [QB, BACK]) for game in (1, 2, 3, 4)]
    first = walk_absences(absence_games(4), parse_player_box(records))
    records[3] = box_game(4, [("1", "Star Passer", 30, 900, 0)])
    second = walk_absences(absence_games(4), parse_player_box(records))
    pd.testing.assert_frame_equal(first[["home_lost_qb"]].head(3), second[["home_lost_qb"]].head(3))


def test_current_values_use_completed_games_only():
    games = absence_games(3).assign(season=2026)
    box = parse_player_box([box_game(game, [QB, BACK]) for game in (1, 2, 3)])
    values = current_player_values(games, box, 2026)
    assert values["T1"]["star passer"].role == "qb"
    assert values["T1"]["star passer"].yards_per_game == 300.0
    assert values["T1"]["top back"].role == "skill"
    assert values["T1"]["top back"].yards_per_game == 100.0


def test_impact_estimate_recovers_a_known_effect():
    rng = np.random.default_rng(0)
    absences = pd.DataFrame(
        {
            "home_lost_qb": rng.choice([0, 250], 5000, p=[0.9, 0.1]),
            "away_lost_qb": rng.choice([0, 250], 5000, p=[0.9, 0.1]),
            "home_lost_skill": rng.choice([0, 80], 5000, p=[0.8, 0.2]),
            "away_lost_skill": rng.choice([0, 80], 5000, p=[0.8, 0.2]),
        }
    )
    design = impact_design(absences)
    errors = -design @ np.array([1.5, 0.5]) + rng.normal(0, 3, 5000)
    assert fit_impact_coefficients(errors, design) == pytest.approx([1.5, 0.5], abs=0.2)


def test_impact_fit_does_not_read_holdout_seasons():
    import inspect

    from cfbsim import adjustments

    source = inspect.getsource(adjustments.fit_player_impact)
    assert "DEVELOPMENT_SEASONS" in source and "HISTORICAL_SEASONS" not in source
    assert not set(HOLDOUT_SEASONS) & set(range(2014, 2023))


def test_forecast_log_records_time_and_skips_when_empty(tmp_path):
    assert write_forecast_log([], tmp_path) is None
    path = write_forecast_log([{"home": "T1"}], tmp_path, datetime(2026, 11, 3, 12, tzinfo=timezone.utc))
    assert "2026-11-03T12" in path.read_text()


@pytest.fixture(scope="module")
def adjusted_bundle():
    today = date.today()
    report = validate_report(
        report_row(
            game_date=str(today + timedelta(days=3)),
            reported_at=f"{today}T08:00:00-05:00",
        )
    )
    inputs = make_inputs(
        fpi={f"T{number}": float(number) for number in range(12)},
        fpi_as_of=today,
        reports={"T1": [report]},
    )
    return make_bundle(kind=PRODUCTION, data_through=today, adjustments=inputs)


def test_prediction_keeps_the_tested_model_beside_the_adjusted_one(adjusted_bundle):
    result = Predictor(adjusted_bundle).predict("T1", "T2", simulations=2000)
    assert result.base_summary is not None
    assert result.adjustment["players"][0]["player"] == "Star Passer"
    # T1's starter is out, so T1 should fall below the tested model's number.
    assert result.summary["home"]["mean"] < result.base_summary["home"]["mean"]


def test_adjusted_factors_add_up_to_the_adjusted_margin(adjusted_bundle):
    predictor = Predictor(adjusted_bundle)
    result = predictor.predict("T1", "T2", simulations=200)
    total = sum(factor["margin"] for factor in result.factors)
    assert total == pytest.approx(result.home_expected - result.away_expected, abs=1e-6)
    labels = {factor["label"] for factor in result.factors}
    assert "Official player availability" in labels and any("FPI" in label for label in labels)


def test_missing_reports_are_never_presented_as_healthy(adjusted_bundle):
    result = Predictor(adjusted_bundle).predict("T1", "T2", simulations=200)
    text = " ".join(notice.text for notice in result.notices)
    assert "No official availability report is on file for T2" in text
    assert "doesn't mean" in text


def test_historical_bundle_has_no_adjustments():
    result = Predictor(make_bundle()).predict("T1", "T2", simulations=200)
    assert result.adjustment is None and result.base_summary is None


def test_public_copy_removes_fpi_and_player_reports(adjusted_bundle):
    public = public_copy(adjusted_bundle)
    assert public.adjustments is None
    result = Predictor(public).predict("T1", "T2", simulations=200)
    assert result.adjustment is None


def test_app_shows_sources_and_says_the_layer_is_untested(tmp_path, adjusted_bundle, monkeypatch):
    path = save_bundle(adjusted_bundle, tmp_path / "production.joblib")
    monkeypatch.setattr(bundle_module, "PRODUCTION_BUNDLE", path)
    monkeypatch.setattr(bundle_module, "DEMO_BUNDLE", tmp_path / "missing.joblib")
    page = AppTest.from_file(APP_PATH, default_timeout=60).run()
    page.button[0].click().run()
    assert not page.exception
    text = page_text(page)
    assert "FPI and player availability" in text
    assert "no pre-game history exists" in text
    assert "Tested model alone" in text
    assert len(page.dataframe) >= 1
