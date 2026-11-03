"""Tests for the production model. All games and stats here are made up."""

import json

import numpy as np
import pandas as pd
import pytest
from conftest import make_record
from test_client import FakeResponse, FakeSession
from test_stage2 import make_league
from test_stage3 import regulation_for

from cfbsim import production
from cfbsim.client import CFBDClient
from cfbsim.config import HISTORICAL_SEASONS
from cfbsim.final_eval import CHECKSUM_NAME, MANIFEST_NAME, RESULT_NAME, _file_hash
from cfbsim.games import parse_games
from cfbsim.predictor import Predictor
from cfbsim.production import (
    PRODUCTION_TRAINING_SEASONS,
    ProductionError,
    ProductionScoreModel,
    ProductionWinModel,
    assemble_production,
    frozen_settings,
    update_plan,
)
from cfbsim.features import ALL_FEATURES

SETTINGS = {
    "elo_params": {"k": 30.0, "home_advantage": 55.0, "season_carryover": 0.7, "lower_division_initial": 1000.0},
    "score_feature_params": {"halflife": 8.0, "season_weight": 0.5},
    "win_feature_params": {"halflife": 8.0, "season_weight": 0.5},
    "alpha": 100.0,
    "bandwidth": 5.0,
    "win_C": 0.1,
}
SEASONS = (*HISTORICAL_SEASONS, 2026)
NOW = pd.Timestamp("2026-11-03", tz="UTC")


@pytest.fixture(scope="module")
def league():
    games, stats = make_league(seasons=SEASONS)
    return games, stats, regulation_for(games)


def upcoming_records():
    return [
        make_record(
            90001, season=2026, week=11, completed=False, homePoints=None, awayPoints=None,
            startDate="2026-11-14T19:00:00.000Z", homeId=1, awayId=2, homeTeam="T1", awayTeam="T2",
        ),
        make_record(
            90002, season=2026, week=1, completed=False, homePoints=None, awayPoints=None,
            startDate="2026-09-01T19:00:00.000Z", homeId=3, awayId=4, homeTeam="T3", awayTeam="T4",
        ),
    ]


@pytest.fixture(scope="module")
def bundle(league):
    games, stats, regulation = league
    return assemble_production(games, stats, regulation, upcoming_records(), SETTINGS, now=NOW)


def test_refuses_before_the_final_evaluation_is_complete(tmp_path):
    with pytest.raises(ProductionError, match="final evaluation"):
        frozen_settings(tmp_path)


def test_refuses_if_the_final_result_was_edited(tmp_path):
    (tmp_path / RESULT_NAME).write_text('{"ok": true}')
    (tmp_path / CHECKSUM_NAME).write_text("not-the-real-hash  final_evaluation.json\n")
    (tmp_path / MANIFEST_NAME).write_text(json.dumps({"settings": SETTINGS}))
    with pytest.raises(ProductionError):
        frozen_settings(tmp_path)


def test_reads_frozen_settings_after_a_valid_evaluation(tmp_path):
    result = tmp_path / RESULT_NAME
    result.write_text('{"ok": true}')
    (tmp_path / CHECKSUM_NAME).write_text(f"{_file_hash(result)}  {RESULT_NAME}\n")
    (tmp_path / MANIFEST_NAME).write_text(json.dumps({"settings": SETTINGS}))
    assert frozen_settings(tmp_path) == SETTINGS


def test_coefficients_train_only_through_2025(bundle):
    assert bundle.training_seasons == list(range(2016, 2026))
    assert 2026 not in PRODUCTION_TRAINING_SEASONS


def test_models_refuse_2026_or_warmup_seasons():
    table = pd.DataFrame({"season": [2026], "target": [10.0]})
    with pytest.raises(ProductionError, match="2016-25"):
        ProductionScoreModel().fit(table)
    with pytest.raises(ProductionError, match="2016-25"):
        ProductionWinModel(ALL_FEATURES).fit(pd.DataFrame({"season": [2014], "home_win": [1]}))


def test_new_2026_games_update_team_data_but_never_the_coefficients(league):
    games, stats, regulation = league
    early = games[(games["season"] < 2026) | (games["week"] <= 5)].reset_index(drop=True)
    first = assemble_production(early, stats, regulation, [], SETTINGS, now=NOW)
    second = assemble_production(games, stats, regulation, [], SETTINGS, now=NOW)
    assert np.array_equal(first.score_model.pipeline[-1].coef_, second.score_model.pipeline[-1].coef_)
    assert np.array_equal(first.win_model.pipeline[-1].coef_, second.win_model.pipeline[-1].coef_)
    assert first.ratings != second.ratings
    assert first.data_through < second.data_through


def test_data_date_is_the_last_completed_game(league, bundle):
    games, *_ = league
    last = games.loc[games["season"] == 2026, "start_date"].max()
    assert bundle.data_through == last.tz_convert("America/New_York").date()
    assert "completed 2026 games" in bundle.data_note


def test_unplayed_games_never_enter_team_data(league):
    """A scheduled game has no score, so parsing removes it and no rating can use it."""
    games, _ = parse_games(upcoming_records())
    assert games.empty


def test_upcoming_lists_only_future_games(bundle):
    assert [(game["home"], game["away"]) for game in bundle.upcoming] == [("T1", "T2")]


def test_forecasts_for_a_scheduled_game_ignore_games_after_data_through(league, bundle):
    predictor = Predictor(bundle)
    result = predictor.predict("T1", "T2", simulations=200)
    assert result.freshness.data_through == bundle.data_through


def test_scoring_this_season_cannot_change_with_later_results(league):
    """Each 2026 forecast must be unchanged when games after it are rewritten."""
    games, stats, regulation = league
    full = assemble_production(games, stats, regulation, [], SETTINGS, now=NOW)
    assert full.live_scorecard["games"] == int((games["season"] == 2026).sum())
    assert 0 < full.live_scorecard["log_loss"] < 1

    early = games[(games["season"] < 2026) | (games["week"] <= 5)].reset_index(drop=True)
    altered = games.copy()
    later = (altered["season"] == 2026) & (altered["week"] > 5)
    altered.loc[later, "margin"] = -altered.loc[later, "margin"]
    altered.loc[later, "home_win"] = 1 - altered.loc[later, "home_win"]
    first = assemble_production(early, stats, regulation, [], SETTINGS, now=NOW)
    second = assemble_production(
        altered[altered["season"].le(2026)].reset_index(drop=True), stats, regulation, [], SETTINGS, now=NOW
    )
    # Same fit and the same early-season ratings, whatever happened afterward.
    assert np.array_equal(first.score_model.pipeline[-1].coef_, second.score_model.pipeline[-1].coef_)


def test_no_current_season_games_is_an_error_not_a_guess(league):
    games, stats, regulation = league
    older = games[games["season"] < 2026].reset_index(drop=True)
    with pytest.raises(ProductionError, match="No completed 2026 games"):
        assemble_production(older, stats, regulation, [], SETTINGS, now=NOW)


def test_rematch_is_labeled_when_teams_already_played(bundle):
    home, away = next((a, b) for a, b, *_ in bundle.completed_matchups)
    texts = [notice.text for notice in Predictor(bundle).predict(home, away, simulations=100).notices]
    assert any("hypothetical rematch" in text for text in texts)


class RoutedSession(FakeSession):
    """Answers each endpoint with its own canned response."""

    def __init__(self, routes):
        super().__init__()
        self.routes = routes

    def get(self, url, params, headers, timeout):
        self.calls.append((url, params, headers))
        for endpoint, payload in self.routes.items():
            if url.endswith(endpoint):
                return FakeResponse(payload=payload)
        return FakeResponse(payload=[])


def test_update_plan_fetches_only_weeks_with_new_games(tmp_path):
    routes = {"games/teams": [{"id": 1, "teams": []}]}
    client = CFBDClient(api_key="x", cache_dir=tmp_path, session=RoutedSession(routes), month=lambda: "2026-11")
    records = [
        {"id": 1, "seasonType": "regular", "week": 1, "completed": True},
        {"id": 2, "seasonType": "regular", "week": 2, "completed": True},
        {"id": 3, "seasonType": "regular", "week": 3, "completed": False},
    ]
    client.get("games/teams", year=2026, week=1, seasonType="regular", classification="fbs")
    plan = update_plan(client, records, 2026)
    weeks = [params["week"] for endpoint, params in plan if endpoint == "games/teams"]
    assert weeks == [2]
    records.append({"id": 9, "seasonType": "regular", "week": 1, "completed": True})
    weeks = [params["week"] for endpoint, params in update_plan(client, records, 2026) if endpoint == "games/teams"]
    assert weeks == [1, 2]


def test_update_season_refuses_before_final_evaluation(tmp_path, monkeypatch):
    monkeypatch.setattr(production, "is_complete", lambda directory=None: False)
    client = CFBDClient(api_key="x", cache_dir=tmp_path, session=FakeSession(), month=lambda: "2026-11")
    with pytest.raises(ProductionError, match="final evaluation"):
        production.update_season(client)
    assert client.session.calls == []


def test_public_copy_removes_scores_and_schedule_but_still_predicts(bundle):
    public = production.public_copy(bundle)
    assert public.upcoming == []
    assert all(score == "" for *_, score in public.completed_matchups)
    assert bundle.upcoming and any(score for *_, score in bundle.completed_matchups)
    home, away = next((a, b) for a, b, *_ in public.completed_matchups)
    result = Predictor(public).predict(home, away, simulations=100)
    rematch = next(notice.text for notice in result.notices if "hypothetical rematch" in notice.text)
    assert "()" not in rematch and " 0, " not in rematch
