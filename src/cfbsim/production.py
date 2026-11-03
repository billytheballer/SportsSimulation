"""The current-season production model. Separate from the frozen final evaluation.

Differences from the evaluated model:

- It uses the same frozen settings, with no new tuning.
- Its coefficients are fit on 2016-25 instead of 2016-20.
- Team ratings and stats also walk through completed 2026 games. Those games never
  refit the coefficients, so scoring them is a genuine out-of-sample test.

It refuses to run until the final evaluation is complete, because it reads 2023-25.
Only completed games enter team data, so a forecast can't see its own or a later result.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from cfbsim.adjustments import (
    FPI_WEIGHT,
    IMPACT_PATH,
    AdjustmentInputs,
    active_reports,
    current_player_values,
    latest_fpi_details,
    latest_fpi_snapshot,
    load_reports,
    parse_player_box,
    save_fpi_snapshot,
    write_forecast_log,
)
from cfbsim.bundle import (
    PRODUCTION,
    PredictionBundle,
    eastern_date,
    latest_teams,
    team_profiles,
    team_records,
)
from cfbsim.client import CFBDClient
from cfbsim.config import HISTORICAL_SEASONS
from cfbsim.elo import EloParams, run_elo
from cfbsim.features import ALL_FEATURES, FeatureParams, walk_forward
from cfbsim.final_eval import FINAL_DIR, MANIFEST_NAME, is_complete
from cfbsim.games import SEASON_TYPES, fetch_season_records, load_games, parse_regulation
from cfbsim.metrics import point_errors, score
from cfbsim.model import WinModel
from cfbsim.scoring import ScoreModel, team_table, team_view
from cfbsim.simulate import PairSimulator
from cfbsim.stage3 import scored_frame, simulate_block
from cfbsim.stats import load_stats

CURRENT_SEASON = 2026
PRODUCTION_TRAINING_SEASONS = tuple(range(2016, CURRENT_SEASON))
CLASSIFICATION = "fbs"


class ProductionError(RuntimeError):
    """The production model can't be built, and says why."""


def _require_training_window(seasons: set[int]) -> None:
    extra = seasons - set(PRODUCTION_TRAINING_SEASONS)
    if extra:
        raise ProductionError(f"Production models train only on 2016-25, but got {sorted(extra)}.")


class ProductionScoreModel(ScoreModel):
    def fit(self, table: pd.DataFrame) -> "ProductionScoreModel":
        _require_training_window(set(table["season"]))
        known = table.dropna(subset=["target"])
        self.pipeline.fit(known[self.features], known["target"].to_numpy(dtype=float))
        return self


class ProductionWinModel(WinModel):
    def fit(self, frame: pd.DataFrame) -> "ProductionWinModel":
        _require_training_window(set(frame["season"]))
        self.pipeline.fit(frame[self.features], frame["home_win"].to_numpy())
        return self


def fit_production_stack(train: pd.DataFrame, alpha: float, bandwidth: float) -> tuple[ProductionScoreModel, PairSimulator]:
    """Fit on 2016-25. Each season's simulator inputs come from a model that skipped that season."""
    known = train.dropna(subset=["home_regulation", "away_regulation"]).reset_index(drop=True)
    fit_data = known.assign(home_expected=np.nan, away_expected=np.nan)
    for season in known["season"].unique():
        inside = (known["season"] == season).to_numpy()
        model = ProductionScoreModel(alpha).fit(team_table(known[~inside]))
        held_out = known[inside]
        fit_data.loc[inside, "home_expected"] = model.predict(team_view(held_out, "home"))
        fit_data.loc[inside, "away_expected"] = model.predict(team_view(held_out, "away"))
    return ProductionScoreModel(alpha).fit(team_table(known)), PairSimulator(fit_data, bandwidth)


def frozen_settings(directory=FINAL_DIR) -> dict[str, Any]:
    if not is_complete(directory):
        raise ProductionError("The final evaluation must be complete and unmodified first.")
    return json.loads((directory / MANIFEST_NAME).read_text())["settings"]


def _boxes_needed(
    client: CFBDClient, records: list[dict[str, Any]], season: int, endpoint: str = "games/teams"
) -> list[dict[str, Any]]:
    """Weeks whose saved box scores are missing, or older than their completed games."""
    needed = []
    for season_type in SEASON_TYPES:
        typed = [record for record in records if record["seasonType"] == season_type and record.get("completed")]
        for week in sorted({record["week"] for record in typed}):
            params = {"year": season, "week": week, "seasonType": season_type, "classification": CLASSIFICATION}
            completed_ids = {record["id"] for record in typed if record["week"] == week}
            if client.is_cached(endpoint, **params):
                saved = {game["id"] for game in client.get(endpoint, **params)}
                if completed_ids <= saved:
                    continue
            needed.append(params)
    return needed


def update_plan(client: CFBDClient, records: list[dict[str, Any]], season: int) -> list[tuple[str, dict[str, Any]]]:
    """The fewest calls that bring a season's cached data up to date."""
    plan: list[tuple[str, dict[str, Any]]] = []
    for season_type in SEASON_TYPES:
        plan.append(("stats/game/advanced", {"year": season, "seasonType": season_type}))
    plan += [("games/teams", params) for params in _boxes_needed(client, records, season)]
    plan += [("games/players", params) for params in _boxes_needed(client, records, season, "games/players")]
    return plan


def update_season(client: CFBDClient, season: int = CURRENT_SEASON) -> dict[str, Any]:
    """Refresh a live season from CFBD. Reports what it found and invents nothing."""
    frozen_settings()
    before = client.calls_this_month()
    records: list[dict[str, Any]] = []
    for season_type in SEASON_TYPES:
        records += client.get(
            "games", refresh=True, year=season, seasonType=season_type, classification=CLASSIFICATION
        )
    for endpoint, params in update_plan(client, records, season):
        client.get(endpoint, refresh=True, **params)
    # Preseason values and benchmarks: fetched once, and then the saved copies are reused.
    for endpoint, params in (
        ("teams/fbs", {"year": season}),
        ("player/returning", {"year": season}),
        ("recruiting/teams", {"year": season}),
        *(("lines", {"year": season, "seasonType": kind}) for kind in SEASON_TYPES),
    ):
        client.get(endpoint, **params)
    fpi_path = save_fpi_snapshot(client.get("ratings/fpi", refresh=True, year=season))
    completed = [record for record in records if record.get("completed") and record.get("homePoints") is not None]
    return {
        "fpi_snapshot": fpi_path.name,
        "season": season,
        "games_listed": len(records),
        "games_completed": len(completed),
        "last_completed_kickoff": max((record["startDate"] for record in completed), default=None),
        "api_calls_used": client.calls_this_month() - before,
    }


def _upcoming(records: list[dict[str, Any]], now: pd.Timestamp) -> list[dict[str, Any]]:
    games = []
    for record in records:
        if record.get("completed") or record.get("homeClassification") != "fbs" or record.get("awayClassification") != "fbs":
            continue
        kickoff = pd.to_datetime(record.get("startDate"), utc=True, errors="coerce")
        if pd.isna(kickoff) or kickoff < now:
            continue
        games.append(
            {
                "kickoff": eastern_date(kickoff),
                "kickoff_time": kickoff,
                "week": record["week"],
                "home": record["homeTeam"],
                "away": record["awayTeam"],
                "neutral_site": bool(record.get("neutralSite")),
                "time_tbd": bool(record.get("startTimeTBD")),
            }
        )
    return sorted(games, key=lambda game: (game["kickoff_time"], game["home"]))


def _completed_matchups(games: pd.DataFrame, season: int) -> list[tuple[str, str, date, str]]:
    latest = games[games["season"] == season]
    return [
        (home, away, eastern_date(start), f"{home} {home_points:.0f}, {away} {away_points:.0f}")
        for home, away, start, home_points, away_points in zip(
            latest["home_team"], latest["away_team"], latest["start_date"], latest["home_points"], latest["away_points"]
        )
    ]


def _live_scorecard(frame: pd.DataFrame, model: ProductionScoreModel, simulator: PairSimulator, season: int) -> dict[str, Any]:
    """Score this season's games so far, each forecast from data before its kickoff."""
    block = frame[frame["season"] == season].reset_index(drop=True)
    if block.empty:
        return {}
    home_expected = model.predict(team_view(block, "home"))
    away_expected = model.predict(team_view(block, "away"))
    result = simulate_block(block, home_expected, away_expected, simulator.simulate)
    outcomes = block["home_win"].to_numpy()
    probabilities = np.clip(result["probability"], 1e-3, 1 - 1e-3)
    win = score(outcomes, probabilities)
    margin = point_errors((block["home_points"] - block["away_points"]).to_numpy(), result["means"]["margin"])
    total = point_errors((block["home_points"] + block["away_points"]).to_numpy(), result["means"]["total"])
    return {
        "games": win["games"],
        "accuracy": win["accuracy"],
        "log_loss": win["log_loss"],
        "brier": win["brier"],
        "margin_mae": margin["mae"],
        "margin_rmse": margin["rmse"],
        "total_mae": total["mae"],
        "total_rmse": total["rmse"],
        "total_bias": total["bias"],
    }


def assemble_production(
    games: pd.DataFrame,
    stats: dict,
    regulation: pd.DataFrame,
    records: list[dict[str, Any]],
    settings: dict[str, Any],
    season: int = CURRENT_SEASON,
    now: pd.Timestamp | None = None,
    today: date | None = None,
) -> PredictionBundle:
    """Fit on 2016-25, walk team data through completed games, and save the snapshot."""
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    current = games[games["season"] == season]
    if current.empty:
        raise ProductionError(f"No completed {season} games are available yet. Run `cfbsim update-season`.")

    elo, ratings = run_elo(games, EloParams(**settings["elo_params"]))
    with_elo = games.merge(elo, on="game_id", validate="one_to_one")
    score_params = FeatureParams(**settings["score_feature_params"])
    win_params = FeatureParams(**settings["win_feature_params"])

    frame = scored_frame(games, elo, stats, score_params, regulation)
    train = frame[frame["season"].isin(PRODUCTION_TRAINING_SEASONS)]
    score_model, simulator = fit_production_stack(train, settings["alpha"], settings["bandwidth"])

    win_frame = scored_frame(games, elo, stats, win_params, regulation)
    win_model = ProductionWinModel(ALL_FEATURES, settings["win_C"]).fit(
        win_frame[win_frame["season"].isin(PRODUCTION_TRAINING_SEASONS)]
    )
    _, score_state = walk_forward(with_elo, stats, score_params)
    _, win_state = walk_forward(with_elo, stats, win_params)

    last_game = current.loc[current["start_date"].idxmax()]
    week_label = f"week {last_game['week']}" if last_game["season_type"] == "regular" else "the postseason"
    teams, conferences = latest_teams(games, season)
    return PredictionBundle(
        kind=PRODUCTION,
        title=f"{season} season",
        season=season,
        training_seasons=list(PRODUCTION_TRAINING_SEASONS),
        data_through=eastern_date(current["start_date"].max()),
        created_on=today or date.today(),
        teams=teams,
        conferences=conferences,
        records=team_records(games, season),
        ratings=ratings,
        score_state=score_state,
        win_state=win_state,
        score_model=score_model,
        win_model=win_model,
        simulator=simulator,
        profiles=team_profiles(score_state, ratings, season, teams),
        data_note=f"{len(current):,} completed {season} games through {week_label}",
        completed_matchups=_completed_matchups(games, season),
        upcoming=_upcoming(
            [record for record in records if record.get("season") == season], now
        ),
        live_scorecard=_live_scorecard(frame, score_model, simulator, season),
    )


def public_copy(bundle: PredictionBundle) -> PredictionBundle:
    """A copy for a public repo, without this season's game scores or schedule.

    CFBD's terms allow publishing derived outputs but not raw data as a dataset. Team
    pairs and dates stay, so the app can still warn when a matchup is a rematch.
    """
    return replace(
        bundle,
        completed_matchups=[(first, second, played_on, "") for first, second, played_on, _ in bundle.completed_matchups],
        upcoming=[],
        # FPI is ESPN's data, and reports name individual players, so neither is published.
        adjustments=None,
        # These are built from this season's raw scores, schedule, and box scores.
        board=[],
        game_logs={},
        season_stats=pd.DataFrame(),
    )


def build_adjustments(
    client: CFBDClient, games: pd.DataFrame, records: list[dict[str, Any]], season: int, data_through: date, today: date
) -> tuple[AdjustmentInputs | None, list[str]]:
    """Gather FPI and official reports. Problems are returned as messages, never papered over."""
    if not IMPACT_PATH.exists():
        return None, ["Player impact hasn't been estimated, so adjustments are off. Run `cfbsim fit-availability`."]
    impact = json.loads(IMPACT_PATH.read_text())["points_per_100_lost_yards"]
    fpi, fpi_as_of = latest_fpi_snapshot(season)
    box: list[dict[str, Any]] = []
    for season_type in SEASON_TYPES:
        typed = [r for r in records if r.get("season") == season and r["seasonType"] == season_type and r.get("completed")]
        for week in sorted({record["week"] for record in typed}):
            params = {"year": season, "week": week, "seasonType": season_type, "classification": CLASSIFICATION}
            box += client.get("games/players", **params)
    values = current_player_values(games, parse_player_box(box), season)
    reports = active_reports(load_reports(), data_through, today)
    known = set(latest_teams(games, season)[0])
    messages = [f"Report names an unknown team: {team}" for team in reports if team not in known]
    if not fpi:
        messages.append("No FPI snapshot found. Run `cfbsim update-season`.")
    inputs = AdjustmentInputs(fpi, fpi_as_of, FPI_WEIGHT, reports, values, impact)
    inputs.fpi_details = latest_fpi_details(season)
    return inputs, messages


def public_adjustment_note() -> str:
    return "This public copy leaves out ESPN FPI values and player reports, so it shows the tested model alone."


def log_upcoming_forecasts(bundle: PredictionBundle, days: int = 7, today: date | None = None) -> Any:
    """Save base and adjusted forecasts for the coming week, so the layer can be scored later."""
    today = today or date.today()
    entries = [
        {
            "kickoff": row["kickoff_time"],
            "home": row["home"],
            "away": row["away"],
            "neutral_site": row["neutral_site"],
            "base_home_win": row["tested_home_win"],
            "adjusted_home_win": row["home_win"],
            "stats_model_home_win": row["stats_model_home_win"],
            "base_margin": row["tested_margin"],
            "adjusted_margin": row["margin"],
            "fpi_used": row["fpi_used"],
            "fpi_as_of": row["fpi_as_of"],
            "players_listed": row["players_listed"],
        }
        for row in bundle.board
        if (row["kickoff"] - today).days <= days
    ]
    return write_forecast_log(entries)


def build_production_bundle(client: CFBDClient, season: int = CURRENT_SEASON) -> PredictionBundle:
    settings = frozen_settings()
    seasons = [*HISTORICAL_SEASONS, season]
    games, _ = load_games(client, seasons)
    stats = load_stats(client, games, seasons)
    records = [record for year in seasons for record in fetch_season_records(client, year)]
    regulation, _ = parse_regulation(records)
    bundle = assemble_production(games, stats, regulation, records, settings, season)
    inputs, messages = build_adjustments(client, games, records, season, bundle.data_through, date.today())
    bundle.adjustments = inputs
    if inputs is not None:
        inputs.notes = messages
    bundle.adjustment_messages = messages

    # Imported here because the board uses the prediction service, which imports bundles.
    from cfbsim.season_board import build_board, season_stat_table, team_details, team_game_logs

    bundle.team_info = team_details(client.get("teams/fbs", year=season))
    bundle.game_logs = team_game_logs(games, season)
    box_records: list[dict[str, Any]] = []
    for season_type in SEASON_TYPES:
        typed = [r for r in records if r.get("season") == season and r["seasonType"] == season_type and r.get("completed")]
        for week in sorted({record["week"] for record in typed}):
            box_records += client.get(
                "games/teams", year=season, week=week, seasonType=season_type, classification=CLASSIFICATION
            )
    season_ids = set(games.loc[games["season"] == season, "game_id"])
    advanced = stats["advanced"][stats["advanced"]["game_id"].isin(season_ids)]
    bundle.season_stats = season_stat_table(box_records, advanced, list(bundle.teams))
    bundle.board = build_board(bundle, date.today())
    return bundle
