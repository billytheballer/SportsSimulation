"""A saved snapshot of fitted models and team data that the app can load quickly.

A bundle holds derived values only: model coefficients, running team averages, and
anonymous score pairs. It needs no API key. Every bundle records what kind it is and
how old its data is, so the interface can't present old data as a current forecast.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from cfbsim.config import DEVELOPMENT_SEASONS, PROJECT_ROOT, REPORTS_DIR, TUNE_SEASONS
from cfbsim.elo import EloParams, run_elo
from cfbsim.features import ALL_FEATURES, METRICS, FeatureParams, WalkState, walk_forward
from cfbsim.games import fetch_season_records, load_games, parse_regulation
from cfbsim.model import WinModel
from cfbsim.simulate import PairSimulator
from cfbsim.scoring import ScoreModel
from cfbsim.stage3 import fit_stack, scored_frame
from cfbsim.stats import load_stats

BUNDLES_DIR = PROJECT_ROOT / "bundles"
DEMO_BUNDLE = BUNDLES_DIR / "demo_2022.joblib"
PRODUCTION_BUNDLE = BUNDLES_DIR / "production_2026.joblib"
BUNDLE_FORMAT = 3
HISTORICAL = "historical_demo"
PRODUCTION = "production"


@dataclass(frozen=True)
class ComparisonMetric:
    """One row of the team comparison chart."""

    key: str
    label: str
    group: str
    higher_is_better: bool
    decimals: int


_side = {"off": "Offense", "def": "Defense"}
COMPARISON_METRICS = [
    ComparisonMetric("elo", "Elo rating", "Overall", True, 0),
    ComparisonMetric("recent_margin", "Recent scoring margin", "Overall", True, 1),
    ComparisonMetric("schedule_strength", "Schedule strength (opponent Elo)", "Overall", True, 0),
    ComparisonMetric("recruiting", "Recruiting (4-year average points)", "Overall", True, 0),
    ComparisonMetric("returning", "Returning production (share of usage)", "Overall", True, 2),
    ComparisonMetric("off_points", "Points scored", _side["off"], True, 1),
    ComparisonMetric("off_yards_per_play", "Yards per play", _side["off"], True, 2),
    ComparisonMetric("off_pass_yards_per_attempt", "Pass yards per attempt", _side["off"], True, 2),
    ComparisonMetric("off_rush_yards_per_attempt", "Rush yards per attempt", _side["off"], True, 2),
    ComparisonMetric("off_success", "Success rate", _side["off"], True, 3),
    ComparisonMetric("off_explosiveness", "Explosiveness", _side["off"], True, 2),
    ComparisonMetric("off_turnovers", "Turnovers committed", _side["off"], False, 2),
    ComparisonMetric("off_plays", "Plays per game", _side["off"], True, 1),
    ComparisonMetric("def_points", "Points allowed", _side["def"], False, 1),
    ComparisonMetric("def_yards_per_play", "Yards per play allowed", _side["def"], False, 2),
    ComparisonMetric("def_pass_yards_per_attempt", "Pass yards per attempt allowed", _side["def"], False, 2),
    ComparisonMetric("def_rush_yards_per_attempt", "Rush yards per attempt allowed", _side["def"], False, 2),
    ComparisonMetric("def_success", "Success rate allowed", _side["def"], False, 3),
    ComparisonMetric("def_explosiveness", "Explosiveness allowed", _side["def"], False, 2),
    ComparisonMetric("def_turnovers", "Turnovers forced", _side["def"], True, 2),
]


@dataclass
class PredictionBundle:
    kind: str
    title: str
    # The season whose team averages this bundle holds.
    season: int
    training_seasons: list[int]
    # Calendar date of the last completed game included (Eastern time).
    data_through: date
    created_on: date
    teams: dict[str, int]
    conferences: dict[str, str]
    records: dict[str, str]
    ratings: dict[int, float]
    score_state: WalkState
    win_state: WalkState
    score_model: ScoreModel
    win_model: WinModel
    simulator: PairSimulator
    profiles: pd.DataFrame
    # Details such as the last completed week, shown beside the data date.
    data_note: str = ""
    # Games already played this season, as (team, team, date, score text).
    completed_matchups: list[tuple[str, str, date, str]] = field(default_factory=list)
    # Upcoming games, for quick selection in the app.
    upcoming: list[dict[str, Any]] = field(default_factory=list)
    # Results scored on games played after training ended, if any.
    live_scorecard: dict[str, Any] = field(default_factory=dict)
    # The FPI and availability layer. None means predictions use the base model alone.
    adjustments: Any = None
    adjustment_messages: list[str] = field(default_factory=list)
    # App content built ahead of time: team details, box-score averages, game logs, predictions.
    team_info: dict[str, dict[str, Any]] = field(default_factory=dict)
    season_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    game_logs: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    board: list[dict[str, Any]] = field(default_factory=list)
    format_version: int = BUNDLE_FORMAT


def eastern_date(moment: pd.Timestamp) -> date:
    return moment.tz_convert("America/New_York").date()


def latest_teams(games: pd.DataFrame, season: int) -> tuple[dict[str, int], dict[str, str]]:
    """FBS teams that appear in a season, by name, with their conference."""
    latest = games[games["season"] == season]
    teams: dict[str, int] = {}
    conferences: dict[str, str] = {}
    for side in ("home", "away"):
        fbs = latest[latest[f"{side}_classification"] == "fbs"]
        teams.update(zip(fbs[f"{side}_team"], fbs[f"{side}_id"]))
        conferences.update(zip(fbs[f"{side}_team"], fbs[f"{side}_conference"]))
    return teams, {team: str(name) for team, name in conferences.items()}


def team_records(games: pd.DataFrame, season: int) -> dict[str, str]:
    """Wins and losses so far in a season, from completed games."""
    latest = games[games["season"] == season]
    wins: dict[str, int] = {}
    losses: dict[str, int] = {}
    for home, away, home_win in zip(latest["home_team"], latest["away_team"], latest["home_win"]):
        winner, loser = (home, away) if home_win else (away, home)
        wins[winner] = wins.get(winner, 0) + 1
        losses[loser] = losses.get(loser, 0) + 1
    return {team: f"{wins.get(team, 0)}-{losses.get(team, 0)}" for team in {*wins, *losses}}


def team_profiles(
    state: WalkState, ratings: dict[int, float], season: int, teams: dict[str, int]
) -> pd.DataFrame:
    """Each team's current averages, with percentile ranks where 100 is always best."""
    rows = []
    for name, team_id in teams.items():
        team = state.teams.get(team_id)
        if team is None:
            continue
        row: dict[str, Any] = {"team": name, "elo": ratings.get(team_id, np.nan)}
        offense, defense = team.offense.read(), team.defense.read()
        for index, metric in enumerate(METRICS):
            row[f"off_{metric}"] = offense[index]
            row[f"def_{metric}"] = defense[index]
        row["recent_margin"] = team.recent_margin.read()[0]
        row["schedule_strength"] = team.schedule.read()[0]
        row["recruiting"] = state.recruiting.get((season, name), np.nan)
        row["returning"] = state.returning.get((season, name), np.nan)
        rows.append(row)
    profiles = pd.DataFrame(rows).set_index("team")
    for metric in COMPARISON_METRICS:
        ranks = profiles[metric.key].rank(pct=True, ascending=metric.higher_is_better)
        profiles[f"{metric.key}_percentile"] = 100 * ranks
    return profiles


def save_bundle(bundle: PredictionBundle, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    joblib.dump(bundle, temporary, compress=3)
    temporary.replace(path)
    return path


def load_bundle(path: Path) -> PredictionBundle:
    bundle = joblib.load(path)
    if bundle.format_version != BUNDLE_FORMAT:
        raise ValueError(f"{path.name} uses an old format. Rebuild it with `cfbsim build-demo`.")
    return bundle


def stage3_settings() -> dict[str, Any]:
    """The settings chosen in Stage 3. Read only; this never changes the report."""
    path = REPORTS_DIR / "stage3_diagnostics.json"
    if not path.exists():
        raise FileNotFoundError("Run `cfbsim stage3` first to choose the model settings.")
    report = json.loads(path.read_text())
    selection = report["selection"]
    return {
        "elo_params": report["elo_params"],
        "score_feature_params": selection["feature_params"],
        "alpha": selection["alpha"],
        "bandwidth": selection["bandwidth"],
        "win_feature_params": report["win_model_settings"]["feature_params"],
        "win_C": report["win_model_settings"]["C"],
    }


def build_demo_bundle(client: Any) -> PredictionBundle:
    """The Stage 3 model with team data through 2022. Never reads 2023 or later."""
    settings = stage3_settings()
    seasons = list(DEVELOPMENT_SEASONS)
    games, _ = load_games(client, seasons)
    stats = load_stats(client, games, seasons)
    records = [record for season in seasons for record in fetch_season_records(client, season)]
    regulation, _ = parse_regulation(records)

    elo, ratings = run_elo(games, EloParams(**settings["elo_params"]))
    with_elo = games.merge(elo, on="game_id", validate="one_to_one")
    score_params = FeatureParams(**settings["score_feature_params"])
    win_params = FeatureParams(**settings["win_feature_params"])
    frame = scored_frame(games, elo, stats, score_params, regulation)
    stack = fit_stack(frame[frame["season"].isin(TUNE_SEASONS)], settings["alpha"], settings["bandwidth"])
    win_frame = scored_frame(games, elo, stats, win_params, regulation)
    win_model = WinModel(ALL_FEATURES, settings["win_C"]).fit(win_frame[win_frame["season"].isin(TUNE_SEASONS)])
    _, score_state = walk_forward(with_elo, stats, score_params)
    _, win_state = walk_forward(with_elo, stats, win_params)

    season = max(seasons)
    teams, conferences = latest_teams(games, season)
    return PredictionBundle(
        kind=HISTORICAL,
        title=f"Historical demonstration: teams as of the end of the {season} season",
        season=season,
        training_seasons=list(TUNE_SEASONS),
        data_through=eastern_date(games["start_date"].max()),
        created_on=date.today(),
        teams=teams,
        conferences=conferences,
        records=team_records(games, season),
        ratings=ratings,
        score_state=score_state,
        win_state=win_state,
        score_model=stack.model,
        win_model=win_model,
        simulator=stack.simulator,
        profiles=team_profiles(score_state, ratings, season, teams),
        data_note=f"End of the {season} season, including bowl games",
    )
