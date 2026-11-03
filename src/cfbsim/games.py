"""Turn raw /games responses into a clean, leak-resistant game table.

Columns are kept by allowlist, so a new post-game field in the API can't reach a
model by accident. Bad rows are dropped and counted, never repaired with guesses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from cfbsim.client import CFBDClient

# Known before kickoff.
PREGAME_FIELDS = {
    "id": "game_id",
    "season": "season",
    "week": "week",
    "seasonType": "season_type",
    "startDate": "start_date",
    "neutralSite": "neutral_site",
    "conferenceGame": "conference_game",
    "homeId": "home_id",
    "homeTeam": "home_team",
    "homeConference": "home_conference",
    "homeClassification": "home_classification",
    "awayId": "away_id",
    "awayTeam": "away_team",
    "awayConference": "away_conference",
    "awayClassification": "away_classification",
}
# CFBD's own pre-game Elo. Used only as a benchmark to compare against, never as a feature.
BENCHMARK_FIELDS = {"homePregameElo": "cfbd_home_elo", "awayPregameElo": "cfbd_away_elo"}
# The outcome: training labels and scoring only.
RESULT_FIELDS = {"completed": "completed", "homePoints": "home_points", "awayPoints": "away_points"}

ALL_FIELDS = {**PREGAME_FIELDS, **BENCHMARK_FIELDS, **RESULT_FIELDS}
OPTIONAL_FIELDS = {"homeClassification", "awayClassification", "homeConference", "awayConference",
                   "conferenceGame", "homePregameElo", "awayPregameElo", "homePoints", "awayPoints"}
SEASON_TYPES = ("regular", "postseason")


@dataclass
class DataQualityReport:
    """Everything removed while cleaning, so nothing disappears silently."""

    rows_read: int = 0
    games_kept: int = 0
    dropped: dict[str, int] = field(default_factory=dict)

    def record(self, reason: str, count: int) -> None:
        if count:
            self.dropped[reason] = self.dropped.get(reason, 0) + int(count)


def parse_games(records: list[dict[str, Any]]) -> tuple[pd.DataFrame, DataQualityReport]:
    """Return completed games involving an FBS team, in kickoff order."""
    report = DataQualityReport(rows_read=len(records))
    columns = list(ALL_FIELDS.values())
    if not records:
        return pd.DataFrame(columns=columns + ["margin", "home_win"]), report

    raw = pd.DataFrame.from_records(records)
    missing = (set(ALL_FIELDS) - OPTIONAL_FIELDS) - set(raw.columns)
    if missing:
        raise ValueError(f"/games response is missing required fields: {sorted(missing)}")
    games = raw.reindex(columns=list(ALL_FIELDS)).rename(columns=ALL_FIELDS)

    duplicated = games.duplicated("game_id")
    report.record("duplicate game id", duplicated.sum())
    games = games[~duplicated]

    is_fbs = games["home_classification"].eq("fbs") | games["away_classification"].eq("fbs")
    report.record("no FBS team", (~is_fbs).sum())
    games = games[is_fbs]

    played = games["completed"].fillna(False).astype(bool)
    report.record("not completed", (~played).sum())
    games = games[played]

    no_score = games["home_points"].isna() | games["away_points"].isna()
    report.record("completed but missing a score", no_score.sum())
    games = games[~no_score]

    games = games.assign(start_date=pd.to_datetime(games["start_date"], utc=True, errors="coerce"))
    needed = ["start_date", "neutral_site", "home_id", "away_id"]
    incomplete = games[needed].isna().any(axis=1)
    report.record("missing kickoff, site, or team id", incomplete.sum())
    games = games[~incomplete]

    tied = games["home_points"] == games["away_points"]
    report.record("tied final score", tied.sum())
    games = games[~tied]

    games = games.assign(
        neutral_site=games["neutral_site"].astype(bool),
        margin=games["home_points"] - games["away_points"],
    )
    games["home_win"] = (games["margin"] > 0).astype(int)
    games = games.sort_values(["start_date", "game_id"], kind="stable").reset_index(drop=True)
    report.games_kept = len(games)
    return games, report


def parse_regulation(records: list[dict[str, Any]]) -> tuple[pd.DataFrame, DataQualityReport]:
    """Points after four quarters, and whether a game went to overtime.

    These come from line scores, which are post-game data. They are training targets
    for the score simulator and must never be used as forecast inputs.
    """
    report = DataQualityReport(rows_read=len(records))
    rows = []
    for game in records:
        home, away = game.get("homeLineScores"), game.get("awayLineScores")
        if not game.get("completed") or game.get("homePoints") is None or game.get("awayPoints") is None:
            continue
        if not home or not away or len(home) != len(away) or len(home) < 4 or any(
            value is None for value in [*home, *away]
        ):
            report.record("missing or partial line scores", 1)
            continue
        if sum(home) != game["homePoints"] or sum(away) != game["awayPoints"]:
            report.record("line scores do not add up to the final", 1)
            continue
        home_regulation, away_regulation = sum(home[:4]), sum(away[:4])
        overtime = len(home) > 4
        if overtime != (home_regulation == away_regulation):
            report.record("overtime does not match a regulation tie", 1)
            continue
        rows.append(
            {
                "game_id": game["id"],
                "home_regulation": home_regulation,
                "away_regulation": away_regulation,
                "overtime": overtime,
            }
        )
    report.games_kept = len(rows)
    return pd.DataFrame(rows, columns=["game_id", "home_regulation", "away_regulation", "overtime"]), report


def fetch_season_records(
    client: CFBDClient, season: int, refresh: bool = False
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for season_type in SEASON_TYPES:
        records += client.get(
            "games", refresh=refresh, year=season, seasonType=season_type, classification="fbs"
        )
    return records


def load_games(
    client: CFBDClient, seasons: tuple[int, ...] | list[int]
) -> tuple[pd.DataFrame, DataQualityReport]:
    records: list[dict[str, Any]] = []
    for season in seasons:
        records += fetch_season_records(client, season)
    return parse_games(records)
