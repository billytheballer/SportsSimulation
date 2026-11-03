"""Per-game team stats, betting lines, and preseason team context from CFBD.

Every value here describes a finished game or a past offseason. Nothing in this
module is safe to use for a game's own forecast until `features.py` has lagged it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cfbsim.client import CFBDClient
from cfbsim.games import SEASON_TYPES

# Recruiting classes are averaged over four years, so earlier classes are needed.
RECRUITING_LOOKBACK = 3


def fetch_plan(games: pd.DataFrame, seasons: list[int]) -> list[tuple[str, dict[str, Any]]]:
    """List every Stage 2 request. Box-score weeks come from games already cached."""
    plan: list[tuple[str, dict[str, Any]]] = []
    for season in seasons:
        for season_type in SEASON_TYPES:
            plan.append(("stats/game/advanced", {"year": season, "seasonType": season_type}))
            plan.append(("lines", {"year": season, "seasonType": season_type}))
            weeks = games.loc[
                (games["season"] == season) & (games["season_type"] == season_type), "week"
            ].unique()
            for week in sorted(int(value) for value in weeks):
                plan.append(
                    ("games/teams",
                     {"year": season, "week": week, "seasonType": season_type, "classification": "fbs"})
                )
        plan.append(("player/returning", {"year": season}))
    first = min(seasons) - RECRUITING_LOOKBACK
    plan += [("recruiting/teams", {"year": year}) for year in range(first, max(seasons) + 1)]
    return plan


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _attempts(completions_attempts: Any) -> float:
    """Pass attempts from a box-score string such as '12-20'."""
    try:
        return float(str(completions_attempts).split("-")[1])
    except (IndexError, ValueError):
        return np.nan


def parse_box_scores(records: list[dict[str, Any]]) -> pd.DataFrame:
    """One row per team per game. Absent stats stay missing rather than becoming zero."""
    rows = []
    for game in records:
        for team in game["teams"]:
            stats = {item["category"]: item["stat"] for item in team["stats"]}
            rush_attempts = _number(stats.get("rushingAttempts"))
            pass_attempts = _attempts(stats.get("completionAttempts"))
            rows.append(
                {
                    "game_id": game["id"],
                    "team_id": team["teamId"],
                    "points": _number(team.get("points")),
                    "total_yards": _number(stats.get("totalYards")),
                    "plays": rush_attempts + pass_attempts,
                    "pass_yards": _number(stats.get("netPassingYards")),
                    "pass_attempts": pass_attempts,
                    "rush_yards": _number(stats.get("rushingYards")),
                    "rush_attempts": rush_attempts,
                    "turnovers": _number(stats.get("turnovers")),
                }
            )
    return pd.DataFrame(rows)


def parse_advanced(records: list[dict[str, Any]]) -> pd.DataFrame:
    """One row per team per game, keyed by team name as the endpoint provides."""
    rows = []
    for game in records:
        row = {"game_id": game["gameId"], "team": game["team"]}
        for side in ("offense", "defense"):
            block = game.get(side) or {}
            prefix = "off" if side == "offense" else "def"
            row[f"{prefix}_ppa"] = block.get("ppa")
            row[f"{prefix}_success"] = block.get("successRate")
            row[f"{prefix}_explosiveness"] = block.get("explosiveness")
            row[f"{prefix}_pass_ppa"] = (block.get("passingPlays") or {}).get("ppa")
            row[f"{prefix}_rush_ppa"] = (block.get("rushingPlays") or {}).get("ppa")
        rows.append(row)
    return pd.DataFrame(rows)


def _pick_line(lines: list[dict[str, Any]], field: str) -> float:
    values = {line["provider"]: line[field] for line in lines if line.get(field) is not None}
    if not values:
        return np.nan
    consensus = values.get("consensus")
    return float(consensus) if consensus is not None else float(np.median(list(values.values())))


def parse_lines(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Listed home spread and total per game: consensus if offered, else the provider median.

    These are benchmarks only, and never model inputs. CFBD doesn't say when in the week
    a listed line was captured, so don't describe them as closing lines. CFBD's `spread`
    is from the home side, so -3 means the home team is favored by 3.
    """
    rows = []
    for game in records:
        lines = game.get("lines") or []
        spread = _pick_line(lines, "spread")
        if np.isnan(spread):
            continue
        has_consensus = any(
            line["provider"] == "consensus" and line.get("spread") is not None for line in lines
        )
        rows.append(
            {
                "game_id": game["id"],
                "home_spread": spread,
                "over_under": _pick_line(lines, "overUnder"),
                "line_source": "consensus" if has_consensus else "median",
            }
        )
    return pd.DataFrame(rows, columns=["game_id", "home_spread", "over_under", "line_source"])


def _collect(client: CFBDClient, plan: list[tuple[str, dict[str, Any]]], endpoint: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for name, params in plan:
        if name == endpoint:
            records += client.get(name, **params)
    return records


def load_stats(client: CFBDClient, games: pd.DataFrame, seasons: list[int]) -> dict[str, pd.DataFrame]:
    plan = fetch_plan(games, seasons)
    box = parse_box_scores(_collect(client, plan, "games/teams"))
    advanced = parse_advanced(_collect(client, plan, "stats/game/advanced"))
    lines = parse_lines(_collect(client, plan, "lines"))
    recruiting = pd.DataFrame.from_records(_collect(client, plan, "recruiting/teams"))
    returning = pd.DataFrame.from_records(_collect(client, plan, "player/returning"))
    return {
        "box": box.drop_duplicates(["game_id", "team_id"]),
        "advanced": advanced.drop_duplicates(["game_id", "team"]),
        "lines": lines.drop_duplicates("game_id"),
        "recruiting": recruiting,
        "returning": returning,
    }
