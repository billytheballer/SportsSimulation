"""Season-level content for the app: team details, box-score stats, and the schedule board.

Everything here is built before the app runs, so pages load instantly. Board numbers come
from the same prediction service and random seed as the Game Center, so the two always agree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from cfbsim.bundle import PredictionBundle, eastern_date
from cfbsim.predictor import Predictor


@dataclass(frozen=True)
class SeasonStat:
    key: str
    label: str
    group: str
    higher_is_better: bool
    decimals: int


SEASON_STATS = [
    SeasonStat("points", "Points per game", "Offense", True, 1),
    SeasonStat("total_yards", "Total yards per game", "Offense", True, 1),
    SeasonStat("yards_per_play", "Yards per play", "Offense", True, 2),
    SeasonStat("pass_yards", "Passing yards per game", "Offense", True, 1),
    SeasonStat("rush_yards", "Rushing yards per game", "Offense", True, 1),
    SeasonStat("first_downs", "First downs per game", "Offense", True, 1),
    SeasonStat("third_down_pct", "Third-down conversion rate", "Offense", True, 1),
    SeasonStat("success", "Success rate", "Offense", True, 3),
    SeasonStat("ppa", "EPA per play", "Offense", True, 3),
    SeasonStat("explosiveness", "Explosiveness", "Offense", True, 2),
    SeasonStat("turnovers", "Turnovers per game", "Offense", False, 2),
    SeasonStat("plays", "Plays per game", "Offense", True, 1),
    SeasonStat("possession_minutes", "Time of possession (minutes)", "Offense", True, 1),
    SeasonStat("points_allowed", "Points allowed per game", "Defense", False, 1),
    SeasonStat("yards_allowed", "Yards allowed per game", "Defense", False, 1),
    SeasonStat("yards_per_play_allowed", "Yards per play allowed", "Defense", False, 2),
    SeasonStat("pass_yards_allowed", "Passing yards allowed per game", "Defense", False, 1),
    SeasonStat("rush_yards_allowed", "Rushing yards allowed per game", "Defense", False, 1),
    SeasonStat("third_down_pct_allowed", "Third-down rate allowed", "Defense", False, 1),
    SeasonStat("success_allowed", "Success rate allowed", "Defense", False, 3),
    SeasonStat("ppa_allowed", "EPA per play allowed", "Defense", False, 3),
    SeasonStat("sacks", "Sacks per game", "Defense", True, 2),
    SeasonStat("tackles_for_loss", "Tackles for loss per game", "Defense", True, 2),
    SeasonStat("takeaways", "Takeaways per game", "Defense", True, 2),
    SeasonStat("penalty_yards", "Penalty yards per game", "Discipline", False, 1),
]


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _split(value: Any, separator: str = "-") -> tuple[float, float] | None:
    parts = str(value).split(separator)
    if len(parts) != 2:
        return None
    first, second = _number(parts[0]), _number(parts[1])
    return None if first is None or second is None else (first, second)


def _team_line(team: dict[str, Any]) -> dict[str, float | None]:
    stats = {item["category"]: item["stat"] for item in team.get("stats", [])}
    third = _split(stats.get("thirdDownEff"))
    attempts = _split(stats.get("completionAttempts"))
    penalties = _split(stats.get("totalPenaltiesYards"))
    possession = _split(stats.get("possessionTime"), ":")
    rushes = _number(stats.get("rushingAttempts"))
    return {
        "points": _number(team.get("points")),
        "total_yards": _number(stats.get("totalYards")),
        "plays": rushes + attempts[1] if rushes is not None and attempts else None,
        "pass_yards": _number(stats.get("netPassingYards")),
        "rush_yards": _number(stats.get("rushingYards")),
        "first_downs": _number(stats.get("firstDowns")),
        "third_made": third[0] if third else None,
        "third_tries": third[1] if third else None,
        "turnovers": _number(stats.get("turnovers")),
        "penalty_yards": penalties[1] if penalties else None,
        "possession_minutes": possession[0] + possession[1] / 60 if possession else None,
        "sacks": _number(stats.get("sacks")),
        "tackles_for_loss": _number(stats.get("tacklesForLoss")),
    }


def season_stat_table(box_records: list[dict[str, Any]], advanced: pd.DataFrame, teams: list[str]) -> pd.DataFrame:
    """Actual per-game box-score averages this season, with ranks where 1 is always best.

    Unlike the model's inputs, these are not adjusted for opponent. A stat missing from
    a game is left out of that average rather than counted as zero.
    """
    rows = []
    for game in box_records:
        lines = game.get("teams", [])
        if len(lines) != 2:
            continue
        for own, other in ((lines[0], lines[1]), (lines[1], lines[0])):
            line = _team_line(own)
            allowed = _team_line(other)
            rows.append(
                {
                    "team": own.get("team"),
                    **line,
                    "points_allowed": allowed["points"],
                    "yards_allowed": allowed["total_yards"],
                    "plays_allowed": allowed["plays"],
                    "pass_yards_allowed": allowed["pass_yards"],
                    "rush_yards_allowed": allowed["rush_yards"],
                    "third_made_allowed": allowed["third_made"],
                    "third_tries_allowed": allowed["third_tries"],
                    "takeaways": allowed["turnovers"],
                }
            )
    if not rows:
        return pd.DataFrame()
    games = pd.DataFrame(rows).apply(lambda column: pd.to_numeric(column) if column.name != "team" else column)
    grouped = games.groupby("team")
    table = grouped.mean(numeric_only=True)
    totals = grouped.sum(numeric_only=True, min_count=1)
    table["games"] = grouped.size()
    table["yards_per_play"] = totals["total_yards"] / totals["plays"]
    table["yards_per_play_allowed"] = totals["yards_allowed"] / totals["plays_allowed"]
    table["third_down_pct"] = 100 * totals["third_made"] / totals["third_tries"]
    table["third_down_pct_allowed"] = 100 * totals["third_made_allowed"] / totals["third_tries_allowed"]

    if not advanced.empty:
        averages = advanced.groupby("team")[
            ["off_ppa", "off_success", "off_explosiveness", "def_ppa", "def_success"]
        ].mean()
        renamed = averages.rename(
            columns={
                "off_ppa": "ppa",
                "off_success": "success",
                "off_explosiveness": "explosiveness",
                "def_ppa": "ppa_allowed",
                "def_success": "success_allowed",
            }
        )
        table = table.join(renamed, how="left")

    table = table[table.index.isin(teams)].copy()
    for stat in SEASON_STATS:
        if stat.key in table:
            table[f"{stat.key}_rank"] = table[stat.key].rank(ascending=not stat.higher_is_better, method="min")
    return table


def team_details(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Display details for each team. Colors and abbreviations are used instead of logos."""
    details = {}
    for team in records:
        location = team.get("location") or {}
        details[team["school"]] = {
            "abbreviation": team.get("abbreviation") or team["school"][:4].upper(),
            "mascot": team.get("mascot") or "",
            "color": team.get("color"),
            "alternate_color": team.get("alternateColor"),
            "stadium": location.get("name") or "",
            "city": location.get("city") or "",
            "state": location.get("state") or "",
        }
    return details


def team_game_logs(games: pd.DataFrame, season: int) -> dict[str, list[dict[str, Any]]]:
    """Each team's completed games this season, oldest first."""
    logs: dict[str, list[dict[str, Any]]] = {}
    current = games[games["season"] == season]
    columns = ["start_date", "week", "home_team", "away_team", "home_points", "away_points", "neutral_site"]
    for start, week, home, away, home_points, away_points, neutral in zip(*(current[column] for column in columns)):
        played_on = eastern_date(start)
        for team, opponent, scored, allowed, site in (
            (home, away, home_points, away_points, "vs"),
            (away, home, away_points, home_points, "vs" if neutral else "at"),
        ):
            logs.setdefault(team, []).append(
                {
                    "date": played_on,
                    "week": int(week),
                    "opponent": opponent,
                    "site": site,
                    "neutral": bool(neutral),
                    "result": "W" if scored > allowed else "L",
                    "points_for": int(scored),
                    "points_against": int(allowed),
                }
            )
    return logs


def confidence_label(probability: float) -> str:
    """Plain words for how strong a favorite is, used across the app."""
    top = max(probability, 1 - probability)
    if top < 0.55:
        return "Toss-up"
    if top < 0.65:
        return "Lean"
    if top < 0.80:
        return "Solid"
    return "Strong"


def build_board(bundle: PredictionBundle, today: Any = None) -> list[dict[str, Any]]:
    """A prediction for every remaining FBS game on the schedule."""
    predictor = Predictor(bundle)
    names = set(predictor.team_names())
    rows = []
    for game in bundle.upcoming:
        if game["home"] not in names or game["away"] not in names:
            continue
        result = predictor.predict(
            game["home"], game["away"], game["neutral_site"], today=today, game_date=game["kickoff"]
        )
        summary = result.summary
        tested = result.base_summary or summary
        adjustment = result.adjustment or {}
        rows.append(
            {
                **game,
                "home_conference": bundle.conferences.get(game["home"], ""),
                "away_conference": bundle.conferences.get(game["away"], ""),
                "home_win": summary["home_win_probability"],
                "away_win": summary["away_win_probability"],
                "home_points": summary["home"]["mean"],
                "away_points": summary["away"]["mean"],
                "margin": summary["margin"]["mean"],
                "total": summary["total"]["mean"],
                "margin_80": summary["margin"]["intervals"]["80%"],
                "tested_home_win": tested["home_win_probability"],
                "tested_margin": tested["margin"]["mean"],
                "stats_model_home_win": result.stats_model_home_probability,
                "fpi_used": adjustment.get("fpi_used", False),
                "fpi_as_of": adjustment.get("fpi_as_of"),
                "players_listed": len(adjustment.get("players", [])),
            }
        )
    return rows


def projected_records(board: list[dict[str, Any]], records: dict[str, str]) -> dict[str, dict[str, float]]:
    """Expected wins in the remaining FBS games on the schedule, added to the current record."""
    remaining: dict[str, list[float]] = {}
    for row in board:
        remaining.setdefault(row["home"], []).append(row["home_win"])
        remaining.setdefault(row["away"], []).append(row["away_win"])
    projected = {}
    for team, record in records.items():
        wins, losses = (int(part) for part in record.split("-"))
        chances = remaining.get(team, [])
        expected = sum(chances)
        projected[team] = {
            "wins": wins,
            "losses": losses,
            "games_left": len(chances),
            "expected_wins_left": expected,
            "projected_wins": wins + expected,
            "projected_losses": losses + len(chances) - expected,
        }
    return projected
