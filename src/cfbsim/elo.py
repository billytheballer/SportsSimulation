"""Walk-forward Elo ratings.

Each game is predicted from ratings as they stood before kickoff, and only then
does its result update those ratings. That order is what keeps this leak-free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

ELO_SCALE = 400.0
FBS_INITIAL = 1500.0


@dataclass(frozen=True)
class EloParams:
    k: float = 25.0
    home_advantage: float = 65.0
    # Share of a team's distance from its baseline kept over each offseason.
    season_carryover: float = 0.7
    # Starting point for FCS and lower teams, which are far weaker on average.
    lower_division_initial: float = 1200.0


def win_probability(rating_diff: float) -> float:
    """Chance of winning for a side with a `rating_diff` edge in Elo points."""
    return 1.0 / (1.0 + 10.0 ** (-rating_diff / ELO_SCALE))


def _margin_multiplier(margin: float, winner_diff: float) -> float:
    """Reward bigger wins, less so for favorites, who are expected to win big."""
    return math.log(abs(margin) + 1.0) * 2.2 / (max(winner_diff, -1000.0) * 0.001 + 2.2)


def run_elo(games: pd.DataFrame, params: EloParams) -> tuple[pd.DataFrame, dict[int, float]]:
    """Return one pre-game row per game, plus every team's latest rating.

    `games` must be in kickoff order, as produced by `parse_games`.
    """
    if not games["start_date"].is_monotonic_increasing:
        raise ValueError("Games must be sorted by kickoff; otherwise ratings would use the future.")

    ratings: dict[int, float] = {}
    last_season: dict[int, int] = {}

    def pregame_rating(team: int, classification: object, season: int) -> float:
        baseline = FBS_INITIAL if classification == "fbs" else params.lower_division_initial
        if team not in ratings:
            ratings[team] = baseline
        elif last_season[team] != season:
            kept = params.season_carryover ** (season - last_season[team])
            ratings[team] = baseline + kept * (ratings[team] - baseline)
        last_season[team] = season
        return ratings[team]

    rows = []
    columns = ["game_id", "season", "home_id", "away_id", "home_classification",
               "away_classification", "neutral_site", "margin"]
    for game_id, season, home, away, home_class, away_class, neutral, margin in zip(
        *(games[column].tolist() for column in columns)
    ):
        home_rating = pregame_rating(home, home_class, season)
        away_rating = pregame_rating(away, away_class, season)
        site_edge = 0.0 if neutral else params.home_advantage
        diff = home_rating - away_rating + site_edge
        expected = win_probability(diff)
        rows.append((game_id, home_rating, away_rating, home_rating - away_rating, expected))

        home_won = margin > 0
        winner_diff = diff if home_won else -diff
        shift = params.k * _margin_multiplier(margin, winner_diff) * (float(home_won) - expected)
        ratings[home] = home_rating + shift
        ratings[away] = away_rating - shift

    predictions = pd.DataFrame(
        rows, columns=["game_id", "elo_home", "elo_away", "elo_diff", "p_elo"]
    )
    return predictions, ratings
