"""Walk-forward team features: one row per game, built only from earlier games.

Like Elo, each game's features are read before that game's stats enter any team's
running averages. Stats are opponent-adjusted using the opponent's averages as they
stood before kickoff, so a big game against a weak defense counts for less.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

METRICS = (
    "points",
    "yards_per_play",
    "pass_yards_per_attempt",
    "rush_yards_per_attempt",
    "turnovers",
    "ppa",
    "pass_ppa",
    "rush_ppa",
    "success",
    "explosiveness",
    "plays",
)
RECENT_HALFLIFE = 2.0
RECRUITING_CLASSES = 4


@dataclass(frozen=True)
class FeatureParams:
    # Games until an old game counts half as much.
    halflife: float = 8.0
    # Extra discount on all earlier games each offseason, for roster turnover.
    season_weight: float = 0.5


def _side_columns(metric: str) -> list[str]:
    return [f"{side}_{metric}_diff" for side in ("off", "def")]


FEATURE_GROUPS = {
    "elo": ["elo_diff", "true_home"],
    "scoring": _side_columns("points"),
    "yardage": [column for metric in METRICS[1:4] for column in _side_columns(metric)],
    "turnovers": _side_columns("turnovers"),
    "efficiency": [column for metric in METRICS[5:10] for column in _side_columns(metric)],
    "schedule": ["schedule_strength_diff"],
    "recent_form": ["recent_margin_diff"],
    "recruiting": ["recruiting_diff"],
    "returning": ["returning_diff"],
}
ALL_FEATURES = [column for columns in FEATURE_GROUPS.values() for column in columns]


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return (numerator / denominator).where(denominator > 0)


def team_game_stats(
    games: pd.DataFrame, box: pd.DataFrame, advanced: pd.DataFrame
) -> dict[tuple[int, int], np.ndarray]:
    """Map (game, team) to that team's own stats for the game, ordered as METRICS."""
    names = pd.concat(
        [
            games[["game_id", "home_team", "home_id"]].set_axis(["game_id", "team", "team_id"], axis=1),
            games[["game_id", "away_team", "away_id"]].set_axis(["game_id", "team", "team_id"], axis=1),
        ]
    )
    advanced = advanced.merge(names, on=["game_id", "team"], how="inner")
    table = box.merge(advanced, on=["game_id", "team_id"], how="outer")
    table = table.assign(
        yards_per_play=_safe_ratio(table["total_yards"], table["plays"]),
        pass_yards_per_attempt=_safe_ratio(table["pass_yards"], table["pass_attempts"]),
        rush_yards_per_attempt=_safe_ratio(table["rush_yards"], table["rush_attempts"]),
        ppa=table["off_ppa"],
        pass_ppa=table["off_pass_ppa"],
        rush_ppa=table["off_rush_ppa"],
        success=table["off_success"],
        explosiveness=table["off_explosiveness"],
        plays=table["plays"].where(table["plays"] > 0),
    )
    values = table[list(METRICS)].to_numpy(dtype=float)
    keys = zip(table["game_id"].tolist(), table["team_id"].tolist())
    return {key: row for key, row in zip(keys, values)}


def recruiting_scores(recruiting: pd.DataFrame) -> dict[tuple[int, str], float]:
    """Average points of a team's latest four classes, all signed before the season starts."""
    points = dict(zip(zip(recruiting["year"], recruiting["team"]), recruiting["points"]))
    scores = {}
    for year in sorted(recruiting["year"].unique()):
        for team in recruiting.loc[recruiting["year"] == year, "team"].unique():
            classes = [points.get((year - back, team)) for back in range(RECRUITING_CLASSES)]
            known = [value for value in classes if value is not None]
            if len(known) == RECRUITING_CLASSES:
                scores[(int(year), team)] = float(np.mean(known))
    return scores


class _RunningMean:
    """Exponentially weighted mean per metric. Reads as missing until a value arrives."""

    def __init__(self, size: int, decay: float) -> None:
        self.mean = np.zeros(size)
        self.weight = np.zeros(size)
        self.decay = decay

    def read(self) -> np.ndarray:
        return np.where(self.weight > 0, self.mean, np.nan)

    def add(self, values: np.ndarray) -> None:
        seen = ~np.isnan(values)
        self.weight[seen] = self.decay * self.weight[seen] + 1.0
        self.mean[seen] += (values[seen] - self.mean[seen]) / self.weight[seen]

    def discount(self, factor: float) -> None:
        self.weight *= factor

    def clear(self) -> None:
        self.mean[:] = 0.0
        self.weight[:] = 0.0


class _TeamState:
    def __init__(self, decay: float, recent_decay: float) -> None:
        self.offense = _RunningMean(len(METRICS), decay)
        self.defense = _RunningMean(len(METRICS), decay)
        self.recent_margin = _RunningMean(1, recent_decay)
        self.schedule = _RunningMean(1, 1.0)
        self.season: int | None = None

    def start_season(self, season: int, season_weight: float) -> None:
        if self.season is not None and self.season != season:
            discount = season_weight ** (season - self.season)
            self.offense.discount(discount)
            self.defense.discount(discount)
            self.recent_margin.clear()
            self.schedule.clear()
        self.season = season


@dataclass
class WalkState:
    """Where every team's averages stood after the last game walked."""

    teams: dict[int, _TeamState]
    recruiting: dict[tuple[int, str], float]
    returning: dict[tuple[int, str], float]


def _read_row(
    state: WalkState, home: _TeamState, away: _TeamState, season: int, home_name: str, away_name: str
) -> dict[str, float]:
    """All pre-game feature values for a matchup, from averages as they stand now."""
    row: dict[str, float] = {}
    sides = (("home", home), ("away", away))
    for name, team in sides:
        offense, defense = team.offense.read(), team.defense.read()
        for index, metric in enumerate(METRICS):
            row[f"{name}_off_{metric}"] = offense[index]
            row[f"{name}_def_{metric}"] = defense[index]
        row[f"{name}_recent_margin"] = team.recent_margin.read()[0]
        row[f"{name}_schedule_strength"] = team.schedule.read()[0]
    row["home_recruiting"] = state.recruiting.get((season, home_name), np.nan)
    row["away_recruiting"] = state.recruiting.get((season, away_name), np.nan)
    row["home_returning"] = state.returning.get((season, home_name), np.nan)
    row["away_returning"] = state.returning.get((season, away_name), np.nan)

    for metric in METRICS:
        row[f"off_{metric}_diff"] = row[f"home_off_{metric}"] - row[f"away_off_{metric}"]
        row[f"def_{metric}_diff"] = row[f"home_def_{metric}"] - row[f"away_def_{metric}"]
    for column, source in (
        ("schedule_strength_diff", "schedule_strength"),
        ("recent_margin_diff", "recent_margin"),
        ("recruiting_diff", "recruiting"),
        ("returning_diff", "returning"),
    ):
        row[column] = row[f"home_{source}"] - row[f"away_{source}"]
    return row


def walk_forward(
    games: pd.DataFrame, stats: dict[str, pd.DataFrame], params: FeatureParams
) -> tuple[pd.DataFrame, WalkState]:
    """Return `games` with pre-game feature columns, plus the final state of every team.

    `games` must already carry walk-forward `elo_home`, `elo_away`, and `elo_diff`.
    """
    if not games["start_date"].is_monotonic_increasing:
        raise ValueError("Games must be sorted by kickoff; otherwise features would use the future.")

    by_game = team_game_stats(games, stats["box"], stats["advanced"])
    state = WalkState(
        teams={},
        recruiting=recruiting_scores(stats["recruiting"]),
        returning=dict(zip(zip(stats["returning"]["season"], stats["returning"]["team"]), stats["returning"]["usage"])),
    )

    decay = 0.5 ** (1.0 / params.halflife)
    recent_decay = 0.5 ** (1.0 / RECENT_HALFLIFE)
    # League averages, so an adjustment measures how far an opponent sits from typical.
    league = _RunningMean(len(METRICS), 1.0)
    missing = np.full(len(METRICS), np.nan)
    points = METRICS.index("points")

    def state_of(team: int, season: int) -> _TeamState:
        current = state.teams.setdefault(team, _TeamState(decay, recent_decay))
        current.start_season(season, params.season_weight)
        return current

    rows = []
    columns = ["game_id", "season", "home_id", "away_id", "home_team", "away_team", "elo_home", "elo_away"]
    for game_id, season, home, away, home_name, away_name, elo_home, elo_away in zip(
        *(games[column].tolist() for column in columns)
    ):
        home_state, away_state = state_of(home, season), state_of(away, season)

        # Forecast inputs are captured here, before this game's stats are used.
        home_off, home_def = home_state.offense.read(), home_state.defense.read()
        away_off, away_def = away_state.offense.read(), away_state.defense.read()
        rows.append({"game_id": game_id, **_read_row(state, home_state, away_state, season, home_name, away_name)})

        # Only now does the game's own data update anyone's averages.
        home_stats = by_game.get((game_id, home), missing)
        away_stats = by_game.get((game_id, away), missing)
        typical = league.read()
        # With no opponent history yet, the raw stat stands unadjusted.
        home_adjusted_off = home_stats - np.nan_to_num(away_def - typical)
        home_adjusted_def = away_stats - np.nan_to_num(away_off - typical)
        away_adjusted_off = away_stats - np.nan_to_num(home_def - typical)
        away_adjusted_def = home_stats - np.nan_to_num(home_off - typical)
        home_state.offense.add(home_adjusted_off)
        home_state.defense.add(home_adjusted_def)
        away_state.offense.add(away_adjusted_off)
        away_state.defense.add(away_adjusted_def)

        home_state.recent_margin.add(np.array([home_adjusted_off[points] - home_adjusted_def[points]]))
        away_state.recent_margin.add(np.array([away_adjusted_off[points] - away_adjusted_def[points]]))
        home_state.schedule.add(np.array([elo_away], dtype=float))
        away_state.schedule.add(np.array([elo_home], dtype=float))
        league.add(home_stats)
        league.add(away_stats)

    frame = games.merge(pd.DataFrame(rows), on="game_id", validate="one_to_one")
    frame["true_home"] = (~frame["neutral_site"]).astype(float)
    return frame, state


def build_features(
    games: pd.DataFrame, stats: dict[str, pd.DataFrame], params: FeatureParams
) -> pd.DataFrame:
    return walk_forward(games, stats, params)[0]


def matchup_row(
    state: WalkState,
    ratings: dict[int, float],
    season: int,
    home_id: int,
    away_id: int,
    home_name: str,
    away_name: str,
    neutral_site: bool,
) -> pd.DataFrame:
    """A one-row feature table for a hypothetical game, using the latest team averages."""
    row = _read_row(state, state.teams[home_id], state.teams[away_id], season, home_name, away_name)
    row.update(
        season=season,
        neutral_site=neutral_site,
        true_home=float(not neutral_site),
        elo_home=ratings[home_id],
        elo_away=ratings[away_id],
        elo_diff=ratings[home_id] - ratings[away_id],
    )
    return pd.DataFrame([row])
