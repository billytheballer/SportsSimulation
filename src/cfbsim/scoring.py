"""Expected points for each team, from its offense against the opponent's defense.

This model never sees a win probability. It predicts points after four quarters,
so that overtime can be simulated as its own step afterward.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cfbsim.config import TUNE_SEASONS
from cfbsim.features import METRICS
from cfbsim.metrics import point_errors
from cfbsim.model import FORWARD_FOLDS

_RATE_METRICS = [metric for metric in METRICS if metric not in ("turnovers", "plays")]

SCORE_FEATURE_GROUPS = {
    "team_offense": [f"team_off_{metric}" for metric in _RATE_METRICS],
    "opponent_defense": [f"opp_def_{metric}" for metric in _RATE_METRICS],
    "pace": ["team_off_plays", "opp_def_plays", "team_def_plays", "opp_off_plays"],
    "turnovers": ["team_off_turnovers", "opp_def_turnovers", "team_def_turnovers", "opp_off_turnovers"],
    "team_strength": ["team_elo", "opp_elo", "team_schedule_strength", "opp_schedule_strength"],
    "recent_form": ["team_recent_margin", "opp_recent_margin"],
    "recruiting": ["team_recruiting", "opp_recruiting"],
    "returning": ["team_returning", "opp_returning"],
    "site": ["is_home", "is_away"],
}
SCORE_FEATURES = [column for columns in SCORE_FEATURE_GROUPS.values() for column in columns]
_GROUP_OF = {column: group for group, columns in SCORE_FEATURE_GROUPS.items() for column in columns}


def team_view(frame: pd.DataFrame, side: str) -> pd.DataFrame:
    """The game table from one team's point of view, keeping the same row order."""
    other = "away" if side == "home" else "home"
    columns = {}
    for column in SCORE_FEATURES:
        if column in ("is_home", "is_away"):
            continue
        owner, rest = column.split("_", 1)
        source = side if owner == "team" else other
        columns[column] = frame[f"elo_{source}" if rest == "elo" else f"{source}_{rest}"].to_numpy()
    neutral = frame["neutral_site"].to_numpy(dtype=bool)
    columns["is_home"] = ((side == "home") & ~neutral).astype(float)
    columns["is_away"] = ((side == "away") & ~neutral).astype(float)
    return pd.DataFrame(columns).assign(season=frame["season"].to_numpy())


def team_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Two rows per game, one per team, with regulation points as the target."""
    home = team_view(frame, "home").assign(target=frame["home_regulation"].to_numpy())
    away = team_view(frame, "away").assign(target=frame["away_regulation"].to_numpy())
    return pd.concat([home, away], ignore_index=True)


class ScoreModel:
    """Regularized linear model of one team's points after four quarters."""

    def __init__(self, alpha: float = 300.0, features: list[str] | None = None) -> None:
        self.features = list(features or SCORE_FEATURES)
        self.pipeline = make_pipeline(
            SimpleImputer(strategy="mean", add_indicator=True), StandardScaler(), Ridge(alpha=alpha)
        )

    def fit(self, table: pd.DataFrame) -> "ScoreModel":
        leaked = set(table["season"]) - set(TUNE_SEASONS)
        if leaked:
            raise ValueError(f"ScoreModel may only train on tune seasons, but got {sorted(leaked)}.")
        known = table.dropna(subset=["target"])
        self.pipeline.fit(known[self.features], known["target"].to_numpy(dtype=float))
        return self

    def predict(self, view: pd.DataFrame) -> np.ndarray:
        return np.maximum(self.pipeline.predict(view[self.features]), 0.0)

    def contributions(self, view: pd.DataFrame) -> pd.DataFrame:
        """Points each feature group adds or removes against an average team, per row."""
        imputer, scaler, ridge = self.pipeline[0], self.pipeline[1], self.pipeline[2]
        scaled = scaler.transform(imputer.transform(view[self.features]))
        names = imputer.get_feature_names_out(self.features)
        groups = [_GROUP_OF[name.removeprefix("missingindicator_")] for name in names]
        effect = pd.DataFrame(scaled * ridge.coef_, columns=names, index=view.index)
        return effect.T.groupby(groups).sum().T


def forward_rmse(table: pd.DataFrame, alpha: float, model: str = "ridge") -> float:
    """Next-season RMSE of team points across the forward folds. Tune seasons only."""
    errors = []
    for train_seasons, test_season in FORWARD_FOLDS:
        train = table[table["season"].isin(train_seasons)].dropna(subset=["target"])
        test = table[table["season"] == test_season].dropna(subset=["target"])
        if model == "ridge":
            predicted = ScoreModel(alpha).fit(train).predict(test)
        else:
            pipeline = make_pipeline(
                SimpleImputer(strategy="mean", add_indicator=True),
                StandardScaler(),
                PoissonRegressor(alpha=alpha, max_iter=1000),
            ).fit(train[SCORE_FEATURES], train["target"])
            predicted = pipeline.predict(test[SCORE_FEATURES])
        errors.append(point_errors(test["target"].to_numpy(), predicted)["rmse"])
    return float(np.mean(errors))
