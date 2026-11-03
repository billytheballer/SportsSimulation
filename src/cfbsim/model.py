"""Regularized logistic regression on pre-game features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cfbsim.config import TUNE_SEASONS
from cfbsim.metrics import log_loss

# Each fold trains on earlier tune seasons and scores the next one.
FORWARD_FOLDS = ((2016, 2017), 2018), ((2016, 2017, 2018), 2019), ((2016, 2017, 2018, 2019), 2020)


class WinModel:
    """Predicts the home team's win probability.

    A missing difference means one team has no history for that stat. It is treated
    as "no known edge" and gets its own indicator column, so the model can tell a
    real zero from an unknown. No stat values are invented.
    """

    def __init__(self, features: list[str], C: float = 0.1) -> None:
        self.features = list(features)
        self.pipeline = make_pipeline(
            SimpleImputer(strategy="constant", fill_value=0.0, add_indicator=True),
            StandardScaler(),
            LogisticRegression(C=C, max_iter=2000),
        )

    def fit(self, frame: pd.DataFrame) -> "WinModel":
        leaked = set(frame["season"]) - set(TUNE_SEASONS)
        if leaked:
            raise ValueError(f"WinModel may only train on tune seasons, but got {sorted(leaked)}.")
        self.pipeline.fit(frame[self.features], frame["home_win"].to_numpy())
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(frame[self.features])[:, 1]

    def coefficients(self) -> dict[str, float]:
        """Log-odds effect of a one standard deviation change in each input."""
        names = self.pipeline[0].get_feature_names_out(self.features)
        return dict(zip(names.tolist(), self.pipeline[-1].coef_[0].tolist()))


def forward_loss(frame: pd.DataFrame, features: list[str], C: float) -> float:
    """Average next-season log loss across the forward folds. Tune seasons only."""
    losses = []
    for train_seasons, test_season in FORWARD_FOLDS:
        train = frame[frame["season"].isin(train_seasons)]
        test = frame[frame["season"] == test_season]
        model = WinModel(features, C).fit(train)
        losses.append(log_loss(test["home_win"].to_numpy(), model.predict(test)))
    return float(np.mean(losses))
