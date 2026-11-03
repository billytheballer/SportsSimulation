"""Tune on early seasons, then score on later seasons the model never trained on."""

from __future__ import annotations

from dataclasses import asdict
from itertools import product

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from cfbsim.config import HOLDOUT_SEASONS, TUNE_SEASONS, VALIDATION_SEASONS
from cfbsim.elo import ELO_SCALE, EloParams, run_elo
from cfbsim.metrics import calibration_table, log_loss, log_loss_edge_interval, score

ELO_GRID = {
    "k": (20.0, 30.0, 40.0, 50.0, 60.0, 80.0),
    "home_advantage": (45.0, 55.0, 65.0, 75.0, 90.0),
    "season_carryover": (0.5, 0.6, 0.7, 0.8),
    "lower_division_initial": (800.0, 900.0, 1000.0, 1100.0, 1200.0),
}


def _fbs_only(frame: pd.DataFrame) -> pd.Series:
    """Score FBS matchups only: that is what the simulator predicts, and FCS games are easy."""
    return frame["home_classification"].eq("fbs") & frame["away_classification"].eq("fbs")


def tune_elo(games: pd.DataFrame) -> EloParams:
    """Grid-search Elo settings by log loss, on tune seasons only.

    Ratings keep running through every season given, so `games` must not include
    anything after the tune seasons.
    """
    if not games["season"].max() <= max(TUNE_SEASONS):
        raise ValueError("tune_elo was given games after the tune seasons.")
    is_scored = (games["season"].isin(TUNE_SEASONS) & _fbs_only(games)).to_numpy()
    outcomes = games["home_win"].to_numpy()[is_scored]

    best_params, best_loss = None, float("inf")
    for values in product(*ELO_GRID.values()):
        params = EloParams(**dict(zip(ELO_GRID, values)))
        predictions, _ = run_elo(games, params)
        loss = log_loss(outcomes, predictions["p_elo"].to_numpy()[is_scored])
        if loss < best_loss:
            best_params, best_loss = params, loss
    return best_params


class EloCalibrator:
    """Logistic regression from Elo gap and home field to a win probability.

    Refusing to fit on evaluation seasons makes that form of leakage an error
    rather than something to remember.
    """

    def __init__(self) -> None:
        self.model = LogisticRegression(C=1e6)

    @staticmethod
    def _features(frame: pd.DataFrame, diff_column: str) -> np.ndarray:
        return np.column_stack(
            [frame[diff_column].to_numpy() / ELO_SCALE, (~frame["neutral_site"]).to_numpy(dtype=float)]
        )

    def fit(self, frame: pd.DataFrame, diff_column: str = "elo_diff") -> "EloCalibrator":
        leaked = set(frame["season"]) - set(TUNE_SEASONS)
        if leaked:
            raise ValueError(f"Calibration may only use tune seasons, but got {sorted(leaked)}.")
        self.model.fit(self._features(frame, diff_column), frame["home_win"].to_numpy())
        return self

    def predict(self, frame: pd.DataFrame, diff_column: str = "elo_diff") -> np.ndarray:
        return self.model.predict_proba(self._features(frame, diff_column))[:, 1]


def _home_rate_forecast(train: pd.DataFrame, frame: pd.DataFrame) -> np.ndarray:
    """Baseline: the home win rate seen in training for true home games, 50% at neutral sites."""
    rate = train.loc[~train["neutral_site"], "home_win"].mean()
    return np.where(frame["neutral_site"], 0.5, rate)


def _evaluate(frame: pd.DataFrame, train: pd.DataFrame, cfbd: EloCalibrator) -> dict:
    """Score the model, baselines, and CFBD's Elo benchmark on one block of games."""
    outcomes = frame["home_win"].to_numpy()
    model = frame["p_model"].to_numpy()
    home_rate = _home_rate_forecast(train, frame)
    coin_flip = np.full(len(frame), 0.5)

    result: dict = {
        "models": {
            "elo_calibrated": score(outcomes, model),
            "elo_uncalibrated": score(outcomes, frame["p_elo"].to_numpy()),
            "baseline_home_rate": score(outcomes, home_rate),
            "baseline_coin_flip": score(outcomes, coin_flip),
        },
        "edge_vs_home_rate": dict(
            zip(("mean", "low", "high"), log_loss_edge_interval(outcomes, model, home_rate))
        ),
        "calibration": calibration_table(outcomes, model).to_dict("records"),
        "by_season": {
            int(season): score(group["home_win"].to_numpy(), group["p_model"].to_numpy())
            for season, group in frame.groupby("season")
        },
    }

    # CFBD lacks Elo for some games, so compare both models on the games it does cover.
    covered = frame.dropna(subset=["cfbd_diff"])
    if len(covered):
        covered_outcomes = covered["home_win"].to_numpy()
        benchmark = cfbd.predict(covered, "cfbd_diff")
        result["cfbd_benchmark"] = {
            "games_covered": len(covered),
            "cfbd_elo": score(covered_outcomes, benchmark),
            "elo_calibrated": score(covered_outcomes, covered["p_model"].to_numpy()),
        }
    return result


def run_backtest(games: pd.DataFrame, include_holdout: bool = False) -> dict:
    """Run the Stage 1 evaluation. Holdout seasons stay unscored unless requested."""
    params = tune_elo(games[games["season"] <= max(TUNE_SEASONS)].reset_index(drop=True))

    predictions, _ = run_elo(games, params)
    frame = games.merge(predictions, on="game_id", validate="one_to_one")
    frame["cfbd_diff"] = frame["cfbd_home_elo"] - frame["cfbd_away_elo"]
    frame = frame[_fbs_only(frame)]

    train = frame[frame["season"].isin(TUNE_SEASONS)]
    calibrator = EloCalibrator().fit(train)
    frame = frame.assign(p_model=calibrator.predict(frame))
    train_covered = train.dropna(subset=["cfbd_diff"])
    cfbd = EloCalibrator().fit(train_covered, "cfbd_diff") if len(train_covered) else None

    blocks = {"validation": VALIDATION_SEASONS}
    if include_holdout:
        blocks["holdout"] = HOLDOUT_SEASONS

    report: dict = {
        "elo_params": asdict(params),
        "calibration_coefficients": {
            "elo_gap_per_400_points": float(calibrator.model.coef_[0][0]),
            "home_field": float(calibrator.model.coef_[0][1]),
            "intercept": float(calibrator.model.intercept_[0]),
        },
        "training_games": len(train),
    }
    for name, seasons in blocks.items():
        block = frame[frame["season"].isin(seasons)]
        if block.empty:
            raise ValueError(f"No games available for the {name} seasons {seasons}.")
        report[name] = {"seasons": list(seasons), **_evaluate(block, train, cfbd)}
    return report


def report_to_markdown(report: dict) -> str:
    lines = ["# Stage 1 backtest: walk-forward Elo", ""]
    lines += [
        "Generated by `cfbsim backtest`. FBS vs FBS games only. Data provided by "
        "CollegeFootballData.com.",
        "",
        f"Tuned on {report['training_games']:,} games. Chosen Elo settings: `{report['elo_params']}`",
    ]
    for name in ("validation", "holdout"):
        if name not in report:
            continue
        block = report[name]
        edge = block["edge_vs_home_rate"]
        lines += ["", f"## {name.title()} seasons: {block['seasons']}", ""]
        lines += ["| Forecast | Games | Accuracy | Log loss | Brier |", "|---|---:|---:|---:|---:|"]
        for model, row in block["models"].items():
            lines.append(
                f"| {model} | {row['games']:,} | {row['accuracy']:.1%} "
                f"| {row['log_loss']:.4f} | {row['brier']:.4f} |"
            )
        lines += [
            "",
            f"Log-loss improvement over the home-rate baseline: {edge['mean']:.4f} "
            f"(95% bootstrap interval {edge['low']:.4f} to {edge['high']:.4f}).",
        ]
        if "cfbd_benchmark" in block:
            benchmark = block["cfbd_benchmark"]
            lines += [
                "",
                f"On the {benchmark['games_covered']:,} games with CFBD's pre-game Elo: "
                f"this model log loss {benchmark['elo_calibrated']['log_loss']:.4f}, "
                f"CFBD Elo {benchmark['cfbd_elo']['log_loss']:.4f}.",
            ]
        lines += ["", "| Forecast range | Games | Mean forecast | Actual home win rate |", "|---|---:|---:|---:|"]
        for row in block["calibration"]:
            lines.append(
                f"| {row['bin']} | {row['games']:,} | {row['forecast']:.1%} | {row['actual']:.1%} |"
            )
    return "\n".join(lines) + "\n"
