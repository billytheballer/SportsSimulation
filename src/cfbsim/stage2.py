"""Stage 2 evaluation: does adding team stats beat Elo on seasons it never trained on?"""

from __future__ import annotations

from dataclasses import asdict
from itertools import product

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from cfbsim.backtest import EloCalibrator, _fbs_only, _home_rate_forecast, tune_elo
from cfbsim.config import HOLDOUT_SEASONS, TUNE_SEASONS, VALIDATION_SEASONS
from cfbsim.elo import run_elo
from cfbsim.features import ALL_FEATURES, FEATURE_GROUPS, FeatureParams, build_features
from cfbsim.metrics import favorite_calibration, log_loss_edge_interval, score
from cfbsim.model import WinModel, forward_loss

HALFLIVES = (4.0, 8.0, 16.0)
SEASON_WEIGHTS = (0.25, 0.5, 1.0)
REGULARIZATION = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
EARLY_SEASON_LAST_WEEK = 4


def _scored_frame(games: pd.DataFrame, elo: pd.DataFrame, stats: dict, params: FeatureParams) -> pd.DataFrame:
    with_elo = games.merge(elo, on="game_id", validate="one_to_one")
    frame = build_features(with_elo, stats, params)
    return frame[_fbs_only(frame)].reset_index(drop=True)


def select_settings(
    games: pd.DataFrame, elo: pd.DataFrame, stats: dict
) -> tuple[FeatureParams, float, float]:
    """Choose feature settings and regularization by forward loss, on tune seasons only."""
    if games["season"].max() > max(TUNE_SEASONS):
        raise ValueError("select_settings was given games after the tune seasons.")
    best = (None, None, float("inf"))
    for halflife, season_weight in product(HALFLIVES, SEASON_WEIGHTS):
        params = FeatureParams(halflife, season_weight)
        frame = _scored_frame(games, elo, stats, params)
        for C in REGULARIZATION:
            loss = forward_loss(frame, ALL_FEATURES, C)
            if loss < best[2]:
                best = (params, C, loss)
    return best


class SpreadModel:
    """Turns a betting spread into a win probability. A benchmark, never a model input."""

    def __init__(self) -> None:
        self.model = LogisticRegression(C=1e6)

    def fit(self, frame: pd.DataFrame) -> "SpreadModel":
        leaked = set(frame["season"]) - set(TUNE_SEASONS)
        if leaked:
            raise ValueError(f"SpreadModel may only fit on tune seasons, but got {sorted(leaked)}.")
        self.model.fit(frame[["home_spread"]].to_numpy() / 10, frame["home_win"].to_numpy())
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(frame[["home_spread"]].to_numpy() / 10)[:, 1]


def _edge(outcomes: np.ndarray, model: np.ndarray, other: np.ndarray) -> dict:
    return dict(zip(("mean", "low", "high"), log_loss_edge_interval(outcomes, model, other)))


def _evaluate(frame: pd.DataFrame, train: pd.DataFrame, drop_models: dict[str, WinModel]) -> dict:
    outcomes = frame["home_win"].to_numpy()
    full = frame["p_full"].to_numpy()
    elo = frame["p_elo_calibrated"].to_numpy()
    result: dict = {
        "models": {
            "stats_model": score(outcomes, full),
            "elo_calibrated": score(outcomes, elo),
            "baseline_home_rate": score(outcomes, _home_rate_forecast(train, frame)),
            "baseline_coin_flip": score(outcomes, np.full(len(frame), 0.5)),
        },
        "edge_vs_elo": _edge(outcomes, full, elo),
        "favorite_calibration": favorite_calibration(outcomes, full).to_dict("records"),
        "by_season": {
            int(season): {
                "stats_model": score(group["home_win"].to_numpy(), group["p_full"].to_numpy()),
                "elo_calibrated": score(group["home_win"].to_numpy(), group["p_elo_calibrated"].to_numpy()),
            }
            for season, group in frame.groupby("season")
        },
    }

    is_early = (frame["season_type"].eq("regular") & (frame["week"] <= EARLY_SEASON_LAST_WEEK)).to_numpy()
    result["by_phase"] = {
        name: {
            "stats_model": score(outcomes[mask], full[mask]),
            "elo_calibrated": score(outcomes[mask], elo[mask]),
        }
        for name, mask in (("weeks_1_to_4", is_early), ("later", ~is_early))
    }

    # Benchmarks are scored on the games they cover, alongside this model on those same games.
    for column, name in (("p_cfbd", "cfbd_elo"), ("p_spread", "listed_line")):
        covered = frame.dropna(subset=[column])
        if len(covered):
            covered_outcomes = covered["home_win"].to_numpy()
            result[f"vs_{name}"] = {
                "games_covered": len(covered),
                name: score(covered_outcomes, covered[column].to_numpy()),
                "stats_model": score(covered_outcomes, covered["p_full"].to_numpy()),
                "edge": _edge(covered_outcomes, covered["p_full"].to_numpy(), covered[column].to_numpy()),
            }

    result["without_group"] = {
        group: score(outcomes, model.predict(frame)) for group, model in drop_models.items()
    }
    return result


def run_stage2(games: pd.DataFrame, stats: dict, include_holdout: bool = False) -> dict:
    """Evaluate on validation seasons. Holdout seasons stay unscored unless requested."""
    tune_games = games[games["season"] <= max(TUNE_SEASONS)].reset_index(drop=True)
    elo_params = tune_elo(tune_games)
    elo_tune, _ = run_elo(tune_games, elo_params)
    feature_params, C, forward = select_settings(tune_games, elo_tune, stats)

    elo_all, _ = run_elo(games, elo_params)
    frame = _scored_frame(games, elo_all, stats, feature_params)
    frame = frame.merge(stats["lines"][["game_id", "home_spread"]], on="game_id", how="left")
    frame["cfbd_diff"] = frame["cfbd_home_elo"] - frame["cfbd_away_elo"]
    train = frame[frame["season"].isin(TUNE_SEASONS)]

    model = WinModel(ALL_FEATURES, C).fit(train)
    elo_calibrator = EloCalibrator().fit(train)
    cfbd = EloCalibrator().fit(train.dropna(subset=["cfbd_diff"]), "cfbd_diff")
    spread = SpreadModel().fit(train.dropna(subset=["home_spread"]))

    frame["p_full"] = model.predict(frame)
    frame["p_elo_calibrated"] = elo_calibrator.predict(frame)
    frame["p_cfbd"] = np.nan
    has_cfbd = frame["cfbd_diff"].notna()
    frame.loc[has_cfbd, "p_cfbd"] = cfbd.predict(frame[has_cfbd], "cfbd_diff")
    frame["p_spread"] = np.nan
    has_spread = frame["home_spread"].notna()
    frame.loc[has_spread, "p_spread"] = spread.predict(frame[has_spread])

    forward_by_group = {"all_features": forward}
    drop_models = {}
    for group, columns in FEATURE_GROUPS.items():
        kept = [column for column in ALL_FEATURES if column not in columns]
        forward_by_group[f"without_{group}"] = forward_loss(train, kept, C)
        drop_models[group] = WinModel(kept, C).fit(train)
    forward_by_group["elo_only"] = forward_loss(train, FEATURE_GROUPS["elo"], C)

    report: dict = {
        "elo_params": asdict(elo_params),
        "feature_params": asdict(feature_params),
        "regularization_C": C,
        "training_games": len(train),
        "forward_log_loss_on_tune_seasons": forward_by_group,
        "coefficients": model.coefficients(),
        "imputed_share_in_training": {
            column: float(train[column].isna().mean())
            for column in ALL_FEATURES
            if train[column].isna().any()
        },
    }
    blocks = {"validation": VALIDATION_SEASONS}
    if include_holdout:
        blocks["holdout"] = HOLDOUT_SEASONS
    for name, seasons in blocks.items():
        block = frame[frame["season"].isin(seasons)]
        if block.empty:
            raise ValueError(f"No games available for the {name} seasons {seasons}.")
        report[name] = {"seasons": list(seasons), **_evaluate(block, train, drop_models)}
    return report


def _row(name: str, values: dict) -> str:
    return (
        f"| {name} | {values['games']:,} | {values['accuracy']:.1%} "
        f"| {values['log_loss']:.4f} | {values['brier']:.4f} |"
    )


def _interval(edge: dict) -> str:
    return f"{edge['mean']:+.4f} (95% interval {edge['low']:+.4f} to {edge['high']:+.4f})"


def report_to_markdown(report: dict) -> str:
    lines = [
        "# Stage 2 backtest: team stats model",
        "",
        "Generated by `cfbsim stage2`. FBS vs FBS games only. Data provided by CollegeFootballData.com.",
        "",
        f"Trained on {report['training_games']:,} games from 2016-20. Settings chosen by forward "
        f"testing within those seasons: `{report['feature_params']}`, "
        f"regularization C = {report['regularization_C']}.",
        "",
        "Positive edges mean the stats model has lower log loss than the comparison.",
    ]
    table_header = ["| Forecast | Games | Accuracy | Log loss | Brier |", "|---|---:|---:|---:|---:|"]
    for name in ("validation", "holdout"):
        if name not in report:
            continue
        block = report[name]
        lines += ["", f"## {name.title()} seasons: {block['seasons']}", "", *table_header]
        lines += [_row(model, values) for model, values in block["models"].items()]
        lines += ["", f"Edge over calibrated Elo: {_interval(block['edge_vs_elo'])}."]
        for key, label in (("cfbd_elo", "CFBD's Elo"), ("listed_line", "the listed betting line (timing unverified)")):
            versus = block.get(f"vs_{key}")
            if versus:
                lines += [
                    "",
                    f"Against {label} on {versus['games_covered']:,} games: stats model log loss "
                    f"{versus['stats_model']['log_loss']:.4f}, {label} {versus[key]['log_loss']:.4f}. "
                    f"Edge {_interval(versus['edge'])}.",
                ]
        lines += ["", "### By season", "", *table_header]
        for season, values in block["by_season"].items():
            lines += [_row(f"{season} stats model", values["stats_model"]),
                      _row(f"{season} Elo", values["elo_calibrated"])]
        lines += ["", "### By part of the season", "", *table_header]
        for phase, values in block["by_phase"].items():
            label = phase.replace("_", " ")
            lines += [_row(f"{label}, stats model", values["stats_model"]),
                      _row(f"{label}, Elo", values["elo_calibrated"])]
        lines += ["", "### Removing one feature group", "", *table_header]
        lines += [_row(f"without {group}", values) for group, values in block["without_group"].items()]
        lines += [
            "",
            "### Calibration from the favorite's side",
            "",
            "| Forecast for favorite | Games | Mean forecast | Favorite won | Standard error |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in block["favorite_calibration"]:
            lines.append(
                f"| {row['bin']} | {row['games']:,} | {row['forecast']:.1%} "
                f"| {row['actual']:.1%} | {row['standard_error']:.1%} |"
            )
    lines += ["", "## Forward log loss within tune seasons", "", "| Features | Log loss |", "|---|---:|"]
    lines += [f"| {name} | {value:.4f} |" for name, value in report["forward_log_loss_on_tune_seasons"].items()]
    lines += ["", "## Coefficients (log-odds per standard deviation)", "", "| Input | Effect |", "|---|---:|"]
    ordered = sorted(report["coefficients"].items(), key=lambda item: -abs(item[1]))
    lines += [f"| {name} | {value:+.3f} |" for name, value in ordered]
    return "\n".join(lines) + "\n"
