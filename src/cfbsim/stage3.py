"""Stage 3: the score model and simulation, tuned on 2016-20 and diagnosed on 2021-22.

Holdout seasons are refused outright here. `run_stage3` raises an error if it receives one.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cfbsim.backtest import EloCalibrator, _fbs_only, tune_elo
from cfbsim.config import HOLDOUT_SEASONS, TUNE_SEASONS, VALIDATION_SEASONS
from cfbsim.elo import run_elo
from cfbsim.features import ALL_FEATURES, FeatureParams, build_features
from cfbsim.metrics import crps, favorite_calibration, log_loss, log_loss_edge_interval, point_errors, score
from cfbsim.model import FORWARD_FOLDS, WinModel
from cfbsim.scoring import ScoreModel, forward_rmse, team_table, team_view
from cfbsim.simulate import (
    INTERVALS,
    PairSimulator,
    SeparateDrawSimulator,
    SimulatedGames,
    regulation_correlation,
)
from cfbsim.stage2 import HALFLIVES, SEASON_WEIGHTS, select_settings

ALPHAS = (10.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0)
POISSON_ALPHAS = (0.0001, 0.001, 0.01)
BANDWIDTHS = (3.0, 5.0, 7.0)
SEPARATE_DRAW_BANDWIDTH = 3.0
SIMULATIONS = 10_000
SELECTION_SIMULATIONS = 2_000
POOLED_DRAWS_PER_GAME = 200
SEED = 2026


def scored_frame(
    games: pd.DataFrame, elo: pd.DataFrame, stats: dict, params: FeatureParams, regulation: pd.DataFrame
) -> pd.DataFrame:
    with_elo = games.merge(elo, on="game_id", validate="one_to_one")
    frame = build_features(with_elo, stats, params)
    frame = frame[_fbs_only(frame)].merge(regulation, on="game_id", how="left")
    # Stored as a number so games without usable line scores can stay missing.
    return frame.assign(overtime=frame["overtime"].astype(float)).reset_index(drop=True)


@dataclass
class ScoringStack:
    model: ScoreModel
    simulator: PairSimulator
    # Out-of-sample expected points and real results for the training games.
    fit_data: pd.DataFrame

    def expected(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return (self.model.predict(team_view(frame, "home")), self.model.predict(team_view(frame, "away")))


def fit_stack(train: pd.DataFrame, alpha: float, bandwidth: float) -> ScoringStack:
    """Fit the score model and simulator on the given seasons.

    The simulator needs outcome spread that reflects real forecast error, so its
    expected points for each season come from a model trained on the other seasons.
    """
    known = train.dropna(subset=["home_regulation", "away_regulation"]).reset_index(drop=True)
    fit_data = known.assign(home_expected=np.nan, away_expected=np.nan)
    for season in known["season"].unique():
        inside = (known["season"] == season).to_numpy()
        model = ScoreModel(alpha).fit(team_table(known[~inside]))
        held_out = known[inside]
        fit_data.loc[inside, "home_expected"] = model.predict(team_view(held_out, "home"))
        fit_data.loc[inside, "away_expected"] = model.predict(team_view(held_out, "away"))
    return ScoringStack(ScoreModel(alpha).fit(team_table(known)), PairSimulator(fit_data, bandwidth), fit_data)


def _forward_folds(
    frame: pd.DataFrame, alpha: float, bandwidth: float, count: int, win_C: float, separate: bool = False
) -> dict:
    """Train on earlier tune seasons, simulate the next one. Tune seasons only."""
    margin_crps, total_crps, sim_probability, outcomes, stats_probability = [], [], [], [], []
    overtime, within_3, key_margins, actual_margins, actual_overtime = [], [], [], [], []
    for train_seasons, test_season in FORWARD_FOLDS:
        train = frame[frame["season"].isin(train_seasons)]
        test = frame[frame["season"] == test_season]
        stack = fit_stack(train, alpha, bandwidth)
        simulator = SeparateDrawSimulator(stack.fit_data, bandwidth) if separate else stack.simulator
        home_expected, away_expected = stack.expected(test)
        rng = np.random.default_rng([SEED, test_season])
        for row, (home, away) in enumerate(zip(home_expected, away_expected)):
            games = simulator.simulate(home, away, count, rng)
            actual_margin = test["home_points"].iloc[row] - test["away_points"].iloc[row]
            actual_total = test["home_points"].iloc[row] + test["away_points"].iloc[row]
            margin_crps.append(crps(games.home - games.away, actual_margin))
            total_crps.append(crps(games.home + games.away, actual_total))
            sim_probability.append(np.mean(games.home > games.away))
            overtime.append(np.mean(games.overtime))
            within_3.append(np.mean(np.abs(games.home - games.away) <= 3))
            key_margins.append(np.mean(np.isin(np.abs(games.home - games.away), (3, 7))))
            actual_margins.append(abs(actual_margin))
        actual_overtime.append(test["overtime"].to_numpy())
        outcomes.append(test["home_win"].to_numpy())
        stats_probability.append(WinModel(ALL_FEATURES, win_C).fit(train).predict(test))
    outcome = np.concatenate(outcomes)
    actual = np.array(actual_margins)
    return {
        "simulated_overtime": float(np.mean(overtime)),
        "actual_overtime": float(np.nanmean(np.concatenate(actual_overtime))),
        "simulated_margin_within_3": float(np.mean(within_3)),
        "actual_margin_within_3": float(np.mean(actual <= 3)),
        "simulated_margin_3_or_7": float(np.mean(key_margins)),
        "actual_margin_3_or_7": float(np.mean(np.isin(actual, (3, 7)))),
        "margin_crps": float(np.mean(margin_crps)),
        "total_crps": float(np.mean(total_crps)),
        "simulation_log_loss": log_loss(outcome, np.clip(sim_probability, 1e-3, 1 - 1e-3)),
        "stats_model_log_loss": log_loss(outcome, np.concatenate(stats_probability)),
    }


def select_scoring(
    tune_games: pd.DataFrame, elo: pd.DataFrame, stats: dict, regulation: pd.DataFrame, win_C: float
) -> dict:
    """Choose all score model settings by forward testing. Tune seasons only."""
    if tune_games["season"].max() > max(TUNE_SEASONS):
        raise ValueError("select_scoring was given games after the tune seasons.")
    best_params, best_alpha, best_rmse = None, None, float("inf")
    for halflife, season_weight in product(HALFLIVES, SEASON_WEIGHTS):
        params = FeatureParams(halflife, season_weight)
        frame = scored_frame(tune_games, elo, stats, params, regulation)
        frame = frame[frame["season"].isin(TUNE_SEASONS)]
        table = team_table(frame)
        for alpha in ALPHAS:
            rmse = forward_rmse(table, alpha)
            if rmse < best_rmse:
                best_params, best_alpha, best_rmse = params, alpha, rmse

    frame = scored_frame(tune_games, elo, stats, best_params, regulation)
    frame = frame[frame["season"].isin(TUNE_SEASONS)]
    poisson = min(forward_rmse(team_table(frame), alpha, "poisson") for alpha in POISSON_ALPHAS)

    forward = {
        bandwidth: _forward_folds(frame, best_alpha, bandwidth, SELECTION_SIMULATIONS, win_C)
        for bandwidth in BANDWIDTHS
    }
    best_bandwidth = min(forward, key=lambda b: forward[b]["margin_crps"] + forward[b]["total_crps"])
    separate = _forward_folds(
        frame, best_alpha, SEPARATE_DRAW_BANDWIDTH, SELECTION_SIMULATIONS, win_C, separate=True
    )
    return {
        "feature_params": asdict(best_params),
        "alpha": best_alpha,
        "bandwidth": best_bandwidth,
        "forward_points_rmse": {"ridge": best_rmse, "poisson_best": poisson},
        "forward_simulation_by_bandwidth": {str(b): value for b, value in forward.items()},
        "forward_separate_draws": separate,
    }


def _normal_draws(
    home: float, away: float, sd: float, rho: float, count: int, rng: np.random.Generator
) -> SimulatedGames:
    """A simple comparison: bell-curve scores, rounded and floored at zero. Ties split evenly."""
    first = rng.standard_normal(count)
    second = rho * first + np.sqrt(1 - rho**2) * rng.standard_normal(count)
    home_score = np.maximum(np.rint(home + sd * first), 0).astype(int)
    away_score = np.maximum(np.rint(away + sd * second), 0).astype(int)
    tied = home_score == away_score
    home_score[tied] += rng.integers(0, 2, tied.sum()) * 3
    away_score[tied] += (home_score[tied] == away_score[tied]) * 3
    return SimulatedGames(home_score, away_score, tied)


def simulate_block(
    frame: pd.DataFrame,
    home_expected: np.ndarray,
    away_expected: np.ndarray,
    draw,
    count: int = SIMULATIONS,
) -> dict:
    """Run `count` simulated games per game and collect every number the diagnostics need."""
    n = len(frame)
    probability = np.zeros(n)
    overtime = np.zeros(n)
    means = {key: np.zeros(n) for key in ("home", "away", "margin", "total")}
    medians = {key: np.zeros(n) for key in means}
    covered = {target: {level: np.zeros(n, dtype=bool) for level in INTERVALS} for target in means}
    below90 = {target: np.zeros(n, dtype=bool) for target in means}
    above90 = {target: np.zeros(n, dtype=bool) for target in means}
    crps_values = {"margin": np.zeros(n), "total": np.zeros(n)}
    pooled_home, pooled_away, pooled_overtime = [], [], []
    actual_home = frame["home_points"].to_numpy()
    actual_away = frame["away_points"].to_numpy()

    for row in range(n):
        rng = np.random.default_rng([SEED, int(frame["game_id"].iloc[row])])
        games = draw(home_expected[row], away_expected[row], count, rng)
        values = {
            "home": games.home,
            "away": games.away,
            "margin": games.home - games.away,
            "total": games.home + games.away,
        }
        probability[row] = np.mean(values["margin"] > 0)
        overtime[row] = np.mean(games.overtime)
        for key, array in values.items():
            means[key][row], medians[key][row] = array.mean(), np.median(array)
        actuals = {
            "home": actual_home[row],
            "away": actual_away[row],
            "margin": actual_home[row] - actual_away[row],
            "total": actual_home[row] + actual_away[row],
        }
        crps_values["margin"][row] = crps(values["margin"], actuals["margin"])
        crps_values["total"][row] = crps(values["total"], actuals["total"])
        for target, array in values.items():
            for level, (low, high) in INTERVALS.items():
                lower, upper = np.percentile(array, [low, high])
                covered[target][level][row] = lower <= actuals[target] <= upper
                if level == "90%":
                    below90[target][row] = actuals[target] < lower
                    above90[target][row] = actuals[target] > upper
        pooled_home.append(games.home[:POOLED_DRAWS_PER_GAME])
        pooled_away.append(games.away[:POOLED_DRAWS_PER_GAME])
        pooled_overtime.append(games.overtime[:POOLED_DRAWS_PER_GAME])

    return {
        "probability": probability,
        "overtime": overtime,
        "means": means,
        "medians": medians,
        "coverage": {
            "team": {
                level: float((covered["home"][level].mean() + covered["away"][level].mean()) / 2)
                for level in INTERVALS
            },
            **{
                target: {level: float(covered[target][level].mean()) for level in INTERVALS}
                for target in ("margin", "total")
            },
        },
        "outside_90": {
            target: {"below": float(below90[target].mean()), "above": float(above90[target].mean())}
            for target in ("margin", "total")
        },
        "crps": {key: float(value.mean()) for key, value in crps_values.items()},
        "pooled_home": np.concatenate(pooled_home),
        "pooled_away": np.concatenate(pooled_away),
        "pooled_overtime": np.concatenate(pooled_overtime),
    }


def _describe(values: np.ndarray) -> dict:
    series = pd.Series(np.asarray(values, dtype=float))
    return {
        "mean": float(series.mean()),
        "sd": float(series.std()),
        "skew": float(series.skew()),
        **{f"p{q}": float(series.quantile(q / 100)) for q in (1, 10, 50, 90, 99)},
    }


def _shape(home: np.ndarray, away: np.ndarray) -> dict:
    """Overall realism statistics, computed the same way for real and simulated games."""
    scores = np.concatenate([home, away])
    margin, total = home - away, home + away
    return {
        "team_points": _describe(scores),
        "total": _describe(total),
        "absolute_margin": _describe(np.abs(margin)),
        "correlation": float(np.corrcoef(home, away)[0, 1]),
        "share_shutout": float(np.mean(scores == 0)),
        "share_team_50_plus": float(np.mean(scores >= 50)),
        "share_total_30_or_less": float(np.mean(total <= 30)),
        "share_total_80_plus": float(np.mean(total >= 80)),
        "share_margin_28_plus": float(np.mean(np.abs(margin) >= 28)),
        "share_margin_within_3": float(np.mean(np.abs(margin) <= 3)),
        "share_margin_exactly_3_or_7": float(np.mean(np.isin(np.abs(margin), (3, 7)))),
        "share_rare_scores_1_2_4_5_8": float(np.mean(np.isin(scores, (1, 2, 4, 5, 8)))),
        "top_scores": {
            int(value): float(np.mean(scores == value)) for value in (0, 3, 7, 10, 14, 17, 21, 24, 28, 31, 35, 38)
        },
    }


class _MarginModel:
    """A linear margin forecast from a win model's inputs, used as a comparison."""

    def __init__(self, features: list[str]) -> None:
        self.features = features
        self.pipeline = make_pipeline(
            SimpleImputer(strategy="constant", fill_value=0.0, add_indicator=True),
            StandardScaler(),
            Ridge(alpha=100.0),
        )

    def fit(self, frame: pd.DataFrame) -> "_MarginModel":
        leaked = set(frame["season"]) - set(TUNE_SEASONS)
        if leaked:
            raise ValueError(f"_MarginModel may only train on tune seasons, but got {sorted(leaked)}.")
        self.pipeline.fit(frame[self.features], frame["margin"].to_numpy())
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(frame[self.features])


def _point_comparison(frame: pd.DataFrame, train: pd.DataFrame, block: dict) -> dict:
    home, away = frame["home_points"].to_numpy(dtype=float), frame["away_points"].to_numpy(dtype=float)
    margin, total = home - away, home + away
    team_actual = np.concatenate([home, away])
    train_home = train["home_points"].to_numpy(dtype=float)
    train_away = train["away_points"].to_numpy(dtype=float)
    average_team = np.full(len(team_actual), np.concatenate([train_home, train_away]).mean())
    neutral = frame["neutral_site"].to_numpy(dtype=bool)
    real_home = ~train["neutral_site"].to_numpy(dtype=bool)
    average_margin = np.where(neutral, 0.0, (train_home - train_away)[real_home].mean())

    means = block["means"]
    report = {
        "team_points": {
            "simulation_mean": point_errors(team_actual, np.concatenate([means["home"], means["away"]])),
            "simulation_median": point_errors(team_actual, np.concatenate([block["medians"]["home"], block["medians"]["away"]])),
            "training_average": point_errors(team_actual, average_team),
        },
        "margin": {
            "simulation_mean": point_errors(margin, means["margin"]),
            "simulation_median": point_errors(margin, block["medians"]["margin"]),
            "stats_regression": point_errors(margin, _MarginModel(ALL_FEATURES).fit(train).predict(frame)),
            "elo_regression": point_errors(margin, _MarginModel(["elo_diff", "true_home"]).fit(train).predict(frame)),
            "training_average": point_errors(margin, average_margin),
        },
        "total": {
            "simulation_mean": point_errors(total, means["total"]),
            "simulation_median": point_errors(total, block["medians"]["total"]),
            "training_average": point_errors(total, np.full(len(total), (train_home + train_away).mean())),
        },
    }
    with_spread = frame["home_spread"].notna().to_numpy()
    report["margin"]["listed_spread_timing_unverified"] = point_errors(
        margin[with_spread], -frame["home_spread"].to_numpy()[with_spread]
    )
    with_total = frame["over_under"].notna().to_numpy()
    report["total"]["listed_total_timing_unverified"] = point_errors(
        total[with_total], frame["over_under"].to_numpy()[with_total]
    )
    return report


def run_stage3(
    games: pd.DataFrame, stats: dict, regulation: pd.DataFrame, regulation_quality: dict | None = None
) -> dict:
    """Fit on 2016-20 and diagnose on 2021-22. There is no option to score the holdout."""
    touched = set(games["season"]) & set(HOLDOUT_SEASONS)
    if touched:
        raise ValueError(f"Stage 3 must not touch holdout seasons, but received {sorted(touched)}.")

    tune_games = games[games["season"] <= max(TUNE_SEASONS)].reset_index(drop=True)
    elo_params = tune_elo(tune_games)
    elo_tune, _ = run_elo(tune_games, elo_params)
    win_params, win_C, _ = select_settings(tune_games, elo_tune, stats)
    selection = select_scoring(tune_games, elo_tune, stats, regulation, win_C)

    elo_all, _ = run_elo(games, elo_params)
    line_columns = stats["lines"][["game_id", "home_spread", "over_under"]]
    frame = scored_frame(games, elo_all, stats, FeatureParams(**selection["feature_params"]), regulation)
    frame = frame.merge(line_columns, on="game_id", how="left")
    train = frame[frame["season"].isin(TUNE_SEASONS)]
    stack = fit_stack(train, selection["alpha"], selection["bandwidth"])

    # The win models keep the settings Stage 2 chose, so they stay directly comparable.
    win_frame = scored_frame(games, elo_all, stats, win_params, regulation)
    win_train = win_frame[win_frame["season"].isin(TUNE_SEASONS)]
    win_model = WinModel(ALL_FEATURES, win_C).fit(win_train)
    elo_model = EloCalibrator().fit(win_train)

    block_rows = frame["season"].isin(VALIDATION_SEASONS).to_numpy()
    block = frame[block_rows].reset_index(drop=True)
    win_block = win_frame[win_frame["season"].isin(VALIDATION_SEASONS)].reset_index(drop=True)
    home_expected, away_expected = stack.expected(block)
    simulator = stack.simulator

    main = simulate_block(block, home_expected, away_expected, simulator.simulate)
    separate = SeparateDrawSimulator(stack.fit_data, SEPARATE_DRAW_BANDWIDTH)
    independent = simulate_block(block, home_expected, away_expected, separate.simulate)
    known = block.dropna(subset=["home_regulation", "away_regulation"])
    residual_sd = float(
        np.std(np.concatenate([
            (train["home_regulation"] - stack.model.predict(team_view(train, "home"))).dropna(),
            (train["away_regulation"] - stack.model.predict(team_view(train, "away"))).dropna(),
        ]))
    )
    normal = simulate_block(
        block, home_expected, away_expected,
        lambda home, away, count, rng: _normal_draws(home, away, residual_sd, separate.correlation, count, rng),
    )

    outcomes = block["home_win"].to_numpy()
    sim_p = np.clip(main["probability"], 1e-3, 1 - 1e-3)
    stats_p = win_model.predict(win_block)
    elo_p = elo_model.predict(win_block)
    if not (block["game_id"].to_numpy() == win_block["game_id"].to_numpy()).all():
        raise ValueError("Win model and score model rows are misaligned.")

    rerun = simulate_rerun(block, home_expected, away_expected, simulator)
    early = (block["season_type"].eq("regular") & (block["week"] <= 4)).to_numpy()
    real_home, real_away = block["home_points"].to_numpy(), block["away_points"].to_numpy()

    report: dict = {
        "elo_params": asdict(elo_params),
        "selection": selection,
        "win_model_settings": {"feature_params": asdict(win_params), "C": win_C},
        "training_games": len(train),
        "regulation_data_quality": regulation_quality or {},
        "fitted": {
            "correlation_after_expectations": separate.correlation,
            "overtime_rate_training": float(train["overtime"].mean()),
            "overtime_games_training": int(train["overtime"].sum()),
            "residual_sd_points": residual_sd,
        },
        "diagnostic_seasons": list(VALIDATION_SEASONS),
        "games": len(block),
        "win_probability": {
            "simulation": score(outcomes, sim_p),
            "stats_model": score(outcomes, stats_p),
            "elo_calibrated": score(outcomes, elo_p),
            "separate_draws": score(outcomes, np.clip(independent["probability"], 1e-3, 1 - 1e-3)),
            "simulation_bell_curve": score(outcomes, np.clip(normal["probability"], 1e-3, 1 - 1e-3)),
        },
        "win_probability_by_phase": {
            name: {
                "simulation": score(outcomes[mask], sim_p[mask]),
                "stats_model": score(outcomes[mask], stats_p[mask]),
                "elo_calibrated": score(outcomes[mask], elo_p[mask]),
            }
            for name, mask in (("weeks_1_to_4", early), ("later", ~early))
        },
        "win_probability_edges": {
            "over_stats_model": dict(zip(("mean", "low", "high"), log_loss_edge_interval(outcomes, sim_p, stats_p))),
            "over_elo": dict(zip(("mean", "low", "high"), log_loss_edge_interval(outcomes, sim_p, elo_p))),
        },
        "factor_importance": _factor_importance(stack, block),
        "largest_disagreements": _largest_gaps(block, sim_p, stats_p),
        "calibration": {
            "simulation": favorite_calibration(outcomes, sim_p).to_dict("records"),
            "stats_model": favorite_calibration(outcomes, stats_p).to_dict("records"),
        },
        "agreement_with_stats_model": {
            "correlation": float(np.corrcoef(sim_p, stats_p)[0, 1]),
            "mean_absolute_gap": float(np.mean(np.abs(sim_p - stats_p))),
            "largest_gap": float(np.max(np.abs(sim_p - stats_p))),
            "share_gap_over_10_points": float(np.mean(np.abs(sim_p - stats_p) > 0.10)),
            "same_favorite_share": float(np.mean((sim_p > 0.5) == (stats_p > 0.5))),
            "mean_home_probability": {"simulation": float(sim_p.mean()), "stats_model": float(stats_p.mean())},
        },
        "points": _point_comparison(block, train, main),
        "points_by_season": {
            int(season): {
                "total": point_errors(
                    (block["home_points"] + block["away_points"]).to_numpy()[mask], main["means"]["total"][mask]
                ),
                "margin": point_errors(
                    (block["home_points"] - block["away_points"]).to_numpy()[mask], main["means"]["margin"][mask]
                ),
            }
            for season in VALIDATION_SEASONS
            for mask in [(block["season"] == season).to_numpy()]
        },
        "coverage": {
            "simulation": main["coverage"],
            "outside_90": main["outside_90"],
            "separate_draws": independent["coverage"],
            "bell_curve": normal["coverage"],
        },
        "crps": {
            "simulation": main["crps"],
            "separate_draws": independent["crps"],
            "bell_curve": normal["crps"],
        },
        "realism": {
            "actual": _shape(real_home, real_away),
            "simulation": _shape(main["pooled_home"], main["pooled_away"]),
            "separate_draws": _shape(independent["pooled_home"], independent["pooled_away"]),
            "bell_curve": _shape(normal["pooled_home"], normal["pooled_away"]),
        },
        "overtime": {
            "actual_rate": float(block["overtime"].mean()),
            "simulated_rate": float(main["overtime"].mean()),
            "separate_draws_rate": float(independent["overtime"].mean()),
            "games_with_known_overtime": int(block["overtime"].notna().sum()),
        },
        "correlation_after_expectations_on_diagnostic_seasons": float(
            regulation_correlation(
                known.assign(
                    home_expected=home_expected[known.index],
                    away_expected=away_expected[known.index],
                ),
                separate.table,
            )
        ),
        "simulation_repeatability": rerun,
    }
    return report


def _factor_importance(stack: ScoringStack, block: pd.DataFrame) -> dict[str, float]:
    """Average size of each factor's push on the projected margin, in points."""
    margin = stack.model.contributions(team_view(block, "home")) - stack.model.contributions(team_view(block, "away"))
    return margin.abs().mean().sort_values(ascending=False).to_dict()


def _largest_gaps(block: pd.DataFrame, sim_p: np.ndarray, stats_p: np.ndarray) -> list[dict]:
    """The games where the simulation and the stats model disagree most, for inspection."""
    return [
        {
            "season": int(block["season"].iloc[row]),
            "week": int(block["week"].iloc[row]),
            "home": block["home_team"].iloc[row],
            "away": block["away_team"].iloc[row],
            "simulation": float(sim_p[row]),
            "stats_model": float(stats_p[row]),
            "home_returning_known": bool(pd.notna(block["home_returning"].iloc[row])),
            "away_returning_known": bool(pd.notna(block["away_returning"].iloc[row])),
        }
        for row in np.argsort(-np.abs(sim_p - stats_p))[:5]
    ]


def simulate_rerun(
    block: pd.DataFrame, home_expected: np.ndarray, away_expected: np.ndarray, simulator: PairSimulator
) -> dict:
    """How much win probabilities move between two independent 10,000-game runs."""
    gaps = []
    for row in range(len(block)):
        first = simulator.simulate(home_expected[row], away_expected[row], SIMULATIONS, np.random.default_rng([1, row]))
        second = simulator.simulate(home_expected[row], away_expected[row], SIMULATIONS, np.random.default_rng([2, row]))
        gaps.append(abs(np.mean(first.home > first.away) - np.mean(second.home > second.away)))
    return {"mean_gap": float(np.mean(gaps)), "largest_gap": float(np.max(gaps))}


def _point_table(rows: dict) -> list[str]:
    lines = ["| Forecast | Games | MAE | RMSE | Bias |", "|---|---:|---:|---:|---:|"]
    for name, values in rows.items():
        lines.append(
            f"| {name} | {values['games']:,} | {values['mae']:.2f} | {values['rmse']:.2f} | {values['bias']:+.2f} |"
        )
    return lines


def _design_rows(selection: dict) -> list[str]:
    forward = selection["forward_simulation_by_bandwidth"]
    designs = {
        "Actual": None,
        "Separate draws": selection["forward_separate_draws"],
        **{f"Real pairs, bandwidth {bandwidth}": row for bandwidth, row in forward.items()},
    }
    sample = selection["forward_separate_draws"]
    rows = [
        f"| Actual | {sample['actual_overtime']:.1%} | {sample['actual_margin_within_3']:.1%} "
        f"| {sample['actual_margin_3_or_7']:.1%} | | | |"
    ]
    for name, row in designs.items():
        if row is not None:
            rows.append(
                f"| {name} | {row['simulated_overtime']:.1%} | {row['simulated_margin_within_3']:.1%} "
                f"| {row['simulated_margin_3_or_7']:.1%} | {row['simulation_log_loss']:.4f} "
                f"| {row['margin_crps']:.3f} | {row['total_crps']:.3f} |"
            )
    return rows


def report_to_markdown(report: dict) -> str:
    selection = report["selection"]
    lines = [
        "# Stage 3 diagnostics: score model and simulation",
        "",
        "Generated by `cfbsim stage3`. FBS vs FBS games only. Data provided by CollegeFootballData.com.",
        "",
        f"Settings chosen on {report['training_games']:,} games from 2016-20: "
        f"`{selection['feature_params']}`, ridge alpha {selection['alpha']}, bandwidth {selection['bandwidth']}.",
        f"Diagnosed on {report['games']:,} games from {report['diagnostic_seasons']}, which were not used for "
        "selection. The 2023-25 holdout was not touched.",
        "",
        "Listed betting lines are benchmarks only and never model inputs. CFBD doesn't say when in the week "
        "they were captured, so they are not described as closing lines.",
        "",
        "## Simulator design test, inside 2016-20 only",
        "",
        "Trained on earlier seasons and simulated 2018, 2019, and 2020. This decided the design.",
        "",
        "| Design | Overtime | Margin within 3 | Margin 3 or 7 | Log loss | Margin CRPS | Total CRPS |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *_design_rows(selection),
        "",
        "## Win probability",
        "",
        "| Forecast | Games | Accuracy | Log loss | Brier |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in report["win_probability"].items():
        lines.append(
            f"| {name} | {row['games']:,} | {row['accuracy']:.1%} | {row['log_loss']:.4f} | {row['brier']:.4f} |"
        )
    agreement = report["agreement_with_stats_model"]
    edges = report["win_probability_edges"]
    lines += [
        "",
        "Log-loss edge of the simulation (above zero means the simulation is better): "
        f"over the stats model {edges['over_stats_model']['mean']:+.4f} "
        f"(95% interval {edges['over_stats_model']['low']:+.4f} to {edges['over_stats_model']['high']:+.4f}), "
        f"over Elo {edges['over_elo']['mean']:+.4f} "
        f"(95% interval {edges['over_elo']['low']:+.4f} to {edges['over_elo']['high']:+.4f}).",
        "",
        f"Agreement with the stats model: correlation {agreement['correlation']:.3f}, average gap "
        f"{agreement['mean_absolute_gap']:.1%}, largest gap {agreement['largest_gap']:.1%}, "
        f"{agreement['share_gap_over_10_points']:.1%} of games differ by over 10 points, "
        f"same favorite in {agreement['same_favorite_share']:.1%}.",
        "",
        "### Calibration from the favorite's side",
        "",
        "| Forecast for favorite | Model | Games | Mean forecast | Favorite won | Standard error |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for model, rows in report["calibration"].items():
        for row in rows:
            lines.append(
                f"| {row['bin']} | {model} | {row['games']:,} | {row['forecast']:.1%} "
                f"| {row['actual']:.1%} | {row['standard_error']:.1%} |"
            )
    lines += ["", "## Point forecasts"]
    for target, rows in report["points"].items():
        lines += ["", f"### {target.replace('_', ' ').title()}", "", *_point_table(rows)]
    lines += ["", "## Interval coverage", "", "Each interval should contain the result about as often as its label says.", ""]
    lines += ["| Interval | Team points | Margin | Total |", "|---|---:|---:|---:|"]
    coverage = report["coverage"]["simulation"]
    for level in coverage["team"]:
        lines.append(
            f"| {level} | {coverage['team'][level]:.1%} | {coverage['margin'][level]:.1%} | {coverage['total'][level]:.1%} |"
        )
    outside = report["coverage"]["outside_90"]
    lines += [
        "",
        f"Outside the 90% interval on totals: {outside['total']['below']:.1%} below and "
        f"{outside['total']['above']:.1%} above (5% each is ideal).",
        "",
        "## Is the distribution realistic?",
        "",
        "| Statistic | Actual | Simulation | Separate draws | Bell curve |",
        "|---|---:|---:|---:|---:|",
    ]
    realism = report["realism"]
    keys = [
        ("correlation", "Correlation of the two scores"),
        ("share_shutout", "Team shut out"),
        ("share_team_50_plus", "Team scores 50+"),
        ("share_total_30_or_less", "Total 30 or less"),
        ("share_total_80_plus", "Total 80+"),
        ("share_margin_28_plus", "Margin 28+"),
        ("share_margin_within_3", "Margin within 3"),
        ("share_margin_exactly_3_or_7", "Margin exactly 3 or 7"),
        ("share_rare_scores_1_2_4_5_8", "Rare scores (1, 2, 4, 5, 8)"),
    ]
    for key, label in keys:
        form = "{:.3f}" if key == "correlation" else "{:.1%}"
        lines.append(
            f"| {label} | " + " | ".join(form.format(realism[name][key]) for name in realism) + " |"
        )
    for key, label in (("team_points", "Team points"), ("total", "Total"), ("absolute_margin", "Absolute margin")):
        for stat in ("mean", "sd", "p1", "p50", "p99"):
            lines.append(
                f"| {label}, {stat} | " + " | ".join(f"{realism[name][key][stat]:.1f}" for name in realism) + " |"
            )
    crps_table = report["crps"]
    lines += [
        "",
        "## Distribution score (CRPS, lower is better)",
        "",
        "| Simulation | Margin | Total |",
        "|---|---:|---:|",
    ]
    lines += [f"| {name} | {row['margin']:.3f} | {row['total']:.3f} |" for name, row in crps_table.items()]
    overtime = report["overtime"]
    rerun = report["simulation_repeatability"]
    lines += [
        "",
        "## Factors behind projected margins",
        "",
        "Average push on the projected margin, in points. These show how the model reached its number. "
        "They overlap and are not proof of cause.",
        "",
        "| Factor | Average push (points) |",
        "|---|---:|",
        *[f"| {name.replace('_', ' ')} | {value:.2f} |" for name, value in report["factor_importance"].items()],
        "",
        "## Overtime and repeatability",
        "",
        f"Overtime rate: actual {overtime['actual_rate']:.1%}, simulated {overtime['simulated_rate']:.1%}, "
        f"separate draws {overtime['separate_draws_rate']:.1%}.",
        "",
        f"Two independent 10,000-game runs differ by {rerun['mean_gap']:.2%} in win probability on average "
        f"(largest {rerun['largest_gap']:.2%}). That is only simulation noise, not forecast accuracy.",
        "",
        "Correlation between the two teams after allowing for expected points: "
        f"{report['fitted']['correlation_after_expectations']:.3f} in training, "
        f"{report['correlation_after_expectations_on_diagnostic_seasons']:.3f} on the diagnostic seasons.",
    ]
    return "\n".join(lines) + "\n"
