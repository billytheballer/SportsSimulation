"""Scoring for win-probability forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd

EPSILON = 1e-12


def log_loss(outcomes: np.ndarray, probabilities: np.ndarray) -> float:
    return float(np.mean(_game_log_loss(outcomes, probabilities)))


def _game_log_loss(outcomes: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, EPSILON, 1 - EPSILON)
    return -(outcomes * np.log(clipped) + (1 - outcomes) * np.log(1 - clipped))


def score(outcomes: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    """Accuracy counts a 50% forecast as half right, so coin flips score 50%."""
    outcomes = np.asarray(outcomes, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    picks = np.where(probabilities == 0.5, 0.5, (probabilities > 0.5).astype(float))
    correct = np.where(picks == 0.5, 0.5, (picks == outcomes).astype(float))
    return {
        "games": int(len(outcomes)),
        "accuracy": float(correct.mean()),
        "log_loss": log_loss(outcomes, probabilities),
        "brier": float(np.mean((probabilities - outcomes) ** 2)),
    }


def calibration_table(
    outcomes: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> pd.DataFrame:
    """Compare forecast probabilities with how often those games were actually won."""
    edges = np.linspace(0, 1, bins + 1)
    frame = pd.DataFrame({"outcome": outcomes, "forecast": probabilities})
    frame["bin"] = pd.cut(frame["forecast"], edges, include_lowest=True)
    table = frame.groupby("bin", observed=True).agg(
        games=("outcome", "size"), forecast=("forecast", "mean"), actual=("outcome", "mean")
    )
    return table.reset_index().assign(bin=lambda t: t["bin"].astype(str))


def crps(samples: np.ndarray, actual: float) -> float:
    """How far a simulated distribution sits from the result. Lower is better; in points."""
    ordered = np.sort(samples)
    count = len(ordered)
    spread = 2.0 * np.sum((2 * np.arange(1, count + 1) - count - 1) * ordered) / count**2
    return float(np.mean(np.abs(ordered - actual)) - 0.5 * spread)


def point_errors(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Error of a point forecast, where a positive bias means the forecast ran high."""
    errors = np.asarray(predicted, dtype=float) - np.asarray(actual, dtype=float)
    return {
        "games": int(len(errors)),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "bias": float(np.mean(errors)),
    }


def favorite_calibration(outcomes: np.ndarray, probabilities: np.ndarray) -> pd.DataFrame:
    """Compare forecasts with results from the favorite's side, with a standard error."""
    home_favored = probabilities >= 0.5
    frame = pd.DataFrame(
        {
            "forecast": np.where(home_favored, probabilities, 1 - probabilities),
            "won": np.where(home_favored, outcomes, 1 - outcomes),
        }
    )
    frame["bin"] = pd.cut(frame["forecast"], [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], include_lowest=True)
    rows = []
    for name, group in frame.groupby("bin", observed=True):
        variance = (group["forecast"] * (1 - group["forecast"])).sum()
        rows.append(
            {
                "bin": str(name),
                "games": len(group),
                "forecast": group["forecast"].mean(),
                "actual": group["won"].mean(),
                "standard_error": float(np.sqrt(variance)) / len(group),
            }
        )
    return pd.DataFrame(rows)


def log_loss_edge_interval(
    outcomes: np.ndarray,
    model: np.ndarray,
    baseline: np.ndarray,
    resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap the model's log-loss improvement. Above zero means the model is better."""
    edge = _game_log_loss(np.asarray(outcomes, float), np.asarray(baseline, float)) - _game_log_loss(
        np.asarray(outcomes, float), np.asarray(model, float)
    )
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(edge), size=(resamples, len(edge)))
    means = edge[indexes].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(edge.mean()), float(low), float(high)
