"""Monte Carlo simulation of final scores, given each team's expected points.

Each simulated game has two steps:

1. Regulation. A pair of scores is drawn from real past games where both teams had
   similar expected points, with weights nudged so the averages match this matchup.
   Drawing real pairs keeps real football numbers such as 17, 24, and 31, and keeps
   how scores move together: close finishes, key margins, ties, and blowouts.
2. Overtime. A regulation tie goes to overtime: a winner is drawn, then both teams
   add points taken from a real past overtime. Games therefore never end tied.

Simulation error shrinks with more runs. Model error does not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri
from sklearn.linear_model import LogisticRegression

GRID_START, GRID_END, GRID_STEP = 3.0, 70.0, 0.25
TILT_LIMIT, TILT_STEPS = 2.0, 50
PAIR_TILT_LIMIT, PAIR_TILT_STEPS = 0.5, 12
# Widen the search for similar games until the draw rests on at least this many.
MINIMUM_EFFECTIVE_GAMES = 200.0
WIDEN_FACTOR = 1.5
MAXIMUM_WIDENINGS = 5
INTERVALS = {"50%": (25, 75), "80%": (10, 90), "90%": (5, 95), "95%": (2.5, 97.5)}


@dataclass
class SimulatedGames:
    home: np.ndarray
    away: np.ndarray
    overtime: np.ndarray


class Overtime:
    """Settles regulation ties using real past overtimes."""

    def __init__(self, games: pd.DataFrame) -> None:
        overtime = games[games["overtime"].astype(bool)]
        home_added = (overtime["home_points"] - overtime["home_regulation"]).to_numpy()
        away_added = (overtime["away_points"] - overtime["away_regulation"]).to_numpy()
        self.adds = np.column_stack(
            [np.maximum(home_added, away_added), np.minimum(home_added, away_added)]
        ).astype(int)
        self.winner = LogisticRegression(C=1.0).fit(
            ((overtime["home_expected"] - overtime["away_expected"]) / 10).to_numpy().reshape(-1, 1),
            (overtime["home_points"] > overtime["away_points"]).astype(int).to_numpy(),
        )

    def settle(
        self, home: np.ndarray, away: np.ndarray, gap: float, rng: np.random.Generator
    ) -> SimulatedGames:
        tied = home == away
        ties = int(tied.sum())
        if ties:
            home_wins = rng.random(ties) < self.winner.predict_proba(np.array([[gap / 10]]))[0, 1]
            winner_add, loser_add = self.adds[rng.integers(len(self.adds), size=ties)].T
            home[tied] += np.where(home_wins, winner_add, loser_add)
            away[tied] += np.where(home_wins, loser_add, winner_add)
        return SimulatedGames(home, away, tied)


class PairSimulator:
    """Draws regulation score pairs from real past games with similar expectations.

    `games` must hold expected points from a model that did not train on those same
    games; otherwise the spread of outcomes would be too narrow. Each game is used from
    both teams' sides, since expected points already include home field.
    """

    def __init__(self, games: pd.DataFrame, bandwidth: float) -> None:
        self.bandwidth = bandwidth
        self.expected = np.column_stack(
            [
                np.concatenate([games["home_expected"], games["away_expected"]]),
                np.concatenate([games["away_expected"], games["home_expected"]]),
            ]
        )
        self.scores = np.column_stack(
            [
                np.concatenate([games["home_regulation"], games["away_regulation"]]),
                np.concatenate([games["away_regulation"], games["home_regulation"]]),
            ]
        ).astype(int)
        self.overtime = Overtime(games)

    def weights(self, home_expected: float, away_expected: float) -> np.ndarray:
        target = np.array([home_expected, away_expected])
        bandwidth = self.bandwidth
        best, best_effective = None, -1.0
        for _ in range(MAXIMUM_WIDENINGS):
            log_kernel = -0.5 * (((self.expected - target) / bandwidth) ** 2).sum(axis=1)
            weights = self._tilted(log_kernel, target)
            effective = 1.0 / np.sum(weights**2)
            if effective >= MINIMUM_EFFECTIVE_GAMES:
                return weights
            if effective > best_effective:
                best, best_effective = weights, effective
            bandwidth *= WIDEN_FACTOR
        # A rare, lopsided matchup may never reach the target, so use the broadest support found.
        return best

    def _tilted(self, log_kernel: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Nudge weights as little as possible so both average scores equal the target."""
        scores = self.scores.astype(float)
        tilt = np.zeros(2)
        for _ in range(PAIR_TILT_STEPS):
            log_weights = log_kernel + scores @ tilt
            weights = np.exp(log_weights - log_weights.max())
            weights /= weights.sum()
            mean = weights @ scores
            centered = scores - mean
            covariance = (centered * weights[:, None]).T @ centered + 1e-6 * np.eye(2)
            tilt = np.clip(tilt + np.linalg.solve(covariance, target - mean), -PAIR_TILT_LIMIT, PAIR_TILT_LIMIT)
        log_weights = log_kernel + scores @ tilt
        weights = np.exp(log_weights - log_weights.max())
        return weights / weights.sum()

    def simulate(
        self, home_expected: float, away_expected: float, count: int, rng: np.random.Generator
    ) -> SimulatedGames:
        picks = rng.choice(len(self.scores), size=count, p=self.weights(home_expected, away_expected))
        home, away = self.scores[picks, 0].copy(), self.scores[picks, 1].copy()
        return self.overtime.settle(home, away, home_expected - away_expected, rng)


def _tilted_weights(log_weights: np.ndarray, scores: np.ndarray, target: float) -> np.ndarray:
    low, high = -TILT_LIMIT, TILT_LIMIT
    for _ in range(TILT_STEPS):
        tilt = (low + high) / 2
        tilted = log_weights + tilt * scores
        weights = np.exp(tilted - tilted.max())
        weights /= weights.sum()
        if weights @ scores < target:
            low = tilt
        else:
            high = tilt
    return weights


class ScoreTable:
    """One team's score distribution by expected points, built from real past scores."""

    def __init__(self, expected: np.ndarray, actual: np.ndarray, bandwidth: float) -> None:
        actual = np.asarray(actual, dtype=int)
        expected = np.asarray(expected, dtype=float)
        self.grid = np.arange(GRID_START, GRID_END + GRID_STEP, GRID_STEP)
        self.pmf = np.zeros((len(self.grid), actual.max() + 1))
        for index, target in enumerate(self.grid):
            log_weights = -0.5 * ((expected - target) / bandwidth) ** 2
            weights = _tilted_weights(log_weights, actual.astype(float), target)
            self.pmf[index] = np.bincount(actual, weights=weights, minlength=actual.max() + 1)
        self.cdf = np.cumsum(self.pmf, axis=1)
        self.cdf[:, -1] = 1.0

    def row_of(self, expected: np.ndarray | float) -> np.ndarray:
        clipped = np.clip(expected, GRID_START, GRID_END)
        return np.rint((clipped - GRID_START) / GRID_STEP).astype(int)

    def draw(self, expected: float, uniforms: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.cdf[self.row_of(expected)], uniforms, side="left")

    def midpoint_rank(self, expected: np.ndarray, actual: np.ndarray) -> np.ndarray:
        """Where each actual score falls in its forecast distribution, from 0 to 1."""
        rows = self.row_of(expected)
        actual = np.minimum(np.asarray(actual, dtype=int), self.pmf.shape[1] - 1)
        below = np.where(actual > 0, self.cdf[rows, np.maximum(actual - 1, 0)], 0.0)
        return np.clip(below + 0.5 * self.pmf[rows, actual], 1e-6, 1 - 1e-6)


class SeparateDrawSimulator:
    """A comparison: each team's score is drawn on its own, linked by one correlation."""

    def __init__(self, games: pd.DataFrame, bandwidth: float) -> None:
        expected = np.concatenate([games["home_expected"], games["away_expected"]])
        actual = np.concatenate([games["home_regulation"], games["away_regulation"]])
        self.table = ScoreTable(expected, actual, bandwidth)
        self.correlation = regulation_correlation(games, self.table)
        self.overtime = Overtime(games)

    def simulate(
        self, home_expected: float, away_expected: float, count: int, rng: np.random.Generator
    ) -> SimulatedGames:
        rho = self.correlation
        first = rng.standard_normal(count)
        second = rho * first + np.sqrt(1 - rho**2) * rng.standard_normal(count)
        home = self.table.draw(home_expected, ndtr(first))
        away = self.table.draw(away_expected, ndtr(second))
        return self.overtime.settle(home, away, home_expected - away_expected, rng)


def regulation_correlation(games: pd.DataFrame, table: ScoreTable) -> float:
    """Correlation between the two teams' results, after allowing for what was expected."""
    home = ndtri(table.midpoint_rank(games["home_expected"].to_numpy(), games["home_regulation"].to_numpy()))
    away = ndtri(table.midpoint_rank(games["away_expected"].to_numpy(), games["away_regulation"].to_numpy()))
    return float(np.corrcoef(home, away)[0, 1])


def summarize(games: SimulatedGames) -> dict:
    """Win chances, averages, medians, and prediction intervals from simulated games."""
    margin, total = games.home - games.away, games.home + games.away
    summary: dict = {
        "simulations": len(margin),
        "home_win_probability": float(np.mean(margin > 0)),
        "away_win_probability": float(np.mean(margin < 0)),
        "overtime_probability": float(np.mean(games.overtime)),
    }
    for name, values in (("home", games.home), ("away", games.away), ("margin", margin), ("total", total)):
        summary[name] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "intervals": {
                level: [float(bound) for bound in np.percentile(values, bounds)]
                for level, bounds in INTERVALS.items()
            },
        }
    pairs, counts = np.unique(np.column_stack([games.home, games.away]), axis=0, return_counts=True)
    top = np.argsort(-counts)[:5]
    summary["common_scores"] = [
        {"home": int(pairs[i, 0]), "away": int(pairs[i, 1]), "share": float(counts[i] / len(margin))} for i in top
    ]
    return summary
