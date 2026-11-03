"""The prediction service: the only part of the project the interface talks to.

It takes two teams and a site, and returns plain results. It holds no interface code,
fits no models, and needs no API key or raw data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from cfbsim.adjustments import apply_adjustments
from cfbsim.bundle import COMPARISON_METRICS, HISTORICAL, PredictionBundle
from cfbsim.features import matchup_row
from cfbsim.matchup import factors_table
from cfbsim.scoring import team_view
from cfbsim.simulate import summarize

DEFAULT_SIMULATIONS = 10_000
DEFAULT_SEED = 2026
# A difference this large between the two win models deserves a visible note.
DISAGREEMENT_NOTICE = 0.10
# In season, team data older than this may be missing recent games.
STALE_AFTER_DAYS = 9
THIN_HISTORY_WEIGHT = 2.0
REPORT_LEAD_DAYS = 3

FACTOR_LABELS = {
    "team_offense": "Offense",
    "opponent_defense": "Defense",
    "site": "Home field",
    "team_strength": "Elo strength and schedule",
    "returning": "Returning production",
    "recruiting": "Recruiting talent",
    "recent_form": "Recent form",
    "pace": "Pace of play",
    "turnovers": "Turnovers",
}


@dataclass(frozen=True)
class Notice:
    # One of "info", "caution".
    level: str
    text: str


@dataclass(frozen=True)
class Freshness:
    """How old the data is. The interface must show this with every prediction."""

    is_current_forecast: bool
    headline: str
    detail: str
    data_through: date


@dataclass
class MatchupPrediction:
    home: str
    away: str
    neutral_site: bool
    summary: dict
    home_expected: float
    away_expected: float
    stats_model_home_probability: float
    factors: list[dict]
    comparison: pd.DataFrame
    home_scores: np.ndarray
    away_scores: np.ndarray
    notices: list[Notice]
    freshness: Freshness
    # The tested model's own results, and the layer on top, when that layer is in use.
    base_summary: dict | None = None
    adjustment: dict | None = None


def assess_freshness(bundle: PredictionBundle, today: date | None = None) -> Freshness:
    today = today or date.today()
    through = bundle.data_through
    if bundle.kind == HISTORICAL:
        return Freshness(
            False,
            f"Historical demonstration, not a forecast of any current or upcoming game",
            f"Team ratings and stats stop at {through:%B %-d, %Y}. Rosters and team strength have changed "
            "since then. Use this to explore how the model works.",
            through,
        )
    age = (today - through).days
    if age > STALE_AFTER_DAYS:
        return Freshness(
            False,
            f"Out-of-date data: team stats are {age} days old",
            f"Team ratings and stats are current through {through:%B %-d, %Y}. Games played since then "
            "are not included, so this is not a reliable forecast until the data is refreshed.",
            through,
        )
    note = f" ({bundle.data_note})" if bundle.data_note else ""
    injuries = (
        "Injuries count only when an official report is on file."
        if bundle.adjustments is not None
        else "Injuries and other late news are never included."
    )
    return Freshness(
        True,
        f"Team stats current through {through:%B %-d, %Y}",
        f"Includes every completed game available through that date{note}. Games played after it are not "
        f"included. {injuries}",
        through,
    )


class Predictor:
    def __init__(self, bundle: PredictionBundle) -> None:
        self.bundle = bundle

    def team_names(self) -> list[str]:
        """Teams this bundle can simulate: FBS in its season, with a known history."""
        return sorted(
            name
            for name, team_id in self.bundle.teams.items()
            if team_id in self.bundle.score_state.teams
            and team_id in self.bundle.win_state.teams
            and team_id in self.bundle.ratings
        )

    def predict(
        self,
        home: str,
        away: str,
        neutral_site: bool = False,
        simulations: int = DEFAULT_SIMULATIONS,
        seed: int = DEFAULT_SEED,
        today: date | None = None,
        game_date: date | None = None,
    ) -> MatchupPrediction:
        names = self.team_names()
        for name in (home, away):
            if name not in names:
                raise ValueError(f"{name} is not available in this model.")
        if home == away:
            raise ValueError("Choose two different teams.")

        bundle = self.bundle
        home_id, away_id = bundle.teams[home], bundle.teams[away]
        row = matchup_row(
            bundle.score_state, bundle.ratings, bundle.season, home_id, away_id, home, away, neutral_site
        )
        win_row = matchup_row(
            bundle.win_state, bundle.ratings, bundle.season, home_id, away_id, home, away, neutral_site
        )
        home_view, away_view = team_view(row, "home"), team_view(row, "away")
        home_expected = float(bundle.score_model.predict(home_view)[0])
        away_expected = float(bundle.score_model.predict(away_view)[0])
        stats_probability = float(bundle.win_model.predict(win_row)[0])
        factors = factors_table(
            bundle.score_model.contributions(home_view).iloc[0],
            bundle.score_model.contributions(away_view).iloc[0],
        )
        for factor in factors:
            factor["label"] = FACTOR_LABELS[factor["factor"]]

        adjustment = None
        base_summary = None
        if bundle.adjustments is not None:
            home_field = next(factor["margin"] for factor in factors if factor["factor"] == "site")
            adjustment = apply_adjustments(
                home, away, home_expected, away_expected, home_field, bundle.adjustments, today or date.today(),
                game_date,
            )
            base_summary = summarize(
                bundle.simulator.simulate(home_expected, away_expected, simulations, np.random.default_rng(seed))
            )
            home_expected, away_expected = adjustment["home_expected"], adjustment["away_expected"]
            factors = self._with_adjustment_factors(factors, adjustment)

        games = bundle.simulator.simulate(
            home_expected, away_expected, simulations, np.random.default_rng(seed)
        )
        summary = summarize(games)
        return MatchupPrediction(
            home=home,
            away=away,
            neutral_site=neutral_site,
            base_summary=base_summary,
            adjustment=adjustment,
            summary=summary,
            home_expected=home_expected,
            away_expected=away_expected,
            stats_model_home_probability=stats_probability,
            factors=factors,
            comparison=self._comparison(home, away),
            home_scores=games.home,
            away_scores=games.away,
            notices=self._notices(home, away, summary, stats_probability, adjustment, game_date, today),
            freshness=assess_freshness(bundle, today),
        )

    @staticmethod
    def _with_adjustment_factors(factors: list[dict], adjustment: dict) -> list[dict]:
        extra = []
        if adjustment["fpi_used"]:
            extra.append(("fpi", f"ESPN FPI ({adjustment['fpi_weight']:.0%} blend)", adjustment["fpi_margin_shift"]))
        if adjustment["players"]:
            extra.append(("availability", "Official player availability", adjustment["availability_margin"]))
        added = [
            {"factor": key, "label": label, "home_points": 0.0, "away_points": 0.0, "margin": float(margin)}
            for key, label, margin in extra
        ]
        return sorted([*factors, *added], key=lambda row: -abs(row["margin"]))

    def _comparison(self, home: str, away: str) -> pd.DataFrame:
        profiles = self.bundle.profiles
        rows = []
        for metric in COMPARISON_METRICS:
            for team, side in ((home, "home"), (away, "away")):
                value = profiles.at[team, metric.key]
                percentile = profiles.at[team, f"{metric.key}_percentile"]
                rows.append(
                    {
                        "metric": metric.label,
                        "group": metric.group,
                        "team": team,
                        "side": side,
                        "value": None if pd.isna(value) else round(float(value), metric.decimals),
                        "percentile": None if pd.isna(percentile) else float(percentile),
                    }
                )
        return pd.DataFrame(rows)

    def _notices(
        self,
        home: str,
        away: str,
        summary: dict,
        stats_probability: float,
        adjustment: dict | None,
        game_date: date | None = None,
        today: date | None = None,
    ) -> list[Notice]:
        bundle = self.bundle
        notices = []
        days_away = None if game_date is None else (game_date - (today or date.today())).days
        if adjustment is not None and days_away is not None and days_away > REPORT_LEAD_DAYS:
            notices.append(
                Notice(
                    "info",
                    "Official availability reports for this game aren't out yet. They usually appear about three days "
                    "before kickoff. Team strength may also change before then, so this forecast will update.",
                )
            )
        elif adjustment is not None:
            for team, has_reports in ((home, adjustment["home_has_reports"]), (away, adjustment["away_has_reports"])):
                if not has_reports:
                    notices.append(
                        Notice(
                            "caution",
                            f"No official availability report is on file for {team}'s next game. That doesn't mean "
                            "the team is healthy, and many non-conference games have no report at all.",
                        )
                    )
        if adjustment is not None and not adjustment["fpi_used"]:
            notices.append(Notice("caution", "FPI isn't included in this prediction because it's missing or out of date."))
        for team in (home, away):
            if pd.isna(bundle.profiles.at[team, "returning"]):
                notices.append(
                    Notice(
                        "caution",
                        f"{team} has no returning-production data for {bundle.season}. In testing, the largest "
                        "gaps between the two win models involved teams like this.",
                    )
                )
            if pd.isna(bundle.profiles.at[team, "recruiting"]):
                notices.append(
                    Notice("info", f"{team} has no complete four-year recruiting data for {bundle.season}.")
                )
            weight = bundle.score_state.teams[bundle.teams[team]].offense.weight.max()
            if weight < THIN_HISTORY_WEIGHT:
                notices.append(Notice("caution", f"{team} has very little game history in this data."))

        gap = abs(summary["home_win_probability"] - stats_probability)
        if gap > DISAGREEMENT_NOTICE:
            notices.append(
                Notice(
                    "caution",
                    f"The simulation and the team-stat model differ by {gap:.0%} here, "
                    "so treat this win probability with extra caution.",
                )
            )
        for first, second, played_on, score in bundle.completed_matchups:
            if {first, second} == {home, away}:
                result = f" ({score})" if score else ""
                notices.append(
                    Notice(
                        "caution",
                        f"These teams already played on {played_on:%B %-d, %Y}{result}. That result is included "
                        "in the team data, so this is a hypothetical rematch and not a pre-game forecast of that game.",
                    )
                )
        return notices
