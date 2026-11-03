"""Simulate one matchup using settings from the last Stage 3 run.

Team averages stop at the end of 2022, because 2023-25 stay untouched until the final
evaluation. Results are a demonstration of the method, not a forecast of a current game.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from cfbsim.backtest import _fbs_only
from cfbsim.client import CFBDClient
from cfbsim.config import DEVELOPMENT_SEASONS, REPORTS_DIR, TUNE_SEASONS
from cfbsim.elo import EloParams, run_elo
from cfbsim.features import ALL_FEATURES, FeatureParams, matchup_row, walk_forward
from cfbsim.games import fetch_season_records, load_games, parse_regulation
from cfbsim.model import WinModel
from cfbsim.scoring import team_view
from cfbsim.simulate import summarize
from cfbsim.stage3 import fit_stack, scored_frame
from cfbsim.stats import load_stats


def load_settings() -> dict:
    path = REPORTS_DIR / "stage3_diagnostics.json"
    if not path.exists():
        raise FileNotFoundError("Run `cfbsim stage3` first to choose the model settings.")
    return json.loads(path.read_text())


def simulate_matchup(
    home_name: str, away_name: str, neutral_site: bool, simulations: int, seed: int
) -> dict:
    report = load_settings()
    client = CFBDClient()
    seasons = list(DEVELOPMENT_SEASONS)
    games, _ = load_games(client, seasons)
    stats = load_stats(client, games, seasons)
    records = [record for season in seasons for record in fetch_season_records(client, season)]
    regulation, _ = parse_regulation(records)

    last_season = max(seasons)
    latest = games[(games["season"] == last_season) & _fbs_only(games)]
    teams = dict(zip(latest["home_team"], latest["home_id"]))
    teams.update(zip(latest["away_team"], latest["away_id"]))
    for name in (home_name, away_name):
        if name not in teams:
            close = [team for team in sorted(teams) if name.lower() in team.lower()]
            hint = f" Did you mean: {', '.join(close)}?" if close else ""
            raise ValueError(f"'{name}' is not an FBS team in {last_season}.{hint}")
    if home_name == away_name:
        raise ValueError("Choose two different teams.")

    selection = report["selection"]
    elo_predictions, ratings = run_elo(games, EloParams(**report["elo_params"]))
    params = FeatureParams(**selection["feature_params"])
    frame = scored_frame(games, elo_predictions, stats, params, regulation)
    stack = fit_stack(frame[frame["season"].isin(TUNE_SEASONS)], selection["alpha"], selection["bandwidth"])

    with_elo = games.merge(elo_predictions, on="game_id", validate="one_to_one")
    _, state = walk_forward(with_elo, stats, params)
    row = matchup_row(
        state, ratings, last_season, teams[home_name], teams[away_name], home_name, away_name, neutral_site
    )
    home_expected, away_expected = stack.expected(row)
    simulated = stack.simulator.simulate(
        home_expected[0], away_expected[0], simulations, np.random.default_rng(seed)
    )

    # The stats model uses the settings Stage 2 chose, so it needs its own team averages.
    win_settings = report["win_model_settings"]
    win_params = FeatureParams(**win_settings["feature_params"])
    win_frame = scored_frame(games, elo_predictions, stats, win_params, regulation)
    win_model = WinModel(ALL_FEATURES, win_settings["C"]).fit(win_frame[win_frame["season"].isin(TUNE_SEASONS)])
    _, win_state = walk_forward(with_elo, stats, win_params)
    win_row = matchup_row(
        win_state, ratings, last_season, teams[home_name], teams[away_name], home_name, away_name, neutral_site
    )

    home_effects = stack.model.contributions(team_view(row, "home")).iloc[0]
    away_effects = stack.model.contributions(team_view(row, "away")).iloc[0]
    return {
        "home_team": home_name,
        "away_team": away_name,
        "neutral_site": neutral_site,
        "as_of": f"end of the {last_season} season",
        "expected_regulation_points": {"home": float(home_expected[0]), "away": float(away_expected[0])},
        "stats_model_home_win_probability": float(win_model.predict(win_row)[0]),
        "simulation": summarize(simulated),
        "margin_factors": factors_table(home_effects, away_effects),
    }


def factors_table(home_effects: pd.Series, away_effects: pd.Series) -> list[dict]:
    """Points each factor adds to the home team's expected margin, largest first."""
    rows = [
        {
            "factor": group,
            "home_points": float(home_effects[group]),
            "away_points": float(away_effects[group]),
            "margin": float(home_effects[group] - away_effects[group]),
        }
        for group in home_effects.index
    ]
    return sorted(rows, key=lambda row: -abs(row["margin"]))


def format_matchup(result: dict) -> str:
    home, away = result["home_team"], result["away_team"]
    simulation = result["simulation"]
    site = "neutral site" if result["neutral_site"] else f"at {home}"
    lines = [
        f"{away} vs {home} ({site}), using team data through the {result['as_of']}",
        f"This demonstrates the method and is not a current forecast. {simulation['simulations']:,} simulations.",
        "",
        f"Win probability:  {home} {simulation['home_win_probability']:.1%}  |  {away} {simulation['away_win_probability']:.1%}",
        f"Stats model (for comparison): {home} {result['stats_model_home_win_probability']:.1%}",
        f"Overtime chance:  {simulation['overtime_probability']:.1%}",
        "",
        f"{'':18}{'Average':>8}{'Median':>8}{'80% interval':>16}{'95% interval':>16}",
    ]
    for key, label in ((("home"), home), (("away"), away), ("margin", "Margin (home)"), ("total", "Total")):
        block = simulation[key]
        eighty, ninety_five = block["intervals"]["80%"], block["intervals"]["95%"]
        lines.append(
            f"{label[:18]:18}{block['mean']:8.1f}{block['median']:8.0f}"
            f"{f'{eighty[0]:.0f} to {eighty[1]:.0f}':>16}{f'{ninety_five[0]:.0f} to {ninety_five[1]:.0f}':>16}"
        )
    lines += ["", "Most common simulated scores:"]
    for item in simulation["common_scores"]:
        lines.append(f"  {home} {item['home']}, {away} {item['away']}: {item['share']:.1%}")
    lines += ["", f"What drives the projected margin (points toward {home}):"]
    for row in result["margin_factors"]:
        lines.append(f"  {row['factor'].replace('_', ' '):18}{row['margin']:+7.1f}")
    lines += [
        "",
        "Factors show how the model reached its number. They overlap and are not proof of cause.",
        "Injuries, weather, coaching changes, and player availability are not included.",
    ]
    return "\n".join(lines)
