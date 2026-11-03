"""The one-time final evaluation on 2023-25.

Safeguards, in order:

1. Freeze: settings plus fingerprints of the code and cached data are saved.
2. Preflight: the same code runs on 2021-22 and must reproduce the Stage 3 numbers
   exactly. This finds bugs without reading a single holdout game.
3. Run: refused if anything changed since the freeze, if preflight hasn't passed, or
   if a result already exists. Results are written read-only and never overwritten.

Models keep the fit from 2016-20. Team ratings walk forward game by game, so each
forecast uses only games played before it. Nothing here tunes anything.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cfbsim.backtest import EloCalibrator
from cfbsim.bundle import stage3_settings
from cfbsim.client import CFBDClient
from cfbsim.config import (
    DEVELOPMENT_SEASONS,
    HISTORICAL_SEASONS,
    HOLDOUT_SEASONS,
    PROJECT_ROOT,
    RAW_DIR,
    REPORTS_DIR,
    TUNE_SEASONS,
    VALIDATION_SEASONS,
)
from cfbsim.elo import EloParams, run_elo
from cfbsim.features import ALL_FEATURES, FeatureParams
from cfbsim.games import fetch_season_records, load_games, parse_regulation
from cfbsim.metrics import favorite_calibration, log_loss_edge_interval, point_errors, score
from cfbsim.model import WinModel
from cfbsim.stage2 import SpreadModel
from cfbsim.stage3 import (
    _factor_importance,
    _largest_gaps,
    _point_comparison,
    _shape,
    fit_stack,
    scored_frame,
    simulate_block,
)
from cfbsim.stats import load_stats

FINAL_DIR = REPORTS_DIR / "final_evaluation"
MANIFEST_NAME = "freeze_manifest.json"
PREFLIGHT_NAME = "preflight.json"
STARTED_NAME = "run_started.json"
RESULT_NAME = "final_evaluation.json"
REPORT_NAME = "final_evaluation.md"
CHECKSUM_NAME = "final_evaluation.sha256"
RUN_PHRASE = "I understand this runs once"

_SOURCE_DIR = PROJECT_ROOT / "src" / "cfbsim"
FROZEN_SOURCES = tuple(
    _SOURCE_DIR / f"{name}.py"
    for name in (
        "backtest", "config", "elo", "features", "final_eval", "games", "metrics",
        "model", "scoring", "simulate", "stage2", "stage3", "stats",
    )
)
STAGE3_REPORT = REPORTS_DIR / "stage3_diagnostics.json"
# Conferences treated as "power" in the breakdown. Decided before seeing any results.
POWER_CONFERENCES = {"SEC", "Big Ten", "ACC", "Big 12", "Pac-12"}
EARLY_LAST_WEEK = 4


class FinalEvaluationError(RuntimeError):
    """The final evaluation can't proceed, and says why."""


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def data_fingerprint(raw_dir: Path = RAW_DIR) -> str:
    """A hash of cached files for past seasons, read as bytes without parsing any data.

    The usage ledger and the live 2026 season change routinely, so they don't count.
    """
    digest = hashlib.sha256()
    for path in sorted(raw_dir.rglob("*.json")):
        if path.name == "_usage.json" or "year=2026" in path.name or "teams_fbs" in str(path.parent):
            continue
        digest.update(str(path.relative_to(raw_dir)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def build_manifest(
    settings: dict[str, Any],
    sources: tuple[Path, ...] = FROZEN_SOURCES,
    stage3_report: Path = STAGE3_REPORT,
    raw_dir: Path = RAW_DIR,
) -> dict[str, Any]:
    return {
        "settings": settings,
        "training_seasons": list(TUNE_SEASONS),
        "evaluation_seasons": list(HOLDOUT_SEASONS),
        "source_hashes": {path.name: _file_hash(path) for path in sources},
        "stage3_report_hash": _file_hash(stage3_report),
        "data_fingerprint": data_fingerprint(raw_dir),
    }


def freeze(
    directory: Path = FINAL_DIR,
    sources: tuple[Path, ...] = FROZEN_SOURCES,
    stage3_report: Path = STAGE3_REPORT,
    raw_dir: Path = RAW_DIR,
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path = directory / MANIFEST_NAME
    if path.exists():
        raise FinalEvaluationError(f"Already frozen at {path}. A freeze is never replaced.")
    manifest = build_manifest(settings or stage3_settings(), sources, stage3_report, raw_dir)
    manifest["frozen_at"] = datetime.now(timezone.utc).isoformat()
    directory.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))
    _make_read_only(path)
    return manifest


def verify_manifest(
    directory: Path = FINAL_DIR,
    sources: tuple[Path, ...] = FROZEN_SOURCES,
    stage3_report: Path = STAGE3_REPORT,
    raw_dir: Path = RAW_DIR,
) -> dict[str, Any]:
    """Return the frozen manifest, or explain exactly what changed since the freeze."""
    path = directory / MANIFEST_NAME
    if not path.exists():
        raise FinalEvaluationError("Nothing is frozen yet. Run `cfbsim final-eval --freeze` first.")
    frozen = json.loads(path.read_text())
    current = build_manifest(frozen["settings"], sources, stage3_report, raw_dir)
    changed = [
        name for name, value in current["source_hashes"].items() if frozen["source_hashes"].get(name) != value
    ]
    if current["stage3_report_hash"] != frozen["stage3_report_hash"]:
        changed.append(stage3_report.name)
    if current["data_fingerprint"] != frozen["data_fingerprint"]:
        changed.append("cached data")
    if changed:
        raise FinalEvaluationError(f"Changed since the freeze: {', '.join(changed)}. The run is refused.")
    return frozen


def _make_read_only(path: Path) -> None:
    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def load_inputs(client: CFBDClient, seasons: list[int]) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    games, _ = load_games(client, seasons)
    stats = load_stats(client, games, seasons)
    records = [record for season in seasons for record in fetch_season_records(client, season)]
    regulation, _ = parse_regulation(records)
    return games, stats, regulation


def _game_type(home_conference: pd.Series, away_conference: pd.Series) -> np.ndarray:
    home_power = home_conference.isin(POWER_CONFERENCES)
    away_power = away_conference.isin(POWER_CONFERENCES)
    return np.select(
        [home_power & away_power, home_power | away_power],
        ["Power vs. power", "Power vs. Group of Five"],
        default="Group of Five vs. Group of Five",
    )


def _segment(frame: pd.DataFrame, mask: np.ndarray, probabilities: dict[str, np.ndarray], sim_margin: np.ndarray, sim_total: np.ndarray) -> dict:
    outcomes = frame["home_win"].to_numpy()[mask]
    margin = (frame["home_points"] - frame["away_points"]).to_numpy()[mask]
    total = (frame["home_points"] + frame["away_points"]).to_numpy()[mask]
    return {
        **{name: score(outcomes, values[mask]) for name, values in probabilities.items()},
        "margin": point_errors(margin, sim_margin[mask]),
        "total": point_errors(total, sim_total[mask]),
    }


def evaluate(
    games: pd.DataFrame,
    stats: dict,
    regulation: pd.DataFrame,
    settings: dict[str, Any],
    seasons: tuple[int, ...],
) -> dict[str, Any]:
    """Fit on 2016-20 with frozen settings, then score the given later seasons."""
    if set(seasons) & set(TUNE_SEASONS):
        raise FinalEvaluationError("Scored seasons must come after the training seasons.")
    elo, _ = run_elo(games, EloParams(**settings["elo_params"]))
    frame = scored_frame(games, elo, stats, FeatureParams(**settings["score_feature_params"]), regulation)
    frame = frame.merge(stats["lines"][["game_id", "home_spread", "over_under"]], on="game_id", how="left")
    frame["cfbd_diff"] = frame["cfbd_home_elo"] - frame["cfbd_away_elo"]
    train = frame[frame["season"].isin(TUNE_SEASONS)]
    stack = fit_stack(train, settings["alpha"], settings["bandwidth"])

    win_frame = scored_frame(games, elo, stats, FeatureParams(**settings["win_feature_params"]), regulation)
    win_train = win_frame[win_frame["season"].isin(TUNE_SEASONS)]
    win_model = WinModel(ALL_FEATURES, settings["win_C"]).fit(win_train)
    elo_model = EloCalibrator().fit(win_train)
    cfbd_model = EloCalibrator().fit(train.dropna(subset=["cfbd_diff"]), "cfbd_diff")
    spread_model = SpreadModel().fit(train.dropna(subset=["home_spread"]))

    block = frame[frame["season"].isin(seasons)].reset_index(drop=True)
    win_block = win_frame[win_frame["season"].isin(seasons)].reset_index(drop=True)
    if not (block["game_id"].to_numpy() == win_block["game_id"].to_numpy()).all():
        raise FinalEvaluationError("Win model and score model rows are misaligned.")
    home_expected, away_expected = stack.expected(block)
    sim = simulate_block(block, home_expected, away_expected, stack.simulator.simulate)

    outcomes = block["home_win"].to_numpy()
    probabilities = {
        "simulation": np.clip(sim["probability"], 1e-3, 1 - 1e-3),
        "stats_model": win_model.predict(win_block),
        "elo": elo_model.predict(win_block),
    }
    actual_margin = (block["home_points"] - block["away_points"]).to_numpy()
    actual_total = (block["home_points"] + block["away_points"]).to_numpy()
    sim_margin, sim_total = sim["means"]["margin"], sim["means"]["total"]

    seasons_array = block["season"].to_numpy()
    by_season = {
        int(season): _segment(block, seasons_array == season, probabilities, sim_margin, sim_total)
        for season in seasons
    }
    for season in seasons:
        mask = seasons_array == season
        by_season[int(season)]["team_points"] = point_errors(
            np.concatenate([block["home_points"].to_numpy()[mask], block["away_points"].to_numpy()[mask]]),
            np.concatenate([sim["means"]["home"][mask], sim["means"]["away"][mask]]),
        )

    is_early = (block["season_type"].eq("regular") & (block["week"] <= EARLY_LAST_WEEK)).to_numpy()
    is_post = block["season_type"].eq("postseason").to_numpy()
    is_conference = block["conference_game"].fillna(False).astype(bool).to_numpy()
    is_neutral = block["neutral_site"].to_numpy(dtype=bool)
    has_gap = (block["home_returning"].isna() | block["away_returning"].isna()).to_numpy()
    game_type = _game_type(block["home_conference"], block["away_conference"])
    confidence = np.maximum(probabilities["simulation"], 1 - probabilities["simulation"])
    segments_masks = {
        "Weeks 1-4": is_early,
        "Week 5 through regular season": ~is_early & ~is_post,
        "Postseason": is_post,
        "Conference games": is_conference,
        "Non-conference games": ~is_conference,
        "True home games": ~is_neutral,
        "Neutral sites": is_neutral,
        "Power vs. power": game_type == "Power vs. power",
        "Power vs. Group of Five": game_type == "Power vs. Group of Five",
        "Group of Five vs. Group of Five": game_type == "Group of Five vs. Group of Five",
        "Team missing returning data": has_gap,
        "Favorite under 60%": confidence < 0.6,
        "Favorite 60-80%": (confidence >= 0.6) & (confidence < 0.8),
        "Favorite 80% or more": confidence >= 0.8,
    }

    report: dict[str, Any] = {
        "seasons": list(seasons),
        "games": len(block),
        "training_games": len(train),
        "win_probability": {name: score(outcomes, values) for name, values in probabilities.items()},
        "baselines": {
            "coin_flip": score(outcomes, np.full(len(block), 0.5)),
            "home_rate": score(
                outcomes,
                np.where(block["neutral_site"], 0.5, train.loc[~train["neutral_site"], "home_win"].mean()),
            ),
        },
        "edges": {
            "simulation_over_stats_model": dict(zip(("mean", "low", "high"), log_loss_edge_interval(
                outcomes, probabilities["simulation"], probabilities["stats_model"]))),
            "simulation_over_elo": dict(zip(("mean", "low", "high"), log_loss_edge_interval(
                outcomes, probabilities["simulation"], probabilities["elo"]))),
            "stats_model_over_elo": dict(zip(("mean", "low", "high"), log_loss_edge_interval(
                outcomes, probabilities["stats_model"], probabilities["elo"]))),
        },
        "calibration": {
            name: favorite_calibration(outcomes, values).to_dict("records") for name, values in probabilities.items()
        },
        "points": _point_comparison(block, train, sim),
        "by_season": by_season,
        "segments": {
            name: {"games": int(mask.sum()), **_segment(block, mask, probabilities, sim_margin, sim_total)}
            for name, mask in segments_masks.items()
            if mask.sum() > 0
        },
        "coverage": sim["coverage"],
        "outside_90": sim["outside_90"],
        "crps": sim["crps"],
        "realism": {
            "actual": _shape(block["home_points"].to_numpy(), block["away_points"].to_numpy()),
            "simulation": _shape(sim["pooled_home"], sim["pooled_away"]),
        },
        "overtime": {
            "actual_rate": float(block["overtime"].mean()),
            "simulated_rate": float(sim["overtime"].mean()),
        },
        "agreement": {
            "correlation": float(np.corrcoef(probabilities["simulation"], probabilities["stats_model"])[0, 1]),
            "mean_absolute_gap": float(np.mean(np.abs(probabilities["simulation"] - probabilities["stats_model"]))),
            "same_favorite_share": float(
                np.mean((probabilities["simulation"] > 0.5) == (probabilities["stats_model"] > 0.5))
            ),
        },
        "factor_importance": _factor_importance(stack, block),
        "largest_disagreements": _largest_gaps(block, probabilities["simulation"], probabilities["stats_model"]),
    }

    # Benchmarks only. Listed lines have unverified timing and never feed any model.
    has_cfbd = block["cfbd_diff"].notna().to_numpy()
    if has_cfbd.any():
        covered = block[has_cfbd]
        report["cfbd_elo_benchmark"] = {
            "games": int(has_cfbd.sum()),
            "cfbd_elo": score(outcomes[has_cfbd], cfbd_model.predict(covered, "cfbd_diff")),
            "simulation": score(outcomes[has_cfbd], probabilities["simulation"][has_cfbd]),
        }
    has_line = block["home_spread"].notna().to_numpy()
    if has_line.any():
        covered = block[has_line]
        listed = spread_model.predict(covered)
        report["listed_line_benchmark"] = {
            "games": int(has_line.sum()),
            "listed_line": score(outcomes[has_line], listed),
            "simulation": score(outcomes[has_line], probabilities["simulation"][has_line]),
            "simulation_edge": dict(zip(("mean", "low", "high"), log_loss_edge_interval(
                outcomes[has_line], probabilities["simulation"][has_line], listed))),
        }
    return report


def run_preflight(
    client: CFBDClient,
    directory: Path = FINAL_DIR,
    stage3_report: Path = STAGE3_REPORT,
) -> dict[str, Any]:
    """Run the evaluator on 2021-22 and require exact agreement with Stage 3.

    Loads only seasons through 2022, so this can't reveal anything about the holdout.
    """
    frozen = verify_manifest(directory, stage3_report=stage3_report)
    games, stats, regulation = load_inputs(client, list(DEVELOPMENT_SEASONS))
    report = evaluate(games, stats, regulation, frozen["settings"], VALIDATION_SEASONS)
    stage3 = json.loads(stage3_report.read_text())
    checks = {
        "simulation": report["win_probability"]["simulation"] == stage3["win_probability"]["simulation"],
        "stats_model": report["win_probability"]["stats_model"] == stage3["win_probability"]["stats_model"],
        "elo": report["win_probability"]["elo"] == stage3["win_probability"]["elo_calibrated"],
        "margin": report["points"]["margin"] == stage3["points"]["margin"],
        "total": report["points"]["total"] == stage3["points"]["total"],
        "team_points": report["points"]["team_points"] == stage3["points"]["team_points"],
        "coverage": report["coverage"] == stage3["coverage"]["simulation"],
    }
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "games": report["games"],
        "seasons": report["seasons"],
        "manifest_hash": _file_hash(directory / MANIFEST_NAME),
        "ran_at": datetime.now(timezone.utc).isoformat(),
    }
    path = directory / PREFLIGHT_NAME
    if path.exists():
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    path.write_text(json.dumps(result, indent=2))
    return result


def is_complete(directory: Path = FINAL_DIR) -> bool:
    """True only if a result exists and still matches its recorded checksum."""
    result, checksum = directory / RESULT_NAME, directory / CHECKSUM_NAME
    if not result.exists() or not checksum.exists():
        return False
    return _file_hash(result) == checksum.read_text().split()[0]


def run_once(
    phrase: str,
    loader: Callable[[], tuple[pd.DataFrame, dict, pd.DataFrame]],
    directory: Path = FINAL_DIR,
    sources: tuple[Path, ...] = FROZEN_SOURCES,
    stage3_report: Path = STAGE3_REPORT,
    raw_dir: Path = RAW_DIR,
    seasons: tuple[int, ...] = HOLDOUT_SEASONS,
    crashed_before_results: bool = False,
) -> dict[str, Any]:
    """The final evaluation. Every guard must pass before any holdout game is loaded."""
    if phrase != RUN_PHRASE:
        raise FinalEvaluationError(f'Confirm with the exact phrase: "{RUN_PHRASE}"')
    if (directory / RESULT_NAME).exists():
        raise FinalEvaluationError("The final evaluation already ran. Its result is permanent; it won't run again.")
    frozen = verify_manifest(directory, sources, stage3_report, raw_dir)

    preflight_path = directory / PREFLIGHT_NAME
    if not preflight_path.exists():
        raise FinalEvaluationError("Run the preflight first: `cfbsim final-eval --preflight`.")
    preflight = json.loads(preflight_path.read_text())
    if not preflight["passed"] or preflight["manifest_hash"] != _file_hash(directory / MANIFEST_NAME):
        raise FinalEvaluationError("The preflight didn't pass for this freeze. The run is refused.")

    started_path = directory / STARTED_NAME
    attempts = []
    if started_path.exists():
        if not crashed_before_results:
            raise FinalEvaluationError(
                "A run already started and left no result. Results are only written at the very end, so a crash "
                "reveals no numbers. Investigate, then pass --crashed-before-results. The retry is recorded."
            )
        attempts = json.loads(started_path.read_text())["attempts"]
        started_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    attempts.append(datetime.now(timezone.utc).isoformat())
    started_path.write_text(json.dumps({"attempts": attempts}, indent=2))

    games, stats, regulation = loader()
    report = evaluate(games, stats, regulation, frozen["settings"], seasons)
    report["run_record"] = {
        "attempts": attempts,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "manifest_hash": _file_hash(directory / MANIFEST_NAME),
        "frozen_at": frozen["frozen_at"],
        "settings": frozen["settings"],
    }

    result_path, report_path = directory / RESULT_NAME, directory / REPORT_NAME
    result_path.write_text(json.dumps(report, indent=2))
    report_path.write_text(report_to_markdown(report))
    (directory / CHECKSUM_NAME).write_text(f"{_file_hash(result_path)}  {RESULT_NAME}\n")
    for path in (result_path, report_path, directory / CHECKSUM_NAME, started_path):
        _make_read_only(path)
    return report


def _score_table(rows: dict[str, dict]) -> list[str]:
    lines = ["| Forecast | Games | Accuracy | Log loss | Brier |", "|---|---:|---:|---:|---:|"]
    for name, row in rows.items():
        lines.append(
            f"| {name} | {row['games']:,} | {row['accuracy']:.1%} | {row['log_loss']:.4f} | {row['brier']:.4f} |"
        )
    return lines


def _error_table(rows: dict[str, dict]) -> list[str]:
    lines = ["| Forecast | Games | MAE | RMSE | Bias |", "|---|---:|---:|---:|---:|"]
    for name, row in rows.items():
        lines.append(
            f"| {name.replace('_', ' ')} | {row['games']:,} | {row['mae']:.2f} | {row['rmse']:.2f} | {row['bias']:+.2f} |"
        )
    return lines


def _interval(edge: dict) -> str:
    return f"{edge['mean']:+.4f} (95% interval {edge['low']:+.4f} to {edge['high']:+.4f})"


def report_to_markdown(report: dict[str, Any]) -> str:
    record = report["run_record"]
    settings = record["settings"]
    edges = report["edges"]
    lines = [
        "# Final evaluation",
        "",
        "**This file is permanent. It was generated once by `cfbsim final-eval` and is read-only.**",
        "",
        f"- Scored on {report['games']:,} FBS vs FBS games from {report['seasons']}.",
        f"- Models were fit on {report['training_games']:,} games from 2016-20 and never refit.",
        "- Team ratings and stats updated game by game, using only games played before each forecast.",
        f"- Settings were frozen at {record['frozen_at']}, before any of these games were scored.",
        f"- Completed at {record['completed_at']}, using {len(record['attempts'])} run attempt(s).",
        f"- Frozen settings: `{settings}`",
        "",
        "Data provided by CollegeFootballData.com. Listed betting lines are benchmarks only, never model "
        "inputs, and their timing is unverified, so they are not described as closing lines.",
        "",
        "## Win probability",
        "",
        *_score_table({**report["win_probability"], **{f"baseline: {k}": v for k, v in report["baselines"].items()}}),
        "",
        "Log-loss edges (above zero means the first model is better):",
        "",
        f"- Simulation over stats model: {_interval(edges['simulation_over_stats_model'])}",
        f"- Simulation over Elo: {_interval(edges['simulation_over_elo'])}",
        f"- Stats model over Elo: {_interval(edges['stats_model_over_elo'])}",
    ]
    if "cfbd_elo_benchmark" in report:
        block = report["cfbd_elo_benchmark"]
        lines += ["", f"CFBD's own Elo on {block['games']:,} games: log loss {block['cfbd_elo']['log_loss']:.4f}, "
                  f"accuracy {block['cfbd_elo']['accuracy']:.1%}. Simulation on those games: "
                  f"{block['simulation']['log_loss']:.4f}, {block['simulation']['accuracy']:.1%}."]
    if "listed_line_benchmark" in report:
        block = report["listed_line_benchmark"]
        lines += ["", f"Listed betting line (timing unverified) on {block['games']:,} games: log loss "
                  f"{block['listed_line']['log_loss']:.4f}. Simulation on those games: "
                  f"{block['simulation']['log_loss']:.4f}. Simulation edge: {_interval(block['simulation_edge'])}."]
    agreement = report["agreement"]
    lines += [
        "",
        f"Simulation and stats model agreement: correlation {agreement['correlation']:.3f}, average gap "
        f"{agreement['mean_absolute_gap']:.1%}, same favorite in {agreement['same_favorite_share']:.1%} of games.",
        "",
        "## Calibration from the favorite's side",
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
    lines += ["", "## Score forecasts"]
    for target, rows in report["points"].items():
        lines += ["", f"### {target.replace('_', ' ').title()}", "", *_error_table(rows)]
    lines += ["", "## By season", ""]
    lines += [
        "| Season | Model | Games | Accuracy | Log loss | Brier |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for season, block in report["by_season"].items():
        for model in ("simulation", "stats_model", "elo"):
            row = block[model]
            lines.append(
                f"| {season} | {model} | {row['games']:,} | {row['accuracy']:.1%} | {row['log_loss']:.4f} | {row['brier']:.4f} |"
            )
    lines += ["", "| Season | Target | Games | MAE | RMSE | Bias |", "|---|---|---:|---:|---:|---:|"]
    for season, block in report["by_season"].items():
        for target in ("team_points", "margin", "total"):
            row = block[target]
            lines.append(
                f"| {season} | {target.replace('_', ' ')} | {row['games']:,} | {row['mae']:.2f} | {row['rmse']:.2f} | {row['bias']:+.2f} |"
            )
    lines += [
        "",
        "## Where the model does well and struggles",
        "",
        "| Group | Games | Sim accuracy | Sim log loss | Stats log loss | Elo log loss | Margin MAE | Total MAE | Total bias |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, block in report["segments"].items():
        lines.append(
            f"| {name} | {block['games']:,} | {block['simulation']['accuracy']:.1%} "
            f"| {block['simulation']['log_loss']:.4f} | {block['stats_model']['log_loss']:.4f} "
            f"| {block['elo']['log_loss']:.4f} | {block['margin']['mae']:.2f} "
            f"| {block['total']['mae']:.2f} | {block['total']['bias']:+.2f} |"
        )
    coverage = report["coverage"]
    outside = report["outside_90"]
    lines += [
        "",
        "## Do the simulated ranges hold up?",
        "",
        "| Interval | Team points | Margin | Total |",
        "|---|---:|---:|---:|",
    ]
    for level in coverage["team"]:
        lines.append(
            f"| {level} | {coverage['team'][level]:.1%} | {coverage['margin'][level]:.1%} | {coverage['total'][level]:.1%} |"
        )
    lines += [
        "",
        f"Outside the 90% range on totals: {outside['total']['below']:.1%} below and "
        f"{outside['total']['above']:.1%} above. On margins: {outside['margin']['below']:.1%} below and "
        f"{outside['margin']['above']:.1%} above. About 5% each way is ideal.",
        "",
        "## Is the distribution realistic?",
        "",
        "| Statistic | Actual | Simulation |",
        "|---|---:|---:|",
    ]
    actual, simulated = report["realism"]["actual"], report["realism"]["simulation"]
    for key, label in (
        ("share_shutout", "Team shut out"),
        ("share_team_50_plus", "Team scores 50+"),
        ("share_total_30_or_less", "Total 30 or less"),
        ("share_total_80_plus", "Total 80+"),
        ("share_margin_28_plus", "Margin 28+"),
        ("share_margin_within_3", "Margin within 3"),
        ("share_margin_exactly_3_or_7", "Margin exactly 3 or 7"),
        ("share_rare_scores_1_2_4_5_8", "Rare scores (1, 2, 4, 5, 8)"),
    ):
        lines.append(f"| {label} | {actual[key]:.1%} | {simulated[key]:.1%} |")
    for key, label in (("team_points", "Team points"), ("total", "Total")):
        for stat_name in ("mean", "sd"):
            lines.append(f"| {label}, {stat_name} | {actual[key][stat_name]:.1f} | {simulated[key][stat_name]:.1f} |")
    overtime = report["overtime"]
    lines += [
        f"| Overtime rate | {overtime['actual_rate']:.1%} | {overtime['simulated_rate']:.1%} |",
        "",
        "## Factors behind projected margins",
        "",
        "| Factor | Average push (points) |",
        "|---|---:|",
    ]
    lines += [f"| {name.replace('_', ' ')} | {value:.2f} |" for name, value in report["factor_importance"].items()]
    return "\n".join(lines) + "\n"
