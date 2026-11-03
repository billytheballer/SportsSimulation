"""Command line entry point: `cfbsim fetch`, `cfbsim backtest`, `cfbsim usage`."""

from __future__ import annotations

import argparse
import json
import sys

from cfbsim import final_eval, production, stage2, stage3
from cfbsim.backtest import report_to_markdown, run_backtest
from cfbsim.bundle import DEMO_BUNDLE, PRODUCTION_BUNDLE, build_demo_bundle, save_bundle
from cfbsim.client import CFBDClient, CFBDError
from cfbsim.config import DEVELOPMENT_SEASONS, HISTORICAL_SEASONS, REPORTS_DIR
from cfbsim.games import SEASON_TYPES, fetch_season_records, load_games, parse_regulation
from cfbsim.matchup import format_matchup, simulate_matchup
from cfbsim.stats import fetch_plan, load_stats


def _fetch(args: argparse.Namespace) -> int:
    client = CFBDClient()
    seasons = list(HISTORICAL_SEASONS) + args.season
    needed = sum(
        not client.is_cached("games", year=season, seasonType=kind, classification="fbs")
        or season in args.season
        for season in seasons
        for kind in SEASON_TYPES
    )
    print(f"{needed} API calls needed; {client.calls_this_month()} used this month.")
    if args.dry_run:
        return 0
    for season in seasons:
        records = fetch_season_records(client, season, refresh=season in args.season)
        print(f"  {season}: {len(records):,} games")
    print(f"Done. {client.calls_this_month()} calls used this month.")
    return 0


def _fetch_stats(args: argparse.Namespace) -> int:
    client = CFBDClient()
    seasons = list(HISTORICAL_SEASONS)
    games, _ = load_games(client, seasons)
    plan = fetch_plan(games, seasons)
    needed = [(endpoint, params) for endpoint, params in plan if not client.is_cached(endpoint, **params)]
    print(f"{len(needed)} of {len(plan)} API calls needed; {client.calls_this_month()} used this month.")
    if args.dry_run:
        return 0
    for endpoint, params in needed:
        client.get(endpoint, **params)
    print(f"Done. {client.calls_this_month()} calls used this month.")
    return 0


def _backtest(args: argparse.Namespace) -> int:
    # Holdout games are not loaded here. Only `cfbsim final-eval` may score them.
    games, quality = load_games(CFBDClient(), list(DEVELOPMENT_SEASONS))
    print(f"Read {quality.rows_read:,} rows, kept {quality.games_kept:,}. Dropped: {quality.dropped}")
    report = run_backtest(games)
    report["data_quality"] = {
        "rows_read": quality.rows_read,
        "games_kept": quality.games_kept,
        "dropped": quality.dropped,
    }
    REPORTS_DIR.mkdir(exist_ok=True)
    markdown = report_to_markdown(report)
    (REPORTS_DIR / "stage1_backtest.md").write_text(markdown)
    (REPORTS_DIR / "stage1_backtest.json").write_text(json.dumps(report, indent=2))
    print(markdown)
    return 0


def _stage3(_: argparse.Namespace) -> int:
    client = CFBDClient()
    seasons = list(DEVELOPMENT_SEASONS)
    games, _ = load_games(client, seasons)
    records = [record for season in seasons for record in fetch_season_records(client, season)]
    regulation, quality = parse_regulation(records)
    quality_summary = {"rows_read": quality.rows_read, "games_kept": quality.games_kept, "dropped": quality.dropped}
    report = stage3.run_stage3(games, load_stats(client, games, seasons), regulation, quality_summary)
    REPORTS_DIR.mkdir(exist_ok=True)
    markdown = stage3.report_to_markdown(report)
    (REPORTS_DIR / "stage3_diagnostics.md").write_text(markdown)
    (REPORTS_DIR / "stage3_diagnostics.json").write_text(json.dumps(report, indent=2))
    print(markdown)
    return 0


def _simulate(args: argparse.Namespace) -> int:
    try:
        result = simulate_matchup(args.home, args.away, args.neutral, args.simulations, args.seed)
    except (ValueError, FileNotFoundError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(format_matchup(result))
    return 0


def _final_eval(args: argparse.Namespace) -> int:
    try:
        if args.freeze:
            manifest = final_eval.freeze()
            print(f"Frozen {len(manifest['source_hashes'])} source files, the settings, and the cached data.")
            return 0
        client = CFBDClient()
        if args.preflight:
            result = final_eval.run_preflight(client)
            print(json.dumps({key: result[key] for key in ("passed", "checks", "games", "seasons")}, indent=2))
            return 0 if result["passed"] else 1
        report = final_eval.run_once(
            args.confirm or "",
            lambda: final_eval.load_inputs(client, list(HISTORICAL_SEASONS)),
            crashed_before_results=args.crashed_before_results,
        )
        print(f"Final evaluation complete on {report['games']:,} games. See reports/final_evaluation/.")
        return 0
    except final_eval.FinalEvaluationError as error:
        print(f"Refused: {error}", file=sys.stderr)
        return 1


def _update_season(_: argparse.Namespace) -> int:
    summary = production.update_season(CFBDClient())
    print(json.dumps(summary, indent=2))
    return 0


def _build_production(args: argparse.Namespace) -> int:
    bundle = production.build_production_bundle(CFBDClient())
    if args.public:
        bundle = production.public_copy(bundle)
    path = save_bundle(bundle, PRODUCTION_BUNDLE)
    print(f"Saved {path}" + (" (public copy, without game scores or schedule)" if args.public else ""))
    print(f"Team data current through {bundle.data_through:%B %-d, %Y} ({bundle.data_note}).")
    print(f"Upcoming games listed: {len(bundle.upcoming)}. This season so far: {json.dumps(bundle.live_scorecard, indent=2)}")
    return 0


def _build_demo(_: argparse.Namespace) -> int:
    path = save_bundle(build_demo_bundle(CFBDClient()), DEMO_BUNDLE)
    print(f"Saved {path}")
    return 0


def _stage2(args: argparse.Namespace) -> int:
    client = CFBDClient()
    seasons = list(DEVELOPMENT_SEASONS)
    games, _ = load_games(client, seasons)
    report = stage2.run_stage2(games, load_stats(client, games, seasons))
    REPORTS_DIR.mkdir(exist_ok=True)
    markdown = stage2.report_to_markdown(report)
    (REPORTS_DIR / "stage2_backtest.md").write_text(markdown)
    (REPORTS_DIR / "stage2_backtest.json").write_text(json.dumps(report, indent=2))
    print(markdown)
    return 0


def _usage(_: argparse.Namespace) -> int:
    client = CFBDClient()
    print(f"{client.calls_this_month()} of {client.monthly_budget} budgeted calls used this month.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cfbsim", description=__doc__)
    commands = parser.add_subparsers(required=True)

    fetch = commands.add_parser("fetch", help="download and cache game data")
    fetch.add_argument("--season", type=int, action="append", default=[],
                       help="also fetch this season, replacing any cached copy (for a live season)")
    fetch.add_argument("--dry-run", action="store_true", help="show the call count and stop")
    fetch.set_defaults(run=_fetch)

    fetch_stats = commands.add_parser("fetch-stats", help="download and cache Stage 2 team data")
    fetch_stats.add_argument("--dry-run", action="store_true", help="show the call count and stop")
    fetch_stats.set_defaults(run=_fetch_stats)

    backtest = commands.add_parser("backtest", help="tune and score on past seasons")
    backtest.set_defaults(run=_backtest)

    stage2_parser = commands.add_parser("stage2", help="train and score the team stats model")
    stage2_parser.set_defaults(run=_stage2)

    stage3_parser = commands.add_parser(
        "stage3", help="fit the score model and test the simulation on 2021-22; never touches 2023-25"
    )
    stage3_parser.set_defaults(run=_stage3)

    simulate = commands.add_parser(
        "simulate", help="simulate one matchup using team data through the end of 2022"
    )
    simulate.add_argument("home", help='home team, for example "Ohio State"')
    simulate.add_argument("away", help="away team")
    simulate.add_argument("--neutral", action="store_true", help="play at a neutral site")
    simulate.add_argument("--simulations", type=int, default=10_000, help="number of simulated games")
    simulate.add_argument("--seed", type=int, default=2026, help="random seed, for repeatable results")
    simulate.set_defaults(run=_simulate)

    final = commands.add_parser(
        "final-eval", help="the one-time final evaluation on 2023-25: freeze, preflight, then run once"
    )
    final.add_argument("--freeze", action="store_true", help="save settings and fingerprints; never replaced")
    final.add_argument("--preflight", action="store_true", help="check the pipeline on 2021-22 only")
    final.add_argument("--confirm", help=f'run for real, with the exact phrase "{final_eval.RUN_PHRASE}"')
    final.add_argument("--crashed-before-results", action="store_true",
                       help="allow a recorded retry after a crash that produced no results")
    final.set_defaults(run=_final_eval)

    update_season = commands.add_parser(
        "update-season", help="download newly completed 2026 games; needs the final evaluation done first"
    )
    update_season.set_defaults(run=_update_season)

    build_production = commands.add_parser(
        "build-production", help="save the 2026 production model for the app"
    )
    build_production.add_argument("--public", action="store_true",
                                  help="leave out this season's scores and schedule, for publishing")
    build_production.set_defaults(run=_build_production)

    build_demo = commands.add_parser(
        "build-demo", help="save the Stage 3 model with data through 2022 for the app"
    )
    build_demo.set_defaults(run=_build_demo)

    usage = commands.add_parser("usage", help="show API calls counted this month")
    usage.set_defaults(run=_usage)

    args = parser.parse_args(argv)
    try:
        return args.run(args)
    except (CFBDError, production.ProductionError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
