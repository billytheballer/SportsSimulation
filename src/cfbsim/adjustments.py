"""An adjustment layer for ESPN FPI and official player availability reports.

No pre-game history exists for FPI or availability reports, so neither can be trained
into the evaluated model or tested on past seasons. This layer therefore sits on top of
the production model's expected points and is kept separate from it:

- Player impact is estimated from 2016-22 box scores: how far teams fell short of the
  model's forecast when players of a given production level didn't appear.
- The FPI weight and the chance a listed player sits are stated assumptions, not fits.

The final evaluation doesn't cover this layer. Forecasts are logged before kickoff so that
it can be tested on future games.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from cfbsim.client import CFBDClient
from cfbsim.config import PROJECT_ROOT, RAW_DIR, REPORTS_DIR, TUNE_SEASONS, VALIDATION_SEASONS

# Assumption: FPI gets a minority weight because nothing yet shows how much it adds.
FPI_WEIGHT = 0.25
# FPI older than this isn't used. ESPN says FPI updates daily.
FPI_MAX_AGE_DAYS = 9

# Assumption: the chance a player with each official designation doesn't play.
ABSENCE_CHANCE = {
    "out": 1.0,
    "out_first_half": 0.5,
    "doubtful": 0.8,
    "game_time_decision": 0.5,
    "questionable": 0.5,
    "probable": 0.1,
    "available": 0.0,
}
# Official conference sites, plus school athletics sites as they are verified.
OFFICIAL_DOMAINS = {
    "bigten.org", "secsports.com", "theacc.com", "big12sports.com", "themw.com", "sunbeltsports.org",
    "pac-12.com", "broncosports.com",
}

REPORTS_PATH = PROJECT_ROOT / "data" / "availability" / "reports.csv"
FPI_SNAPSHOT_DIR = PROJECT_ROOT / "data" / "snapshots" / "fpi"
FORECAST_LOG_DIR = PROJECT_ROOT / "data" / "forecast_log"
# Kept apart from data/raw so the final evaluation's data fingerprint stays valid.
PLAYER_HISTORY_DIR = PROJECT_ROOT / "data" / "player_history"
IMPACT_PATH = REPORTS_DIR / "availability_impact.json"

MINIMUM_GAMES = 3
# Past this many straight missed games, the team's running stats already reflect the absence.
MAXIMUM_PRIOR_MISSES = 2
# Box scores list only players with stats, so a low-volume player can look absent after a
# quiet game. Higher thresholds reduce that false signal; the estimate stays conservative too.
MINIMUM_SKILL_YARDS = 70.0
STARTER_PASS_SHARE = 0.5
REPORT_FIELDS = ["team", "player", "status", "game_date", "reported_at", "source_url"]


class AvailabilityError(ValueError):
    """An availability report is incomplete or doesn't come from an official source."""


def normalize_name(name: str) -> str:
    cleaned = re.sub(r"[^a-z ]", "", name.lower())
    return " ".join(word for word in cleaned.split() if word not in {"jr", "sr", "ii", "iii", "iv"})


def _number(text: Any) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return 0.0


def parse_player_box(records: list[dict[str, Any]]) -> dict[tuple[int, str], dict[str, dict[str, Any]]]:
    """Each player's passing, rushing, and receiving line, by (game, team) and player id."""
    games: dict[tuple[int, str], dict[str, dict[str, Any]]] = {}
    for game in records:
        for team in game.get("teams", []):
            players = games.setdefault((game["id"], team["team"]), {})
            for category in team.get("categories", []):
                name = category.get("name")
                if name not in ("passing", "rushing", "receiving"):
                    continue
                for stat in category.get("types", []):
                    for athlete in stat.get("athletes", []):
                        player_id = str(athlete.get("id"))
                        label = str(athlete.get("name", ""))
                        if player_id in ("None", "-9999") or label.strip().lower() == "team":
                            continue
                        line = players.setdefault(
                            player_id, {"name": label, "attempts": 0.0, "pass_yards": 0.0, "other_yards": 0.0}
                        )
                        if name == "passing" and stat.get("name") == "C/ATT":
                            parts = str(athlete.get("stat", "")).split("/")
                            line["attempts"] = _number(parts[1]) if len(parts) == 2 else 0.0
                        elif stat.get("name") == "YDS":
                            key = "pass_yards" if name == "passing" else "other_yards"
                            line[key] += _number(athlete.get("stat"))
    return games


@dataclass
class _Player:
    name: str
    games: int = 0
    attempts: float = 0.0
    pass_yards: float = 0.0
    other_yards: float = 0.0
    missed_in_a_row: int = 0

    def yards_per_game(self) -> float:
        return (self.pass_yards + self.other_yards) / max(self.games, 1)

    def skill_yards_per_game(self) -> float:
        return self.other_yards / max(self.games, 1)


@dataclass(frozen=True)
class PlayerValue:
    name: str
    role: str
    yards_per_game: float
    games: int
    missed_in_a_row: int


class _TeamLedger:
    """A team's production so far this season. Read before a game, updated after it."""

    def __init__(self) -> None:
        self.players: dict[str, _Player] = {}
        self.team_attempts = 0.0

    def starter(self) -> str | None:
        passers = [(player.attempts, key) for key, player in self.players.items() if player.attempts > 0]
        if not passers or self.team_attempts == 0:
            return None
        attempts, key = max(passers)
        return key if attempts / self.team_attempts >= STARTER_PASS_SHARE else None

    def role_of(self, key: str) -> str | None:
        """'qb' or 'skill' for a player whose absence can be valued, otherwise None."""
        player = self.players[key]
        if player.games < MINIMUM_GAMES:
            return None
        if key == self.starter():
            return "qb"
        if player.attempts == 0 and player.skill_yards_per_game() >= MINIMUM_SKILL_YARDS:
            return "skill"
        return None

    def lost_yards(self, played: set[str]) -> dict[str, float]:
        """Per-game yards of valued players who didn't appear in a game."""
        lost = {"qb": 0.0, "skill": 0.0}
        for key, player in self.players.items():
            if key in played or player.missed_in_a_row > MAXIMUM_PRIOR_MISSES:
                continue
            role = self.role_of(key)
            if role == "qb":
                lost["qb"] += player.yards_per_game()
            elif role == "skill":
                lost["skill"] += player.skill_yards_per_game()
        return lost

    def update(self, lines: dict[str, dict[str, Any]]) -> None:
        for key, line in lines.items():
            player = self.players.setdefault(key, _Player(line["name"]))
            player.games += 1
            player.attempts += line["attempts"]
            player.pass_yards += line["pass_yards"]
            player.other_yards += line["other_yards"]
            player.missed_in_a_row = 0
            self.team_attempts += line["attempts"]
        for key, player in self.players.items():
            if key not in lines:
                player.missed_in_a_row += 1

    def values(self) -> dict[str, PlayerValue]:
        result = {}
        for key, player in self.players.items():
            role = self.role_of(key)
            if role:
                yards = player.yards_per_game() if role == "qb" else player.skill_yards_per_game()
                result[normalize_name(player.name)] = PlayerValue(
                    player.name, role, round(yards, 1), player.games, player.missed_in_a_row
                )
        return result


def walk_absences(games: pd.DataFrame, box: dict) -> pd.DataFrame:
    """Per game, the production of valued players who didn't appear, in kickoff order.

    A game with no box score leaves every ledger unchanged. Used only to estimate impact:
    who actually played is known only after the game.
    """
    ledgers: dict[tuple[int, str], _TeamLedger] = {}
    rows = []
    for game_id, season, home, away in zip(games["game_id"], games["season"], games["home_team"], games["away_team"]):
        row = {"game_id": game_id}
        has_box = (game_id, home) in box and (game_id, away) in box
        for side, team in (("home", home), ("away", away)):
            ledger = ledgers.setdefault((season, team), _TeamLedger())
            lines = box.get((game_id, team), {})
            lost = ledger.lost_yards(set(lines)) if has_box else {"qb": 0.0, "skill": 0.0}
            row[f"{side}_lost_qb"], row[f"{side}_lost_skill"] = lost["qb"], lost["skill"]
        if has_box:
            ledgers[(season, home)].update(box[(game_id, home)])
            ledgers[(season, away)].update(box[(game_id, away)])
        row["has_box"] = has_box
        rows.append(row)
    return pd.DataFrame(rows)


def current_player_values(games: pd.DataFrame, box: dict, season: int) -> dict[str, dict[str, PlayerValue]]:
    """Every team's valued players, by normalized name, from completed games this season."""
    ledgers: dict[str, _TeamLedger] = {}
    current = games[games["season"] == season]
    for game_id, home, away in zip(current["game_id"], current["home_team"], current["away_team"]):
        for team in (home, away):
            if (game_id, team) in box:
                ledgers.setdefault(team, _TeamLedger()).update(box[(game_id, team)])
    return {team: ledger.values() for team, ledger in ledgers.items()}


def impact_design(absences: pd.DataFrame) -> np.ndarray:
    """Two columns, each the home team's lost yards minus the away team's, per 100 yards."""
    return np.column_stack(
        [
            (absences["home_lost_qb"] - absences["away_lost_qb"]) / 100.0,
            (absences["home_lost_skill"] - absences["away_lost_skill"]) / 100.0,
        ]
    )


def fit_impact_coefficients(errors: np.ndarray, design: np.ndarray) -> np.ndarray:
    """Points of margin per 100 lost yards, floored at zero: losing a starter never helps."""
    intercept = np.column_stack([np.ones(len(errors)), design])
    solution, *_ = np.linalg.lstsq(intercept, errors, rcond=None)
    return -solution[1:]


def bootstrap_interval(errors: np.ndarray, design: np.ndarray, resamples: int = 1000, seed: int = 0) -> list[list[float]]:
    rng = np.random.default_rng(seed)
    draws = np.array(
        [
            fit_impact_coefficients(errors[index], design[index])
            for index in (rng.integers(0, len(errors), len(errors)) for _ in range(resamples))
        ]
    )
    return [np.percentile(draws[:, column], [2.5, 97.5]).tolist() for column in range(design.shape[1])]


def _history_client() -> CFBDClient:
    return CFBDClient(cache_dir=PLAYER_HISTORY_DIR, ledger_path=RAW_DIR / "_usage.json")


def fit_player_impact(client: CFBDClient, output: Path = IMPACT_PATH) -> dict[str, Any]:
    """Estimate player impact from 2016-22 box scores. Never loads 2023-25."""
    # Imported here so the app can load adjustments without the training code.
    from cfbsim.config import DEVELOPMENT_SEASONS
    from cfbsim.elo import EloParams, run_elo
    from cfbsim.features import FeatureParams
    from cfbsim.final_eval import FINAL_DIR, MANIFEST_NAME, load_inputs
    from cfbsim.stage3 import fit_stack, scored_frame

    seasons = list(DEVELOPMENT_SEASONS)
    games, stats, regulation = load_inputs(client, seasons)
    if games["season"].max() > max(VALIDATION_SEASONS):
        raise AvailabilityError("Player impact may only use seasons through 2022.")
    settings = json.loads((FINAL_DIR / MANIFEST_NAME).read_text())["settings"]

    history = _history_client()
    records: list[dict[str, Any]] = []
    for (season, season_type, week), _ in games.groupby(["season", "season_type", "week"]):
        if season in (*TUNE_SEASONS, *VALIDATION_SEASONS):
            records += history.get(
                "games/players", year=int(season), week=int(week), seasonType=season_type, classification="fbs"
            )
    box = parse_player_box(records)
    absences = walk_absences(games, box)

    elo, _ = run_elo(games, EloParams(**settings["elo_params"]))
    frame = scored_frame(games, elo, stats, FeatureParams(**settings["score_feature_params"]), regulation)
    stack = fit_stack(frame[frame["season"].isin(TUNE_SEASONS)], settings["alpha"], settings["bandwidth"])
    actual = frame.assign(actual_margin=frame["home_points"] - frame["away_points"])

    # Tune seasons use forecasts from a model that skipped that season, as the simulator does.
    tune = stack.fit_data[["game_id", "home_expected", "away_expected"]]
    later = actual[actual["season"].isin(VALIDATION_SEASONS)]
    home, away = stack.expected(later)
    later_expected = pd.DataFrame({"game_id": later["game_id"], "home_expected": home, "away_expected": away})
    expected = pd.concat([tune, later_expected])
    data = actual[["game_id", "season", "actual_margin"]].merge(expected, on="game_id").merge(absences, on="game_id")
    data = data[data["has_box"]].reset_index(drop=True)
    data["error"] = data["actual_margin"] - (data["home_expected"] - data["away_expected"])

    design = impact_design(data)
    errors = data["error"].to_numpy()
    is_later = data["season"].isin(VALIDATION_SEASONS).to_numpy()
    early = fit_impact_coefficients(errors[~is_later], design[~is_later])
    corrected = errors[is_later] + design[is_later] @ early
    coefficients = fit_impact_coefficients(errors, design)
    interval = bootstrap_interval(errors, design)
    # QB estimates hold steady across thresholds. Rusher and receiver estimates don't, and can be
    # inflated by the false signal above, so those use the low end of their interval.
    conservative_skill = max(interval[1][0], 0.0)
    report = {
        "seasons": sorted(int(season) for season in data["season"].unique()),
        "games": len(data),
        "games_with_a_valued_absence": int((np.abs(design).sum(axis=1) > 0).sum()),
        "points_per_100_lost_yards": {
            "qb": max(float(coefficients[0]), 0.0),
            "skill": float(conservative_skill),
        },
        "unfloored_estimates": {"qb": float(coefficients[0]), "skill": float(coefficients[1])},
        "interval_95": {"qb": interval[0], "skill": interval[1]},
        "check_on_2021_22": {
            "fit_on_2016_20": {"qb": float(early[0]), "skill": float(early[1])},
            "margin_mae_without": float(np.mean(np.abs(errors[is_later]))),
            "margin_mae_with_who_actually_played": float(np.mean(np.abs(corrected))),
            "note": "An upper limit, since it uses who actually played, known only after each game.",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    return report


def validate_report(row: dict[str, str]) -> dict[str, Any]:
    """Accept a report only if it's complete and comes from an official source."""
    missing = [name for name in REPORT_FIELDS if not str(row.get(name, "")).strip()]
    if missing:
        raise AvailabilityError(f"Report is missing {', '.join(missing)}.")
    status = row["status"].strip().lower().replace(" ", "_").replace("-", "_").replace("(1st_half)", "first_half")
    status = status.replace("out_1st_half", "out_first_half")
    if status not in ABSENCE_CHANCE:
        raise AvailabilityError(f"Unknown status '{row['status']}'. Use one of: {', '.join(ABSENCE_CHANCE)}.")
    parts = urlparse(row["source_url"].strip())
    host = (parts.hostname or "").lower()
    official = host.endswith(".edu") or any(host == d or host.endswith(f".{d}") for d in OFFICIAL_DOMAINS)
    if parts.scheme != "https" or not official:
        raise AvailabilityError(
            f"'{row['source_url']}' isn't a recognized official source. Use a conference or school athletics page."
        )
    try:
        game_date = date.fromisoformat(row["game_date"].strip())
        reported_at = datetime.fromisoformat(row["reported_at"].strip())
    except ValueError as error:
        raise AvailabilityError(f"Bad date in report for {row['player']}: {error}") from None
    if reported_at.tzinfo is None:
        raise AvailabilityError(f"reported_at for {row['player']} needs a time zone.")
    if reported_at.date() > game_date + timedelta(days=1):
        raise AvailabilityError(f"Report for {row['player']} is dated after its game, so it can't be used.")
    return {
        "team": row["team"].strip(),
        "player": row["player"].strip(),
        "status": status,
        "game_date": game_date,
        "reported_at": reported_at,
        "source_url": row["source_url"].strip(),
    }


def load_reports(path: Path = REPORTS_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return [validate_report(row) for row in csv.DictReader(handle)]


def add_report(row: dict[str, str], path: Path = REPORTS_PATH) -> dict[str, Any]:
    report = validate_report(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({name: row[name] for name in REPORT_FIELDS})
    return report


def active_reports(reports: list[dict[str, Any]], data_through: date, today: date) -> dict[str, list[dict[str, Any]]]:
    """Latest report per player for each team's next game. Old games' reports don't carry over."""
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for report in reports:
        if not today <= report["game_date"] <= today + timedelta(days=7):
            continue
        if report["game_date"] <= data_through:
            continue
        key = (report["team"], normalize_name(report["player"]))
        if key not in latest or report["reported_at"] > latest[key]["reported_at"]:
            latest[key] = report
    by_team: dict[str, list[dict[str, Any]]] = {}
    for (team, _), report in sorted(latest.items()):
        by_team.setdefault(team, []).append(report)
    return by_team


@dataclass
class AdjustmentInputs:
    """What the prediction service needs to apply this layer."""

    fpi: dict[str, float]
    fpi_as_of: date | None
    fpi_weight: float
    reports: dict[str, list[dict[str, Any]]]
    player_values: dict[str, dict[str, PlayerValue]]
    points_per_100_yards: dict[str, float]
    notes: list[str] = field(default_factory=list)


def team_availability(team: str, inputs: AdjustmentInputs) -> tuple[float, list[dict[str, Any]]]:
    """Expected points a team loses to listed absences, plus a row for each listed player."""
    total = 0.0
    rows = []
    values = inputs.player_values.get(team, {})
    for report in inputs.reports.get(team, []):
        value = values.get(normalize_name(report["player"]))
        chance = ABSENCE_CHANCE[report["status"]]
        points = 0.0
        note = "No estimated effect: the model values only starting QBs and top rushers and receivers"
        if value is not None:
            if value.missed_in_a_row > MAXIMUM_PRIOR_MISSES:
                note = "No added effect: this absence is already in the team's recent stats"
            else:
                rate = inputs.points_per_100_yards[value.role]
                points = chance * rate * value.yards_per_game / 100.0
                note = f"{value.yards_per_game:.0f} yards per game"
        total += points
        rows.append(
            {
                "team": team,
                "player": report["player"],
                "status": report["status"].replace("_", " ").title(),
                "points": points,
                "note": note,
                "source_url": report["source_url"],
                "reported_at": report["reported_at"],
            }
        )
    return total, rows


def fpi_is_fresh(inputs: AdjustmentInputs, today: date) -> bool:
    return inputs.fpi_as_of is not None and (today - inputs.fpi_as_of).days <= FPI_MAX_AGE_DAYS


def apply_adjustments(
    home: str,
    away: str,
    home_expected: float,
    away_expected: float,
    home_field: float,
    inputs: AdjustmentInputs,
    today: date,
) -> dict[str, Any]:
    """Move expected points for availability, then blend the margin toward FPI.

    Lost points come off the listed player's own team. The FPI blend moves the margin
    only, so it leaves the projected total unchanged.
    """
    home_lost, home_rows = team_availability(home, inputs)
    away_lost, away_rows = team_availability(away, inputs)
    adjusted_home, adjusted_away = home_expected - home_lost, away_expected - away_lost

    use_fpi = fpi_is_fresh(inputs, today) and home in inputs.fpi and away in inputs.fpi
    fpi_margin = None
    fpi_shift = 0.0
    if use_fpi:
        fpi_margin = inputs.fpi[home] - inputs.fpi[away] + home_field
        fpi_shift = inputs.fpi_weight * (fpi_margin - (adjusted_home - adjusted_away))
        adjusted_home += fpi_shift / 2
        adjusted_away -= fpi_shift / 2
    return {
        "home_expected": max(adjusted_home, 0.0),
        "away_expected": max(adjusted_away, 0.0),
        "availability_margin": away_lost - home_lost,
        "fpi_used": use_fpi,
        "fpi_margin": fpi_margin,
        "fpi_margin_shift": fpi_shift,
        "fpi_home": inputs.fpi.get(home),
        "fpi_away": inputs.fpi.get(away),
        "fpi_as_of": inputs.fpi_as_of,
        "fpi_weight": inputs.fpi_weight,
        "players": home_rows + away_rows,
        "home_has_reports": bool(inputs.reports.get(home)),
        "away_has_reports": bool(inputs.reports.get(away)),
    }


def save_fpi_snapshot(rows: list[dict[str, Any]], directory: Path = FPI_SNAPSHOT_DIR, now: datetime | None = None) -> Path:
    """Keep a dated copy, since CFBD offers only the latest value and history can't be rebuilt."""
    now = now or datetime.now(timezone.utc)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{now:%Y-%m-%d}.json"
    path.write_text(json.dumps({"saved_at": now.isoformat(), "ratings": rows}))
    return path


def latest_fpi_snapshot(season: int, directory: Path = FPI_SNAPSHOT_DIR) -> tuple[dict[str, float], date | None]:
    paths = sorted(directory.glob("*.json")) if directory.exists() else []
    if not paths:
        return {}, None
    content = json.loads(paths[-1].read_text())
    ratings = {
        row["team"]: float(row["fpi"])
        for row in content["ratings"]
        if row.get("year") == season and row.get("fpi") is not None
    }
    return ratings, datetime.fromisoformat(content["saved_at"]).date()


def write_forecast_log(entries: list[dict[str, Any]], directory: Path = FORECAST_LOG_DIR, now: datetime | None = None) -> Path | None:
    """Record forecasts before kickoff. A later test can score them once games are played."""
    if not entries:
        return None
    now = now or datetime.now(timezone.utc)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{now:%Y-%m-%dT%H%M%SZ}.json"
    path.write_text(json.dumps({"logged_at": now.isoformat(), "forecasts": entries}, indent=2, default=str))
    return path


def inputs_to_dict(inputs: AdjustmentInputs) -> dict[str, Any]:
    return asdict(inputs)
