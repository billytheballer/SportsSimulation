"""The app's pages: Game Center, Schedule, Teams, Injuries, and About the Model."""

from __future__ import annotations

import json
from datetime import date
from html import escape
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import streamlit as st

from cfbsim.app import charts
from cfbsim.app.theme import feature, footer, html, section, team_accent, tile
from cfbsim.bundle import PredictionBundle
from cfbsim.config import REPORTS_DIR
from cfbsim.predictor import Freshness, MatchupPrediction
from cfbsim.season_board import SEASON_STATS, confidence_label, projected_records

ALL = "All"
OFFICIAL_REPORT_PAGES = {
    "Big Ten": "https://bigten.org/sports/2026/9/10/FB_Availability_Reports.aspx",
    "SEC": "https://www.secsports.com/fbreports",
    "ACC": "https://theacc.com/sports/2025/8/28/availability-reporting-football.aspx",
    "Big 12": "https://big12sports.com/sports/2025/8/14/FBreporting.aspx",
}
Prediction = Callable[..., MatchupPrediction]


def brand_header(bundle: PredictionBundle, freshness: Freshness) -> None:
    chips = [f"<b>Stats through</b> {freshness.data_through:%b %-d, %Y}"]
    inputs = bundle.adjustments
    if inputs is not None:
        if inputs.fpi_as_of:
            chips.append(f"<b>FPI</b> {inputs.fpi_as_of:%b %-d}")
        count = sum(len(reports) for reports in inputs.reports.values())
        chips.append(f"<b>Injury reports</b> {count}")
    html(
        '<div class="brand"><div><div class="brand-name">Gridiron <span>Forecast</span></div>'
        '<div class="brand-note">College football predictions from 10,000 simulated games</div></div>'
        f'<div class="chips">{"".join(f"<div class=chip>{chip}</div>" for chip in chips)}</div></div>'
    )
    if not freshness.is_current_forecast:
        html(f'<div class="status"><strong>⚠ {escape(freshness.headline)}</strong>{escape(freshness.detail)}</div>')


def show_chart(chart_and_table: tuple[Any, pd.DataFrame], table_title: str = "View as a table") -> None:
    chart, table = chart_and_table
    st.altair_chart(chart, width="stretch", theme=None)
    with st.expander(table_title):
        st.dataframe(table, hide_index=True, width="stretch")


def kickoff_label(game: dict[str, Any]) -> str:
    day = f"{game['kickoff']:%a, %b %-d}"
    moment = game.get("kickoff_time")
    if moment is None or game.get("time_tbd"):
        return f"{day} · Time TBD"
    return f"{day} · {moment.tz_convert('America/New_York'):%-I:%M %p} ET"


def game_label(game: dict[str, Any]) -> str:
    joiner = "vs." if game["neutral_site"] else "at"
    return f"Week {game['week']} · {game['kickoff']:%b %-d} · {game['away']} {joiner} {game['home']}"


def remaining_games(bundle: PredictionBundle) -> list[dict[str, Any]]:
    """Predicted games that haven't kicked off, in schedule order."""
    now = pd.Timestamp.now(tz="UTC")
    today = date.today()
    return [
        row
        for row in bundle.board
        if (row.get("kickoff_time") is not None and row["kickoff_time"] >= now)
        or (row.get("kickoff_time") is None and row["kickoff"] >= today)
    ]


def filter_games(games: list[dict[str, Any]], week: str, conference: str, team: str) -> list[dict[str, Any]]:
    result = games
    if week != ALL:
        result = [row for row in result if str(row["week"]) == week.removeprefix("Week ")]
    if conference != ALL:
        result = [row for row in result if conference in (row["home_conference"], row["away_conference"])]
    if team != ALL:
        result = [row for row in result if team in (row["home"], row["away"])]
    return result


def filter_controls(games: list[dict[str, Any]], prefix: str, default_week: str | None = None) -> list[dict[str, Any]]:
    """Week, conference, and team filters, shared by Game Center and Schedule."""
    weeks = [ALL, *[f"Week {week}" for week in sorted({row["week"] for row in games})]]
    conferences = [ALL, *sorted({row[key] for row in games for key in ("home_conference", "away_conference") if row[key]})]
    teams = [ALL, *sorted({row[key] for row in games for key in ("home", "away")})]
    for key, options in ((f"{prefix}_week", weeks), (f"{prefix}_conference", conferences), (f"{prefix}_team", teams)):
        if st.session_state.get(key) not in options:
            st.session_state[key] = (default_week or ALL) if key.endswith("_week") else ALL
    first, second, third = st.columns(3)
    week = first.selectbox("Week", weeks, key=f"{prefix}_week")
    conference = second.selectbox("Conference", conferences, key=f"{prefix}_conference")
    team = third.selectbox("Team", teams, key=f"{prefix}_team")
    return filter_games(games, week, conference, team)


def _team_side(prediction: MatchupPrediction, bundle: PredictionBundle, team: str, share: float, leads: bool, right: bool) -> str:
    info = bundle.team_info.get(team, {})
    accent, on_accent = team_accent(info)
    abbreviation = info.get("abbreviation") or team[:4].upper()
    record = bundle.records.get(team, "")
    detail = " · ".join(part for part in (info.get("mascot", ""), record, bundle.conferences.get(team, "")) if part)
    win_class = "win" if leads else "win dim"
    side = "team-side right" if right else "team-side"
    dot = "home" if team == prediction.home else "away"
    return (
        f'<div class="{side}"><div class="team-head"><div class="badge" style="background:{escape(accent)};'
        f'color:{on_accent}">{escape(abbreviation)}</div><div><div class="team-title">{escape(team)}</div>'
        f'<div class="team-sub">{escape(detail)}</div></div></div>'
        f'<div class="{win_class}">{share:.0%}</div><div class="win-label"><span class="role-dot {dot}"></span>Win chance</div></div>'
    )


def scoreboard(prediction: MatchupPrediction, bundle: PredictionBundle, meta: str) -> None:
    summary = prediction.summary
    home_share, away_share = summary["home_win_probability"], summary["away_win_probability"]
    home_points, away_points = summary["home"]["mean"], summary["away"]["mean"]
    margin = summary["margin"]["mean"]
    leader = prediction.home if margin >= 0 else prediction.away
    home_accent, _ = team_accent(bundle.team_info.get(prediction.home))
    away_accent, _ = team_accent(bundle.team_info.get(prediction.away))
    favorite = prediction.home if home_share >= away_share else prediction.away
    underdog = prediction.away if favorite == prediction.home else prediction.home
    top, low = max(home_share, away_share), min(home_share, away_share)
    if top < 0.55:
        outlook = "This is close to a coin flip."
    elif low >= 0.005:
        outlook = f"{escape(underdog)} still wins about 1 game in {round(1 / low)}."
    else:
        outlook = f"{escape(underdog)} almost never wins, though simulations can't see everything."

    tested = ""
    if prediction.base_summary is not None:
        adjustment = prediction.adjustment
        parts = []
        if adjustment["fpi_used"]:
            parts.append("ESPN FPI")
        if adjustment["players"]:
            parts.append("official injury reports")
        if parts:
            base = prediction.base_summary["home_win_probability"]
            tested = (
                f"<small>Includes {' and '.join(parts)}, an untested add-on. "
                f"Tested model alone: {escape(prediction.home)} {base:.1%}. "
                f"Team-stat model: {escape(prediction.home)} {prediction.stats_model_home_probability:.1%}.</small>"
            )
    if not tested:
        tested = (
            f"<small>Team-stat model for comparison: {escape(prediction.home)} "
            f"{prediction.stats_model_home_probability:.1%}.</small>"
        )

    html(
        f"""<div class="scoreboard">
        <div class="score-stripe"><div style="background:{escape(away_accent)}"></div><div style="background:{escape(home_accent)}"></div></div>
        <div class="score-meta">{escape(meta)}</div>
        <div class="score-grid">
        {_team_side(prediction, bundle, prediction.away, away_share, away_share > home_share, False)}
        <div class="center"><div class="center-label">Projected score</div>
        <div class="center-score">{away_points:.0f}<span>–</span>{home_points:.0f}</div>
        <div class="tile-sub">{escape(leader)} by {abs(margin):.1f}</div>
        <span class="pill">{confidence_label(home_share)}</span></div>
        {_team_side(prediction, bundle, prediction.home, home_share, home_share >= away_share, True)}
        </div>
        <div class="winbar"><div class="away" style="width:{away_share * 100:.2f}%"></div>
        <div class="home" style="width:{home_share * 100:.2f}%"></div></div>
        <div class="verdict"><b>{escape(favorite)}</b> won {top:.1%} of 10,000 simulated games. {outlook} {tested}</div>
        </div>"""
    )


def _range(block: dict, level: str) -> str:
    low, high = block["intervals"][level]
    return f"{level} range: {low:.0f} to {high:.0f}"


def projection_tiles(prediction: MatchupPrediction) -> None:
    summary = prediction.summary
    margin = summary["margin"]

    def lines(block: dict, median: str) -> list[str]:
        return [f"Median: {median}", _range(block, "80%"), _range(block, "95%")]

    leader = prediction.home if margin["mean"] >= 0 else prediction.away
    html(
        '<div class="tiles">'
        + tile(prediction.away, f"{summary['away']['mean']:.1f}", lines(summary["away"], f"{summary['away']['median']:.0f}"))
        + tile(prediction.home, f"{summary['home']['mean']:.1f}", lines(summary["home"], f"{summary['home']['median']:.0f}"))
        + tile(
            f"Point differential ({prediction.home} minus {prediction.away})",
            f"{margin['mean']:+.1f}",
            lines(margin, f"{margin['median']:+.0f}") + [f"Average: {leader} by {abs(margin['mean']):.1f}"],
        )
        + tile("Total points", f"{summary['total']['mean']:.1f}", lines(summary["total"], f"{summary['total']['median']:.0f}"))
        + "</div>"
    )


def notices(prediction: MatchupPrediction) -> None:
    for notice in prediction.notices:
        extra = "" if notice.level == "caution" else " info"
        html(f'<div class="note{extra}">{escape(notice.text)}</div>')


def overview_tab(prediction: MatchupPrediction) -> None:
    projection_tiles(prediction)
    notices(prediction)
    section("Score ranges", "The thick bar holds 80% of simulated scores and the thin bar holds 95%. The dot is the median.")
    show_chart(charts.score_ranges(prediction))
    section(
        "Reading these numbers",
        "A win chance isn't a guarantee: a 75% favorite should lose about 1 game in 4. The projected score is an average "
        "across simulations, so the ranges matter more than any one exact score.",
    )


def drivers_tab(prediction: MatchupPrediction) -> None:
    section(
        "What drives this pick",
        "Points each factor adds to the projected margin. They overlap and don't prove cause, but together they explain "
        "how the model reached its number.",
    )
    top = [factor for factor in prediction.factors if abs(factor["margin"]) >= 0.05][:3]
    if top:
        html(
            '<div class="tiles">'
            + "".join(
                tile(
                    factor["label"],
                    f"{abs(factor['margin']):.1f}",
                    [f"points toward {prediction.home if factor['margin'] >= 0 else prediction.away}"],
                )
                for factor in top
            )
            + "</div>"
        )
    show_chart(charts.factor_chart(prediction))


def simulations_tab(prediction: MatchupPrediction) -> None:
    section(
        "Results of 10,000 simulated games",
        "Each bar counts how often a result came up. More simulations reduce random noise, but not model error.",
    )
    margin, total, scores, grid, common = st.tabs(
        ["Point differential", "Total points", "Team scores", "Final score grid", "Most common scores"]
    )
    with margin:
        show_chart(charts.margin_distribution(prediction))
    with total:
        show_chart(charts.total_distribution(prediction))
    with scores:
        show_chart(charts.team_score_distribution(prediction))
    with grid:
        st.caption("Each cell covers a 7-point range. Cells above the line are games the away team won.")
        show_chart(charts.final_score_heatmap(prediction))
    with common:
        rows = [
            {prediction.away: item["away"], prediction.home: item["home"], "Share of simulations": f"{item['share']:.1%}"}
            for item in prediction.summary["common_scores"]
        ]
        st.caption("Even the most common exact score is unlikely, which is why the ranges matter more.")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def season_stat_comparison(bundle: PredictionBundle, home: str, away: str) -> pd.DataFrame:
    table = bundle.season_stats
    rows = []
    for stat in SEASON_STATS:
        if stat.key not in table or home not in table.index or away not in table.index:
            continue
        home_rank, away_rank = table.at[home, f"{stat.key}_rank"], table.at[away, f"{stat.key}_rank"]
        rows.append(
            {
                "Group": stat.group,
                "Stat": stat.label,
                away: round(float(table.at[away, stat.key]), stat.decimals) if pd.notna(table.at[away, stat.key]) else None,
                f"{away} rank": int(away_rank) if pd.notna(away_rank) else None,
                home: round(float(table.at[home, stat.key]), stat.decimals) if pd.notna(table.at[home, stat.key]) else None,
                f"{home} rank": int(home_rank) if pd.notna(home_rank) else None,
                "Edge": (
                    home if pd.notna(home_rank) and pd.notna(away_rank) and home_rank < away_rank
                    else away if pd.notna(home_rank) and pd.notna(away_rank) and away_rank < home_rank else "Even"
                ),
            }
        )
    return pd.DataFrame(rows)


def recent_results(bundle: PredictionBundle, team: str, count: int = 5) -> pd.DataFrame:
    rows = []
    for game in reversed(bundle.game_logs.get(team, [])[-count:]):
        site = "vs." if game["site"] == "vs" else "at"
        rows.append(
            {
                "Date": f"{game['date']:%b %-d}",
                "Opponent": f"{site} {game['opponent']}",
                "Result": game["result"],
                "Score": f"{game['points_for']}-{game['points_against']}",
            }
        )
    return pd.DataFrame(rows)


def team_stats_tab(prediction: MatchupPrediction, bundle: PredictionBundle) -> None:
    home, away = prediction.home, prediction.away
    comparison = season_stat_comparison(bundle, home, away)
    if not comparison.empty:
        section("Season box-score stats", "Actual per-game averages this season. Rank 1 is best among FBS teams.")
        offense, defense, discipline = st.tabs(["Offense", "Defense", "Discipline"])
        for tab, group in ((offense, "Offense"), (defense, "Defense"), (discipline, "Discipline")):
            with tab:
                st.dataframe(
                    comparison[comparison["Group"] == group].drop(columns="Group"), hide_index=True, width="stretch"
                )
    section(
        "How the model sees them",
        "These are the model's inputs: running averages adjusted for opponent strength, with recent games counting more. "
        "A higher percentile is always better.",
    )
    overall, offense, defense = st.tabs(["Overall", "Offense", "Defense"])
    for tab, group in ((overall, "Overall"), (offense, "Offense"), (defense, "Defense")):
        with tab:
            show_chart(charts.comparison_chart(prediction, group))
    if bundle.game_logs:
        section("Recent results")
        left, right = st.columns(2)
        for column, team in ((left, away), (right, home)):
            with column:
                st.markdown(f"**{team}** ({bundle.records.get(team, '')})")
                st.dataframe(recent_results(bundle, team), hide_index=True, width="stretch")


def fpi_table(bundle: PredictionBundle, teams: list[str]) -> pd.DataFrame:
    inputs = bundle.adjustments
    rows = []
    for team in teams:
        detail = inputs.fpi_details.get(team, {})
        efficiency, ranks = detail.get("efficiencies", {}), detail.get("ranks", {})
        rows.append(
            {
                "Team": team,
                "FPI": inputs.fpi.get(team),
                "FPI rank": ranks.get("fpi"),
                "Offense": efficiency.get("offense"),
                "Defense": efficiency.get("defense"),
                "Special teams": efficiency.get("specialTeams"),
                "Schedule rank": ranks.get("strengthOfSchedule"),
                "Remaining schedule rank": ranks.get("remainingStrengthOfSchedule"),
            }
        )
    return pd.DataFrame(rows)


def player_table(rows: list[dict[str, Any]]) -> None:
    table = pd.DataFrame(
        [
            {
                "Team": row["team"],
                "Player": row["player"],
                "Official status": row["status"],
                "Points off team": round(row["points"], 1),
                "Basis": row["note"],
                "Source": row["source_url"],
            }
            for row in rows
        ]
    )
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={"Source": st.column_config.LinkColumn("Source", display_text="Official report")},
    )


def availability_tab(prediction: MatchupPrediction, bundle: PredictionBundle) -> None:
    adjustment = prediction.adjustment
    if adjustment is None:
        html(
            '<div class="note info">FPI and official injury reports are used only in current-season forecasts, '
            "so they aren't part of this view.</div>"
        )
        return
    section(
        "Official injury reports",
        "From conference and school reports only. A listed player's expected cost comes off their own team's points.",
    )
    if adjustment["players"]:
        player_table(adjustment["players"])
        st.caption("Chances that a listed player sits are assumptions. Defensive players and linemen have no estimated effect.")
    else:
        html(
            '<div class="note info">No official reports are on file for this game. That doesn\'t mean either team is healthy. '
            "Reports usually appear about three days before conference games, and many non-conference games have none.</div>"
        )
    section("ESPN FPI", "ESPN's rating of team strength, in points above or below an average team.")
    st.dataframe(fpi_table(bundle, [prediction.away, prediction.home]), hide_index=True, width="stretch")
    if adjustment["fpi_used"]:
        fpi_favorite = prediction.home if adjustment["fpi_margin"] >= 0 else prediction.away
        shift = adjustment["fpi_margin_shift"]
        moved = prediction.home if shift >= 0 else prediction.away
        html(
            f'<div class="note info">FPI as of {adjustment["fpi_as_of"]:%B %-d, %Y} favors {escape(fpi_favorite)} by '
            f'{abs(adjustment["fpi_margin"]):.1f} points, with home field. It gets an assumed {adjustment["fpi_weight"]:.0%} '
            f'weight here, which moved the margin {abs(shift):.1f} points toward {escape(moved)}.</div>'
        )
    else:
        html('<div class="note">FPI isn\'t included here because it\'s missing or out of date.</div>')
    st.caption(
        "FPI and injury reports are an untested add-on: no pre-game history exists to test them on past seasons. "
        "Forecasts are being logged before kickoff so they can be scored on future games."
    )


def prediction_view(prediction: MatchupPrediction, bundle: PredictionBundle, meta: str) -> None:
    scoreboard(prediction, bundle, meta)
    overview, drivers, simulations, stats, availability = st.tabs(
        ["Overview", "What Drives It", "Simulations", "Team Stats", "Injuries & FPI"]
    )
    with overview:
        overview_tab(prediction)
    with drivers:
        drivers_tab(prediction)
    with simulations:
        simulations_tab(prediction)
    with stats:
        team_stats_tab(prediction, bundle)
    with availability:
        availability_tab(prediction, bundle)


def custom_matchup(bundle: PredictionBundle, names: list[str], path: str, run: Prediction) -> None:
    def index_of(name: str, fallback: int) -> int:
        return names.index(name) if name in names else min(fallback, len(names) - 1)

    with st.form("custom_matchup"):
        first, second, third, fourth = st.columns([3, 3, 2, 2], vertical_alignment="bottom")
        home = first.selectbox("Home team", names, index=index_of("Ohio State", 0))
        away = second.selectbox("Away team", names, index=index_of("Oregon", 1))
        neutral = third.checkbox("Neutral site")
        submitted = fourth.form_submit_button("Simulate", type="primary", width="stretch")
    if submitted:
        st.session_state["custom_request"] = (path, home, away, neutral)
    request = st.session_state.get("custom_request")
    if request is None or request[0] != path:
        st.info("Choose two teams and select **Simulate**.")
        return
    _, home, away, neutral = request
    if home == away:
        st.error("Choose two different teams.")
        return
    site = "Neutral site" if neutral else f"At {home}"
    prediction = run(home, away, neutral, None)
    prediction_view(prediction, bundle, f"Hypothetical matchup · {site}")


def game_center(bundle: PredictionBundle, names: list[str], path: str, run: Prediction) -> None:
    games = remaining_games(bundle)
    if not games:
        custom_matchup(bundle, names, path, run)
        return
    build_own = st.toggle("Build your own matchup", key="build_own")
    if build_own:
        custom_matchup(bundle, names, path, run)
        return
    next_week = f"Week {min(row['week'] for row in games)}"
    chosen = filter_controls(games, "game", next_week)
    if not chosen:
        st.info("No remaining games match those filters.")
        return
    labels = {game_label(row): row for row in chosen}
    if st.session_state.get("game_choice") not in labels:
        st.session_state["game_choice"] = next(iter(labels))
    label = st.selectbox(f"Game ({len(labels)} remaining)", list(labels), key="game_choice")
    game = labels[label]
    prediction = run(game["home"], game["away"], game["neutral_site"], game["kickoff"])
    info = bundle.team_info.get(game["home"], {})
    stadium, city = info.get("stadium", ""), info.get("city", "")
    place = "Neutral site" if game["neutral_site"] else (
        stadium if city and city in stadium else ", ".join(part for part in (stadium, city) if part)
    )
    meta = " · ".join(part for part in (f"Week {game['week']}", kickoff_label(game), place) if part)
    prediction_view(prediction, bundle, meta)


def _open_game_from_schedule(rows_key: str, table_key: str) -> None:
    selected = st.session_state[table_key]["selection"]["rows"]
    if selected:
        st.session_state["game_week"] = ALL
        st.session_state["game_conference"] = ALL
        st.session_state["game_team"] = ALL
        st.session_state["game_choice"] = st.session_state[rows_key][selected[0]]
        st.session_state["build_own"] = False
        st.session_state["view"] = "Game Center"


def schedule_page(bundle: PredictionBundle) -> None:
    games = remaining_games(bundle)
    section(
        "Rest of the regular season",
        f"Predictions for {len(games)} remaining games between FBS teams. Select a row to open that game. "
        "Later weeks use today's team strength, so those forecasts will change as games are played.",
    )
    if not games:
        st.info("No remaining games are available.")
        return
    next_week = f"Week {min(row['week'] for row in games)}"
    chosen = filter_controls(games, "schedule", next_week)
    if not chosen:
        st.info("No remaining games match those filters.")
        return
    closest = min(chosen, key=lambda row: abs(row["home_win"] - 0.5))
    biggest = max(chosen, key=lambda row: abs(row["home_win"] - 0.5))
    highest = max(chosen, key=lambda row: row["total"])

    def matchup(row: dict) -> str:
        return f"{row['away']} {'vs.' if row['neutral_site'] else 'at'} {row['home']}"

    def favorite(row: dict) -> tuple[str, float]:
        return (row["home"], row["home_win"]) if row["home_win"] >= row["away_win"] else (row["away"], row["away_win"])

    close_team, close_share = favorite(closest)
    big_team, big_share = favorite(biggest)
    html(
        '<div class="tiles" style="grid-template-columns:repeat(auto-fit,minmax(230px,1fr))">'
        + feature("Closest game", matchup(closest), f"{close_team} {close_share:.0%}")
        + feature("Biggest favorite", matchup(biggest), f"{big_team} {big_share:.0%}")
        + feature("Highest scoring", matchup(highest), f"{highest['total']:.0f} projected points")
        + "</div>"
    )
    rows = []
    for row in chosen:
        team, share = favorite(row)
        rows.append(
            {
                "Date": kickoff_label(row),
                "Game": matchup(row),
                "Pick": team,
                "Win chance": share,
                "Confidence": confidence_label(share),
                "Projected": f"{row['away_points']:.0f}-{row['home_points']:.0f}",
                "Margin": abs(row["margin"]),
                "Total": row["total"],
                "Injury reports": row["players_listed"],
            }
        )
    st.session_state["schedule_rows"] = [game_label(row) for row in chosen]
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        height=min(620, 38 * (len(rows) + 1) + 4),
        key="schedule_table",
        on_select=lambda: _open_game_from_schedule("schedule_rows", "schedule_table"),
        selection_mode="single-row",
        column_config={
            "Win chance": st.column_config.ProgressColumn("Win chance", format="percent", min_value=0.5, max_value=1.0),
            "Margin": st.column_config.NumberColumn("Margin", format="%.1f"),
            "Total": st.column_config.NumberColumn("Total", format="%.1f"),
            "Projected": st.column_config.TextColumn("Projected (away-home)"),
        },
    )
    if len(games) < len(bundle.board):
        st.caption(f"{len(bundle.board) - len(games)} games have kicked off since the last update and are hidden.")
    st.caption("Games against non-FBS opponents and postseason games aren't included.")


def team_table(bundle: PredictionBundle, projections: dict[str, dict[str, float]]) -> pd.DataFrame:
    inputs = bundle.adjustments
    rows = []
    for team, profile in bundle.profiles.iterrows():
        projection = projections.get(team, {})
        details = inputs.fpi_details.get(team, {}) if inputs else {}
        rows.append(
            {
                "Team": team,
                "Conference": bundle.conferences.get(team, ""),
                "Record": bundle.records.get(team, ""),
                "Elo": round(float(profile["elo"])) if pd.notna(profile["elo"]) else None,
                "FPI": inputs.fpi.get(team) if inputs else None,
                "FPI rank": details.get("ranks", {}).get("fpi"),
                "Offense percentile": round(float(profile["off_points_percentile"])) if pd.notna(profile["off_points_percentile"]) else None,
                "Defense percentile": round(float(profile["def_points_percentile"])) if pd.notna(profile["def_points_percentile"]) else None,
                "Projected wins": projection.get("projected_wins"),
                "Games left": projection.get("games_left"),
            }
        )
    table = pd.DataFrame(rows).sort_values("Elo", ascending=False, na_position="last").reset_index(drop=True)
    table.insert(0, "Rank", table.index + 1)
    return table


def teams_page(bundle: PredictionBundle) -> None:
    section(
        "Team rankings",
        "Ranked by the model's Elo rating. Offense and defense percentiles use opponent-adjusted scoring, where higher is "
        "always better. Projected wins add current wins to win chances in remaining FBS games, so it's an average and not "
        "an exact record.",
    )
    projections = projected_records(remaining_games(bundle), bundle.records)
    table = team_table(bundle, projections)
    left, right = st.columns(2)
    conferences = [ALL, *sorted({value for value in table["Conference"] if value})]
    conference = left.selectbox("Conference", conferences, key="teams_conference")
    shown = table if conference == ALL else table[table["Conference"] == conference]
    profile_options = shown["Team"].tolist() or table["Team"].tolist()
    if st.session_state.get("teams_team") not in profile_options:
        st.session_state["teams_team"] = profile_options[0]
    team = right.selectbox("Team profile", profile_options, key="teams_team")
    st.dataframe(
        shown,
        hide_index=True,
        width="stretch",
        height=min(480, 38 * (len(shown) + 1) + 4),
        column_config={
            "Projected wins": st.column_config.NumberColumn("Projected wins", format="%.1f"),
            "FPI": st.column_config.NumberColumn("FPI", format="%.1f"),
        },
    )
    team_profile(bundle, team, projections)


def team_profile(bundle: PredictionBundle, team: str, projections: dict[str, dict[str, float]]) -> None:
    info = bundle.team_info.get(team, {})
    stadium, city, state = info.get("stadium", ""), info.get("city", ""), info.get("state", "")
    place = stadium if city and city in stadium else ", ".join(part for part in (stadium, city, state) if part)
    section(f"{team} {info.get('mascot', '')}".strip(), place)
    projection = projections.get(team)
    profile = bundle.profiles.loc[team]
    tiles = [tile("Record", bundle.records.get(team, "n/a"), [bundle.conferences.get(team, "")])]
    tiles.append(tile("Elo rating", f"{profile['elo']:.0f}", [f"Percentile: {profile['elo_percentile']:.0f}"]))
    inputs = bundle.adjustments
    if inputs is not None and team in inputs.fpi:
        rank = inputs.fpi_details.get(team, {}).get("ranks", {}).get("fpi")
        tiles.append(tile("ESPN FPI", f"{inputs.fpi[team]:+.1f}", [f"Rank: {rank}" if rank else ""]))
    if projection:
        tiles.append(
            tile(
                "Projected wins",
                f"{projection['projected_wins']:.1f}",
                [f"{projection['expected_wins_left']:.1f} expected in {projection['games_left']} games left"],
            )
        )
    html(f'<div class="tiles">{"".join(tiles)}</div>')

    games = [row for row in remaining_games(bundle) if team in (row["home"], row["away"])]
    if games:
        st.markdown("**Remaining games**")
        rows = []
        for row in games:
            at_home = row["home"] == team
            opponent = row["away"] if at_home else row["home"]
            site = "vs." if at_home or row["neutral_site"] else "at"
            share = row["home_win"] if at_home else row["away_win"]
            own, other = (row["home_points"], row["away_points"]) if at_home else (row["away_points"], row["home_points"])
            rows.append(
                {
                    "Date": f"{row['kickoff']:%b %-d}",
                    "Opponent": f"{site} {opponent}",
                    "Win chance": share,
                    "Projected": f"{own:.0f}-{other:.0f}",
                }
            )
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            width="stretch",
            column_config={"Win chance": st.column_config.ProgressColumn("Win chance", format="percent", min_value=0, max_value=1)},
        )
    if bundle.game_logs.get(team):
        st.markdown("**Results so far**")
        st.dataframe(recent_results(bundle, team, count=20), hide_index=True, width="stretch")
    table = bundle.season_stats
    if team in table.index:
        st.markdown("**Season stats** (rank 1 is best among FBS teams)")
        rows = [
            {
                "Group": stat.group,
                "Stat": stat.label,
                "Value": round(float(table.at[team, stat.key]), stat.decimals) if pd.notna(table.at[team, stat.key]) else None,
                "Rank": int(table.at[team, f"{stat.key}_rank"]) if pd.notna(table.at[team, f"{stat.key}_rank"]) else None,
            }
            for stat in SEASON_STATS
            if stat.key in table
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def injuries_page(bundle: PredictionBundle) -> None:
    inputs = bundle.adjustments
    section(
        "Official injury reports",
        "Players listed in official conference or school reports for upcoming games. Reports count only for their own game.",
    )
    if inputs is None:
        html('<div class="note info">Injury reports are used only in current-season forecasts.</div>')
        return
    from cfbsim.adjustments import team_availability

    rows = []
    for team in sorted(inputs.reports):
        rows += team_availability(team, inputs)[1]
    if rows:
        teams = [ALL, *sorted({row["team"] for row in rows})]
        team = st.selectbox("Team", teams, key="injury_team")
        player_table([row for row in rows if team == ALL or row["team"] == team])
        st.caption("Chances that a listed player sits are assumptions. Defensive players and linemen have no estimated effect.")
    else:
        html(
            '<div class="note info">No official reports are on file yet. A team that isn\'t listed may still have injuries. '
            "Conferences usually post reports about three days before kickoff.</div>"
        )
    section("How injuries affect a forecast")
    rates = inputs.points_per_100_yards
    html(
        '<div class="tiles" style="grid-template-columns:repeat(auto-fit,minmax(230px,1fr))">'
        + tile("Starting QB out", f"{rates['qb']:.2f}", ["points per 100 lost yards per game", "Estimated from 2016-22"])
        + tile("Top rusher or receiver out", f"{rates['skill']:.2f}", ["points per 100 lost yards per game", "Low end of a less stable estimate"])
        + tile("Other players", "0", ["No estimated effect, including defense", "Free data has no comparable value"])
        + "</div>"
    )
    st.markdown(
        "A player's cost also depends on their status, using assumed chances of sitting: Out 100%, Doubtful 80%, "
        "Questionable 50%, Game Time Decision 50%, Out 1st Half 50%, Probable 10%."
    )
    links = " · ".join(f"[{name}]({url})" for name, url in OFFICIAL_REPORT_PAGES.items())
    st.markdown(f"**Official report pages:** {links}")


def _read_report(name: str, reports_dir: Path) -> dict | None:
    path = reports_dir / name
    return json.loads(path.read_text()) if path.exists() else None


def about_page(bundle: PredictionBundle, reports_dir: Path = REPORTS_DIR) -> None:
    section("How it works")
    html(
        '<div class="tiles" style="grid-template-columns:repeat(auto-fit,minmax(240px,1fr))">'
        + tile("1. Rate the teams", "Elo + stats", ["Ratings and opponent-adjusted stats, built only from earlier games"])
        + tile("2. Project the points", "Expected score", ["A separate model estimates each team's points, offense against defense"])
        + tile("3. Add current news", "FPI + injuries", ["An untested add-on using ESPN FPI and official reports"])
        + tile("4. Play it 10,000 times", "Simulation", ["Draws real score patterns from similar past games, with overtime"])
        + "</div>"
    )
    final = _read_report("final_evaluation/final_evaluation.json", reports_dir)
    if final:
        section(
            "How accurate is it?",
            f"A one-time test on {final['games']:,} games from 2023-25, using a model frozen beforehand. "
            "This measures the tested model, without the FPI and injury add-on.",
        )
        win = final["win_probability"]
        listed = final.get("listed_line_benchmark", {}).get("listed_line", {})
        rows = [
            {"Forecast": "Simulation (this app's tested model)", "Picked the winner": win["simulation"]["accuracy"], "Log loss": win["simulation"]["log_loss"]},
            {"Forecast": "Team-stat model", "Picked the winner": win["stats_model"]["accuracy"], "Log loss": win["stats_model"]["log_loss"]},
            {"Forecast": "Elo only", "Picked the winner": win["elo"]["accuracy"], "Log loss": win["elo"]["log_loss"]},
            {"Forecast": "Coin flip", "Picked the winner": final["baselines"]["coin_flip"]["accuracy"], "Log loss": final["baselines"]["coin_flip"]["log_loss"]},
        ]
        if listed:
            rows.insert(0, {"Forecast": "Listed betting line (benchmark, timing unverified)", "Picked the winner": None, "Log loss": listed["log_loss"]})
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            width="stretch",
            column_config={
                "Picked the winner": st.column_config.NumberColumn("Picked the winner", format="percent"),
                "Log loss": st.column_config.NumberColumn("Log loss (lower is better)", format="%.4f"),
            },
        )
        margin, total = final["points"]["margin"]["simulation_mean"], final["points"]["total"]["simulation_mean"]
        st.markdown(
            f"Typical miss: **{margin['mae']:.1f} points** on the margin and **{total['mae']:.1f} points** on the total. "
            "Listed betting lines were more accurate."
        )
    live = bundle.live_scorecard
    if live:
        st.markdown(
            f"**{bundle.season} season so far:** {live['accuracy']:.1%} of winners on {live['games']:,} FBS games played after "
            f"training ended, with log loss {live['log_loss']:.3f}. That's the tested model on a partial season, so it may not hold up."
        )
    section("Where to be careful")
    st.markdown(
        "- **Bowl games and neutral sites:** close to a coin flip in testing.\n"
        "- **New FBS teams:** missing some data, and no better than Elo in testing.\n"
        "- **The FPI and injury add-on:** untested, with assumed weights. The tested number is always shown beside it.\n"
        "- **Injuries:** only official reports on file count, and defenders and linemen have no estimated effect.\n"
        "- **Not modeled:** weather, coaching changes, and team styles.\n"
        "- **Score ranges:** tend to run slightly wide."
    )
    section("What the terms mean")
    st.markdown(
        "- **Win chance:** how often a team won across 10,000 simulated games.\n"
        "- **Confidence:** Toss-up is under 55%, Lean under 65%, Solid under 80%, and Strong is 80% or more.\n"
        "- **80% range:** 8 of every 10 simulated results fell inside it.\n"
        "- **Elo:** a team rating that rises and falls with results.\n"
        "- **FPI:** ESPN's team rating, in points above or below average.\n"
        "- **EPA and success rate:** how much plays improved a team's scoring chances, and how often they did.\n"
        "- **Log loss:** a score for probability forecasts, where lower is better and confident misses cost the most."
    )
    st.caption(f"The model is fit on seasons {min(bundle.training_seasons)}-{max(bundle.training_seasons)}.")


__all__ = [
    "about_page", "brand_header", "custom_matchup", "footer", "game_center", "injuries_page", "schedule_page", "teams_page",
]
