"""Page sections for the Streamlit app. All team names are escaped before display."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from cfbsim.app import charts
from cfbsim.bundle import PredictionBundle
from cfbsim.config import REPORTS_DIR
from cfbsim.predictor import Freshness, MatchupPrediction

STYLES = """
<style>
:root {
  --surface: #fcfcfb; --plane: #f9f9f7; --ink: #0b0b0b; --ink-2: #52514e; --muted: #6f6d68;
  --hairline: rgba(11,11,11,0.10); --home: #2a78d6; --away: #eb6834; --navy: #0d2340;
}
.stApp { background: var(--plane); }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2rem; max-width: 1120px; }
.brand { background: var(--navy); color: #fff; border-radius: 12px; padding: 22px 26px; margin-bottom: 14px; }
.brand h1 { color: #fff; font-size: 26px; font-weight: 700; margin: 0; padding: 0; letter-spacing: -0.01em; }
.brand p { color: #c8d4e3; font-size: 14px; margin: 4px 0 0 0; }
.status { border-radius: 10px; padding: 12px 16px; margin: 0 0 14px 0; font-size: 14px; line-height: 1.45;
  border: 1px solid; color: var(--ink); }
.status strong { display: block; font-size: 15px; margin-bottom: 2px; }
.status.old { background: #fff6e0; border-color: #f0c766; }
.status.current { background: #e9f6e9; border-color: #8fce8f; }
.note { border-left: 3px solid #e0a100; background: #fffaf0; padding: 9px 14px; border-radius: 0 8px 8px 0;
  margin: 6px 0; font-size: 14px; color: var(--ink); }
.note.info { border-left-color: #8a97a8; background: #f5f6f8; }
.card { background: var(--surface); border: 1px solid var(--hairline); border-radius: 12px; padding: 18px 20px; height: 100%; }
.tiles { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 8px; }
@media (max-width: 860px) { .tiles { grid-template-columns: repeat(2, 1fr); } }
.label { color: var(--ink-2); font-size: 13px; margin-bottom: 4px; }
.value { color: var(--ink); font-size: 30px; font-weight: 650; line-height: 1.15; }
.sub { color: var(--ink-2); font-size: 13px; margin-top: 4px; }
.matchup { font-size: 14px; color: var(--ink-2); margin-bottom: 12px; }
.teams { display: flex; justify-content: space-between; gap: 18px; align-items: flex-end; flex-wrap: wrap; }
.team-name { display: flex; gap: 8px; align-items: center; font-size: 16px; font-weight: 600; color: var(--ink); }
.swatch { width: 12px; height: 12px; border-radius: 3px; display: inline-block; flex: none; }
.swatch.home { background: var(--home); } .swatch.away { background: var(--away); border-radius: 2px; transform: rotate(45deg); }
.hero { font-size: 56px; font-weight: 700; line-height: 1.05; color: var(--ink); }
.hero.secondary { font-size: 40px; color: var(--ink-2); font-weight: 600; }
.right { text-align: right; } .right .team-name { justify-content: flex-end; }
.split { display: flex; gap: 2px; height: 12px; margin-top: 14px; }
.split div { height: 100%; }
.split .home { background: var(--home); border-radius: 4px 0 0 4px; }
.split .away { background: var(--away); border-radius: 0 4px 4px 0; }
.section { font-size: 20px; font-weight: 650; color: var(--ink); margin: 26px 0 2px 0; }
.section-note { color: var(--ink-2); font-size: 14px; margin: 0 0 10px 0; }
.foot { color: var(--ink-2); font-size: 13px; margin-top: 30px; padding-top: 14px; border-top: 1px solid var(--hairline); }
@media (max-width: 640px) { .hero { font-size: 42px; } .value { font-size: 24px; } }
</style>
"""


def apply_styles() -> None:
    st.markdown(STYLES, unsafe_allow_html=True)


def brand_header() -> None:
    st.markdown(
        '<div class="brand"><h1>College Football Matchup Simulator</h1>'
        "<p>Statistical win probabilities and 10,000 simulated games. Every number comes from a model "
        "tested on games it never saw.</p></div>",
        unsafe_allow_html=True,
    )


def status_banner(freshness: Freshness) -> None:
    kind = "current" if freshness.is_current_forecast else "old"
    icon = "✓" if freshness.is_current_forecast else "⚠"
    st.markdown(
        f'<div class="status {kind}"><strong>{icon} {escape(freshness.headline)}</strong>{escape(freshness.detail)}</div>',
        unsafe_allow_html=True,
    )


def show_notices(prediction: MatchupPrediction) -> None:
    for notice in prediction.notices:
        extra = "" if notice.level == "caution" else " info"
        st.markdown(f'<div class="note{extra}">{escape(notice.text)}</div>', unsafe_allow_html=True)


def _table_view(table: pd.DataFrame, title: str = "View as a table") -> None:
    with st.expander(title):
        st.dataframe(table, hide_index=True, use_container_width=True)


def _chart(chart_and_table: tuple[Any, pd.DataFrame], table_title: str = "View as a table") -> None:
    chart, table = chart_and_table
    st.altair_chart(chart, use_container_width=True)
    _table_view(table, table_title)


def win_probability_card(prediction: MatchupPrediction) -> None:
    summary = prediction.summary
    home_share = summary["home_win_probability"]
    away_share = summary["away_win_probability"]
    home_class = "hero" if home_share >= away_share else "hero secondary"
    away_class = "hero" if away_share > home_share else "hero secondary"
    site = "Neutral site" if prediction.neutral_site else f"At {prediction.home}"
    st.markdown(
        f"""<div class="card">
        <div class="matchup">{escape(prediction.away)} vs. {escape(prediction.home)} · {escape(site)} · Win probability</div>
        <div class="teams">
          <div><div class="team-name"><span class="swatch home"></span>{escape(prediction.home)}</div>
          <div class="{home_class}">{home_share:.1%}</div></div>
          <div class="right"><div class="team-name"><span class="swatch away"></span>{escape(prediction.away)}</div>
          <div class="{away_class}">{away_share:.1%}</div></div>
        </div>
        <div class="split"><div class="home" style="width:{home_share * 100:.2f}%"></div>
        <div class="away" style="width:{away_share * 100:.2f}%"></div></div>
        <div class="sub" style="margin-top:12px">From {summary['simulations']:,} simulated games.
        Team-stat model for comparison: {escape(prediction.home)} {prediction.stats_model_home_probability:.1%},
        {escape(prediction.away)} {1 - prediction.stats_model_home_probability:.1%}.
        Overtime in {summary['overtime_probability']:.1%} of simulations.</div>
        </div>""",
        unsafe_allow_html=True,
    )


def _tile(label: str, value: str, lines: list[str]) -> str:
    details = "".join(f"<div>{escape(line)}</div>" for line in lines)
    return (
        f'<div class="card"><div class="label">{escape(label)}</div>'
        f'<div class="value">{escape(value)}</div><div class="sub">{details}</div></div>'
    )


def _signed_margin(prediction: MatchupPrediction, value: float, decimals: int) -> str:
    leader = prediction.home if value >= 0 else prediction.away
    return f"{leader} by {abs(value):.{decimals}f}"


def _range(block: dict, level: str) -> str:
    low, high = block["intervals"][level]
    return f"{level} range: {low:.0f} to {high:.0f}"


def _detail_lines(block: dict) -> list[str]:
    return [f"Median: {block['median']:.0f}", _range(block, "80%"), _range(block, "95%")]


def projection_tiles(prediction: MatchupPrediction) -> None:
    summary = prediction.summary
    margin = summary["margin"]
    tiles = [
        _tile(f"{prediction.home} projected points", f"{summary['home']['mean']:.1f}", _detail_lines(summary["home"])),
        _tile(f"{prediction.away} projected points", f"{summary['away']['mean']:.1f}", _detail_lines(summary["away"])),
        _tile(
            f"Point differential ({prediction.home} minus {prediction.away})",
            f"{margin['mean']:+.1f}",
            [
                f"Median: {margin['median']:+.0f}",
                _range(margin, "80%"),
                _range(margin, "95%"),
                f"Average result: {_signed_margin(prediction, margin['mean'], 1)}",
            ],
        ),
        _tile("Total points", f"{summary['total']['mean']:.1f}", _detail_lines(summary["total"])),
    ]
    st.markdown(f'<div class="tiles">{"".join(tiles)}</div>', unsafe_allow_html=True)


def section(title: str, note: str = "") -> None:
    extra = f'<div class="section-note">{escape(note)}</div>' if note else ""
    st.markdown(f'<div class="section">{escape(title)}</div>{extra}', unsafe_allow_html=True)


def score_range_section(prediction: MatchupPrediction) -> None:
    section(
        "Score ranges",
        "The thick bar is the 80% range and the thin bar is the 95% range. The dot marks the median.",
    )
    _chart(charts.score_ranges(prediction))


def simulation_section(prediction: MatchupPrediction) -> None:
    section(
        f"Results of {prediction.summary['simulations']:,} simulated games",
        "These show the range of possible outcomes. More simulations reduce random noise, not model error.",
    )
    margin, total, scores, grid, common = st.tabs(
        ["Point differential", "Total points", "Team scores", "Final score grid", "Most common scores"]
    )
    with margin:
        _chart(charts.margin_distribution(prediction))
    with total:
        _chart(charts.total_distribution(prediction))
    with scores:
        _chart(charts.team_score_distribution(prediction))
    with grid:
        st.caption("Each cell covers a 7-point range. Cells above the line are games where the away team won.")
        _chart(charts.final_score_heatmap(prediction))
    with common:
        rows = [
            {
                prediction.home: item["home"],
                prediction.away: item["away"],
                "Share of simulations": f"{item['share']:.1%}",
            }
            for item in prediction.summary["common_scores"]
        ]
        st.caption("Even the most common exact score is unlikely, which is why the ranges matter more.")
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def factors_section(prediction: MatchupPrediction) -> None:
    section(
        "What drives this prediction",
        "How many points each factor adds to the projected margin. Factors overlap and don't prove cause. "
        "Together they add up to the difference in expected points after four quarters.",
    )
    _chart(charts.factor_chart(prediction))
    top = [factor for factor in prediction.factors if abs(factor["margin"]) >= 0.05][:3]
    if top:
        sentences = []
        for factor in top:
            team = prediction.home if factor["margin"] >= 0 else prediction.away
            sentences.append(f"{factor['label']} favors {team} by {abs(factor['margin']):.1f} points.")
        st.markdown(" ".join(sentences))


def comparison_section(prediction: MatchupPrediction) -> None:
    section(
        "Team comparison",
        "Opponent-adjusted running averages, with recent games counting more. "
        "Ranks are out of all FBS teams in this data, and a higher percentile is always better.",
    )
    overall, offense, defense = st.tabs(["Overall", "Offense", "Defense"])
    with overall:
        _chart(charts.comparison_chart(prediction, "Overall"))
    with offense:
        _chart(charts.comparison_chart(prediction, "Offense"))
    with defense:
        _chart(charts.comparison_chart(prediction, "Defense"))


def _read_report(name: str, reports_dir: Path) -> dict | None:
    path = reports_dir / name
    return json.loads(path.read_text()) if path.exists() else None


def model_accuracy_section(bundle: PredictionBundle, reports_dir: Path = REPORTS_DIR) -> None:
    with st.expander("How accurate is this model?"):
        final = _read_report("final_evaluation/final_evaluation.json", reports_dir)
        if final:
            win = final["win_probability"]
            st.markdown(
                f"**Final evaluation on {final['games']:,} games from 2023-25, run once.** Simulation: "
                f"{win['simulation']['accuracy']:.1%} of winners, log loss {win['simulation']['log_loss']:.3f}. "
                f"Team-stat model: {win['stats_model']['accuracy']:.1%}, {win['stats_model']['log_loss']:.3f}. "
                f"Elo: {win['elo']['accuracy']:.1%}, {win['elo']['log_loss']:.3f}. "
                f"Margin error (MAE): {final['points']['margin']['simulation_mean']['mae']:.1f} points. "
                f"Total error (MAE): {final['points']['total']['simulation_mean']['mae']:.1f} points."
            )
        else:
            diagnostics = _read_report("stage3_diagnostics.json", reports_dir)
            if diagnostics:
                win = diagnostics["win_probability"]
                st.markdown(
                    f"**Development results on {diagnostics['games']:,} games from 2021-22.** Simulation: "
                    f"{win['simulation']['accuracy']:.1%} of winners, log loss {win['simulation']['log_loss']:.3f}. "
                    f"Team-stat model: {win['stats_model']['accuracy']:.1%}, {win['stats_model']['log_loss']:.3f}. "
                    f"Elo: {win['elo_calibrated']['accuracy']:.1%}, {win['elo_calibrated']['log_loss']:.3f}. "
                    "The final test on 2023-25 hasn't been run yet."
                )
        live = bundle.live_scorecard
        if live:
            st.markdown(
                f"**{bundle.season} season so far, on {live['games']:,} FBS games played after training ended.** "
                f"Simulation: {live['accuracy']:.1%} of winners, log loss {live['log_loss']:.3f}. "
                f"Margin error (MAE): {live['margin_mae']:.1f} points. This is a partial season."
            )
        st.markdown(
            "**Where it struggles:** bowl games and other neutral-site games, where it has done little better "
            "than a coin flip. A listed betting line has also been more accurate.\n\n"
            "**What the model can't see:** injuries, player availability, suspensions, weather, and coaching "
            "changes. Simulated score ranges have tended to run slightly wide."
        )
        st.caption(f"Trained on seasons {min(bundle.training_seasons)}-{max(bundle.training_seasons)}.")


def footer() -> None:
    st.markdown(
        '<div class="foot">Data provided by CollegeFootballData.com. Not official NCAA, conference, or school data. '
        "For analysis and education only, and not betting advice. Betting lines are never used as model inputs.</div>",
        unsafe_allow_html=True,
    )
