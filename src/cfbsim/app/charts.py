"""Altair charts for a matchup. Each function returns a chart and its table-view data.

Home is always the first palette slot and away the second, so a team's color follows
its role in every chart. These are the palette's dark-mode steps, documented as validated
together for color-blind and normal vision. Tables beside each chart hold every value.
"""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd

from cfbsim.predictor import MatchupPrediction

HOME_COLOR = "#3987e5"
AWAY_COLOR = "#d95926"
HEAT_LOW, HEAT_HIGH = "#184f95", "#cde2fb"
SURFACE = "#131b2e"
INK = "#f0f3f8"
INK_SECONDARY = "#bcc5d3"
MUTED = "#939dae"
GRID = "#2a3448"
BASELINE = "#46536a"
FONT = '"Barlow", system-ui, -apple-system, "Segoe UI", sans-serif'
HEATMAP_BIN = 7


def _style(chart: alt.Chart, height: int) -> alt.Chart:
    return (
        chart.properties(height=height, width="container", background=SURFACE)
        .configure_view(strokeWidth=0)
        .configure_axis(
            gridColor=GRID,
            domainColor=BASELINE,
            tickColor=BASELINE,
            labelColor=INK_SECONDARY,
            titleColor=INK_SECONDARY,
            labelFont=FONT,
            titleFont=FONT,
            labelFontSize=12,
            titleFontSize=12,
            titleFontWeight="normal",
        )
        .configure_legend(
            labelColor=INK_SECONDARY,
            titleColor=INK_SECONDARY,
            labelFont=FONT,
            titleFont=FONT,
            labelFontSize=12,
            orient="top",
            title=None,
        )
    )


def _team_scale(prediction: MatchupPrediction) -> alt.Scale:
    return alt.Scale(domain=[prediction.home, prediction.away], range=[HOME_COLOR, AWAY_COLOR])


def score_ranges(prediction: MatchupPrediction) -> tuple[alt.Chart, pd.DataFrame]:
    """Each team's 95% range, 80% range, and median, on one shared points axis."""
    rows = []
    for team, side in ((prediction.home, "home"), (prediction.away, "away")):
        block = prediction.summary[side]
        low_95, high_95 = block["intervals"]["95%"]
        low_80, high_80 = block["intervals"]["80%"]
        rows.append(
            {
                "Team": team,
                "95% low": low_95,
                "80% low": low_80,
                "Median": block["median"],
                "Average": round(block["mean"], 1),
                "80% high": high_80,
                "95% high": high_95,
            }
        )
    table = pd.DataFrame(rows)
    color = alt.Color("Team:N", scale=_team_scale(prediction), legend=None)
    y = alt.Y("Team:N", sort=[prediction.home, prediction.away], title=None, axis=alt.Axis(ticks=False, domain=False, labelFontSize=13))
    tooltip = list(table.columns)
    wide = alt.Chart(table).mark_rule(strokeWidth=4, opacity=0.3).encode(
        x=alt.X("95% low:Q", title="Points", scale=alt.Scale(domainMin=0), axis=alt.Axis(tickCount=8)),
        x2="95% high:Q",
        y=y,
        color=color,
        tooltip=tooltip,
    )
    narrow = alt.Chart(table).mark_rule(strokeWidth=12, strokeCap="round").encode(
        x="80% low:Q", x2="80% high:Q", y=y, color=color, tooltip=tooltip
    )
    median = alt.Chart(table).mark_point(
        filled=True, size=130, color="#ffffff", stroke=SURFACE, strokeWidth=2, opacity=1
    ).encode(x="Median:Q", y=y, tooltip=tooltip)
    label = alt.Chart(table).mark_text(dy=-18, color=INK, font=FONT, fontSize=13, fontWeight=600).encode(
        x="Median:Q", y=y, text=alt.Text("Median:Q", format=".0f")
    )
    return _style(alt.layer(wide, narrow, median, label), 150), table


def _counts(values: np.ndarray, minimum: int, maximum: int) -> pd.DataFrame:
    edges = np.arange(minimum, maximum + 2)
    counts, _ = np.histogram(values, bins=edges)
    return pd.DataFrame({"value": edges[:-1], "games": counts, "share": counts / len(values)})


def margin_distribution(prediction: MatchupPrediction) -> tuple[alt.Chart, pd.DataFrame]:
    """How many simulated games ended at each point margin, by which team won."""
    margin = prediction.home_scores - prediction.away_scores
    table = _counts(margin, int(margin.min()), int(margin.max()))
    table = table[table["games"] > 0].copy()
    table["Winner"] = np.where(table["value"] > 0, prediction.home, prediction.away)
    table["Result"] = [
        f"{prediction.home} by {value}" if value > 0 else f"{prediction.away} by {-value}" for value in table["value"]
    ]
    bars = alt.Chart(table).mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2, width={"band": 0.86}).encode(
        x=alt.X(
            "value:O",
            title=f"Final margin (positive means {prediction.home} wins)",
            axis=alt.Axis(values=list(range(-70, 71, 7)), labelAngle=0, grid=False),
            sort="ascending",
        ),
        y=alt.Y("games:Q", title="Simulated games", axis=alt.Axis(tickCount=4)),
        color=alt.Color("Winner:N", scale=_team_scale(prediction)),
        tooltip=[
            "Result:N",
            alt.Tooltip("games:Q", title="Simulated games", format=","),
            alt.Tooltip("share:Q", title="Share", format=".1%"),
        ],
    )
    return _style(bars, 300), table[["Result", "games", "share"]].rename(columns={"games": "Simulated games", "share": "Share"})


def total_distribution(prediction: MatchupPrediction) -> tuple[alt.Chart, pd.DataFrame]:
    """How many simulated games reached each combined point total."""
    total = prediction.home_scores + prediction.away_scores
    table = _counts(total, int(total.min()), int(total.max()))
    table = table[table["games"] > 0]
    chart = alt.Chart(table).mark_bar(
        color=HOME_COLOR, cornerRadiusTopLeft=2, cornerRadiusTopRight=2, width={"band": 0.86}
    ).encode(
        x=alt.X("value:O", title="Combined points", axis=alt.Axis(values=list(range(0, 201, 10)), labelAngle=0, grid=False)),
        y=alt.Y("games:Q", title="Simulated games", axis=alt.Axis(tickCount=4)),
        tooltip=[
            alt.Tooltip("value:O", title="Total"),
            alt.Tooltip("games:Q", title="Simulated games", format=","),
            alt.Tooltip("share:Q", title="Share", format=".1%"),
        ],
    )
    return _style(chart, 300), table.rename(columns={"value": "Total", "games": "Simulated games", "share": "Share"})


def team_score_distribution(prediction: MatchupPrediction) -> tuple[alt.Chart, pd.DataFrame]:
    """Each team's simulated points, with lines that show their shapes on one axis."""
    highest = int(max(prediction.home_scores.max(), prediction.away_scores.max()))
    frames = []
    for team, scores in ((prediction.home, prediction.home_scores), (prediction.away, prediction.away_scores)):
        counts = _counts(scores, 0, highest).assign(Team=team)
        counts["smoothed"] = counts["share"].rolling(3, center=True, min_periods=1).mean()
        frames.append(counts)
    table = pd.concat(frames, ignore_index=True)
    chart = alt.Chart(table).mark_line(strokeWidth=2, strokeJoin="round", strokeCap="round").encode(
        x=alt.X("value:Q", title="Points scored", scale=alt.Scale(domainMin=0, nice=False)),
        y=alt.Y("smoothed:Q", title="Share of simulated games", axis=alt.Axis(format=".0%", tickCount=4)),
        color=alt.Color("Team:N", scale=_team_scale(prediction)),
        tooltip=[
            "Team:N",
            alt.Tooltip("value:Q", title="Points"),
            alt.Tooltip("share:Q", title="Exactly this score", format=".1%"),
        ],
    )
    wide = table.pivot(index="value", columns="Team", values="games").reset_index().rename(columns={"value": "Points"})
    return _style(chart, 300), wide


def final_score_heatmap(prediction: MatchupPrediction) -> tuple[alt.Chart, pd.DataFrame]:
    """Which combinations of final scores came up most, grouped into 7-point cells."""
    home_bin = prediction.home_scores // HEATMAP_BIN * HEATMAP_BIN
    away_bin = prediction.away_scores // HEATMAP_BIN * HEATMAP_BIN
    table = pd.DataFrame({"home": home_bin, "away": away_bin}).value_counts().reset_index(name="games")
    table["share"] = table["games"] / len(home_bin)
    table["home_range"] = table["home"].map(lambda low: f"{low}-{low + HEATMAP_BIN - 1}")
    table["away_range"] = table["away"].map(lambda low: f"{low}-{low + HEATMAP_BIN - 1}")
    table["home_end"] = table["home"] + HEATMAP_BIN - 0.5
    table["away_end"] = table["away"] + HEATMAP_BIN - 0.5
    table = table[table["share"] >= 0.0005]
    cells = alt.Chart(table).mark_rect(cornerRadius=2).encode(
        x=alt.X("home:Q", title=f"{prediction.home} points", scale=alt.Scale(domainMin=0, nice=False), axis=alt.Axis(tickCount=8, grid=False)),
        x2="home_end:Q",
        y=alt.Y("away:Q", title=f"{prediction.away} points", scale=alt.Scale(domainMin=0, nice=False), axis=alt.Axis(tickCount=8, grid=False)),
        y2="away_end:Q",
        color=alt.Color(
            "share:Q",
            title="Share of simulated games",
            scale=alt.Scale(range=[HEAT_LOW, HEAT_HIGH]),
            legend=alt.Legend(format=".0%", gradientLength=160, orient="top"),
        ),
        tooltip=[
            alt.Tooltip("home_range:N", title=prediction.home),
            alt.Tooltip("away_range:N", title=prediction.away),
            alt.Tooltip("share:Q", title="Share", format=".1%"),
        ],
    )
    limit = int(max(table["home_end"].max(), table["away_end"].max()))
    tie_line = alt.Chart(pd.DataFrame({"x": [0, limit], "y": [0, limit]})).mark_line(color=MUTED, strokeWidth=1).encode(x="x:Q", y="y:Q")
    view = table.sort_values("share", ascending=False)[["home_range", "away_range", "share"]].rename(
        columns={"home_range": prediction.home, "away_range": prediction.away, "share": "Share"}
    )
    return _style(alt.layer(cells, tie_line), 380), view


def factor_chart(prediction: MatchupPrediction) -> tuple[alt.Chart, pd.DataFrame]:
    """What pushed the projected margin toward each team, in points, largest first."""
    table = pd.DataFrame(
        {
            "Factor": [factor["label"] for factor in prediction.factors],
            "Points": [round(factor["margin"], 1) for factor in prediction.factors],
        }
    )
    table["Favors"] = np.where(table["Points"] >= 0, prediction.home, prediction.away)
    table["Size"] = table["Points"].abs()
    table["Edge"] = [f"{team} +{size:.1f}" for team, size in zip(table["Favors"], table["Size"])]
    order = table["Factor"].tolist()
    limit = max(1.0, float(table["Size"].max()) * 1.35)
    y = alt.Y("Factor:N", sort=order, title=None, axis=alt.Axis(ticks=False, domain=False, labelFontSize=13, labelLimit=220))
    bars = alt.Chart(table).mark_bar(height=18, cornerRadius=3).encode(
        x=alt.X(
            "Points:Q",
            title=f"Points toward {prediction.away}  ←  0  →  Points toward {prediction.home}",
            scale=alt.Scale(domain=[-limit, limit]),
            axis=alt.Axis(tickCount=7),
        ),
        y=y,
        color=alt.Color("Favors:N", scale=_team_scale(prediction)),
        tooltip=["Factor:N", "Edge:N"],
    )
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=BASELINE, strokeWidth=1).encode(x="x:Q")
    positive = alt.Chart(table[table["Points"] >= 0]).mark_text(align="left", dx=6, color=INK, font=FONT, fontSize=12).encode(
        x="Points:Q", y=y, text=alt.Text("Size:Q", format=".1f")
    )
    negative = alt.Chart(table[table["Points"] < 0]).mark_text(align="right", dx=-6, color=INK, font=FONT, fontSize=12).encode(
        x="Points:Q", y=y, text=alt.Text("Size:Q", format=".1f")
    )
    return _style(alt.layer(bars, zero, positive, negative), 44 * len(table)), table[["Factor", "Edge"]]


def comparison_chart(prediction: MatchupPrediction, group: str) -> tuple[alt.Chart, pd.DataFrame]:
    """Each stat as a percentile among FBS teams, where 100 is always the best."""
    data = prediction.comparison[prediction.comparison["group"] == group]
    known = data.dropna(subset=["percentile"])
    order = data["metric"].drop_duplicates().tolist()
    y = alt.Y("metric:N", sort=order, title=None, axis=alt.Axis(ticks=False, domain=False, labelFontSize=12, labelLimit=260, grid=True))
    pivot = known.pivot(index="metric", columns="team", values="percentile")
    both = pivot.dropna().reset_index()
    both = both.rename(columns={prediction.home: "home_value", prediction.away: "away_value"})
    connector = alt.Chart(both).mark_rule(color=BASELINE, strokeWidth=2).encode(
        x=alt.X(
            "home_value:Q",
            title="Percentile among FBS teams (higher is always better)",
            scale=alt.Scale(domain=[0, 100]),
            axis=alt.Axis(values=[0, 25, 50, 75, 100], grid=True),
        ),
        x2="away_value:Q",
        y=y,
    )
    points = alt.Chart(known).mark_point(filled=True, size=130, opacity=1, stroke=SURFACE, strokeWidth=2).encode(
        x="percentile:Q",
        y=y,
        color=alt.Color("team:N", scale=_team_scale(prediction)),
        shape=alt.Shape("side:N", scale=alt.Scale(domain=["home", "away"], range=["circle", "diamond"]), legend=None),
        tooltip=[
            alt.Tooltip("metric:N", title="Stat"),
            alt.Tooltip("team:N", title="Team"),
            alt.Tooltip("value:Q", title="Value"),
            alt.Tooltip("percentile:Q", title="Percentile", format=".0f"),
        ],
    )
    value = data.pivot(index="metric", columns="team", values="value").reindex(order)
    rank = data.pivot(index="metric", columns="team", values="percentile").reindex(order).round(0)
    table = pd.DataFrame(index=value.index)
    for team in (prediction.home, prediction.away):
        table[team] = value[team]
        table[f"{team} percentile"] = rank[team]
    return _style(alt.layer(connector, points), 42 * len(order) + 50), table.reset_index().rename(columns={"metric": "Stat"})
