"""The app's look: colors, type, and reusable page pieces. Team names are always escaped."""

from __future__ import annotations

from html import escape

import streamlit as st

from cfbsim.app.charts import AWAY_COLOR, HOME_COLOR

STYLES = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700;800&family=Barlow:wght@400;500;600&display=swap');
:root {{
  --bg: #0a1020; --card: #131b2e; --card-2: #1a2439; --ink: #f0f3f8; --ink-2: #bcc5d3; --muted: #939dae;
  --line: rgba(255,255,255,0.11); --home: {HOME_COLOR}; --away: {AWAY_COLOR}; --gold: #f5b82e;
  --sans: 'Barlow', system-ui, -apple-system, 'Segoe UI', sans-serif;
  --display: 'Barlow Condensed', 'Barlow', system-ui, -apple-system, 'Segoe UI', sans-serif;
}}
html, body, .stApp, [data-testid="stMarkdownContainer"] {{ font-family: var(--sans); }}
.stApp {{ background: var(--bg); color: var(--ink); }}
[data-testid="stHeader"] {{ background: transparent; height: 0; }}
.block-container {{ padding: 1.2rem 1.25rem 3rem; max-width: 1180px; }}

.brand {{ display: flex; justify-content: space-between; align-items: center; gap: 16px; flex-wrap: wrap;
  padding: 6px 2px 14px; }}
.brand-name {{ font-family: var(--display); font-size: 34px; font-weight: 800; letter-spacing: 0.02em;
  text-transform: uppercase; line-height: 1; color: var(--ink); }}
.brand-name span {{ color: var(--gold); }}
.brand-note {{ color: var(--muted); font-size: 13px; margin-top: 4px; }}
.chips {{ display: flex; gap: 8px; flex-wrap: wrap; }}
.chip {{ background: var(--card); border: 1px solid var(--line); border-radius: 999px; padding: 5px 12px;
  font-size: 12.5px; color: var(--ink-2); white-space: nowrap; }}
.chip b {{ color: var(--ink); font-weight: 600; }}

div[data-testid="stRadio"] > label {{ display: none; }}
div[data-testid="stRadio"] [role="radiogroup"] {{ gap: 6px; flex-wrap: wrap; border-bottom: 1px solid var(--line);
  padding-bottom: 12px; margin-bottom: 8px; }}
div[data-testid="stRadio"] [role="radiogroup"] label {{ background: var(--card); border: 1px solid var(--line);
  border-radius: 8px; padding: 7px 15px; margin: 0; cursor: pointer; }}
div[data-testid="stRadio"] [role="radiogroup"] label > div > div:not(:has(p)) {{ display: none; }}
div[data-testid="stRadio"] [role="radiogroup"] label p {{ font-family: var(--display); font-size: 17px;
  font-weight: 700; letter-spacing: 0.03em; text-transform: uppercase; color: var(--ink-2); }}
div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {{ background: var(--gold);
  border-color: var(--gold); }}
div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) p {{ color: #0a1020; }}

.scoreboard {{ background: linear-gradient(180deg, var(--card-2), var(--card)); border: 1px solid var(--line);
  border-radius: 16px; overflow: hidden; margin: 8px 0 12px; }}
.score-stripe {{ display: flex; height: 6px; }}
.score-stripe div {{ flex: 1; }}
.score-meta {{ text-align: center; color: var(--ink-2); font-size: 13px; padding: 12px 16px 0; }}
.score-grid {{ display: grid; grid-template-columns: 1fr auto 1fr; gap: 12px; align-items: center; padding: 14px 24px 8px; }}
.team-side {{ display: flex; flex-direction: column; gap: 4px; min-width: 0; }}
.team-side.right {{ align-items: flex-end; text-align: right; }}
.team-head {{ display: flex; align-items: center; gap: 12px; min-width: 0; }}
.team-side.right .team-head {{ flex-direction: row-reverse; }}
.badge {{ width: 58px; height: 58px; border-radius: 12px; display: flex; align-items: center; justify-content: center;
  font-family: var(--display); font-weight: 800; font-size: 20px; flex: none; }}
.team-title {{ font-family: var(--display); font-size: 30px; font-weight: 700; line-height: 1.02; color: var(--ink); }}
.team-sub {{ font-size: 13px; color: var(--ink-2); margin-top: 3px; }}
.win {{ font-family: var(--display); font-size: 58px; font-weight: 800; line-height: 1; color: var(--ink); margin-top: 8px; }}
.win.dim {{ color: var(--muted); font-size: 44px; }}
.win-label {{ font-size: 12px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); display: flex; align-items: center; gap: 6px; }}
.role-dot {{ width: 10px; height: 10px; flex: none; }}
.role-dot.home {{ background: var(--home); border-radius: 50%; }}
.role-dot.away {{ background: var(--away); transform: rotate(45deg); }}
.center {{ text-align: center; padding: 0 8px; }}
.center-label {{ font-size: 12px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); }}
.center-score {{ font-family: var(--display); font-size: 62px; font-weight: 800; line-height: 1.02; color: var(--ink); white-space: nowrap; }}
.center-score span {{ color: var(--muted); padding: 0 6px; font-weight: 600; }}
.pill {{ display: inline-block; background: var(--gold); color: #0a1020; font-weight: 600; font-size: 12.5px;
  border-radius: 999px; padding: 3px 11px; margin-top: 4px; }}
.winbar {{ display: flex; gap: 2px; height: 12px; margin: 8px 24px 4px; }}
.winbar .home {{ background: var(--home); border-radius: 0 4px 4px 0; }}
.winbar .away {{ background: var(--away); border-radius: 4px 0 0 4px; }}
.verdict {{ padding: 8px 24px 16px; color: var(--ink); font-size: 15.5px; line-height: 1.45; }}
.verdict small {{ display: block; color: var(--muted); font-size: 12.5px; margin-top: 5px; }}

.tiles {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 8px 0; }}
.tile {{ background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 13px 15px; min-width: 0; }}
.tile-label {{ color: var(--ink-2); font-size: 13px; }}
.tile-value {{ font-family: var(--display); font-size: 34px; font-weight: 700; color: var(--ink); line-height: 1.1; }}
.tile-sub {{ color: var(--ink-2); font-size: 12.5px; line-height: 1.5; margin-top: 2px; }}
.section {{ font-family: var(--display); font-size: 24px; font-weight: 700; letter-spacing: 0.01em; color: var(--ink); margin: 20px 0 0; }}
.section-note {{ color: var(--ink-2); font-size: 14px; margin: 1px 0 8px; line-height: 1.45; }}
.note {{ border-left: 3px solid var(--gold); background: rgba(245,184,46,0.09); padding: 8px 13px; border-radius: 0 8px 8px 0;
  margin: 6px 0; font-size: 13.5px; color: var(--ink); }}
.note.info {{ border-left-color: #7f8ca0; background: rgba(255,255,255,0.05); }}
.status {{ border-radius: 10px; padding: 11px 15px; margin: 2px 0 10px; font-size: 14px; line-height: 1.45;
  border: 1px solid rgba(245,184,46,0.55); background: rgba(245,184,46,0.1); color: var(--ink); }}
.status strong {{ display: block; font-size: 15px; }}
.feature {{ background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 13px 15px; }}
.feature-label {{ color: var(--gold); font-size: 12px; letter-spacing: 0.07em; text-transform: uppercase; font-weight: 600; }}
.feature-title {{ font-family: var(--display); font-size: 23px; font-weight: 700; color: var(--ink); line-height: 1.1; margin-top: 3px; }}
.feature-sub {{ color: var(--ink-2); font-size: 13px; margin-top: 3px; }}
.foot {{ color: var(--muted); font-size: 12.5px; margin-top: 26px; padding-top: 12px; border-top: 1px solid var(--line); line-height: 1.5; }}
button[kind="primary"], button[data-testid="stBaseButton-primaryFormSubmit"] {{ background: var(--gold) !important;
  border-color: var(--gold) !important; color: #0a1020 !important; font-weight: 600; }}
[data-testid="stTabs"] button p {{ font-size: 15px; font-weight: 600; }}
@media (max-width: 760px) {{
  .score-grid {{ grid-template-columns: 1fr; padding: 12px 16px 4px; }}
  .team-side.right {{ align-items: flex-start; text-align: left; }}
  .team-side.right .team-head {{ flex-direction: row; }}
  .center {{ order: -1; }}
  .win {{ font-size: 44px; }} .win.dim {{ font-size: 36px; }} .center-score {{ font-size: 50px; }}
  .team-title {{ font-size: 25px; }} .winbar {{ margin: 8px 16px 4px; }} .verdict {{ padding: 8px 16px 14px; }}
  .tiles {{ grid-template-columns: repeat(2, 1fr); }} .tile-value {{ font-size: 28px; }}
}}
</style>
"""


def apply_styles() -> None:
    st.markdown(STYLES, unsafe_allow_html=True)


def _luminance(color: str) -> float | None:
    code = str(color or "").lstrip("#")
    if len(code) != 6:
        return None
    try:
        channels = [int(code[index : index + 2], 16) / 255 for index in (0, 2, 4)]
    except ValueError:
        return None
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def team_accent(info: dict | None) -> tuple[str, str]:
    """A team color that shows up on the dark page, and readable text to put on it."""
    info = info or {}
    fallback = "#5a6880"
    for color in (info.get("color"), info.get("alternate_color")):
        luminance = _luminance(color)
        if luminance is not None and luminance >= 0.035 and luminance <= 0.95:
            return color, "#0a1020" if luminance > 0.25 else "#ffffff"
    return fallback, "#ffffff"


def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def section(title: str, note: str = "") -> None:
    extra = f'<div class="section-note">{escape(note)}</div>' if note else ""
    html(f'<div class="section">{escape(title)}</div>{extra}')


def tile(label: str, value: str, lines: list[str]) -> str:
    details = "".join(f"<div>{escape(line)}</div>" for line in lines)
    return (
        f'<div class="tile"><div class="tile-label">{escape(label)}</div>'
        f'<div class="tile-value">{escape(value)}</div><div class="tile-sub">{details}</div></div>'
    )


def feature(label: str, title: str, sub: str) -> str:
    return (
        f'<div class="feature"><div class="feature-label">{escape(label)}</div>'
        f'<div class="feature-title">{escape(title)}</div><div class="feature-sub">{escape(sub)}</div></div>'
    )


def footer() -> None:
    html(
        '<div class="foot">Data provided by CollegeFootballData.com. FPI is an ESPN rating. Not official NCAA, conference, '
        "or school data, and team names belong to their owners. For analysis and education only, and not betting advice. "
        "Betting lines are never used as model inputs.</div>"
    )
