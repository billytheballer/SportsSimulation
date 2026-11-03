"""College Football Matchup Simulator. Run with: uv run streamlit run streamlit_app.py"""

from __future__ import annotations

import sys
from pathlib import Path

# Lets the app run on hosts that don't install this project as a package.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st

from cfbsim.app import ui
from cfbsim.bundle import DEMO_BUNDLE, PRODUCTION_BUNDLE, PredictionBundle, load_bundle
from cfbsim.predictor import MatchupPrediction, Predictor, assess_freshness

st.set_page_config(page_title="CFB Matchup Simulator", page_icon="🏈", layout="wide")


@st.cache_resource(show_spinner="Loading the model...")
def get_bundle(path: str) -> PredictionBundle:
    return load_bundle(Path(path))


@st.cache_data(show_spinner=False)
def run_prediction(path: str, home: str, away: str, neutral: bool) -> MatchupPrediction:
    return Predictor(get_bundle(path)).predict(home, away, neutral)


def available_bundles() -> dict[str, Path]:
    options = {}
    if PRODUCTION_BUNDLE.exists():
        options["Current season"] = PRODUCTION_BUNDLE
    if DEMO_BUNDLE.exists():
        options["Historical demonstration (2022)"] = DEMO_BUNDLE
    return options


def index_of(names: list[str], preferred: str, fallback: int) -> int:
    return names.index(preferred) if preferred in names else min(fallback, len(names) - 1)


def upcoming_game_picker(bundle: PredictionBundle, names: list[str]) -> dict | None:
    """Optional shortcut that fills in the controls from the published schedule."""
    games = [game for game in bundle.upcoming if game["home"] in names and game["away"] in names]
    if not games:
        return None
    labels = {
        f"{game['kickoff']:%b %-d} · {game['away']} {'vs.' if game['neutral_site'] else 'at'} {game['home']}": game
        for game in games
    }
    choice = st.selectbox("Upcoming games (optional)", ["Choose your own matchup", *labels])
    return labels.get(choice)


def main() -> None:
    ui.apply_styles()
    ui.brand_header()

    options = available_bundles()
    if not options:
        st.error("No model bundle found. Build one first: `uv run cfbsim build-demo`")
        st.stop()
    if len(options) > 1:
        choice = st.radio("Team data", list(options), horizontal=True)
    else:
        choice = next(iter(options))
    path = str(options[choice])
    bundle = get_bundle(path)
    ui.status_banner(assess_freshness(bundle))

    predictor = Predictor(bundle)
    names = predictor.team_names()
    preset = upcoming_game_picker(bundle, names)
    preferred_home = preset["home"] if preset else "Ohio State"
    preferred_away = preset["away"] if preset else "Oregon"
    # A new key resets the controls when a different upcoming game is picked.
    form_key = f"{path}-{preferred_home}-{preferred_away}"
    with st.form("matchup"):
        home_column, away_column, site_column, go_column = st.columns([3, 3, 2, 2], vertical_alignment="bottom")
        home = home_column.selectbox("Home team", names, index=index_of(names, preferred_home, 0), key=f"home-{form_key}")
        away = away_column.selectbox("Away team", names, index=index_of(names, preferred_away, 1), key=f"away-{form_key}")
        neutral = site_column.checkbox("Neutral site", value=bool(preset and preset["neutral_site"]), key=f"site-{form_key}")
        submitted = go_column.form_submit_button("Generate prediction", type="primary", use_container_width=True)

    if submitted:
        st.session_state["requested_matchup"] = (path, home, away, neutral)

    request = st.session_state.get("requested_matchup")
    if request is None or request[0] != path:
        st.info("Choose two teams and select **Generate prediction**.")
        ui.footer()
        return
    if request[1] == request[2]:
        st.error("Choose two different teams.")
        ui.footer()
        return

    with st.spinner("Simulating 10,000 games..."):
        prediction = run_prediction(*request)

    ui.win_probability_card(prediction)
    st.write("")
    ui.projection_tiles(prediction)
    ui.show_notices(prediction)
    ui.score_range_section(prediction)
    ui.simulation_section(prediction)
    ui.factors_section(prediction)
    ui.comparison_section(prediction)
    st.write("")
    ui.model_accuracy_section(bundle)
    ui.footer()


main()
