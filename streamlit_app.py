"""Gridiron Forecast, a college football prediction app. Run with: uv run streamlit run streamlit_app.py"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

# Lets the app run on hosts that don't install this project as a package.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st

from cfbsim.app import theme, ui
from cfbsim.bundle import DEMO_BUNDLE, PRODUCTION_BUNDLE, PredictionBundle, load_bundle
from cfbsim.predictor import MatchupPrediction, Predictor, assess_freshness

st.set_page_config(page_title="Gridiron Forecast", page_icon="🏈", layout="wide")


@st.cache_resource(show_spinner="Loading the model...")
def get_bundle(path: str, modified: float) -> PredictionBundle:
    # `modified` makes a rebuilt bundle load without restarting the app.
    return load_bundle(Path(path))


def modified_time(path: str) -> float:
    return Path(path).stat().st_mtime


@st.cache_data(show_spinner=False)
def run_prediction(
    path: str, modified: float, home: str, away: str, neutral: bool, game_date: date | None
) -> MatchupPrediction:
    return Predictor(get_bundle(path, modified)).predict(home, away, neutral, game_date=game_date)


def main() -> None:
    theme.apply_styles()
    # The current season is the main view. The 2022 demonstration appears only without it.
    source = PRODUCTION_BUNDLE if PRODUCTION_BUNDLE.exists() else DEMO_BUNDLE
    if not source.exists():
        st.error("No model bundle found. Build one first: `uv run cfbsim build-demo`")
        st.stop()
    path = str(source)
    modified = modified_time(path)
    bundle = get_bundle(path, modified)
    ui.brand_header(bundle, assess_freshness(bundle))

    def run(home: str, away: str, neutral: bool, game_date: date | None) -> MatchupPrediction:
        with st.spinner("Simulating 10,000 games..."):
            return run_prediction(path, modified, home, away, neutral, game_date)

    names = Predictor(bundle).team_names()
    has_board = bool(bundle.board)
    views = ["Game Center", *(["Schedule"] if has_board else []), "Teams", "Injuries", "About the Model"]
    if st.session_state.get("view") not in views:
        st.session_state["view"] = views[0]
    view = st.radio("View", views, horizontal=True, key="view", label_visibility="collapsed")

    if view == "Game Center":
        ui.game_center(bundle, names, path, run)
    elif view == "Schedule":
        ui.schedule_page(bundle)
    elif view == "Teams":
        ui.teams_page(bundle)
    elif view == "Injuries":
        ui.injuries_page(bundle)
    else:
        ui.about_page(bundle)
    theme.footer()


main()
