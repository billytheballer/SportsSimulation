"""Hand-built API-shaped records. These exist only to test code and never reach a report."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from cfbsim.games import parse_games


def make_record(game_id: int = 1, **overrides: Any) -> dict[str, Any]:
    record = {
        "id": game_id,
        "season": 2019,
        "week": 1,
        "seasonType": "regular",
        "startDate": "2019-09-07T19:00:00.000Z",
        "completed": True,
        "neutralSite": False,
        "conferenceGame": False,
        "attendance": 50000,
        "venueId": 7,
        "homeId": 1,
        "homeTeam": "Home U",
        "homeConference": "A",
        "homeClassification": "fbs",
        "homePoints": 28,
        "homeLineScores": [7, 7, 7, 7],
        "homePostgameWinProbability": 0.9,
        "homePregameElo": 1600,
        "homePostgameElo": 1620,
        "awayId": 2,
        "awayTeam": "Away State",
        "awayConference": "B",
        "awayClassification": "fbs",
        "awayPoints": 14,
        "awayLineScores": [0, 7, 0, 7],
        "awayPostgameWinProbability": 0.1,
        "awayPregameElo": 1500,
        "awayPostgameElo": 1480,
        "excitementIndex": 3.2,
    }
    return {**record, **overrides}


@pytest.fixture
def schedule() -> pd.DataFrame:
    """Three fake teams playing each other over two seasons."""
    pairings = [(1, 2, 21), (2, 3, -7), (3, 1, 3), (1, 3, 14), (2, 1, -10), (3, 2, 24)]
    records = []
    for index, (home, away, margin) in enumerate(pairings):
        season = 2019 if index < 3 else 2020
        records.append(
            make_record(
                index + 1,
                season=season,
                startDate=f"{season}-09-{index + 1:02d}T19:00:00.000Z",
                homeId=home,
                awayId=away,
                homePoints=30 + margin,
                awayPoints=30,
            )
        )
    games, _ = parse_games(records)
    return games
