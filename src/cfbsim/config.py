"""Project paths and the fixed season splits used for evaluation."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORTS_DIR = PROJECT_ROOT / "reports"

API_BASE_URL = "https://api.collegefootballdata.com"
# The free tier allows 1,000 calls a month. Stop short so a miscount can't exhaust it.
DEFAULT_MONTHLY_BUDGET = 900

# Splits are by whole season and in time order, so nothing is trained on the future.
WARMUP_SEASONS = (2014, 2015)  # Ratings need time to settle; these games are never scored.
TUNE_SEASONS = (2016, 2017, 2018, 2019, 2020)  # Choose hyperparameters, fit calibration.
VALIDATION_SEASONS = (2021, 2022)  # Compare ideas while developing.
HOLDOUT_SEASONS = (2023, 2024, 2025)  # Scored only with --holdout, once a model is final.

DEVELOPMENT_SEASONS = WARMUP_SEASONS + TUNE_SEASONS + VALIDATION_SEASONS
HISTORICAL_SEASONS = DEVELOPMENT_SEASONS + HOLDOUT_SEASONS
