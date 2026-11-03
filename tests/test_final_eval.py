"""Tests for the final evaluation's safeguards. All games here are made up.

These run the guards on a fake league in temporary folders, so they never touch the real
holdout seasons or the real locked report.
"""

import json
import stat

import pytest
from test_stage2 import make_league
from test_stage3 import regulation_for

from cfbsim import final_eval
from cfbsim.config import DEVELOPMENT_SEASONS, HISTORICAL_SEASONS, HOLDOUT_SEASONS, VALIDATION_SEASONS
from cfbsim.final_eval import (
    RUN_PHRASE,
    FinalEvaluationError,
    data_fingerprint,
    evaluate,
    freeze,
    is_complete,
    report_to_markdown,
    run_once,
    verify_manifest,
)

SETTINGS = {
    "elo_params": {"k": 30.0, "home_advantage": 55.0, "season_carryover": 0.7, "lower_division_initial": 1000.0},
    "score_feature_params": {"halflife": 8.0, "season_weight": 0.5},
    "win_feature_params": {"halflife": 8.0, "season_weight": 0.5},
    "alpha": 100.0,
    "bandwidth": 5.0,
    "win_C": 0.1,
}


@pytest.fixture(scope="module")
def full_league():
    games, stats = make_league(seasons=HISTORICAL_SEASONS)
    return games, stats, regulation_for(games)


@pytest.fixture(autouse=True)
def fewer_simulations(monkeypatch):
    monkeypatch.setattr("cfbsim.stage3.SIMULATIONS", 100)


@pytest.fixture
def workspace(tmp_path):
    """A temporary project with its own frozen sources, Stage 3 report, and cached data."""
    source = tmp_path / "model.py"
    source.write_text("frozen code\n")
    report = tmp_path / "stage3_diagnostics.json"
    report.write_text("{}\n")
    raw = tmp_path / "raw"
    (raw / "games").mkdir(parents=True)
    (raw / "games" / "year=2024.json").write_text("[1]\n")
    directory = tmp_path / "final"
    kwargs = {"directory": directory, "sources": (source,), "stage3_report": report, "raw_dir": raw}
    freeze(settings=SETTINGS, **kwargs)
    return {"source": source, "report": report, "raw": raw, "directory": directory, "kwargs": kwargs}


def pass_preflight(directory):
    (directory / final_eval.PREFLIGHT_NAME).write_text(
        json.dumps({"passed": True, "manifest_hash": final_eval._file_hash(directory / final_eval.MANIFEST_NAME)})
    )


def test_freeze_is_never_replaced(workspace):
    with pytest.raises(FinalEvaluationError, match="never replaced"):
        freeze(settings=SETTINGS, **workspace["kwargs"])


def test_manifest_file_is_read_only(workspace):
    mode = (workspace["directory"] / final_eval.MANIFEST_NAME).stat().st_mode
    assert not mode & stat.S_IWUSR


def test_verification_passes_when_nothing_changed(workspace):
    assert verify_manifest(**workspace["kwargs"])["settings"] == SETTINGS


def test_changed_code_is_refused(workspace):
    workspace["source"].write_text("tuned after peeking\n")
    with pytest.raises(FinalEvaluationError, match="model.py"):
        verify_manifest(**workspace["kwargs"])


def test_changed_stage3_report_is_refused(workspace):
    workspace["report"].write_text('{"changed": true}\n')
    with pytest.raises(FinalEvaluationError, match="stage3_diagnostics"):
        verify_manifest(**workspace["kwargs"])


def test_changed_past_data_is_refused(workspace):
    (workspace["raw"] / "games" / "year=2024.json").write_text("[2]\n")
    with pytest.raises(FinalEvaluationError, match="cached data"):
        verify_manifest(**workspace["kwargs"])


def test_live_season_updates_do_not_count_as_changed_data(workspace):
    before = data_fingerprint(workspace["raw"])
    (workspace["raw"] / "games" / "year=2026.json").write_text("[1]\n")
    (workspace["raw"] / "_usage.json").write_text("{}\n")
    assert data_fingerprint(workspace["raw"]) == before


def test_wrong_phrase_is_refused_before_any_loading(workspace):
    def loader():
        pytest.fail("Holdout data was loaded before the guards passed.")

    with pytest.raises(FinalEvaluationError, match="exact phrase"):
        run_once("yes", loader, **workspace["kwargs"])


def test_run_needs_a_passed_preflight(workspace):
    def loader():
        pytest.fail("Holdout data was loaded before the guards passed.")

    with pytest.raises(FinalEvaluationError, match="preflight"):
        run_once(RUN_PHRASE, loader, **workspace["kwargs"])
    (workspace["directory"] / final_eval.PREFLIGHT_NAME).write_text(
        json.dumps({"passed": False, "manifest_hash": "x"})
    )
    with pytest.raises(FinalEvaluationError, match="didn't pass"):
        run_once(RUN_PHRASE, loader, **workspace["kwargs"])


def test_changed_code_blocks_the_run_before_any_loading(workspace):
    pass_preflight(workspace["directory"])
    workspace["source"].write_text("tuned after peeking\n")

    def loader():
        pytest.fail("Holdout data was loaded before the guards passed.")

    with pytest.raises(FinalEvaluationError, match="Changed since"):
        run_once(RUN_PHRASE, loader, **workspace["kwargs"])


def test_runs_once_then_refuses_and_locks_results(workspace, full_league):
    directory = workspace["directory"]
    pass_preflight(directory)
    assert not is_complete(directory)
    report = run_once(RUN_PHRASE, lambda: full_league, **workspace["kwargs"])
    assert report["seasons"] == list(HOLDOUT_SEASONS)
    assert is_complete(directory)
    for name in (final_eval.RESULT_NAME, final_eval.REPORT_NAME, final_eval.CHECKSUM_NAME):
        assert not (directory / name).stat().st_mode & stat.S_IWUSR

    def loader():
        pytest.fail("A second run loaded data.")

    with pytest.raises(FinalEvaluationError, match="already ran"):
        run_once(RUN_PHRASE, loader, **workspace["kwargs"])


def test_edited_results_stop_counting_as_complete(workspace, full_league):
    directory = workspace["directory"]
    pass_preflight(directory)
    run_once(RUN_PHRASE, lambda: full_league, **workspace["kwargs"])
    result = directory / final_eval.RESULT_NAME
    result.chmod(stat.S_IRUSR | stat.S_IWUSR)
    result.write_text('{"simulation": "much better"}')
    assert not is_complete(directory)


def test_crash_blocks_retry_until_acknowledged_and_records_it(workspace, full_league):
    directory = workspace["directory"]
    pass_preflight(directory)

    def crashing_loader():
        raise RuntimeError("power failure")

    with pytest.raises(RuntimeError):
        run_once(RUN_PHRASE, crashing_loader, **workspace["kwargs"])
    assert not (directory / final_eval.RESULT_NAME).exists()
    with pytest.raises(FinalEvaluationError, match="crash"):
        run_once(RUN_PHRASE, lambda: full_league, **workspace["kwargs"])
    report = run_once(RUN_PHRASE, lambda: full_league, crashed_before_results=True, **workspace["kwargs"])
    assert len(report["run_record"]["attempts"]) == 2


def test_evaluation_uses_only_earlier_seasons_to_fit(full_league):
    """Changing every holdout result must leave earlier seasons' scores untouched."""
    games, stats, regulation = full_league
    first = evaluate(games, stats, regulation, SETTINGS, VALIDATION_SEASONS)
    altered = games.copy()
    later = altered["season"].isin(HOLDOUT_SEASONS)
    altered.loc[later, "margin"] = -altered.loc[later, "margin"]
    altered.loc[later, "home_win"] = 1 - altered.loc[later, "home_win"]
    second = evaluate(altered, stats, regulation, SETTINGS, VALIDATION_SEASONS)
    assert first["win_probability"] == second["win_probability"]
    assert first["points"]["margin"] == second["points"]["margin"]


def test_refuses_to_score_training_seasons(full_league):
    games, stats, regulation = full_league
    with pytest.raises(FinalEvaluationError, match="after the training"):
        evaluate(games, stats, regulation, SETTINGS, (2019,))


def test_report_covers_everything_required_and_never_says_closing_line(full_league):
    games, stats, regulation = full_league
    report = evaluate(games, stats, regulation, SETTINGS, HOLDOUT_SEASONS)
    report["run_record"] = {
        "attempts": ["then"], "completed_at": "now", "frozen_at": "before", "manifest_hash": "x", "settings": SETTINGS,
    }
    text = report_to_markdown(report)
    for season in HOLDOUT_SEASONS:
        assert f"| {season} | simulation |" in text
    for piece in ("Accuracy", "Log loss", "MAE", "RMSE", "Calibration", "elo", "stats_model"):
        assert piece in text
    assert "closing line" not in text.lower().replace("not described as closing lines", "")


def test_preflight_reads_only_development_seasons():
    assert not set(DEVELOPMENT_SEASONS) & set(HOLDOUT_SEASONS)
