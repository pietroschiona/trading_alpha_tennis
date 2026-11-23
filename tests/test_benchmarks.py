import sys

import pandas as pd
import pytest

import run_benchmarks
from run_benchmarks import (
    _edge_verdict,
    bootstrap_roi,
    brier_metrics,
    calibration_metrics,
    load_results,
)


def test_load_results_parses_provenance(tmp_path):
    p = tmp_path / "walkforward_results.txt"
    p.write_text(
        "Bets: 10\nWins: 6/10 (60.0%)\nDataset rows: 20\n"
        "Dataset sha256: abc\nOpening odds coverage: 0.0%\n",
        encoding="utf-8",
    )
    r = load_results(p)
    assert r["total_bets"] == 10
    assert r["wins"] == 6
    assert r["dataset_rows"] == 20


def test_load_results_rejects_missing_provenance(tmp_path):
    p = tmp_path / "walkforward_results.txt"
    p.write_text("Bets: 10\nWins: 6/10\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        load_results(p)


def test_bootstrap_roi_positive():
    bets = pd.DataFrame({"stake_head": [10.0] * 100, "pnl_head": [1.0] * 100})
    res = bootstrap_roi(bets, n_boot=1000)
    assert res["roi"] == pytest.approx(0.1)
    assert res["ci_95"][0] > 0
    assert res["p_value"] < 0.05


def test_bootstrap_roi_zero():
    bets = pd.DataFrame({"stake_head": [10.0] * 100, "pnl_head": [1.0, -1.0] * 50})
    res = bootstrap_roi(bets, n_boot=1000)
    assert res["roi"] == pytest.approx(0.0)
    assert res["p_value"] > 0.05


def test_bootstrap_roi_reports_unavailable_on_empty_or_missing_columns():
    empty = pd.DataFrame({"stake_head": [], "pnl_head": []})
    assert bootstrap_roi(empty)["available"] is False
    assert bootstrap_roi(pd.DataFrame({"date": ["2020-01-01"]}))["available"] is False
    assert bootstrap_roi(None)["available"] is False

def test_brier_metrics_compares_model_and_market_on_same_bets():
    bets = pd.DataFrame({
        "prob": [0.7, 0.3],
        "won": [True, False],
        "market_prob": [0.6, 0.4],
    })
    scores = brier_metrics(bets)
    assert scores["available"] is True
    assert scores["model_brier"] == pytest.approx(0.09)
    assert scores["market_brier"] == pytest.approx(0.16)
    assert scores["delta"] == pytest.approx(-0.07)

def test_brier_metrics_is_unavailable_without_probabilities():
    assert brier_metrics(pd.DataFrame({"won": [True]})) == {"available": False}
    assert calibration_metrics(None) == {"available": False}

def test_calibration_metrics_reports_bins_and_ece():
    bets = pd.DataFrame({"prob": [0.2, 0.8], "won": [False, True]})
    cal = calibration_metrics(bets)
    assert cal["available"] is True
    assert cal["n"] == 2
    assert len(cal["bins"]) == 2
    # The model under-predicts both outcomes by 0.2, so ECE is 0.2.
    assert cal["ece"] == pytest.approx(0.2)

def test_edge_verdict_is_no_when_the_95pct_ci_contains_zero():
    # p <= 0.05 certifies the 90% interval, so it must not be enough on its own.
    ret = {"available": True, "roi": 0.02, "p_value": 0.01, "ci_95": [-0.01, 0.05]}
    assert _edge_verdict(ret) == "NO"
    ret["ci_95"] = [0.005, 0.05]
    assert _edge_verdict(ret) == "YES"
    assert _edge_verdict(None) == "NOT MEASURED"

def test_main_writes_void_report_and_exits_1_on_sha_mismatch(tmp_path, monkeypatch):
    results = tmp_path / "walkforward_results.txt"
    results.write_text(
        "Code commit: abc+dirty\nBets: 10\nWins: 6/10 (60.0%)\n"
        "Dataset rows: 2\nDataset sha256: deadbeef\n"
        "Opening odds coverage: 0.0%\n",
        encoding="utf-8",
    )
    dataset = tmp_path / "tennis_features_v36.csv"
    dataset.write_text("Date,winner_name,loser_name\n2020-01-01,a,b\n", encoding="utf-8")
    meta = tmp_path / "tennis_features_v36.meta.json"
    meta.write_text('{"rows": 2, "sha256": "deadbeef"}', encoding="utf-8")

    monkeypatch.setattr(run_benchmarks, "BENCH_DIR", tmp_path)
    monkeypatch.setattr(run_benchmarks, "RESULTS_PATH", results)
    monkeypatch.setattr(run_benchmarks, "DATASET_PATH", dataset)
    monkeypatch.setattr(run_benchmarks, "META_PATH", meta)
    monkeypatch.setattr(sys, "argv", ["run_benchmarks.py", "--quick"])

    assert run_benchmarks.main() == 1

    report = (tmp_path / "benchmarks" / "validation_report.md").read_text(encoding="utf-8")
    assert "VOID" in report
    # The only place a p-value can appear is the return-test table row, so its
    # absence proves no number was published.
    assert "| p-value" not in report
    assert "Edge demonstrated: NOT MEASURED" in report
