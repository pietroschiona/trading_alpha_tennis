import pandas as pd
import pytest

from run_benchmarks import bootstrap_roi, load_results


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
