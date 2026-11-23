import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tensorflow")

from hydra_model import (
    _build_swapped_sample,
    _code_commit,
    _market_prob_index,
    _swap_market_context,
    compute_ewma_features,
    write_results,
)

def test_code_commit_prefers_the_launcher_value(monkeypatch):
    monkeypatch.setenv("TRADING_ALPHA_CODE_COMMIT", "abc123")
    assert _code_commit() == "abc123"

def test_code_commit_never_returns_an_empty_string(monkeypatch):
    monkeypatch.delenv("TRADING_ALPHA_CODE_COMMIT", raising=False)
    assert _code_commit()

def test_compute_ewma_does_not_use_future_values():
    df = pd.DataFrame({
        "name": ["a", "a"],
        "Date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
        "x": [1.0, 10.0],
    })
    out = compute_ewma_features(df, ["x"], "name", com=1)
    assert out.loc[0, "ewma_x"] == 0.0
    assert out.loc[1, "ewma_x"] == 1.0

def test_swap_market_context_swaps_probabilities_and_odds():
    cols = ["fair_prob_A", "implied_A", "implied_B", "Market_Inefficiency"]
    x = np.array([0.7, 1 / 1.5, 1 / 3.0, 0.0])
    y = _swap_market_context(x, cols)
    assert y[0] == pytest.approx(0.3)
    assert y[1] == pytest.approx(1 / 3.0)
    assert y[2] == pytest.approx(1 / 1.5)
    assert y[3] == 0.0

def test_build_swapped_sample_swaps_features_targets_and_market_row():
    cols = ["fair_prob_A", "implied_A", "implied_B", "Market_Inefficiency"]
    xa = np.array([1.0, 2.0], dtype="float32")
    xb = np.array([3.0, 4.0], dtype="float32")
    xphy = np.array([0.5], dtype="float32")
    xmkt = np.array([0.7, 1 / 1.5, 1 / 3.0, 0.0], dtype="float32")

    xa_s, xb_s, xphy_s, xmkt_s, y_prob, y_stake, xgb_x, xgb_y = _build_swapped_sample(
        xa, xb, xphy, xmkt, cols, 1.0, 2.75)

    assert np.array_equal(xa_s, xb)
    assert np.array_equal(xb_s, xa)
    assert np.array_equal(xmkt_s, _swap_market_context(xmkt, cols))
    assert y_prob == pytest.approx(0.0)
    assert y_stake[0] == pytest.approx(0.0)
    assert y_stake[1] == pytest.approx(2.75)
    assert np.array_equal(xgb_x[-len(xmkt_s):], xmkt_s)
    assert xgb_y == pytest.approx(0.0 - (1.0 - 0.7))

def test_market_prob_index_falls_back_to_implied_a():
    assert _market_prob_index(
        ["implied_A", "implied_B", "fair_prob_A"]) == 2
    assert _market_prob_index(
        ["Market_Inefficiency", "implied_A", "implied_B"]) == 1

def test_write_results_reports_ruin_as_zero(tmp_path):
    path = tmp_path / "walkforward_results.txt"
    equity = {"Flat": [-186.0], "Perc": [1000.0], "Kelly": [1000.0],
              "Reference": [1000.0], "StakeHead": [1000.0]}
    write_results(path, equity, [], {"rows": 1}, 0.0, {})
    text = path.read_text(encoding="utf-8")
    assert "Flat: EUR 0 (P: -1000) (bankrupt)" in text
    assert "EUR -186" not in text

def test_write_results_contains_provenance(tmp_path):
    path = tmp_path / "walkforward_results.txt"
    equity = {"Flat": [1000.0], "Perc": [1000.0], "Kelly": [1000.0],
              "Reference": [1000.0], "StakeHead": [1000.0]}
    write_results(path, equity, [], {"rows": 10, "sha256": "abc"}, 0.0, {})
    text = path.read_text(encoding="utf-8")
    for expected in [
        "Bets: 0",
        "Wins: 0/0",
        "Dataset rows: 10",
        "Dataset sha256: abc",
        "Opening odds coverage: 0.0%",
        "StakeHead: EUR",
    ]:
        assert expected in text
