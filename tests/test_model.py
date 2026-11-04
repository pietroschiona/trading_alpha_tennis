import numpy as np
import pandas as pd
import pytest

from hydra_model import (
    _swap_market_context,
    compute_ewma_features,
    write_results,
)

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
