import pandas as pd
import pytest

from leak_checks import duplicate_groups, h2h_correlation, market_brier


def test_duplicate_groups_counts_both_rows():
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
        "winner_name": ["a", "a"],
        "loser_name": ["b", "b"],
    })
    assert duplicate_groups(df)["duplicate_rows"] == 2


def test_market_brier_coin_flip_is_quarter():
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2019-01-01"]),
        "MaxW": [2.0],
        "MaxL": [2.0],
    })
    assert market_brier(df) == pytest.approx(0.25)


def test_h2h_correlation_pinned_on_six_matches():
    # One pair, a always beats b: the lagged per-pair win ratio tracks the
    # outcome every time after the first match, so the correlation must be
    # strongly positive (sqrt(5/6)). This pins the long-format shift(1).cumsum
    # recipe: swapping it for a same-row computation would push it to 1.0.
    df = pd.DataFrame({
        "Date": pd.to_datetime([
            "2020-01-01", "2020-02-01", "2020-03-01",
            "2020-04-01", "2020-05-01", "2020-06-01",
        ]),
        "winner_name": ["a"] * 6,
        "loser_name": ["b"] * 6,
        "surface": ["Hard"] * 6,
    })
    assert h2h_correlation(df) == pytest.approx(0.9129, abs=1e-4)
