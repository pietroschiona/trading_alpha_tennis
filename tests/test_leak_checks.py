import pandas as pd
import pytest

from leak_checks import duplicate_groups, market_brier


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
