import hashlib
import json
from pathlib import Path

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


def test_sidecar_and_dropped_columns():
    root = Path(__file__).resolve().parents[1]
    csv = root / "tennis_features_v36.csv"
    meta_path = root / "tennis_features_v36.meta.json"
    if not csv.exists() or not meta_path.exists():
        pytest.skip("local dataset or sidecar missing")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    columns = pd.read_csv(csv, nrows=0).columns
    for col in ("score", "Wsets", "_w_sets", "_odds_w_sets", "dedup_key"):
        assert col not in columns, f"outcome helper column {col} leaked into the dataset"

    h = hashlib.sha256()
    with open(csv, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    assert meta["sha256"] == h.hexdigest()
    assert meta["rows"] == len(pd.read_csv(csv, usecols=["Date"]))
    assert meta["date_min"] <= meta["date_max"]

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
