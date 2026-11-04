"""Integrity checks shared by verify_leak_fix.py-style audits and run_benchmarks.py.

Each function takes a DataFrame of the shipped dataset (Date, winner_name,
loser_name, surface, MaxW, MaxL) and returns a plain value, so the benchmark
report can show the number instead of asserting a verdict.
"""

import numpy as np
import pandas as pd

# corr(h2h_ratio, outcome) above this is the signature of the 2026-10 leak
# (leaked dataset scored +0.336; leak-free is ~0.10).
H2H_CORR_MAX = 0.15

_KEY = ["Date", "winner_name", "loser_name"]


def duplicate_groups(df: pd.DataFrame) -> dict:
    """Count duplicate (Date, winner, loser) rows: >0 means the many-to-many merge is back."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=_KEY)
    mask = df.duplicated(subset=_KEY, keep=False)
    n_dup = int(mask.sum())
    return {
        "total_rows": len(df),
        "duplicate_rows": n_dup,
        "duplicate_groups": int(df.loc[mask].groupby(_KEY).ngroups),
        "passed": n_dup == 0,
    }


def h2h_correlation(df: pd.DataFrame) -> float:
    """corr(h2h_ratio, outcome) as the model computes it: long format, shift(1).cumsum per pair+surface."""
    cols = ["Date", "winner_name", "loser_name", "surface"]
    df = df[cols].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "winner_name", "loser_name"])
    if df.empty:
        return float("nan")

    w = df[["Date", "winner_name", "loser_name", "surface"]]
    w.columns = ["Date", "name", "opp_name", "surface"]
    w["target"] = 1
    l = df[["Date", "winner_name", "loser_name", "surface"]]
    l.columns = ["Date", "opp_name", "name", "surface"]
    l["target"] = 0
    long = pd.concat([w, l]).sort_values("Date").reset_index(drop=True)

    keys = ["name", "opp_name", "surface"]
    if "surface" not in long.columns:
        long["surface"] = ""
    long["h2h_wins"] = long.groupby(keys)["target"].transform(
        lambda x: x.shift(1).cumsum()).fillna(0)
    long["h2h_games"] = long.groupby(keys)["target"].transform(
        lambda x: x.shift(1).expanding().count()).fillna(0)
    long["h2h_ratio"] = (
        long["h2h_wins"] / long["h2h_games"].replace(0, np.nan)).fillna(0.5)
    return float(long["h2h_ratio"].corr(long["target"]))


def market_brier(df: pd.DataFrame, start_year: int = 2018) -> float:
    """Market Brier on the winning row, odds-normalised implied probability."""
    b = df[pd.to_datetime(df["Date"], errors="coerce").dt.year >= start_year].copy()
    b = b.dropna(subset=["MaxW", "MaxL"])
    b = b[(b["MaxW"] > 0) & (b["MaxL"] > 0)]
    if b.empty:
        return float("nan")
    p = (1 / b["MaxW"]) / ((1 / b["MaxW"]) + (1 / b["MaxL"]))
    return float(((p - 1.0) ** 2).mean())
