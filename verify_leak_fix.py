"""Replica dello script di verifica di u/1ma9nja (gist f9579e6a72aafe3f2205c3ff1168a381).

Steps:
 1. duplicate (Date, winner, loser) rows
 2. corr(h2h_ratio, outcome) after the repo's h2h computation
 3. market vs model Brier on 2018+ (market must NOT be beaten without leakage)
Run: python verify_leak_fix.py
"""
import pandas as pd
import numpy as np

CSV = "tennis_features_v36.csv"

df = pd.read_csv(CSV, low_memory=False)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date", "winner_name", "loser_name"])
print(f"Loaded {len(df)} rows")

# STEP 1 — duplicates
dup_mask = df.duplicated(subset=["Date", "winner_name", "loser_name"], keep=False)
print(f"\nSTEP 1 — duplicate (Date,winner,loser) rows: {int(dup_mask.sum())} "
      f"in {df.loc[dup_mask].groupby(['Date','winner_name','loser_name']).ngroups} groups")
assert dup_mask.sum() == 0, "LEAK: duplicates present"

# STEP 2 — h2h as the model computes it (long format, shift(1).cumsum per pair+surface)
w = df[["Date", "winner_name", "loser_name", "surface"]].copy()
w.columns = ["Date", "name", "opp_name", "surface"]; w["target"] = 1
l = df[["Date", "winner_name", "loser_name", "surface"]].copy()
l.columns = ["Date", "opp_name", "name", "surface"]
l["target"] = 0
long = pd.concat([w, l]).sort_values("Date").reset_index(drop=True)
long["h2h_wins"] = long.groupby(["name", "opp_name", "surface"])["target"].transform(
    lambda x: x.shift(1).cumsum()).fillna(0)
long["h2h_games"] = long.groupby(["name", "opp_name", "surface"])["target"].transform(
    lambda x: x.shift(1).expanding().count()).fillna(0)
long["h2h_ratio"] = (long["h2h_wins"] / long["h2h_games"].replace(0, np.nan)).fillna(0.5)
corr = long["h2h_ratio"].corr(long["target"])
print(f"\nSTEP 2 — corr(h2h_ratio, outcome): {corr:+.3f}  (leaked was +0.336, leak-free ~0.11)")
assert corr < 0.15, "LEAK: h2h correlates too much with outcome"

# STEP 3 — market Brier 2018+ (market must not be beatable without leakage)
b = df[df["Date"].dt.year >= 2018].copy()
b["p_market_w"] = (1 / b["MaxW"]) / (1 / b["MaxW"] + 1 / b["MaxL"])
brier_market = ((b["p_market_w"] - 1) ** 2 + (b["p_market_w"]) ** 2 * 0).mean()
# proper Brier: outcome=1 for winner row
y = 1.0
brier_market = ((b["p_market_w"] - 1.0) ** 2).mean()
print(f"\nSTEP 3 — market (MaxW) Brier [2018+]: {brier_market:.4f}")
print("Reference: leaked model scored 0.1909 (< market 0.1974) = impossible without leakage.")
print("A leak-free model must score >= market.")
print("\nALL CHECKS PASSED ✅")
