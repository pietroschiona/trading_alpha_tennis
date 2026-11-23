"""Independent anti-leak audit for the shipped dataset.

Run this BEFORE trusting any performance number:

    python verify_leak_fix.py

Checks:
  1. duplicate (Date, winner, loser) rows must be 0;
  2. corr(h2h_ratio, outcome) must stay below the leak threshold (0.15);
  3. the market Brier index (2018+) is printed as the reference to beat;
  4. if a bet log exists, the model Brier is printed next to the market Brier
     on the same bets.

Exit code 0 only when checks 1 and 2 pass.  This is the same recipe as
u/1ma9nja's independent verification script (r/algobetting thread 1tviwt7).
"""
import sys
from pathlib import Path

import pandas as pd

from leak_checks import H2H_CORR_MAX, duplicate_groups, h2h_correlation, market_brier

DATASET = Path("tennis_features_v36.csv")
BETS = Path("walkforward_bets.csv")


def _brier(prob, outcome) -> float:
    p = pd.to_numeric(prob, errors="coerce")
    y = outcome.astype(str).str.lower().isin(["true", "1", "1.0"]).astype(float)
    keep = p.notna()
    return float(((p[keep] - y[keep]) ** 2).mean())


def main() -> int:
    if not DATASET.exists():
        print(f"ERROR: {DATASET} not found. Run data_pipeline.py first.")
        return 1

    df = pd.read_csv(DATASET, low_memory=False)
    print(f"Loaded {len(df):,} rows from {DATASET}")

    print("\nSTEP 1 - duplicate (Date, winner, loser) rows")
    dup = duplicate_groups(df)
    print(f"  rows: {dup['total_rows']:,}")
    print(f"  duplicates: {dup['duplicate_rows']} in {dup['duplicate_groups']} groups")
    print(f"  verdict: {'PASS' if dup['passed'] else 'FAIL'}")

    print("\nSTEP 2 - corr(h2h_ratio, outcome)")
    corr = h2h_correlation(df)
    print(f"  correlation: {corr:+.4f} (leaked dataset: +0.336; threshold: {H2H_CORR_MAX})")
    print(f"  verdict: {'PASS' if abs(corr) < H2H_CORR_MAX else 'FAIL'}")

    print("\nSTEP 3 - market Brier reference (2018+)")
    mb = market_brier(df)
    print(f"  market Brier: {mb:.4f} (leaked dataset: 0.1974; repaired reference: 0.2035)")
    print("  A leak-free model must not beat the market Brier without a reason.")

    if BETS.exists():
        bets = pd.read_csv(BETS)
        if {"prob", "won"} <= set(bets.columns):
            model_b = _brier(bets["prob"], bets["won"])
            print("\nSTEP 4 - model vs market Brier on the logged bets")
            print(f"  bets: {len(bets):,}")
            print(f"  model Brier:  {model_b:.4f}")
            if "market_prob" in bets.columns:
                market_b = _brier(bets["market_prob"], bets["won"])
                print(f"  market Brier: {market_b:.4f}")
                print(f"  delta (model - market): {model_b - market_b:+.4f} "
                      "(positive = worse than the market)")
    else:
        print("\nSTEP 4 - skipped: no walkforward_bets.csv yet")

    ok = dup["passed"] and abs(corr) < H2H_CORR_MAX
    print("\n" + ("ALL CHECKS PASSED" if ok else "LEAK CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
