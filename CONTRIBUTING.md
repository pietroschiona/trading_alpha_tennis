# Contributing

This repository is primarily a research post-mortem: the value is in the audit trail,
not in adding features.

## The one rule that matters

Any change that touches data loading or feature engineering must keep these properties:

1. **One match, one odds line.** Never join odds to matches on a loose key
   (`year + players`, name similarity, nearest date alone). Use real match identity.
2. **No future.** Every historical feature (`ELO`, `EWMA`, `h2h`, fatigue) uses only
   strictly earlier dates.
3. **No silent fallback.** If opening odds are missing, the run fails or drops the match
   explicitly. It never substitutes closing odds without recording `closing_fallback`.
4. **Evidence or it did not happen.** A performance claim needs a committed
   `walkforward_results.txt` with provenance, a `walkforward_bets.csv`, and a benchmark
   report whose dataset hash matches the sidecar.

## Before opening a PR

```bash
python -m pytest tests -q
python verify_leak_fix.py
python run_benchmarks.py --quick
```

All three must pass. If `verify_leak_fix.py` fails, the numbers are meaningless and
the PR is rejected regardless of the ROI.

## Reporting a result

State the model Brier **and** the market Brier on the same bets. A ROI significantly
above zero with a Brier above the market is a data problem, not an edge.
