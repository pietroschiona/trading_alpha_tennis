# Reproducibility

Every published number is tied to an artifact you can inspect and, with the same inputs, rebuild.

## What is pinned

| Item | Where |
|---|---|
| Code | git commit recorded in `walkforward_results.txt` (`Code commit:`) and in the manifest |
| Dataset | `tennis_features_v36.meta.json`: rows, date range, sha256, seed, tolerance |
| Run config | `walkforward_manifest.json`: years, seed, data mode, modeling rows, results |
| Bets | `walkforward_bets.csv`: date, match, probability, market probability, odds, stake, PnL, outcome |
| Environment | `requirements.txt` pins; the reference image is `python:3.12-slim` with those pins |
| Randomness | `RNG_SEED = 42` for Python, NumPy and TensorFlow; XGBoost runs with its library default seed and `n_jobs = -1` |

## What is not reproducible bit-for-bit

- TensorFlow floating-point reductions can differ across CPUs and library versions. Expect the
  same decisions in practice, not identical 12th decimals.
- The raw sources are live: tennis-data.co.uk rotates its URL segment and TML-Database changes.
  A rebuild in a future year will not produce the same sha256. The sidecar records the old hash,
  so a mismatch is visible instead of silent.
- The opening-odds CSV is scraped and not redistributable. Without it the canonical run refuses
  to start; the labelled degraded mode is not comparable.

## Commands

```bash
# 1. data
python download_data.py
python data_pipeline.py

# 2. audit (must pass before any number)
python verify_leak_fix.py

# 3. canonical run (~8 min on the reference CPU image)
python -c "import hydra_model as m; raise SystemExit(m.main())"

# 4. statistical report
python run_benchmarks.py

# 5. README figures from the published bets
python make_figures.py
```

## How to check a claim in one minute

1. Open `walkforward_results.txt`; compare `Dataset sha256` with the sidecar and `Code commit` with
   `git log`.
2. Open `walkforward_manifest.json`; check `data_mode: opening_odds` and the coverage.
3. Open `benchmarks/validation_report.md`; look at the return CI and the model-vs-market Brier.
4. If any of the three disagrees with the README, the README is wrong: open an issue.

`run_benchmarks.py` already enforces step 1 and 2: a mismatch produces a report marked **VOID**
with no p-values and exit code 1.
