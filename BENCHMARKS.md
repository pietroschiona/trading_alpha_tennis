# Benchmarks — measured, not claimed

**Result: no demonstrated edge.** The leak-free system picks winners more often than the market's
break-even rate, but it loses money because its winners are short-priced and its losers are long.
The model is also less well calibrated and scores a worse Brier than the market on the exact bets
it chose.

Every number below comes from the committed artifacts: `walkforward_results.txt`,
`walkforward_bets.csv`, `walkforward_manifest.json` and `benchmarks/validation_report.md`.

## Run provenance

| Field | Value |
|---|---|
| Code commit | `e2a3af3` |
| Generated | 2026-11-23 |
| Dataset rows | 29,712 |
| Dataset SHA-256 | `467c8a17…e897c5bb` |
| Opening-odds coverage | 71.1% overall, 67.9% of 2017+ matches |
| Data mode | `opening_odds` (no closing fallback) |
| Modeling sample | 21,134 matches |
| Period | 2013-2016 warmup, 2017-2025 out-of-sample |
| Bets | 1,789 |
| Integrity checks | PASS (duplicates 0, h2h corr +0.0936) |

## Headline

| Metric | Value | How to read it |
|---|---|---|
| Bets | 1,789 | stake-head bets with a positive signal |
| Win rate | 42.5% (760/1,789) | descriptive; not an edge test |
| Average odds | 2.75 | break-even win rate at these prices ≈ 36.4% |
| ROI | **-8.05%** | stake-weighted return over €11,898 staked |
| 95% bootstrap CI | **[-15.97%, -0.13%]** | entirely below zero: no edge, and the loss is unlikely to be noise |
| p-value (one-sided, ROI ≤ 0) | 0.9769 | cannot reject "no profit" |
| Model Brier | **0.2357** | lower is better |
| Market Brier, same bets | **0.2167** | the opponent |
| Delta (model − market) | **+0.0190** | positive = model worse |
| Calibration ECE | **0.1144** | 11.4 points of miscalibration |

### Why a 42% win rate still loses money

A 42.5% win rate at average odds 2.75 looks profitable: 0.425 × 2.75 ≈ 1.17. It is not, because
winners and losers do not share the same average price:

| Group | Count | Win rate | PnL (EUR) |
|---|---:|---:|---:|
| Odds 1.00-1.50 | 177 | 77% | +14 |
| Odds 1.50-2.00 | 417 | 55% | -86 |
| Odds 2.00-3.00 | 702 | 40% | -184 |
| Odds 3.00-5.00 | 379 | 28% | -443 |
| Odds > 5.00 | 114 | 11% | -259 |

- Average odds of the bets it won: **2.234**. Of the bets it lost: **3.130**.
- The model's mean probability is **0.539**; the realized win rate is **0.4248**.
- The market's mean probability on those same bets is **0.4236**, almost exactly the realized rate.

The model is systematically overconfident, especially on long shots: it buys 11%-win, 5x+ underdogs
with money that the market prices correctly. The 2% execution slippage turns that into a loss.

## Strategy comparison

All five strategies share the same signal; they differ only in staking. A bankroll that reaches
zero stops betting (ruin is absorbing).

| Strategy | Final bankroll | PnL | Note |
|---|---:|---:|---|
| Flat (€10) | €0 | -€1,000 | bankrupt at bet 1,031 |
| Perc (1%) | €250 | -€750 | |
| Kelly (capped) | €29 | -€971 | |
| Reference (1.5% Kelly) | €985 | -€15 | near break-even, narrow stakes |
| StakeHead | €42 | -€958 | the head that decides |

The Reference strategy ending near break-even is not an edge: its stakes are a small fraction of
Kelly, so it barely bets.

## Calibration

| Predicted bin | Bets | Mean predicted | Observed win rate |
|---|---:|---:|---:|
| 0.1-0.2 | 6 | 0.169 | 0.000 |
| 0.2-0.3 | 62 | 0.262 | 0.177 |
| 0.3-0.4 | 285 | 0.362 | 0.298 |
| 0.4-0.5 | 460 | 0.451 | 0.315 |
| 0.5-0.6 | 384 | 0.547 | 0.417 |
| 0.6-0.7 | 284 | 0.647 | 0.518 |
| 0.7-0.8 | 186 | 0.747 | 0.640 |
| 0.8-0.9 | 105 | 0.843 | 0.762 |
| 0.9-1.0 | 17 | 0.911 | 0.765 |

Every bin below 0.9 sits under the diagonal: the model says 45% and wins 31.5%. That gap is the
leak-free result, not a tuning opportunity.

## Integrity and anti-leak checks

| Check | Result | Threshold |
|---|---|---|
| Duplicate `(Date, winner, loser)` rows | 0 in 0 groups | must be 0 |
| `corr(h2h_ratio, outcome)` | **+0.0936** | < 0.15 (leaked: +0.336) |
| Morning input | opening odds only | closing odds are calibration-only |
| Opening vs closing coverage (2017+) | 67.9% (13,904 / 20,475) | reported, not hidden |
| Provenance match (results ↔ sidecar) | verified | sha256 equality |

## The void historical claim

The original repository advertised **63.3% win rate / +30% ROI on 6,067 bets (p < 1e-95)**. That
was measured on a dataset corrupted by a many-to-many odds merge:

| Quantity | Leaked dataset | Repaired |
|---|---:|---:|
| Rows | 55,049 | 29,712 |
| Duplicate match rows | 1,579 | 0 |
| `corr(h2h, outcome)` | +0.336 | +0.0936 |
| Market Brier (2018+) | 0.1974 | 0.2036 |
| Model Brier | 0.1909 (below market) | 0.2357 (above market)† |

† The leaked figure is the pre-fix run's own score. The repaired figure is measured on the 1,789
logged bets, where the market scores 0.2167 on the same bets.

A model that beats Pinnacle's Brier is a leakage symptom first. The repaired model does not.

## Reproduce

```bash
python verify_leak_fix.py
python -c "import hydra_model as m; raise SystemExit(m.main())"
python run_benchmarks.py
python make_figures.py
```

The full report, including the drift test (Test 5), is regenerated at
`benchmarks/validation_report.md`. If the dataset hash does not match the sidecar, the report is
marked **VOID** and no p-value is published.

## Limitations

- The betting sample is the 71.1% of matches with opening odds, skewed toward well-covered events.
  Coverage falls to 57.8% in 2024.
- One walk-forward, one seed: the all-or-nothing CI is wide by construction.
- Slippage is a flat 2%; real execution varies by market and stake.
- No closing-line-value computation: the bet log records the opening price, so CLV is left as
  future work rather than estimated.
