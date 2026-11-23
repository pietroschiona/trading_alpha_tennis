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
| Code commit | `3cbaf57` |
| Generated | 2026-11-23 |
| Dataset rows | 29,535 |
| Dataset SHA-256 | `362c5f1e…23b9c3c0d` |
| Opening-odds coverage | 71.3% overall, 68.0% of 2017+ matches |
| Data mode | `opening_odds` (no closing fallback) |
| Modeling sample | 21,069 matches |
| Period | 2013-2016 warmup, 2017-2025 out-of-sample |
| Bets | 1,825 |
| Integrity checks | PASS (duplicates 0, h2h corr +0.0938) |

## Headline

| Metric | Value | How to read it |
|---|---|---|
| Bets | 1,825 | stake-head bets with a positive signal |
| Win rate | 42.2% (771/1,825) | descriptive; not an edge test |
| Average odds | 2.83 | break-even win rate at these prices ≈ 35.4% |
| ROI | **-5.22%** | stake-weighted return over €18,350 staked |
| 95% bootstrap CI | **[-12.72%, +2.28%]** | contains zero: no edge proven |
| p-value (one-sided, ROI ≤ 0) | 0.9127 | cannot reject "no profit" |
| Model Brier | **0.2373** | lower is better |
| Market Brier, same bets | **0.2186** | the opponent |
| Delta (model − market) | **+0.0186** | positive = model worse |
| Calibration ECE | **0.1161** | 11.6 points of miscalibration |

### Why a 42% win rate still loses money

A 42.2% win rate at average odds 2.83 looks profitable: 0.422 × 2.83 ≈ 1.19. It is not, because
winners and losers do not share the same average price:

| Group | Count | Win rate | PnL (EUR) |
|---|---:|---:|---:|
| Odds 1.00-1.50 | 176 | 70% | -37 |
| Odds 1.50-2.00 | 432 | 56% | -87 |
| Odds 2.00-3.00 | 719 | 39% | -291 |
| Odds 3.00-5.00 | 370 | 29% | +41 |
| Odds > 5.00 | 128 | 9% | -584 |

- Average odds of the bets it won: **2.254**. Of the bets it lost: **3.248**.
- The model's mean probability is **0.539**; the realized win rate is **0.4225**.
- The market's mean probability on those same bets is **0.421**, almost exactly the realized rate.

The model is systematically overconfident, especially on long shots: it buys 9%-win, 5x+ underdogs
with money that the market prices correctly. The 2% execution slippage turns that into a loss.

## Strategy comparison

All five strategies share the same signal; they differ only in staking. A bankroll that reaches
zero stops betting (ruin is absorbing).

| Strategy | Final bankroll | PnL | Note |
|---|---:|---:|---|
| Flat (€10) | €0 | -€1,000 | bankrupt at bet 1,562 |
| Perc (1%) | €256 | -€744 | |
| Kelly (capped) | €35 | -€965 | |
| Reference (1.5% Kelly) | €986 | -€14 | near break-even, narrow stakes |
| StakeHead | €42 | -€958 | the head that decides |

The Reference strategy ending near break-even is not an edge: its stakes are a small fraction of
Kelly, so it barely bets.

## Calibration

| Predicted bin | Bets | Mean predicted | Observed win rate |
|---|---:|---:|---:|
| 0.1-0.2 | 6 | 0.185 | 0.167 |
| 0.2-0.3 | 62 | 0.259 | 0.113 |
| 0.3-0.4 | 281 | 0.357 | 0.299 |
| 0.4-0.5 | 468 | 0.451 | 0.344 |
| 0.5-0.6 | 411 | 0.549 | 0.384 |
| 0.6-0.7 | 307 | 0.649 | 0.531 |
| 0.7-0.8 | 175 | 0.746 | 0.634 |
| 0.8-0.9 | 95 | 0.847 | 0.726 |
| 0.9-1.0 | 20 | 0.915 | 0.850 |

Every bin below 0.9 sits under the diagonal: the model says 45% and wins 34.4%. That gap is the
leak-free result, not a tuning opportunity.

## Integrity and anti-leak checks

| Check | Result | Threshold |
|---|---|---|
| Duplicate `(Date, winner, loser)` rows | 0 in 0 groups | must be 0 |
| `corr(h2h_ratio, outcome)` | **+0.0938** | < 0.15 (leaked: +0.336) |
| Morning input | opening odds only | closing odds are calibration-only |
| Opening vs closing coverage (2017+) | 68.0% (13,895 / 20,438) | reported, not hidden |
| Provenance match (results ↔ sidecar) | verified | sha256 equality |

## The void historical claim

The original repository advertised **63.3% win rate / +30% ROI on 6,067 bets (p < 1e-95)**. That
was measured on a dataset corrupted by a many-to-many odds merge:

| Quantity | Leaked dataset | Repaired |
|---|---:|---:|
| Rows | 55,049 | 29,535 |
| Duplicate match rows | 1,579 | 0 |
| `corr(h2h, outcome)` | +0.336 | +0.0938 |
| Market Brier (2018+) | 0.1974 | 0.2037 |
| Model Brier | 0.1909 (below market) | 0.2373 (above market)† |

† The leaked figure is the pre-fix run's own score. The repaired figure is measured on the 1,825
logged bets, where the market scores 0.2186 on the same bets.

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

- The betting sample is the 71.3% of matches with opening odds, skewed toward well-covered events.
  Coverage falls to 57.8% in 2024.
- One walk-forward, one seed: the all-or-nothing CI is wide by construction.
- Slippage is a flat 2%; real execution varies by market and stake.
- No closing-line-value computation: the bet log records the opening price, so CLV is left as
  future work rather than estimated.
