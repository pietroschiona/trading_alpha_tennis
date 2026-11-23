# Trading Alpha - Validation Report

**Generated:** 2026-11-23 12:42:10
**Integrity checks:** PASS
**Edge demonstrated:** NO

## Provenance

| Field | Value |
|---|---|
| Model run commit (produced these bets) | `e2a3af3` |
| Benchmark commit | `unknown` |
| Dataset rows | 29712 |
| Dataset sha256 | `467c8a17369bf9126969f06793149fd65e98cd9d930f16572e1fb4e9e897c5bb` |
| Meta rows | 29712 |
| Meta sha256 | `467c8a17369bf9126969f06793149fd65e98cd9d930f16572e1fb4e9e897c5bb` |
| Provenance match | yes |
| Opening odds coverage | 71.1% |
| Bets | 1789 |
| Wins | 760/1789 (42.5%) |

## Return test (primary evidence)

| Metric | Value |
|---|---|
| Bets | 1789 |
| Total staked | 11898.16 |
| Total PnL | -957.83 |
| ROI | -8.05% |
| 95% bootstrap CI | [-15.97%, -0.13%] |
| p-value (one-sided, ROI <= 0) | 0.9769 |
| Max drawdown | 1256.74 |

Stakes and PnL come from `walkforward_bets.csv` (bankroll = 1000, stake_head strategy). The confidence interval is the only claim that matters: if it contains zero, the strategy has not been shown to make money.

## Probability quality (same bets)

| Metric | Value |
|---|---|
| Bets with a probability | 1789 |
| Model Brier | 0.2357 |
| Market Brier | 0.2167 (market_prob) |
| Delta (model - market) | +0.0190 |

A positive delta means the model is *worse* than the market on the exact bets it chose. Win rate alone cannot show this.

### Calibration (reliability)

Expected calibration error: **0.1144** over 1789 bets.

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

Bins below the diagonal are overconfident: the predicted probability sits above the observed win rate. That gap is the leak-free result, not a tuning problem.

Market Brier reference (2018+, MaxW/MaxL): **0.2036** This is the bar a model must beat without leakage.

## Descriptive: win-rate statistics

**A win rate above 50% is not evidence of profit.**

| Metric | Value |
|---|---|
| Win rate | 42.48% (95% CI [40.20%, 44.78%]) |
| Binomial p (H0: 50%) | 1.00e+00 |
| Z-score | -6.36 |
| Monte Carlo p | 1.00e+00 (200,000 sims) |
| Bayes factor BF10 (50% vs observed rate) | 6.6e+08 (decisive) |
| Cohen's h | 0.1509 (small) |

These describe how often the picker was right. They say nothing about whether the picks were priced correctly, which is where the money is made or lost.

## Test 5: Opening vs closing market drift (descriptive)

Betting matches (2017+): **20475**
With opening odds: **13904** (67.9% coverage)
Mean (closing - opening) implied probability: +0.0081 (sd 0.0530)

## Test 6: Duplicate-match integrity check

Rows checked: **29712**
Duplicate (Date, winner, loser) rows: **0** in 0 groups

**Integrity:** PASS - no duplicates, the year+player merge leak is not back.

## Test 7: h2h leakage check

corr(h2h_ratio, outcome) = **+0.094** (threshold < 0.15; leaked dataset was +0.336)

**Integrity:** PASS - no outcome information in the h2h feature.

---
## Summary

**Integrity checks: PASS**
**Edge demonstrated: NO**

- Return test: ROI -8.05%, 95% CI [-15.97%, -0.13%], p = 0.9769
- Duplicates: 0
- h2h corr: 0.0936
- Provenance: verified