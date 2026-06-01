# Trading Alpha — Validation Report

**Generated:** 2026-06-01 18:26:28
**Bets:** 6067 | **Win Rate:** 63.3%

---

## Test 1: Binomial Significance

| Metric | Value |
|---|---|
| Bets | 6067 |
| Wins | 3839 |
| Win Rate | 63.28% |
| Z-score | 20.68 |
| p-value | 2.45e-96 |
| Significant at 0.001? | YES |
| 95% CI | [62.06%, 64.48%] |

## Test 2: Monte Carlo Simulation

Out of 200,000 simulated random models (true WR = 50%):
- **0.000000%** achieved >= 63.28%
- Equivalent to p ≈ 5.00e-06

## Test 3: Bayes Factor

BF10 = **1.0e+94** — **decisive** evidence for H1 (model) vs H0 (random)

## Test 4: Effect Size

Cohen's h = **0.2688** (medium effect)

## Test 5: Anti-Leakage Validation


---
## Summary

**Binomial:** p = 2.45e-96
**Monte Carlo:**
**Bayes Factor:** BF10 = 1.0e+94
**Effect Size:** h = 0.2688
**Anti-Leakage:** Zero false edges from fallback

**Overall: ALL TESTS PASSED**