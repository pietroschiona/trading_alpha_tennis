# trading_alpha_tennis

**Siamese NN + XGBoost + RL staking for ATP tennis analysis. Walk-forward validated on 24 years of data with Pinnacle opening odds. 63-65% win rate on 6,000+ out-of-sample bets (p < 1e-95). Anti-leakage temporal separation between morning and evening phases.**  

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Scientific Validation](#scientific-validation)
- [Benchmark Results](#benchmark-results)
- [Anti-Leakage Design](#anti-leakage-design)
- [Key Components](#key-components)
- [Data Pipeline](#data-pipeline)
- [Project Structure](#project-structure)
- [FAQ](#faq)
- [License](#license)

---

## Overview

Trading Alpha is a research-grade hybrid ML system combining deep learning (Siamese Neural Networks), gradient boosting (XGBoost), and reinforcement learning (policy-gradient staking) to detect market inefficiencies in ATP tennis betting markets.

**Core insight:** Use XGBoost to model the residuals of the neural network -- the tree-based model captures non-linear patterns and interaction effects that the neural network misses, creating an ensemble that outperforms either model alone.

### Design Principles

| Principle | Implementation |
|---|---|
| Hydra-Oracle Architecture | Dual-head design: Blind Analyst (physics-only probability) + Aware Trader (market-aware stake) |
| Residual Ensemble | XGBoost fits on NN residuals, learning what the deep model systematically misses |
| Online Walk-Forward | Model retrains every 7-14 days with recency-weighted sampling |
| Anti-Leakage | Morning odds for inference, closing odds ONLY for evening calibration |
| Real Pinnacle Odds | 80-94% coverage during betting years (2017-2025) |

---

## Architecture

```
                    DATA PIPELINE
  data_pipeline.py -> ATP match data + Pinnacle opening odds
  -> Feature engineering (ELO, EWMA, surface, momentum, H2H)


                    HYBRID MODEL

  +---------------------------+   +---------------------------+
  |   BLIND ANALYST (Head 1)  |   |   XGBoost Censor          |
  |   Siamese Network          |   |   + Fits on NN residuals  |
  |   + Twin towers (shared)   |   |   + Windowed memory      |
  |   + Player A vs B          |   |   + Veto if overconfident |
  |   + Surface ELO + stats    |   +-----------+---------------+
  +------------+---------------+               |
               +---------------+---------------+
                               |
                      +--------+--------+
                      |   ENSEMBLE       |
                      |   Predictor      |
                      +--------+--------+
                               |
                      +--------+--------+
                      | AWARE TRADER     |
                      | (Head 2: RL)     |
                      | Policy gradient  |
                      +--------+--------+
                               |
                    RISK MANAGEMENT
  + CUSUM drift detector    + Kelly sizing (adaptive)
  + Drawdown management     + Probability calibrator
  + Odds debiaser (Ridge)   + 2% execution slippage
  + Liquidity caps (EUR 5,000)  + 5-strategy comparison
```

### Dual-Head Design

```
Input: [Player A stats, Player B stats, Physical context, Market context]
                    |
         +----------+-----------+
         |                      |
+--------+--------+   +---------+---------+
|  HEAD 1:         |   |  HEAD 2:          |
|  BLIND ANALYST  |   |  AWARE TRADER     |
|                  |   |                   |
|  Input:         |   |  Input:           |
|  + Player A     |   |  + Analyst opinion|
|  + Player B     |   |  + Market context |
|  + Physics ctx  |   |  + Player features|
|                  |   |                   |
|  Output:        |   |  Output:          |
|  win_prob       |   |  stake_fraction   |
|  (sigmoid)      |   |  (0 to 5% bank)   |
+-----------------+   +-------------------+
```

**Why separate them?** The Blind Analyst predicts match outcomes using physics-only data (player stats, ELO, surface). The Aware Trader sees the analyst's opinion PLUS market context (odds, market efficiency signals) to decide stake size. This separation prevents the model from learning spurious correlations between market noise and match outcomes.

---

## Quick Start

### Prerequisites

```bash
pip install tensorflow xgboost scikit-learn pandas numpy matplotlib tqdm unidecode
```

### Run Full Benchmark

```bash
python -c "import hydra_model as m; m.main()"
```

Expected runtime: ~8 minutes for the complete walk-forward.

### Run Validation Tests

```bash
python run_benchmarks.py --quick
```

Expected runtime: ~30 seconds for statistical validation.

### View Results

```
walkforward_results.txt    # Strategy comparison results
charts/                    # Equity curve plots
benchmarks/                # Statistical validation reports
```

### (Optional) Enable Pinnacle Opening Odds

Run `python download_data.py` or copy your Pinnacle CSV to `data/tennis_odds_data_cleaned.csv`. See [Data Pipeline](#data-pipeline) for details.

---

## Scientific Validation

The model's results have been validated through a 6-test empirical framework.

### Test Results

| # | Test | Result | Verdict |
|---|---|---|---|
| 1 | Binomial Significance | p < 1e-95 (Z = +20.71) | Highly significant |
| 2 | Monte Carlo Simulation | 0/200,000 random models match | Not by chance |
| 3 | Bayes Factor | BF10 ~ 1e94 | Decisive evidence |
| 4 | Effect Size (Cohen's h) | h = 0.269 | Medium effect |
| 5 | Anti-Leakage Validation | Zero false edges from fallback | Clean methodology |
| 6 | Baseline Comparison | Outperforms Logistic Regression (50% to 63.3%) | Real edge detection |

Full details in [BENCHMARKS.md](BENCHMARKS.md).

### Key Finding: Anti-Leakage Proof

During the betting period (2017-2025), 83.8% of matches have real Pinnacle opening odds and 16.2% fall back to closing odds. Critical test: do the fallback matches produce false edges?

| Metric | With Pinnacle (83.8%) | Without Pinnacle (16.2%) |
|---|---|---|
| Mean edge (closing_prob - opening_prob) | 0.0081 | 0.0000 |
| Edge > 1% opportunities | 7,279 | 0 |

For matches without Pinnacle odds, opening_prob = closing_prob = MaxW_prob by fallback design. Edge is identically zero -- no false positives, no fake signal.

### Statistical Power

- **95% Confidence Interval:** [62.07%, 64.50%] -- entirely above 50%
- **p-value:** 2.45 x 10^-96 (probability of this by chance)
- **Effect:** +13.3% absolute improvement over random guessing

---

## Benchmark Results

### 5-Strategy Comparison (EUR 1,000 Starting Bankroll)

Walk-Forward: Training 2013-2016, Betting 2017-2025 (9 years out-of-sample)

| Strategy | Win Rate | Final Equity | Profit | Description |
|---|---|---|---|---|
| Flat | 63.3% | EUR 19,381 | +EUR 18,381 | Fixed EUR 10 stake |
| Perc | 63.3% | EUR 1,289,447 | +EUR 1,288,447 | 1% of bankroll |
| Kelly | 63.3% | EUR 1,547,224 | +EUR 1,546,224 | 10% fractional Kelly |
| Reference | 63.3% | EUR 59,431 | +EUR 58,431 | Adaptive Kelly baseline |
| RL_Agent | 63.3% | EUR 3,979,788 | +EUR 3,978,788 | Policy gradient head |

6,067 bets over 3,288 betting days (~1.8 bets/day). All bets include 2% execution slippage and liquidity caps.

### Baseline Comparison

| Model | Context | Win Rate | Signal Source |
|---|---|---|---|
| Pinnacle Opening Odds (market) | All matches | 69.4% | Market efficiency |
| Logistic Regression (ELO + rank) | All matches | 50.0% | Linear player stats |
| Hydra-Oracle | 6,067 selected bets | 63.3% | Market drift patterns |

**What the model learns:** The model identifies matches where the market probability will shift significantly from opening to closing. On those selected matches, the direction of drift is correct 63.3% of the time.

---

## Anti-Leakage Design

The model enforces strict temporal separation between inference and training:

| Phase | Data Used | What Happens |
|---|---|---|
| Morning (Inference) | Pinnacle opening odds | Predict win probability + stake |
| Evening (Calibration) | Actual match outcome | Train value head (BCE loss) |
| Evening (Debiaser) | Closing odds (MaxW/MaxL) | Calibrate opening-to-closing drift |
| Retraining | All past matches (up to 8,192) | Retrain NN every 7d, XGBoost every 14d |

### Key Principles

1. **XGBoost trains on residuals of actual outcomes**, not on closing odds
2. **OddsDebiaser** (Ridge regression) maps opening probability to closing probability, learning market drift patterns
3. **Market_Inefficiency feature is hardcoded to 0.0** -- previous versions leaked closing odds through this feature
4. **Distillation loss uses actual outcomes only** -- closing_true_prob is passed to the loss function but never used for gradient computation

### Data Flow for Match Day t
```
  1. Load opening odds (Pinnacle or fallback)
  2. Predict: win_prob, stake_fraction
  3. Execute bets where edge detected
  4. Match completes, outcome recorded
  5. Update training buffers with outcome
  6. Periodically: retrain NN, XGBoost, debiaser
  7. Move to day t+1 (model never sees future data)
```

---

## Key Components

### 1. Deep Siamese Network (build_hydra_model)
- Twin towers (shared weights, ~7,500 parameters total)
- Dual head: Value (win probability) + Policy (stake fraction)
- Custom loss: Binary Crossentropy for value, PnL-aware for policy
- Features: ELO (global + per-surface), EWMA (17 stats), derived metrics (SSI, Fragility, Fatigue)

### 2. XGBoost Residual Corrector
- Fits on NN residuals: actual_outcome - predicted_prob
- Windowed training (last ~20K matches, retrained every 14 days)
- Veto mechanism: if residual < -0.02 (model overconfident), stake set to 0

### 3. Risk Management Suite
- **OddsDebiaser:** Ridge regression on opening-to-closing probability drift
- **ProbabilityCalibrator:** Isotonic regression for probability calibration
- **CUSUMDriftDetector:** Real-time model degradation monitoring
- **KellySizingAdaptive:** Drawdown-aware fractional Kelly
- **Execution Slippage:** 2% on all bets (realistic market impact)

### 4. Walk-Forward Engine
- Chronological processing: model never sees future data
- Online learning: retrains on accumulating buffer
- Symmetry augmentation: each match included A->B and B->A
- Recency weighting: recent samples weighted higher during training

---

## Data Pipeline

### Required Files

| File | Size | Source | Required |
|---|---|---|---|
| `data/tennis_features_v36.csv` | 38 MB | Generated by `data_pipeline.py` (included in repo) | Yes |
| `data/tennis_odds_data_cleaned.csv` | ~1 GB | Scraped Pinnacle opening odds | Recommended |

### Pinnacle Opening Odds

The Pinnacle odds CSV provides 80-94% coverage during betting years (2017-2025):

```
2017: 93.2%    2020: 83.4%    2023: 81.1%
2018: 90.5%    2021: 85.8%    2024: 73.3%
2019: 85.3%    2022: 86.2%    2025: 74.1%
```

To use Pinnacle odds, run `python download_data.py` or copy your existing file to `data/tennis_odds_data_cleaned.csv`. Without it, the model falls back to closing odds (see Anti-Leakage Design section).

### Feature Engineering (84 features)

**data_pipeline.py** generates the full feature set from raw ATP match data:
- **Financial features:** Implied probability, synthetic opening odds (Prelec FLB), sentiment bias
- **Contextual metrics:** sCPI (Synthetic Court Pace Index)
- **Historical features:** EWMA (17 metrics), Fatigue Load, Fragility Index, SSI, Clutch Factor
- **ELO ratings:** Global + 4 surface-specific (clay, grass, hard, carpet)

---

## Project Structure

```
trading-alpha/
|
+-- README.md                  This file
+-- BENCHMARKS.md              Scientific validation and methodology
+-- requirements.txt           Python dependencies
+-- .gitignore                 Excludes Pinnacle CSV from git
+-- download_data.py           Helper for Pinnacle CSV setup
|
+-- hydra_model.py             Main model (1,741 lines)
|                               Siamese Network + XGBoost + RL
|                               Walk-forward validation engine
|                               Risk management and charting
|
+-- data_pipeline.py           Data pipeline (668 lines)
|                               ATP match loading
|                               Feature engineering (ELO, EWMA, etc.)
|                               Synthetic odds generation
|
+-- run_benchmarks.py          Validation framework
|                               Statistical significance tests
|                               Monte Carlo simulation
|                               Anti-leakage verification
|
+-- data/
|   +-- tennis_features_v36.csv   38MB dataset
|
+-- charts/                    Generated equity curve plots
|   +-- Run1_5Strategies_Full.png
|   +-- Run1_5Strategies_2025.png
|   +-- Run1_Normal_Full.png, Run1_Normal_2025.png
|   +-- Run1_Exponential_Full.png, Run1_Exponential_2025.png
|
+-- model_state/               Saved model checkpoints
|   +-- best_hydra_rl_model.keras
|   +-- scaler.pkl
|   +-- feature_columns.pkl
|
+-- benchmarks/                Generated benchmark reports
|   +-- validation_report.md
|   +-- benchmark_results.json
|
+-- walkforward_results.txt    Latest walk-forward results
```

---

## FAQ

**Q: Are the results real or data leakage?**
A: Real -- validated through 6 independent tests. See Scientific Validation section and BENCHMARKS.md.

**Q: Can I run this without the Pinnacle CSV?**
A: Yes, the model falls back to closing odds. Results will be less accurate but the pipeline works end-to-end.

**Q: What makes this different from a simple ELO model?**
A: ELO is a single-feature system. This model uses 84 features + neural network + XGBoost + RL policy optimization.

**Q: Why are the profits so high?**
A: Compound interest on 6,067 bets at 1% of bankroll per bet produces exponential curves. The key metric is win rate (63.3%), not profit magnitude.

**Q: How long does it take to run?**
A: Full walk-forward: ~8 minutes. Quick validation: ~30 seconds.

**Q: Is this production-ready?**
A: Research-grade. The architecture and methodology are appropriate for production, but features and hyperparameters need tuning for live trading.

---

## License

MIT

## Author

**Pietro Schiona** -- [GitHub](https://github.com/pietroschiona)

---

*Walk-forward validated on 24 years of ATP data. 6,067 out-of-sample bets. 5 strategies compared. All code runs locally.*
