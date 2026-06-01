# Trading Alpha -- Runbook

**Complete guide to running, interpreting, and reproducing the Trading Alpha system.**

---

## Table of Contents

- [Quick Start](#quick-start)
- [System Requirements](#system-requirements)
- [Data Setup](#data-setup)
- [Running the Model](#running-the-model)
- [Running Benchmarks](#running-benchmarks)
- [Interpreting Results](#interpreting-results)
- [Configuration Reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)
- [Reproducing Published Results](#reproducing-published-results)

---

## Quick Start

```bash
# 1. Install dependencies
pip install tensorflow xgboost scikit-learn pandas numpy matplotlib tqdm unidecode

# 2. Run the full walk-forward validation
python -c "import hydra_model as m; m.main()"

# 3. Run statistical validation
python run_benchmarks.py --quick
```

Expected runtime: ~8 minutes for full walk-forward, ~30 seconds for quick validation.

---

## System Requirements

### Hardware
- **CPU:** Any modern x86-64 processor (the model is CPU-only, TF forced to CPU)
- **RAM:** Minimum 8 GB, recommended 16 GB (the Pinnacle odds CSV is ~1 GB)
- **Disk:** ~40 MB for the repo + 38 MB for the dataset + 1 GB optional for Pinnacle odds
- **GPU:** Not required (GPU disabled via `CUDA_VISIBLE_DEVICES=-1`)

### Software
- **Python:** 3.10 or later (tested on 3.13)
- **TensorFlow:** 2.10 or later
- **XGBoost:** 1.7 or later
- **Scikit-learn:** 1.2 or later
- **Pandas:** 1.5 or later
- **NumPy:** 1.24 or later
- **Matplotlib:** 3.6 or later

### Operating System
- **Windows:** Tested on Windows 11 with Python 3.13
- **Linux/macOS:** Should work (paths are relative, use `python download_data.py` for data setup)

---

## Data Setup

### Included Dataset

The repository includes `data/tennis_features_v36.csv` (38 MB), which contains 55,049 ATP tennis matches (2001-2025) with 84 engineered features including:
- Match metadata (tournament, surface, round, date)
- Player statistics (height, age, handedness, rank)
- Match statistics (aces, double faults, serve percentages, break points)
- Market odds (MaxW/MaxL: closing odds from tennis-data.co.uk)
- Engineered features (ELO ratings, EWMA metrics, fatigue, fragility, SSI)

This file is ready to use immediately and is included in the repository.

### Optional: Pinnacle Opening Odds

For valid anti-leakage results, the Pinnacle opening odds CSV is required (not included due to size, ~1 GB).

**To add it:**
```bash
# Option A: Quick check (looks in common locations)
python download_data.py

# Option B: Manual copy
# Copy your tennis_odds_data_cleaned.csv to data/tennis_odds_data_cleaned.csv

# Option C: Symlink on Linux/macOS
ln -s /path/to/tennis_odds_data_cleaned.csv data/
```

**Without this file:** The model falls back to closing odds (MaxW/MaxL) as the "morning" input. This means the anti-leakage separation between morning and evening phases collapses, and results will be artificially inflated. The pipeline still runs, but results are NOT reliable.

**Pinnacle coverage by year (when available):**
```
2013: 93.5%    2017: 93.2%    2021: 85.8%
2014: 90.9%    2018: 90.5%    2022: 86.2%
2015: 89.8%    2019: 85.3%    2023: 81.1%
2016: 86.0%    2020: 83.4%    2024: 73.3%
```

---

## Running the Model

### Method 1: Full Pipeline (Recommended)

```bash
python -c "import hydra_model as m; m.main()"
```

This executes the complete pipeline:
1. Loads the dataset from `data/tennis_features_v36.csv`
2. Loads Pinnacle opening odds if available
3. Runs the full feature engineering pipeline (ELO, EWMA, scaling)
4. Builds the Hydra-Oracle model (Siamese NN + XGBoost + RL head)
5. Executes the walk-forward validation (2013-2016 training, 2017-2025 betting)
6. Generates equity curve charts in `charts/`
7. Saves model checkpoints to `model_state/`

**Output files:**

| File | Description |
|---|---|
| `charts/Run1_5Strategies_Full.png` | Full equity curves, all 5 strategies |
| `charts/Run1_5Strategies_2025.png` | 2025 focus, all 5 strategies |
| `charts/Run1_Normal_Full.png` | Non-compound strategies (Flat, Reference) |
| `charts/Run1_Exponential_Full.png` | Compound strategies (Perc, Kelly, RL_Agent) |
| `model_state/best_hydra_rl_model.keras` | Best RL model checkpoint |
| `model_state/scaler.pkl` | Feature scaler for normalization |
| `model_state/feature_columns.pkl` | Feature column names |
| `walkforward_results.txt` | Updated with latest results |

### Method 2: Quick Run (Skip Charts)

```bash
python -c "
import hydra_model as m
# Reduce to 1 iteration, skip charting by modifying config
m.NUM_LOOPS = 1
# Run main which includes charting anyway
m.main()
"
```

### Method 3: Custom Configuration

```bash
python -c "
import hydra_model as m

# Customize parameters
m.START_YEAR = 2001  # Use all available data (default: 2013)
m.BETTING_START_YEAR = 2010  # Earlier betting start (default: 2017)
m.INITIAL_BANKROLL = 10000  # Start with EUR 10,000
m.NUM_LOOPS = 3  # Run 3 times for averaged results

# Run
m.main()
"
```

---

## Running Benchmarks

### Quick Validation

```bash
python run_benchmarks.py --quick
```

Runs tests 1-4 only (~30 seconds):
- Binomial significance test
- Monte Carlo simulation
- Bayes Factor
- Effect size (Cohen's h)

### Full Validation (Requires Pinnacle CSV)

```bash
python run_benchmarks.py
```

Runs all 6 tests (~2 minutes):
- Tests 1-4: Statistical validation
- Test 5: Anti-leakage verification (requires Pinnacle odds)
- Generates report in `benchmarks/validation_report.md`

### Output Format

```
benchmarks/
  validation_report.md     # Human-readable markdown report
  benchmark_results.json   # Machine-readable JSON data
```

---

## Interpreting Results

### Understanding the walkforward_results.txt

```
TRADING ALPHA - Walk-Forward Results
Periodo: 2017-2025 (9 years out-of-sample)
Training: 2013-2016 (4 years warmup, Pinnacle coverage >90%)
Bets: 6271
Wins: 4101/6271 (65.4%)

Flat:   EUR 20,447  (P: +EUR 19,447)
Perc:   EUR 1,094,295 (P: +EUR 1,093,295)
Kelly:  EUR 1,318,427 (P: +EUR 1,317,427)
Reference: EUR 48,747  (P: +EUR 47,747)
RL_Agent: EUR 3,293,446 (P: +EUR 3,292,446)
```

**Key metrics to focus on:**

1. **Win Rate (65.4%):** The proportion of bets where the predicted winner actually won. Statistically significant (p < 1e-95). Compare against the market opening odds accuracy (~69.4% on ALL matches).

2. **Number of Bets (6,271):** More bets = more reliable statistics. 6,271 bets over 9 years is sufficient for statistical significance.

3. **Strategy comparison:**
   - **Flat** (EUR 10 fixed): Shows the effect of stake sizing on raw edge.
   - **Perc** (1% of bankroll): Compound effect on the same signal.
   - **Kelly** (fractional Kelly): Theoretical optimal growth.
   - **Reference** (adaptive): Conservative baseline for comparison.
   - **RL_Agent** (policy gradient): Learned adaptive staking. Outperformance vs Reference shows the RL agent is learning useful dynamics.

**Profit numbers should be interpreted with caution:**
- Compound strategies (Perc, Kelly, RL_Agent) produce exponential curves
- EUR 3M from EUR 1K over 6,271 bets is primarily compound interest on a modest edge
- The meaningful metric is win rate + edge detection, not final equity

### Understanding the Charts

**5Strategies charts:** All 5 strategies on one plot. May use log scale if strategies diverge significantly. The log scale is automatically applied when equity exceeds EUR 1M.

**Normal charts:** Flat and Reference strategies only, on linear scale. Shows the raw prediction signal without compounding effects.

**Exponential charts:** Perc, Kelly, and RL_Agent on log scale. Shows the compound growth trajectory.

**2025 focus charts:** Same as above but zoomed into 2025 only. Useful for recent performance.

### Validating Results

Use the benchmarks to validate:
1. Check `benchmarks/validation_report.md` for p-values and confidence intervals
2. Run `python run_benchmarks.py --quick` to regenerate
3. If p < 0.001 and the 95% CI is above 50%, the result is statistically significant

---

## Configuration Reference

All configuration parameters are at the top of `hydra_model.py` (lines 1-90).

### Data Parameters

| Parameter | Default | Description |
|---|---|---|
| `START_YEAR` | 2013 | First year of data to include (Pinnacle coverage >90% after 2013) |
| `BETTING_START_YEAR` | 2017 | First year of out-of-sample betting |
| `DATASET_FILE` | data/tennis_features_v36.csv | Path to feature CSV |
| `SCRAPER_CSV` | auto-detect | Path to Pinnacle odds CSV |

### Model Architecture Parameters

| Parameter | Default | Description |
|---|---|---|
| `TOWER_DIM_1` | 64 | First Siamese tower layer size |
| `TOWER_DIM_2` | 32 | Second Siamese tower layer size |
| `DENSE_DIM` | 32 | Final dense layer size |
| `DROPOUT_RATE` | 0.3 | Dropout for regularization |
| `BATCH_SIZE` | 256 | Training batch size |
| `BUFFER_SIZE` | 8192 | Training buffer capacity |
| `LR_VALUE` | 0.0001 | Learning rate for value head |
| `LR_POLICY` | 0.0002 | Learning rate for policy head |

### Betting Parameters

| Parameter | Default | Description |
|---|---|---|
| `INITIAL_BANKROLL` | 1000.0 | Starting bankroll in EUR |
| `MIN_EDGE` | 0.01 | Minimum edge (1%) to place a bet |
| `PERC_STAKE` | 0.01 | Percentage stake (1% of bankroll) |
| `FLAT_STAKE` | 10.0 | Fixed stake in EUR |
| `KELLY_FRACTION` | 0.015 | Fractional Kelly (1.5%) |
| `MAX_STAKE_LEGAL_CAP` | 5000.0 | Maximum stake cap in EUR |
| `EXECUTION_SLIPPAGE` | 0.02 | 2% slippage on odds execution |

### Retraining Schedule

| Parameter | Default | Description |
|---|---|---|
| `RETRAIN_DEEP_DAYS` | 7 | Retrain neural network every N days |
| `RETRAIN_XGB_DAYS` | 14 | Retrain XGBoost every N days |
| `RETRAIN_DEBIASER_DAYS` | 30 | Retrain odds debiaser every N days |
| `XGB_WINDOW` | 20000 | XGBoost memory window (~7 years) |

### Risk Management

| Parameter | Default | Description |
|---|---|---|
| `RL_STAKE_MIN` | 0.00 | Minimum RL stake (0% of bankroll) |
| `RL_STAKE_MAX` | 0.05 | Maximum RL stake (5% of bankroll) |
| `CUSUM_THRESHOLD` | 5.0 | CUSUM drift detection threshold |
| `SHARPE_PENALTY` | 0.08 | PnL variance penalty in RL loss |

---

## Troubleshooting

### "FileNotFoundError: data/tennis_features_v36.csv not found"

**Cause:** The script was run from the wrong directory.

**Fix:**
```bash
# Run from the trading-alpha directory
cd path/to/trading-alpha
python -c "import hydra_model as m; m.main()"
```

### "ModuleNotFoundError: No module named 'xgboost'"

**Fix:**
```bash
pip install xgboost tensorflow scikit-learn pandas numpy matplotlib tqdm unidecode
```

### "Pinnacle CSV not found - proceeding without Opening odds"

**Cause:** The model found the main dataset but not the Pinnacle odds CSV (1 GB).

**Effect:** Results will use closing odds for both morning and evening phases. The pipeline runs, but anti-leakage guarantees are weakened.

**Fix:** See [Data Setup](#data-setup) section.

### "CUDA errors" or GPU-related issues

The model explicitly disables GPU via `os.environ["CUDA_VISIBLE_DEVICES"] = "-1"`. If you see CUDA errors, they're from TensorFlow initialization. The model runs on CPU only. These warnings are harmless:

```
oneDNN custom operations are on. You may see slightly different numerical results...
```

### "WARNING: X does not have valid feature names, but StandardScaler was fitted with feature names"

This is a scikit-learn warning that occurs when the scaler is applied to numpy arrays instead of DataFrames. It does not affect the numerical results.

### Model produces identical results every run?

The walk-forward is deterministic by design (data is time-ordered, no shuffling). Minor variations can occur due to:
- Floating-point non-determinism in TensorFlow
- Different Pinnacle CSV coverage

For genuinely averaged results, set `NUM_LOOPS = 3` or higher.

---

## Reproducing Published Results

To reproduce the results documented in BENCHMARKS.md:

```bash
# 1. Ensure Pinnacle odds CSV is available
python download_data.py

# 2. Verify configuration matches
#    START_YEAR = 2013
#    BETTING_START_YEAR = 2017

# 3. Run the walk-forward
python -c "import hydra_model as m; m.main()"

# 4. Run validation
python run_benchmarks.py --quick
```

Expected output should match within statistical tolerance (minor variations due to TF non-determinism). The key indicators should be:
- Win rate: approximately 63-65%
- Bets: approximately 6,000-6,300
- p-value: < 0.001
- 95% CI entirely above 50%

---

## Paper References

The papers in `references/` correspond to the following techniques used in the code:

| File | Paper | Relevance to Code |
|---|---|---|
| `xgboost_chen_guestrin_2016.pdf` | Chen & Guestrin, "XGBoost: A Scalable Tree Boosting System" | The XGBoost library used for residual correction and meta-labeling |
| `ppo_schulman_2017.pdf` | Schulman et al., "Proximal Policy Optimization Algorithms" | Policy gradient framework for the RL staking head |
| `siamese_koch_2015.pdf` | Koch et al., "Siamese Neural Networks for One-shot Image Recognition" | Siamese/twin tower architecture for player comparison |
| `ensemble_stacking_zhang_2021.pdf` | Zhang & Ma, "Ensemble Learning" | Theoretical foundation for blending NN + XGBoost predictions |
| `kelly_thorp_2008.pdf` | Thorp, "The Kelly Criterion in Blackjack Sports Betting and the Stock Market" | Kelly criterion derivation used in stake sizing |
