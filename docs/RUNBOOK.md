# Runbook

How to run the system, read its output, and check whether a result is trustworthy.
Everything here has been executed on a clean checkout; commands that need the 1 GB
opening-odds file say so explicitly.

---

## 1. Environment

| Item | Value |
|---|---|
| Python | 3.12 (tested in the `tat-runner` image: `python:3.12-slim`) |
| TensorFlow | `tensorflow-cpu==2.16.1` (CPU only; GPU is disabled in code) |
| Other pins | see `requirements.txt` |
| RAM | 8 GB minimum; the opening-odds CSV alone is ~1 GB |
| Disk | ~40 MB repo + ~20 MB dataset + ~1 GB opening odds (optional) |

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt   # pytest
```

On this Ubuntu host the system Python is 3.14 and has no TensorFlow build, so every
command below runs inside a container:

```bash
docker build -t tat-runner - <<'EOF'
FROM python:3.12-slim
RUN pip install --no-cache-dir \
    tensorflow-cpu==2.16.1 xgboost==2.1.3 scikit-learn==1.5.2 \
    pandas==2.2.3 numpy==1.26.4 scipy==1.14.1 matplotlib==3.9.2 \
    tqdm==4.67.0 unidecode==1.3.8 openpyxl==3.1.5 pytest==8.3.4
WORKDIR /w
EOF
docker run --rm -v "$PWD":/w -w /w tat-runner python --version
```

The exact image used for the published run is described by the pins above; the run
manifest records the code commit, the dataset hash and the configuration.

---

## 2. Data

```bash
python download_data.py      # ATP match CSVs + closing odds from tennis-data.co.uk
python data_pipeline.py      # rebuilds tennis_features_v36.csv (~5 min)
```

The pipeline needs `data/atp_matches_TML/` and `data/match_odds/`. It writes
`tennis_features_v36.csv` (20 MB, gitignored) and `tennis_features_v36.meta.json`
(rows, date range, sha256 — tracked, because it is the dataset's identity card).

Opening odds are a separate, ~1 GB file: `data/tennis_odds_data_cleaned.csv`, scraped
from tennisexplorer.com with `scraper/`. It is **not** in git and is **required** for the
honest run. Without it `main()` exits with code 1; it does not silently substitute
closing odds.

Coverage measured on the published dataset with the loader used by `main()`:

```
overall            71.3%   (21,069 / 29,535 matches)
2013-2016          82.8 / 80.2 / 81.3 / 71.7 %
2017-2025          77.7 / 78.6 / 70.5 / 68.5 / 70.0 / 67.4 / 62.3 / 57.8 / 59.0 %
```

`MaxW`/`AvgW` in tennis-data.co.uk are closing prices. They are never used as morning
odds.

---

## 3. Before any number: the leak audit

```bash
python verify_leak_fix.py
```

Exits 0 only if, on the rebuilt dataset:

- 0 duplicate `(Date, winner, loser)` rows;
- `corr(h2h_ratio, won)` below 0.15 (currently 0.094);
- the market Brier is printed as the reference the model must match.

If this fails, stop. Numbers produced downstream are not interpretable.

---

## 4. The canonical run

```bash
python -c "import hydra_model as m; raise SystemExit(m.main())"
```

Runtime: ~8 minutes on the reference CPU image for the published configuration.
What it does:

1. loads `tennis_features_v36.csv`;
2. enriches it with opening odds and **drops matches without them** (never fills them);
3. builds ELO, EWMA, h2h and market features using only earlier dates;
4. runs the walk-forward: 2013-2016 warmup, 2017-2025 out-of-sample, one day at a time;
5. retrains the networks (7 d), XGBoost (14 d) and debiaser (30 d) on past matches only;
6. writes the evidence files and the charts.

### Degraded mode (explicit, labelled)

Only for debugging the pipeline, never for results:

```bash
TRADING_ALPHA_ALLOW_CLOSING_FALLBACK=1 python -c "import hydra_model as m; raise SystemExit(m.main())"
```

The run records `Data mode: closing_fallback` and `Opening odds coverage: 0.0%`, and the
README numbers do not come from this mode. Morning inference on closing prices is the
classic way to manufacture a fake edge.

### Outputs

| File | Content |
|---|---|
| `walkforward_results.txt` | headline numbers + provenance (commit, dataset sha256, coverage, mode) |
| `walkforward_bets.csv` | one row per bet: date, players, prob, market_prob, odds, stake, pnl, won |
| `walkforward_manifest.json` | full config + results, machine-readable |
| `charts/*.png` | equity curves written by the run (dark theme, for the run itself) |
| `model_state/` | last model checkpoints and scaler (not portable across TF versions) |

---

## 5. Benchmarks and probability quality

```bash
python run_benchmarks.py --quick   # ~30 s, skips the slow drift test
python run_benchmarks.py           # all tests
```

The report in `benchmarks/validation_report.md` contains:

- **Return test** — ROI, 95% bootstrap CI, p-value. Bets are compared to zero, so a
  positive ROI with a CI containing zero is **not** an edge;
- **Probability quality** — Brier score of the model vs the market **on the same bets**,
  plus the delta. The market is the reference, not 50%;
- **Calibration** — reliability bins and ECE from `walkforward_bets.csv`;
- **Data tests** — duplicates, opening/closing separation, h2h correlation.

`run_benchmarks.py` refuses to publish a p-value if `walkforward_results.txt` lacks
provenance or if the dataset sha256 does not match the sidecar: the report is marked
**VOID** and the process exits 1.

`make_figures.py` reads `walkforward_bets.csv` and writes the light-theme figures in
`docs/figures/` (equity, calibration, Brier comparison). It cannot invent data: every
pixel comes from the published bets.

---

## 6. Reading a result without fooling yourself

1. `verify_leak_fix.py` passes → the dataset is clean.
2. `walkforward_results.txt` says `Data mode: opening_odds` and coverage > 0.
3. `run_benchmarks.py` reports model Brier **at or below** the market Brier on the same
   bets, with a calibration table that is not systematically above the diagonal.
4. Only then look at ROI. Treat the bootstrap CI as the answer: if it contains 0, the
   honest conclusion is "no demonstrated edge".

A model that beats Pinnacle's Brier by a large margin is a reason to look for a leak,
not a reason to celebrate.

---

## 7. Configuration

All constants live at the top of `hydra_model.py`.

| Group | Parameter | Default |
|---|---|---|
| Years | `START_YEAR` / `BETTING_START_YEAR` | 2013 / 2017 |
| Model | `BUFFER_SIZE` / `BATCH_SIZE` / `DROPOUT_RATE` | 8192 / 256 / 0.3 |
| Towers | `TOWER_DIM_1` / `TOWER_DIM_2` / `DENSE_DIM` | 64 / 32 / 32 |
| XGBoost | `XGB_WINDOW` / `XGB_MAX_DEPTH` / `XGB_VETO_THRESHOLD` | 20000 / 3 / -0.02 |
| Betting | `MIN_EDGE` / `KELLY_FRACTION` / `FLAT_STAKE` | 0.01 / 0.015 / 10.0 |
| Caps | `MAX_STAKE_LEGAL_CAP` / `MAX_REALISTIC_STAKE` | 5000 / 5000 |
| Execution | `EXECUTION_SLIPPAGE` | 0.02 |
| Retraining | `RETRAIN_DEEP_DAYS` / `RETRAIN_XGB_DAYS` / `RETRAIN_DEBIASER_DAYS` | 7 / 14 / 30 |
| Reproducibility | `RNG_SEED` | 42 |

The walk-forward is deterministic in ordering (no shuffling); TensorFlow may still
produce small floating-point differences between runs. `NUM_LOOPS = 1` is the published
configuration; higher values average several runs and are slower.

---

## 8. Tests and CI

```bash
python -m pytest tests -q
```

CI (`.github/workflows/tests.yml`) installs the pinned requirements on Python 3.12 and
runs the suite. The workflow does **not** run the walk-forward: it needs the 1 GB odds
file, and the published artifacts are committed instead.

---

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Opening odds CSV missing` and exit 1 | correct behaviour: morning odds are required | provide `data/tennis_odds_data_cleaned.csv`, or use the labelled degraded mode |
| `no match was paired` and exit 1 | the CSV exists but the loader matched nothing (wrong file, encoding) | check the path and the columns; do not force a result |
| `ModuleNotFoundError: tensorflow` | Python 3.13+ or missing pin | use Python 3.12 and `requirements.txt` |
| `X does not have valid feature names` warning | scaler fitted on a DataFrame, applied to an array | harmless |
| identical-looking numbers every run | walk-forward is chronological and seeded | expected; small TF float differences are normal |
| charts are missing | run failed before the plotting stage | read the traceback; `main()` returns non-zero on failure |

---

## 10. Papers

`references/` keeps the papers actually used by the code:

| File | Used for |
|---|---|
| `xgboost_chen_guestrin_2016.pdf` | the XGBoost residual corrector |
| `siamese_koch_2015.pdf` | twin-tower comparison architecture |
| `ensemble_stacking_zhang_2021.pdf` | NN + tree blending |
| `kelly_thorp_2008.pdf` | Kelly reference for stake sizing |

There is no reinforcement learning in this codebase; a PPO paper was removed because it
described a component that does not exist.
