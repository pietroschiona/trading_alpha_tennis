# Model card

## What this is

A research system that predicts ATP match outcomes from pre-match features and simulates betting
against opening odds. It exists to test whether a hybrid Siamese/XGBoost model adds information
beyond the market. The published answer, after removing a data leak, is **no demonstrated edge**.

## Intended use

- Studying leak-free sports forecasting and walk-forward evaluation.
- Auditing a data pipeline for temporal leakage.
- Teaching why a win-rate test is not an edge test.

## Out of scope

- Live betting, bankroll advice, or financial advice.
- Any claim of profitability. The published result is at or below break-even.
- WTA, doubles, or non-ATP tours: the data and the feature set are ATP singles only.

## Training data

- ATP singles 2000-2025 from TML-Database (Sackmann schema), filtered to 2013+ for the model.
- Closing odds 2013-2025 from tennis-data.co.uk.
- Opening odds scraped from tennisexplorer.com, 71.1% match coverage.
- Feature dataset: 29,712 matches, 83 columns, sha256 in `tennis_features_v36.meta.json`.
- Details and limitations: [`dataset.md`](dataset.md).

## Evaluation

- Walk-forward: warmup 2013-2016, out-of-sample betting 2017-2025, one day at a time.
- Primary metric: ROI with a 95% bootstrap CI, and Brier score against the market on the same bets.
- Window and metrics: [`../BENCHMARKS.md`](../BENCHMARKS.md).

## Ethical and legal notes

- Odds data are subject to the source sites' terms; the 1 GB opening-odds file is not redistributed.
- Betting can be harmful. Nothing here encourages it; the measured result argues against using the
  model to make money.
- No personal data is used. Player names are public sports records.

## Known failure modes

- Missing opening odds → match dropped, sample shrinks and skews toward well-covered tournaments.
- Coverage falls in recent years (57.8% in 2024), so the later out-of-sample period is less
  representative.
- The market is the opponent; a model that appears to beat it by a lot is a leakage symptom first.
