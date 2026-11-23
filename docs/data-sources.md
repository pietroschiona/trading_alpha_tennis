# Data sources

Use case: ATP value betting 2013-2025. Opening odds are a pre-match feature; closing odds
are a calibration target. Comparison run on 2026-10-01 and re-measured on 2026-11-23 with
the loader used by `hydra_model.py`.

## Comparison

| Criterion | tennisexplorer scraper (`scraper/`) | OddsHarvester (OddsPortal) | tennis-data.co.uk | The Odds API |
|---|---|---|---|---|
| Historical opening odds | Yes, Pinnacle + 8 secondary books | Yes (`--odds-history`) | **No** | Only from 2020-06-06 |
| Closing odds | Partial | Yes | **Excellent** (PSW/PSL ~99%, `MaxW/AvgW` are Oddsportal aggregates at **closing**) | Yes |
| Cost | 0 | 0 | 0 | 500 credits/month free; history paid |
| Effort | Already in the repo | Low (pip) | Almost none (annual xlsx) | Low (REST) |
| Blocking risk | Medium | Medium-high (OddsPortal anti-bot) | None | None |
| Fresh updates | To be written | Yes (`upcoming`) | Weekly | Real-time |

Verified facts:

- tennis-data `MaxW`/`AvgW` are **not** opening odds: they are Oddsportal "most recent
  before play starts" aggregates, i.e. closing. https://www.tennis-data.co.uk/notes.txt
- OddsHarvester is active (owner jordantete, tennis supported):
  https://github.com/jordantete/OddsHarvester
- The Odds API history starts 2020-06-06 and is paid:
  https://the-odds-api.com/historical-odds-data/

## Chosen stack

1. **tennis-data.co.uk** — closing baseline 2013-2025 (`data/match_odds/`, annual xlsx).
   Pinnacle `PSW`/`PSL` coverage ~99%. Used for calibration, never for morning inference.
2. **tennisexplorer scraper** (`scraper/`) — the only free source of pre-2020 Pinnacle
   **opening** odds found. Output: `data/tennis_odds_data_cleaned.csv` (~1 GB, not in git).
   Every match in the feature dataset gets at most one opening line, from a bookmaker
   priority list (Pinnacle first).
3. **OddsHarvester** — planned backfill for the missing opening-odds coverage and a
   fallback if tennisexplorer changes layout. Not wired into the pipeline yet; when it is,
   it must respect the same one-to-one merge rules.

Optional: The Odds API free tier for next-day matches (500 credits/month).

## Measured opening-odds coverage

Computed by `HybridDataLoader.enrich_with_opening_odds()` on the published dataset
(29,535 matches). A match is covered only when both opening prices exist.

```
overall   71.3%   (21,069 / 29,535)
Pinnacle  62.5%   (18,457 matches)
bet365    5.7%    (1,696)
Betfair   2.7%    (811)
other     0.4%    (105 across Unibet, 10Bet, BetVictor, Betway, Betsafe)
none      28.7%   (8,466)
```

| Year | Coverage |
|---|---|
| 2012 | 30.8% (partial year) |
| 2013 | 82.8% |
| 2014 | 80.2% |
| 2015 | 81.3% |
| 2016 | 71.7% |
| 2017 | 77.7% |
| 2018 | 78.6% |
| 2019 | 70.5% |
| 2020 | 68.5% |
| 2021 | 70.0% |
| 2022 | 67.4% |
| 2023 | 62.3% |
| 2024 | 57.8% |
| 2025 | 59.0% |

Earlier notes claimed "Pinnacle ~88.7%". That figure counted coverage only among matches
that already had a scraped line, not among all matches: the honest denominator is the
full dataset, hence 71.3% overall. The canonical run drops uncovered matches instead of
pricing them with closing odds.

## Data pipeline

```
data/atp_matches_TML/                  <- TML-Database (2024 missing), 2000-2025
data/match_odds/                       <- tennis-data.co.uk annual xlsx, 2013-2025
data/tennis_odds_data_cleaned.csv      <- scraped opening odds, ~1 GB (not in git)
```

Download the raw sources:

```bash
python download_data.py
```

URLs verified on 2026-10-01 (tennis-data paths change over time):

| Source | Pattern | Notes |
|---|---|---|
| TML-Database | `https://raw.githubusercontent.com/Tennismylife/TML-Database/master/<year>.csv` | CSVs at the repo root, not in `csv/` |
| tennis-data.co.uk | `http://tennis-data.co.uk/<obfuscated-segment>/<year>/<year>.xlsx` | the segment rotates: copy it from [data.php](https://www.tennis-data.co.uk/data.php). A `User-Agent` header is required, otherwise 403 |
| odds 2000-2012 | `.../<year>/<year>.xls` | `.xls`, not `.xlsx`; 2000 has results only |

Rebuild the dataset:

```bash
python data_pipeline.py
```

It writes `tennis_features_v36.csv` (20 MB) and `tennis_features_v36.meta.json`
(rows, date range, sha256, tolerance). The sidecar is tracked in git; the CSV is not.

Anti-leakage rules applied by the pipeline: one-to-one nearest-date merge, identity check
on the sets won by the winner, dedup on the real `(date, winner, loser)` key,
`DATE_TOLERANCE_DAYS = 7`.

Mandatory check before trusting any number:

```bash
python verify_leak_fix.py
```
