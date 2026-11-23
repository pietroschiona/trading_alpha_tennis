# Data card: `tennis_features_v36.csv`

The feature dataset used by every published number. It is not committed (20 MB); it is
rebuilt deterministically from the public raw sources, and its identity is pinned by the
sidecar committed next to it.

| Field | Value |
|---|---|
| Rows (matches) | 29,712 |
| Date range | 2012-12-31 .. 2025-11-16 |
| Columns | 83 |
| Seed | 42 |
| Date tolerance in the odds merge | 7 days |
| SHA-256 | `467c8a17369bf9126969f06793149fd65e98cd9d930f16572e1fb4e9e897c5bb` |
| Sidecar | `tennis_features_v36.meta.json` (tracked) |
| Generator | `python data_pipeline.py` (~5 min) |

## Sources

| Raw input | Origin | License / terms |
|---|---|---|
| ATP match statistics 2000-2025 | [TML-Database](https://github.com/Tennismylife/TML-Database), Jeff Sackmann schema | Check the upstream repo |
| Closing odds 2013-2025 (Pinnacle `PSW`/`PSL`) | [tennis-data.co.uk](https://www.tennis-data.co.uk/) annual xlsx | Free for personal use; check the site notes |
| Opening odds (scraped) | tennisexplorer.com, via `scraper/` | Not redistributed: the ~1 GB file stays local |

The derived feature CSV mixes all three. It is published as a checksum, not as a
redistribution of the odds: to obtain the exact file you rebuild it from the same sources.

## Columns

| Group | Examples | Notes |
|---|---|---|
| Match identity | `Date`, `winner_name`, `loser_name`, `tourney_name`, `surface`, `round` | one row = one match |
| Player context | height, age, handedness, rank | as of the match |
| Match statistics | aces, double faults, serve %, break points | post-match, but not used as pre-match input |
| Market | closing odds (`MaxW`/`MaxL` etc.), implied probabilities | calibration target only |
| Engineered | ELO (global + surface), EWMA (17 stats), fatigue, fragility, SSI, h2h | every historical feature uses strictly earlier dates |

Opening odds are **not** in this file: they are merged at run time from the scraped CSV,
one line per match maximum. A match without an opening line is dropped by the canonical
run, never priced with closing odds.

## Known limitations

- Opening-odds coverage is 71.1% overall, and falls to 57.8% in 2024 (see
  [`data-sources.md`](data-sources.md)). The betting sample is therefore not the full ATP
  population.
- Matches between the same two players within 7 days are matched by nearest date plus an
  identity check on sets won; the residual ambiguity is measured by
  `verify_leak_fix.py` (0 duplicate rows, `corr(h2h, outcome) = 0.094`).
- The dataset is rebuilt from live sources whose URLs and layouts change. A rebuild in a
  future year will not reproduce this hash exactly; the sidecar records the gap honestly.

## Audit

```bash
python verify_leak_fix.py
python run_benchmarks.py --quick
```

A mismatch between the dataset hash and the sidecar makes `run_benchmarks.py` mark the
report **VOID** and exit 1.
