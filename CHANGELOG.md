# Changelog

## 1.0.0 — 2026-11-23

The honest release. The headline result is negative, and that is the point.

### Fixed

- **Data leakage in the odds-to-match merge.** `smart_merge()` joined odds on
  `year + players` with a 15-day window and deduplicated on a key that collapsed
  distinct matches between the same players. `h2h_ratio` therefore encoded the
  current result (`corr = 0.336`). The merge is now one-to-one, nearest date,
  with an identity check on sets won and a dedup on `(date, winner, loser)`
  (`DATE_TOLERANCE_DAYS = 7`).
- **Morning inference used closing odds.** The canonical run now requires real
  opening odds; matches without them are dropped, not filled. Closing odds are
  used only for evening calibration.
- Benchmark parsing now reads the `Wins:` line the model actually writes, and
  refuses to publish p-values when the dataset hash does not match its sidecar.

### Added

- Market-relative scoring: model vs market Brier on the same bets, plus a
  calibration table and ECE.
- Provenance for every run: code commit, dataset SHA-256, opening-odds coverage,
  data mode, modeling rows, per-bet CSV and a JSON manifest.
- `make_figures.py` regenerates the README figures from the published bets.
- Unit tests and a CI workflow (Python 3.12, pinned dependencies).
- Documentation: architecture, dataset data card, data sources, runbook, honest
  benchmark report.

### Removed

- The PPO paper reference and all claims of reinforcement learning: the stake
  head is supervised, not policy-gradient.
- The historical "63.3% win rate / +30% ROI" claims from the README and the
  repository description. They were measured on the leaked dataset and are not
  reproducible.
