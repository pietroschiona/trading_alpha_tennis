# Documentation

Start here, then follow the path that matches your question.

| I want to… | Read | Time |
|---|---|---|
| Understand what happened and why the result is negative | [README](../README.md) | 10 min |
| See the measured numbers and how they were tested | [BENCHMARKS](../BENCHMARKS.md) | 10 min |
| Understand the system and the morning/evening split | [architecture](architecture.md) | 8 min |
| Understand ELO, EWMA, the residual model, staking | [methodology](methodology.md) | 10 min |
| Run it myself | [RUNBOOK](RUNBOOK.md) | 5 min + runtime |
| Rebuild the dataset and know its limits | [dataset](dataset.md), [data-sources](data-sources.md) | 8 min |
| Verify that a published number is reproducible | [reproducibility](reproducibility.md) | 5 min |
| Know what the model may and may not be used for | [model-card](model-card.md) | 4 min |
| Cite it | [CITATION.cff](../CITATION.cff), [references](references.md) | 1 min |

## The three checks that precede every number

1. `python verify_leak_fix.py` — 0 duplicate matches, h2h correlation below 0.15.
2. `walkforward_results.txt` — `Data mode: opening_odds`, coverage above zero.
3. `run_benchmarks.py` — return CI and model-vs-market Brier, same bets.

If one fails, the number is not a result. This rule comes from the project's own history: the
original 63% win rate was a data leak, documented in the [README](../README.md#the-2026-10-leak).
