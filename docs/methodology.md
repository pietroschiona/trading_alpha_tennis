# Methodology

The system predicts a single ATP match from pre-match information and decides whether the price
offers value. This page explains every moving part and the reason it exists.

## 1. From names to a match key

Player names come from two sources with different spellings: TML-Database (`Jannik Sinner`) and
the odds files (`Sinner J.`). `data_pipeline.standardize_name_strict` reduces both to a compact
token (`sinner j`). The match key is `(date, winner, loser)`; the odds key adds the bookmaker.

`smart_merge` joins the two sides one-to-one: for each match it keeps the nearest odds line within
±7 days, checks that the number of sets won by the winner matches (a fingerprint of the score),
and deduplicates on the real key. The old version joined on `year + players` and deduplicated on a
key that collapsed two distinct matches — that was the leak.

## 2. Features

| Family | Examples | Causality |
|---|---|---|
| ELO | global + clay/grass/hard/carpet | updated after the match, read before |
| EWMA | 17 serve/return stats, `com = 20` | `shift(1)` per player, first value 0 |
| Surface and context | court pace index, round, tournament level | known before play |
| Physical | fatigue load, fragility, SSI | built from earlier matches |
| Market | opening implied probability, fair probability after vig removal | available in the morning |
| Head-to-head | prior meetings only | `shift(1)` |

`corr(h2h_ratio, outcome) = 0.094` on the repaired dataset. Above 0.15 on a single pre-match
feature is treated as a leak.

## 3. Model

### Blind Analyst

A Siamese network with two shared-weight towers, one per player. Inputs are player blocks
(`cols_A`, `cols_B`), physical context and market context. Output: win probability for player A.
The same model scores the mirrored sample (A↔B) during training, so the network cannot learn a
player-position bias.

### XGBoost residual corrector

The trees are trained on `outcome − opening market probability`, not on the outcome alone. A
positive residual means the market may under-price the player; a residual below -0.02 triggers a
veto and the stake is set to zero. Depth 3, learning rate 0.03, window 20,000 matches, retrained
every 14 days.

### Stake head

A supervised head that maps the analyst output plus market context to a bankroll fraction in
`[0, 0.05]`. The loss is PnL-shaped with a variance penalty (`create_pnl_loss`). It is **not**
reinforcement learning and there is no policy gradient: earlier descriptions said otherwise and
were wrong.

## 4. Morning and evening

```
morning:  opening odds -> probability -> edge vs price -> stake or veto
evening:  outcome -> training buffer; closing odds -> OddsDebiaser (calibration target)
```

Closing odds never enter the morning decision. If opening odds are missing for a match, the match
is dropped. The `OddsDebiaser` learns the drift from opening to closing probability with a rolling
Ridge regression (window 4,000); it is a calibration aid, not a source of edge.

## 5. Walk-forward

One pass, chronological, no shuffle:

1. 2013-2016 warm up ELO, EWMA and buffers; no bets.
2. From 2017-01-01, for each match day: predict using the current model, decide, then append the
   outcome to the buffers and update ELO.
3. Retrain the network every 7 days, XGBoost every 14, the debiaser every 30, always on data
   strictly before the current day.

Seeds are fixed (`RNG_SEED = 42`). Bit-for-bit reproducibility is not promised across TensorFlow
versions or CPU instruction sets; ordering and data are deterministic.

## 6. Staking and costs

Five strategies are simulated on the same signal: `Flat` (€10), `Perc` (1% of bankroll), `Kelly`
(fraction capped at 10%), `Reference` (1.5% Kelly), and `StakeHead` (the supervised head). Every
execution pays 2% slippage on the price. Stakes are capped by tournament liquidity and by a €5,000
hard cap. A strategy whose bankroll reaches zero stops betting.

## 7. Evaluation

| Metric | What it answers |
|---|---|
| ROI + 95% bootstrap CI | did the bets make money, with uncertainty |
| Brier model vs market, same bets | did the model price better than the closing line |
| Calibration table + ECE | is the model overconfident |
| Win rate | descriptive only — it does not price the bets |

The market is the reference. A win rate above 50% says the picker was right more often than a coin;
it says nothing about whether the picks were worth their price. See
[`../BENCHMARKS.md`](../BENCHMARKS.md) for the measured result and [`RUNBOOK.md`](RUNBOOK.md) for
the commands.
