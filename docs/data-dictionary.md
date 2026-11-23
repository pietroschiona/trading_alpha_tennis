# Data dictionary

83 columns in `tennis_features_v36.csv`, grouped. One row is one match, winner and loser named
explicitly. Raw match statistics describe the match that already happened; only the features
listed as pre-match are legal model inputs.

## Identity and tournament (7)

| Column | Meaning |
|---|---|
| `tourney_id`, `tourney_name`, `tourney_level`, `surface`, `draw_size`, `indoor`, `round` | tournament context, known before play |
| `match_num` | order inside the tournament |

## Date

| Column | Meaning |
|---|---|
| `tourney_date` | TML convention: the Monday of the tournament week |
| `Date_Tourney`, `Date` | normalized match date used for ordering and the odds merge |

## Winner block (13) and loser block (13)

`winner_id`, `winner_seed`, `winner_entry`, `winner_name`, `winner_hand`, `winner_ht`,
`winner_ioc`, `winner_age`, `winner_rank`, `winner_rank_points`, `winner_std` — and the same
`loser_*` columns. Ranks and ages are as of the match. `*_std` is the standardized name used to
join the two data sources.

## Match statistics (18)

`minutes`, `best_of`, plus `w_*` / `l_*` for aces, double faults, service points, first serves
in/won, second serves won, service games, break points saved/faced, and `Comment`.

These are **post-match** values. They are useful for building histories; they must never be fed
to the model as if known before the match. The model consumes EWMA/fatigue features built with
`shift(1)`, not these raw columns.

## Closing odds and market (14)

| Column | Meaning |
|---|---|
| `AvgW`, `AvgL`, `MaxW`, `MaxL` | tennis-data.co.uk closing aggregates — calibration target only |
| `implied_W`, `implied_L`, `overround` | implied probabilities and bookmaker margin |
| `true_prob_W`, `true_prob_L` | vig-free probabilities |
| `subj_prob_W`, `subj_prob_L` | probability with the favourite-longshot bias applied (Prelec) |
| `synth_odds_W_FLB`, `synth_odds_L_FLB` | synthetic prices from the FLB model |
| `synth_odds_W`, `synth_odds_L`, `rank_gap`, `noise_scale` | synthetic price and noise context |

Opening odds are **not** stored here. They are merged at run time from the scraped CSV and become
`Pin_Open_W` / `Pin_Open_L` inside `hydra_model.py`.

## Engineered context (17)

| Column | Meaning |
|---|---|
| `year` | match year |
| `sCPI` | synthetic court pace index |
| `w_Fatigue_Load`, `l_Fatigue_Load` | cumulative minutes and matches in the recent window |
| `w_Fragility_Index`, `l_Fragility_Index` | collapse-risk indicator |
| `w_SSI`, `l_SSI` | serve strength index |
| `w_Pythagorean`, `l_Pythagorean` | expected win rate from points won |
| `w_Clutch_Factor`, `l_Clutch_Factor` | performance in high-pressure points |

## Not stored, computed at run time

ELO (global and per surface), EWMA of the 17 match statistics, head-to-head ratio, and the market
context vector (`fair_prob_A`, `implied_A`, `implied_B`, `Market_Inefficiency = 0`). They are
rebuilt by `hydra_model.enrich_features_residual` from strictly earlier dates; storing them would
tempt a future merge to leak the current result.
