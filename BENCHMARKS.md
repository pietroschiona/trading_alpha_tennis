# Trading Alpha -- Scientific Benchmarks

> ## ⚠️ DOCUMENTO ANNULATO (2026-10)
>
> Tutti i risultati qui sotto descrivono il dataset **leakato** (55.049 righe, merge
> many-to-many in `smart_merge`). Non sono riproducibili e non sono un edge: il modello batteva
> il benchmark di mercato perché le quote erano abbinate ai match sbagliati, non perché avesse
> trovato segnale.
>
> Il punto di riferimento onesto è il **Brier di mercato 0.2035** (2018+), misurato sul dataset
> riparato. Non 0.1974, che era il numero del dataset contaminato.
>
> Verifica: `python verify_leak_fix.py`. Dettagli: [README § Post-Mortem](README.md#post-mortem-il-leak).
>
> **Cosa insegna**: un test statistico che passa con p < 1e-95 *non* dimostra un edge. Dimostra
> solo che il campione è grande e che il segnale è nonzero — il p-value è insensibile al fatto che
> il segnale sia informazione legittima o rumore di allineamento. Il test anti-leakage va **prima**
> di ogni test di significatività, non dopo.

**Methodology:** 6-test empirical validation framework — **results VOID**
**Data (leakato):** 55,049 ATP matches (2001-2025)
**Data (riparato):** 24,494 ATP matches (2013-2025), TML-Database + tennis-data.co.uk
**Model:** Hydra-Oracle (Siamese NN + XGBoost + RL) -- walk-forward validated
**Status:** leak corretto e verificato; prestazioni sul dataset riparato **non ancora misurate**

---

## 0. Anti-Leakage (ora prima di tutto)

| Controllo | Leakato | Riparato | Atteso |
|---|---|---|---|
| Righe duplicate (Date, winner, loser) | 1.579 | **0** | 0 |
| `corr(h2h_ratio, outcome)` | +0.336 | **+0.091** | ~0.11 |
| Brier di mercato (2018+) | 0.1974 | **0.2035** | riferimento |
| Brier del modello | 0.1909 (**sotto il mercato**) | da misurare | ≥ mercato |

---

## 1. Statistical Significance Test — [ANNULATO]

I test seguenti restano utili come esempio di *cosa non basta*. Erano tutti "passati" su dati
leakati.

**Question (storica):** Is the 63.3% win rate on 6,067 bets statistically different from random (50%)?

| Test | Result | Verdict |
|---|---|---|
| Binomial test | Z = +20.71, p < 1e-95 | Highly significant |
| Monte Carlo (200K trials) | 0/200,000 random models reach >=63.3% | Not by chance |
| 95% CI (Jeffrey's interval) | [62.07%, 64.50%] | Entirely above 50% |
| Bayes Factor BF10 | ~ 1e94 | Decisive evidence for H1 |
| Cohen's h | 0.269 | Medium effect size |

**Conclusione storica (annullata):** ~~The probability of achieving 63.3% on 6,067 bets by chance is less than 1 in 1e95. The model is genuinely learning predictive patterns.~~

**Conclusione corretta:** la probabilità è bassa, ma la quantità misurata non era "accuratezza
predittiva": era la quantità di informazione che il dataset leakato dava al modello per free.

---

## 2. Anti-Leakage Validation (storico — [ANNULATO])

**Question:** Does the ~16% fallback to closing odds (when Pinnacle opening odds are unavailable) introduce data leakage?

### Design

Split the 17,991 betting-period matches into two groups:
- With Pinnacle opening odds: 15,074 matches (83.8%)
- Without Pinnacle (fallback): 2,917 matches (16.2%)

For each match, compute:
- opening_prob = fair probability from morning odds
- closing_prob = fair probability from closing odds (MaxW/MaxL)
- edge = closing_prob - opening_prob

### Results

| Metric | With Pinnacle | Without Pinnacle |
|---|---|---|
| Mean edge | 0.0081 | 0.0000 |
| Edge std | 0.0523 | 0.0000 |
| Edge > 1% opportunities | 7,279 (100%) | 0 (0%) |

**For non-Pinnacle matches:** opening = closing = MaxW (by fallback design), so edge = 0 identically. The fallback produces zero edge opportunities.

**For Pinnacle matches:** The edge closing_prob - opening_prob captures real market drift. At edge > 1%, the actual win rate of the favorite is 82.5%.

### Proof of Sound Methodology

The model's edge detection pipeline:
1. Sees input: Pinnacle opening odds (83.8%) OR MaxW closing odds (16.2%, edge-neutral)
2. Predicts: Win probability approximating closing probability (via training + debiaser)
3. Computes edge: predicted_prob - opening_prob
4. For Pinnacle matches: Edge approximates closing - opening = real market drift signal
5. For fallback matches: Edge approximates MaxW - MaxW = 0 (no fake signal)

The model cannot find false edges on fallback matches because opening and closing probabilities are identical.

---

## 3. Ablation Study: Debiasing Effect — [ANNULATO]

**Question:** What does the OddsDebiaser contribute?

The debiaser is a Ridge regression trained online to map opening probability (from Pinnacle) to closing probability (from MaxW/MaxL).

**Empirical edge statistics:**

| Threshold | Matches | Actual Win Rate | Interpretation |
|---|---|---|---|
| Edge > 0.5% | 8,061 | 81.6% | Market drift is predictive |
| Edge > 1% | 7,279 | 82.5% | Stronger drift, higher accuracy |
| Edge > 2% | 5,651 | 84.3% | |
| Edge > 3% | 4,136 | 85.6% | |
| Edge > 5% | 1,951 | 88.0% | Extreme drift is rare but reliable |

**Key insight:** The debiaser captures the systematic tendency of markets to move from opening to closing odds. Matches where the market moves significantly in one direction (without new information) have a higher probability of that direction being correct.

---

## 4. Strategy Comparison — [ANNULATO]

**Question:** How do different betting strategies perform on the same prediction signal?

All strategies use the same win probability from the model's value head. They differ only in how they convert probability into stake size.

| Strategy | Stake Rule | Final Equity (1K start) | Max Drawdown |
|---|---|---|---|
| Flat | Fixed EUR 10/bet | EUR 19,381 | ~EUR 500 |
| Perc | 1% of bankroll | EUR 1,289,447 | ~EUR 90K |
| Kelly | 10% fractional Kelly | EUR 1,547,224 | ~EUR 110K |
| Reference | Adaptive Kelly | EUR 59,431 | ~EUR 4K |
| RL_Agent | Policy gradient (0-5%) | EUR 3,979,788 | ~EUR 180K |

**RL_Agent outperforms by 67x vs Reference strategy** *(annullato — su dati leakati)* -- the policy gradient learns:
- When to stake aggressively (high confidence, low variance)
- When to scale back (unfamiliar market conditions)
- When to sit out entirely (CUSUM drift detected or XGB veto active)

---

## 5. Baseline Comparison — [ANNULATO]

**Question:** How does the full model compare against simpler approaches?

| Model | Win Rate | AUC | Brier Score |
|---|---|---|---|
| Pinnacle Opening Odds (market) | 69.4% | 0.762 | 0.199 |
| Logistic Regression (ELO + rank) | 50.0% | 0.500 | 0.250 |
| Hybrid Model (on 6,067 selected bets) | 63.3% | -- | -- |

**Interpretation (storica, annullata):**
- ~~The market predicts all matches at 69.4% — this is the efficiency baseline~~
- ~~Logistic Regression on ELO alone fails — ELO needs non-linear interactions~~
- ~~The Hybrid Model achieves 63.3% on selected bets only — it identifies a subset where the market drift is predictable~~

**Correzione:** la riga "Hybrid Model" è il Brier 0.1909 contro 0.199 di mercato — cioè il modello
*vinceva*. Su un mercato efficiente come Pinnacle, un modello che vince stabilmente è più
probabilmente disallineato che bravo. Il Brier corretto sul dataset riparato è **0.2035** e
costituisce il riferimento da battere.

Il confronto "mercato 69.4% vs modello 63.3% su subset" era inoltre un confronto scorretto: WR
selezionato e WRglobale non sono confrontabili, perché il subset è scelto dall'edge del modello.

---

## 6. Sensitivity Analysis — [ANNULATO]

**Question:** How stable are the results to parameter changes?

| Parameter | Value Used | Impact |
|---|---|---|
| START_YEAR | 2013 | Pinnacle coverage >90% after 2013 |
| BETTING_START_YEAR | 2017 | 4 years warmup for ELO stabilization |
| Slippage | 2% | Realistic execution cost |
| Min Edge | 0.01 (1%) | Standard threshold |
| Kelly Fraction | 0.015 (1.5%) | Conservative sizing |
| Liquidity cap | EUR 5,000 | Legal constraint |

**With 2001 START (16 years training, includes pre-Pinnacle data):**
- WR: 68.4% on 7,353 bets (partially trained on data without Pinnacle separation)

**With 2013 START (4 years training, clean data):**
- WR: 63.3% on 6,067 bets (validated anti-leakage)

The 5.1 percentage point drop confirms that the earlier training data (2001-2012, with 0-60% Pinnacle coverage) was contaminating the model. After removing it, results are lower but methodologically sound.

---

## 7. Reproducibility Checklist

- [x] Deterministic random seed? Walk-forward is sequential (deterministic)
- [x] Walk-forward validation (no look-ahead bias)?
- [x] Online learning (no data snooping)?
- [x] Symmetry augmentation documented?
- [x] Hyperparameters reported? (top of hydra_model.py)
- [x] **Merge match↔odds one-to-one? (post-fix in `data_pipeline.py`)**
- [x] **Zero duplicate (Date, winner, loser)? (`verify_leak_fix.py`)**
- [x] **`corr(h2h_ratio, outcome) < 0.15`? (`verify_leak_fix.py`)**
- [ ] **Prestazioni sul dataset riparato misurate?**
- [ ] **Brier del modello ≥ Brier di mercato (0.2035)?**

Le ultime due voci restano aperte: non c'è ancora un edge dimostrato.

---

## Quick Replication

```bash
# 1. Dependencies
pip install -r requirements.txt

# 2. Raw data
python download_data.py

# 3. Rebuild dataset
python data_pipeline.py

# 4. ANTI-LEAK CHECK — non saltarlo
python verify_leak_fix.py

# 5. Train + backtest
python -c "import hydra_model as m; m.main()"
python run_benchmarks.py --quick

# Output: walkforward_results.txt, charts/*.png, benchmarks/
```

---

*I test statistici di questo documento descrivevano un dataset leakato e sono annullati. La
verifica riproducibile oggi è `verify_leak_fix.py`.*
