# trading_alpha_tennis

> ## ⚠️ I risultati storici di questo progetto erano un data leak
>
> Il claim originale **"63.3% WR / +30% ROI"** non era un edge: era artefatto di un merge
> difettoso in `data_pipeline.py`, segnalato da u/1ma9nja su r/algobetting (thread `1tviwt7`,
> [script di verifica](https://gist.github.com/boristobbe/9ec6a6b96234b7dced1f6195a1a5c1d9)).
>
> **Causa:** `smart_merge()` abbinava le quote ai match su chiave `anno + giocatori` entro una
> finestra di 15 giorni. Due match distinti degli stessi giocatori (R16 e QF, o due tornei
> vicini) collassavano sulla stessa riga; `drop_duplicates` li cancellava a caso e le quote
> finivano sulla partita sbagliata. Ne consegue che `h2h_ratio` — calcolato con
> `shift(1).cumsum` sulla data — finiva per contenere l'esito del match corrente:
> `corr(h2h_ratio, outcome) = 0.336` contro ~0.11 atteso da una feature lecita.
>
> **Correzione (2026-10):** merge one-to-one con selezione nearest-date, identity check sui
> set vinti dal vincitore, dedup su chiave reale. Dataset ricostruito da zero.
>
> **Stato attuale:** il fix ai dati è **verificato** (vedi [Post-Mortem](#post-mortem-il-leak)).
> Le prestazioni del modello sul dataset ripulito sono **in corso di misurazione** e non sono
> ancora pubblicate: non si ripubblicano i numeri storici finché non escono da un re-run pulito.
>
> Cosa *non* è mai stato provato qui: che il sistema batta il mercato. Su dati senza leak il
> confronto con il Brier di mercato è l'unico riferimento valido.

**Siamese NN + XGBoost + RL staking for ATP tennis analysis.** I dati grezzi sono match stats
(schema Jeff Sackmann) e quote closing Pinnacle (tennis-data.co.uk, ~99% di copertura).  

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Post-Mortem: il leak](#post-mortem-il-leak)
- [Scientific Validation](#scientific-validation)
- [Benchmark Results](#benchmark-results)
- [Anti-Leakage Design](#anti-leakage-design)
- [Key Components](#key-components)
- [Data Pipeline](#data-pipeline)
- [Project Structure](#project-structure)
- [FAQ](#faq)
- [License](#license)

---

## Overview

Trading Alpha is a research-grade hybrid ML system combining deep learning (Siamese Neural Networks), gradient boosting (XGBoost), and reinforcement learning (policy-gradient staking) to detect market inefficiencies in ATP tennis betting markets.

**Core insight:** Use XGBoost to model the residuals of the neural network -- the tree-based model captures non-linear patterns and interaction effects that the neural network misses, creating an ensemble that outperforms either model alone.

### Design Principles

| Principle | Implementation |
|---|---|
| Hydra-Oracle Architecture | Dual-head design: Blind Analyst (physics-only probability) + Aware Trader (market-aware stake) |
| Residual Ensemble | XGBoost fits on NN residuals, learning what the deep model systematically misses |
| Online Walk-Forward | Model retrains every 7-14 days with recency-weighted sampling |
| Anti-Leakage | Morning odds for inference, closing odds ONLY for evening calibration (non ancora alimentato da dati reali) |
| Real Pinnacle Odds | closing odds da tennis-data.co.uk (PSW/PSL), ~99% copertura 2013-2025 |

---

## Architecture

```
                    DATA PIPELINE
  data_pipeline.py -> ATP match data (TML-Database) + closing odds (tennis-data.co.uk)
  -> Feature engineering (ELO, EWMA, surface, momentum, H2H)


                    HYBRID MODEL

  +---------------------------+   +---------------------------+
  |   BLIND ANALYST (Head 1)  |   |   XGBoost Censor          |
  |   Siamese Network         |   |   + Fits on NN residuals  |
  |   + Twin towers (shared)  |   |   + Windowed memory       |
  |   + Player A vs B         |   |   + Veto if overconfident |
  |   + Surface ELO + stats   |   +-----------+---------------+
  +------------+--------------+               |
               |                              |
               +---------------+--------------+
                               |
                      +--------+--------+
                      |   ENSEMBLE      |
                      |   Predicto      |
                      +--------+--------+
                               |
                      +--------+--------+
                      | AWARE TRADER    |
                      | (Head 2: RL)    |
                      | Policy gradient |
                      +--------+--------+
                               |
                    RISK MANAGEMENT
  + CUSUM drift detector    + Kelly sizing (adaptive)
  + Drawdown management     + Probability calibrator
  + Odds debiaser (Ridge)   + 2% execution slippage
  + Liquidity caps (EUR 5,000)  + 5-strategy comparison
```

### Dual-Head Design

```
Input: [Player A stats, Player B stats, Physical context, Market context]
                    |
         +----------+-----------+
         |                      |
+--------+--------+   +---------+---------+
|  HEAD 1:        |   |  HEAD 2:          |
|  BLIND ANALYST  |   |  AWARE TRADER     |
|                 |   |                   |
|  Input:         |   |  Input:           |
|  + Player A     |   |  + Analyst opinion|
|  + Player B     |   |  + Market context |
|  + Physics ctx  |   |  + Player features|
|                 |   |                   |
|  Output:        |   |  Output:          |
|  win_prob       |   |  stake_fraction   |
|  (sigmoid)      |   |  (0 to 5% bank)   |
+-----------------+   +-------------------+
```

**Why separate them?** The Blind Analyst predicts match outcomes using physics-only data (player stats, ELO, surface). The Aware Trader sees the analyst's opinion PLUS market context (odds, market efficiency signals) to decide stake size. This separation prevents the model from learning spurious correlations between market noise and match outcomes.

---

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### 1. Scarica i dati grezzi

```bash
python download_data.py
```

Popola `data/atp_matches_TML/` (26 CSV, TML-Database) e `data/match_odds/` (13 XLSX,
tennis-data.co.uk). I comandi esatti e le licenze sono in [`docs/data-sources.md`](docs/data-sources.md).

### 2. Rigenera il dataset

```bash
python data_pipeline.py
```

~5 min. Produce `tennis_features_v36.csv` (~24.500 righe, 2013-2025).

### 3. Verifica che non ci sia leak — **prima di credere a qualsiasi numero**

```bash
python verify_leak_fix.py
```

Deve stampare 0 duplicati e `corr(h2h_ratio, outcome)` sotto 0.15. Se fallisce, fermati:
i risultati che verranno fuori non significano niente.

### 4. Train + backtest

```bash
python -c "import hydra_model as m; m.main()"   # ~8 min
python run_benchmarks.py --quick                 # test statistici, ~30 s
```

### View Results

```
walkforward_results.txt    # Strategy comparison results
charts/                    # Equity curve plots
benchmarks/                # Statistical validation reports
```

### (Optional) Quote opening

Solo per odds fresche: servono le proxy (`scraper/test_proxies.py`) e Chrome installato.
Non serve per il rebuild storico. Vedi [Data Pipeline](#data-pipeline).

---

## Post-Mortem: il leak

### Il difetto

`smart_merge()` in `data_pipeline.py` costruiva le coppie match↔quote così:

1. join su chiave `anno + winner + loser` (molti-a-molti);
2. filtro `|Date_match - Date_odds| <= 15 giorni`;
3. `drop_duplicates(['Date_match', winner, loser], keep='first')`.

Il passo 3 è il bug: due match realmente distinti che condividono i due giocatori e cadono
nella stessa finestra di 15 giorni producono la **stessa chiave di dedup**. La riga sopravvissuta
era scelta arbitrariamente, quindi il match conservato poteva avere le quote dell'altro, o nessuna.
Le quote finivano sulla partita sbagliata e `h2h_ratio` — funzione di `shift(1).cumsum` ordinato per
data — non poteva più garantire "solo partite precedenti".

### Il fix

- **One-to-one**: ogni match prende al massimo una riga quote, scelta per data più vicina.
- **Identity check**: il numero di set vinti dal vincitore (impronta del punteggio, indipendente
  dall'esito) deve coincidere tra match e quote. È ciò che distingue "R16 e QF" da "un match solo".
- **Dedup su chiave reale** `(data, winner, loser)`.
- `DATE_TOLERANCE_DAYS = 7`: la `tourney_date` di TML è il lunedì del torneo, il match si gioca
  fino a 7 giorni dopo. Con 3 giorni si perdevano ~25% delle coppie legittime; la tolleranza non è
  il problema di qualità, l'ambiguità lo era.

### Verifica

`python verify_leak_fix.py` — replica indipendente dello script di u/1ma9nja sul dataset
ricostruito:

| Controllo | Leakato | Riparato | Atteso |
|---|---|---|---|
| Righe duplicate (Date, winner, loser) | 1.579 | **0** | 0 |
| `corr(h2h_ratio, outcome)` | +0.336 | **+0.091** | ~0.11 |
| Brier di mercato (MaxW, 2018+) | 0.1974 | **0.2035** | riferimento |
| Brier del modello | **0.1909** (sotto il mercato) | da misurare | ≥ mercato |

Un modello che batte il mercato sul Brier è, da solo, sospetto: le quote Pinnacle sono la
migliore stima disponibile e superarle richiede o informazione non disponibile al momento della
scommessa, o una perdita di allineamento dei dati. Nel dataset leakato il modello vinceva
proprio perché allineava i dati al contrario rispetto alla realtà.

### Costo della riparazione

Il dataset riparato ha **24.494 righe** (2013–2025) contro 55.049: il vecchio count era
gonfiato dai match duplicati e dai match senza quote. La copertura è ~1.900 partite/anno con
quote, coerente con le ~2.600 partite ATP/anno di cui ~75% hanno quote su tennis-data.co.uk.

---

## Scientific Validation

`run_benchmarks.py` esegue 6 test statistici. **I risultati storici (p < 1e-95, BF₁₀ ~ 1e94,
h = 0.269, "0/200.000 modelli casuali") descrivevano il dataset leakato e non sono più
riproducibili.** Il test anti-leakage ora eseguibile è quello di `verify_leak_fix.py`, che è la
verifica prioritaria.

Regola operativa: `corr(feature, target) > 0.15` su una singola feature pre-match è un sospetto
di leak, non un edge.

---

## Benchmark Results

### Storico (VOID — dataset leakato)

Non ripubblicato. I numeri qui sotto erano generati su dati con il merge difettoso e
non rappresentano il sistema:

- ~~63.3% win rate su 6.067 scommesse~~
- ~~Brier 0.1909, sotto il benchmark di mercato 0.1974~~
- ~~EUR 3.979.788 di equity finale con RL_Agent~~

### Attuale

Da misurare: `python -c "import hydra_model as m; m.main()"` poi `python run_benchmarks.py`.
Il riferimento da battere è il **Brier di mercato (0.2035, 2018+)**, non il 50% della moneta.
Se il modello non sta almeno sul mercato, il risultato onesto è "nessun edge".

---

## Anti-Leakage Design

> **Stato reale dei dati:** il dataset ricostruato contiene **solo quote closing**
> (tennis-data.co.uk). Non c'è una serie di opening odds pulita e verificata a copertura
> sufficiente. Quindi la separazione "mattina si decide, sera si calibra" descritta sotto è
> **l'architettura prevista, non ciò che i dati attuali supportano**: oggi l'inferenza e la
> calibrazione usano la stessa fonte. Va risolta prima di rivendicare un edge — usare una
> closing price per decidere una scommessa da mettersi è il caso più comune di edge fittizio dopo
> il disallineamento dei dati.

L'architettura prevista separa i due tempi:

| Fase | Dati usati | Cosa succede |
|---|---|---|
| Mattino (inferenza) | quote opening | predice win probability + stake |
| Sera (calibrazione) | esito reale | allena la value head (BCE) |
| Sera (debiaser) | quote closing | calibra il drift opening→closing |
| Retraining | solo match passati (max 8.192) | NN ogni 7g, XGBoost ogni 14g |

### Key Principles

1. **XGBoost trains on residuals of actual outcomes**, not on closing odds
2. **OddsDebiaser** (Ridge regression) maps opening probability to closing probability, learning market drift patterns
3. **Market_Inefficiency feature is hardcoded to 0.0** -- previous versions leaked closing odds through this feature
4. **Distillation loss uses actual outcomes only** -- closing_true_prob is passed to the loss function but never used for gradient computation
5. **Ogni feature storica usa `shift(1)`**: `h2h_ratio`, EWMA e fatigue guardano solo date
   strettamente precedenti. Corr(`h2h_ratio`, outcome) deve restare sotto ~0.15

### Data Flow for Match Day t
```
  1. Load opening odds (Pinnacle or fallback)
  2. Predict: win_prob, stake_fraction
  3. Execute bets where edge detected
  4. Match completes, outcome recorded
  5. Update training buffers with outcome
  6. Periodically: retrain NN, XGBoost, debiaser
  7. Move to day t+1 (model never sees future data)
```

---

## Key Components

### 1. Deep Siamese Network (build_hydra_model)
- Twin towers (shared weights, ~7,500 parameters total)
- Dual head: Value (win probability) + Policy (stake fraction)
- Custom loss: Binary Crossentropy for value, PnL-aware for policy
- Features: ELO (global + per-surface), EWMA (17 stats), derived metrics (SSI, Fragility, Fatigue)

### 2. XGBoost Residual Corrector
- Fits on NN residuals: actual_outcome - predicted_prob
- Windowed training (last ~20K matches, retrained every 14 days)
- Veto mechanism: if residual < -0.02 (model overconfident), stake set to 0

### 3. Risk Management Suite
- **OddsDebiaser:** Ridge regression on opening-to-closing probability drift
- **ProbabilityCalibrator:** Isotonic regression for probability calibration
- **CUSUMDriftDetector:** Real-time model degradation monitoring
- **KellySizingAdaptive:** Drawdown-aware fractional Kelly
- **Execution Slippage:** 2% on all bets (realistic market impact)

### 4. Walk-Forward Engine
- Chronological processing: model never sees future data
- Online learning: retrains on accumulating buffer
- Symmetry augmentation: each match included A->B and B->A
- Recency weighting: recent samples weighted higher during training

---

## Data Pipeline

### Required Files

| Cosa | Dove | Fonte | Scaricabile |
|---|---|---|---|
| Match stats 2000–2025 | `data/atp_matches_TML/` | TML-Database (schema Jeff Sackmann), 26 CSV annuali | Sì, da GitHub |
| Quote closing 2013–2025 | `data/match_odds/` | tennis-data.co.uk, 13 XLSX annuali (PSW/PSL = Pinnacle) | Sì, da GitHub |
| Quote opening | `data/tennis_odds_data_cleaned.csv` (~1 GB) | scrapate da tennisexplorer.com con `scraper/` | Solo se si hanno le proxy |
| Dataset features | `tennis_features_v36.csv` | generato da `data_pipeline.py` | No, si rigenera |

Nessun CSV grande è nel repo: tutto si ri-scarica o si rigenera. I comandi esatti sono in
[`docs/data-sources.md`](docs/data-sources.md).

**Attenzione alla copertura:** `MaxW`/`AvgW` da tennis-data.co.uk sono **closing**, non opening.
Trattarli come opening e poi usarli come se fossero l'informazione disponibile *prima* del match è
un leak di per sé, oltre a quello già corretto.

### Feature Engineering (84 features)

**data_pipeline.py** generates the full feature set from raw ATP match data:
- **Financial features:** Implied probability, synthetic opening odds (Prelec FLB), sentiment bias
- **Contextual metrics:** sCPI (Synthetic Court Pace Index)
- **Historical features:** EWMA (17 metrics), Fatigue Load, Fragility Index, SSI, Clutch Factor
- **ELO ratings:** Global + 4 surface-specific (clay, grass, hard, carpet)

---

## Project Structure

```
trading-alpha/
|
+-- README.md                  This file
+-- CLAUDE.md                  Regole di progetto e lezione anti-leak
+-- BENCHMARKS.md              Scientific validation and methodology
+-- requirements.txt           Python dependencies
+-- requirements-scraping.txt  Selenium, per lo scraper delle quote
+-- .gitignore                 Esclude CSV grandi e log dal git
+-- download_data.py           Helper per il CSV delle quote
+-- verify_leak_fix.py         Replica indipendente del test anti-leak
|
+-- hydra_model.py             Main model (1,741 lines)
|                               Siamese Network + XGBoost + RL
|                               Walk-forward validation engine
|                               Risk management and charting
|
+-- data_pipeline.py           Data pipeline (668 lines)
|                               ATP match loading
|                               Feature engineering (ELO, EWMA, etc.)
|                               Merge match<->odds one-to-one (post-fix)
|
+-- run_benchmarks.py          Validation framework
|                               Statistical significance tests
|                               Monte Carlo simulation
|                               Anti-leakage verification
|
+-- docs/
|   +-- data-sources.md        Fonti dati, licenze, comandi di download
|   +-- RUNBOOK.md             Runbook operativo
|
+-- scraper/                   Selenium scraper di tennisexplorer.com
|   +-- scraper.py             Estrae le quote dai link dei match
|   +-- orchestrator.py        Gira i match su più proxy (resume-safe)
|   +-- test_proxies.py        Verifica le proxy prima di partire
|   +-- links_*.txt            Lista dei match ATP
|
+-- data/                      Raw, non nel git
|   +-- atp_matches_TML/       26 CSV annuali, match stats (schema Sackmann)
|   +-- match_odds/            13 XLSX annuali, quote closing
|   +-- tennis_odds_data_cleaned.csv   ~1GB, quote opening scrapate
|
+-- tennis_features_v36.csv    Dataset ricostruito (non nel git)
+-- charts/                    Generated equity curve plots
|   +-- Run1_5Strategies_Full.png
|   +-- Run1_5Strategies_2025.png
|   +-- Run1_Normal_Full.png, Run1_Normal_2025.png
|   +-- Run1_Exponential_Full.png, Run1_Exponential_2025.png
|
+-- model_state/               Saved model checkpoints
|   +-- best_hydra_rl_model.keras
|   +-- scaler.pkl
|   +-- feature_columns.pkl
|
+-- benchmarks/                Generated benchmark reports
|   +-- validation_report.md
|   +-- benchmark_results.json
|
+-- walkforward_results.txt    Latest walk-forward results
```

---

## FAQ

**Q: Are the results real or data leakage?**
A: **Erano leakage.** I test statistici passavano perché misuravano la contaminazione, non
l'edge: un modico disallineamento delle quote produce un segnale enorme e perfettamente
"significativo". La verifica corretta è `verify_leak_fix.py`.

**Q: Can I run this without the Pinnacle CSV?**
A: Yes, the model falls back to closing odds. Results will be less accurate but the pipeline works end-to-end.

**Q: What makes this different from a simple ELO model?**
A: ELO is a single-feature system. This model uses 84 features + neural network + XGBoost + RL policy optimization.

**Q: Why are the historical profits so high?**
A: Erano finti. Venivano da un merge difettoso che disallineava le quote dai match e faceva
leggere al modello l'esito corrente nelle feature storiche. Vedi
[Post-Mortem](#post-mortem-il-leak).

**Q: How long does it take to run?**
A: Full walk-forward: ~8 minutes. Quick validation: ~30 seconds.

**Q: Is this production-ready?**
A: No. Il progetto è alla stato di ricerca e, dopo la riparazione del leak, **non ha ancora un
edge dimostrato sul mercato**. Prima di usarlo serve un re-run pulito con il Brier di mercato
(0.2035) come riferimento.

---

## License

Apache License 2.0

## Author

**Pietro Schiona** -- [GitHub](https://github.com/pietroschiona)

---

*Dataset ricostruito su TML-Database (schema Jeff Sackmann) + quote closing tennis-data.co.uk.
Verifica anti-leak: 0 duplicati, corr(h2h, outcome) = 0.091. Le prestazioni storiche sono
annullate.*
