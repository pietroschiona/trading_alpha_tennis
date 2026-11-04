# CLAUDE.md — trading_alpha_tennis

Sistema ML per value betting tennistico (Hydra-Oracle: Siamese NN a 38 feature + XGBoost residual +
stake head supervisionato; **niente RL**, vedi `hydra_model.py:76` e `hydra_model.py:838`).
Python ≥3.10, Windows. Dipendenze: `pip install -r requirements.txt` (+ `requirements-scraping.txt`
per lo scraper, `requirements-dev.txt` per i test).

## ⚠️ Lezione fondamentale: il data leakage del 2026-10

Il claim originale "63.3% WR / +30% ROI" era un **leak**, non un edge (segnalato da u/1ma9nja su
r/algobetting, thread `1tviwt7`). Causa: `smart_merge` in `data_pipeline.py` faceva join odds↔match
su anno+giocatore con tolerance 15 giorni → match distinti fra gli stessi giocatori collassati su
una riga, `h2h_ratio` (shift/cumsum su `Date`) encodava l'esito del match corrente (corr 0.336 vs ~0.11 reale).
Fix: merge one-to-one nearest-date, identity check sui set vinti dal vincitore, dedup su chiave reale,
`DATE_TOLERANCE_DAYS = 7` (la `tourney_date` TML è il lunedì del torneo; con 3 si perdeva il 25% delle
coppie legittime). Non regredire.

**Regole anti-leakage (non negoziabili):**
- Mai merge su chiave anno+giocatore senza filtro data stretta; ogni riga match ↔ max una riga odds.
- Qualunque feature storica (EWMA, h2h, fatigue) usa solo date **strettamente precedenti** (`shift(1)`).
- Prima di credere a un edge: corr(feature, target) > 0.15 su una singola feature pre-match = sospetto leak.
- Test di regressione: `python verify_leak_fix.py` (0 duplicati, corr(h2h,outcome) < 0.15) e
  `python run_benchmarks.py` (Test 6, duplicati). Entrambi devono passare **prima** di guardare i numeri.

## Stato al 2026-11-04

- ✅ Fix leak applicato e verificato: 29.535 righe, 0 duplicati, corr(h2h,outcome) **0.094**.
- ✅ Scraper, raw e fonti recuperati e documentati in `docs/data-sources.md`.
- ✅ TensorFlow installato: il run è stato misurato. **Nessun edge dimostrato.**
  - 3.700 scommesse finanziate dalla stake head, 1.183 vittorie (**32.0%**), avg odds 4.27.
  - Staked **32.229,04**, PnL **-1.000**, **ROI -3.10%**, CI 95% **[-10.19%, +4.13%]**, p=0.81.
  - Flat, Kelly e StakeHead finiscono a 0 (bankrupt); Perc 173; Reference 985.
  - Provenienza: ROI/CI/p vengono dal run `98fcd55+dirty` del 2026-11-04 e **non sono riproducibili
    da un commit** finché non si rifà un run pulito; i numeri del dataset e del leak vengono dal
    sidecar `tennis_features_v36.meta.json`.
- Riferimento da battere: **Brier di mercato 0.2037** (2018+), non 0.1974 (quello del dataset leakato).
- Quote opening opzionali: copertura **0.0%** nel run misurato, quindi la separazione mattina/sera
  dell'architettura non è alimentata da dati reali.

## Fonti dati (dettagli in `docs/data-sources.md`)

| Cosa | Dove |
|---|---|
| Match stats raw | `data/atp_matches_TML/` — TML-Database (github.com/Tennismylife/TML-Database), 2000–2025, schema Jeff Sackmann |
| Closing odds | `data/match_odds/` — tennis-data.co.uk annuali 2013–2025 (xlsx). PSW/PSL = Pinnacle closing ~99%. **MaxW/AvgW NON sono opening** |
| Opening odds | `data/tennis_odds_data_cleaned.csv` (~1 GB, gitignored, non è in git) — scraped da tennisexplorer.com. Opzionali: senza, il rebuild usa le sole closing |
| Scraper | `scraper/` — Selenium, tennisexplorer.com |

Nessun CSV grande in git. I raw si ri-scaricano con `python download_data.py` (URL e trappole in `docs/data-sources.md`).

## Comandi

```powershell
python download_data.py                        # popola data/ con i raw
python data_pipeline.py                        # rigenera tennis_features_v36.csv (~5 min)
python verify_leak_fix.py                      # verifica anti-leak, DA FARE PRIMA DI OGNI NUMERO
python -c "import hydra_model as m; m.main()"  # training + backtest (~8 min)
python run_benchmarks.py                       # return bootstrap + Test 5 drift opening/closing + Test 6 duplicati + Test 7 h2h
python run_benchmarks.py --quick               # salta solo il Test 5 (~30 s)
python -m pytest tests -q                      # test suite (requirements-dev.txt)
```

Config: costanti in testa a `hydra_model.py` (START_YEAR=2013, BETTING_START_YEAR=2017) e
`data_pipeline.py` (DATE_TOLERANCE_DAYS=7, YEAR range). Output: `benchmarks/`, `charts/`, `model_state/`.

`run_benchmarks.py` rifiuta di pubblicare p-value se il sidecar `tennis_features_v36.meta.json`
manca o se `sha256`/`rows` non coincidono con il dataset: in quel caso marca il report **VOID**
ed esce con codice 1.

## Scraper

```powershell
cd scraper && pip install -r ../requirements-scraping.txt
python test_proxies.py     # proxy da proxifly
python scraper.py          # input links_tennisexplorer_ATP_post2000_filtered.txt, resume-safe
```

Serve solo per odds fresche/aggiornamenti; il rebuild storico non lo richiede.

## Note

- README contiene il post-mortem del leak: i risultati onesti post-fix sono il riferimento, i claim
  vecchi (63.3% WR) sono storici e non riproducibili.
- Riferimenti di progetto nel vault Obsidian: `6 - note principali/Creazione Dataset Tennis SOTA.md`
  (progetto gold standard), `1 - appunti grezzi/Guida al contenuto dei dati scommesse AI sul Tennis.md` (schema fonti).

*Regole*
- E' mandatorio il continuo aggiornamento di un voult obsidian interno per l'allineamento dei dati
- Per capire il contesto è mandatorio che interagisci con il voult di obsidian in caso di dubbi o riguardo a ciò che è stato fatto recentemente e nelle ultime sessioni a riguardo
