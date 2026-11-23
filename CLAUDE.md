# CLAUDE.md — trading_alpha_tennis

Sistema ML per value betting tennistico (Hydra-Oracle: Siamese NN + XGBoost sui residui di mercato +
stake head supervisionata; **niente RL**, vedi `hydra_model.py:76-83` e `create_pnl_loss`).
Python 3.10-3.12, CPU. Dipendenze: `pip install -r requirements.txt` (+ `requirements-scraping.txt`
per lo scraper, `requirements-dev.txt` per i test).

## ⚠️ Lezione fondamentale: il data leakage del 2026-10

Il claim originale "63.3% WR / +30% ROI" era un **leak**, non un edge (segnalato da u/1ma9nja su
r/algobetting, thread `1tviwt7`). Causa: `smart_merge` in `data_pipeline.py` faceva join odds↔match
su anno+giocatore con tolerance 15 giorni → match distinti fra gli stessi giocatori collassati su
una riga, `h2h_ratio` (shift/cumsum su `Date`) encodava l'esito del match corrente (corr 0.336 vs
0.0938 reale).
Fix: merge one-to-one nearest-date, identity check sui set vinti dal vincitore, dedup su chiave
reale, `DATE_TOLERANCE_DAYS = 7`. Non regredire.

**Regole anti-leakage (non negoziabili):**
- Mai merge su chiave anno+giocatore senza filtro data stretta; ogni riga match ↔ max una riga odds.
- Qualunque feature storica (EWMA, h2h, fatigue) usa solo date **strettamente precedenti**
  (`shift(1)`; niente `bfill`).
- Prima di credere a un edge: corr(feature, target) > 0.15 su una singola feature pre-match = sospetto leak.
- Test di regressione: `python verify_leak_fix.py` (0 duplicati, corr(h2h,outcome) < 0.15) e
  `python run_benchmarks.py` (Test 6 duplicati). Entrambi devono passare **prima** di guardare i numeri.

## Stato al 2026-11-23 — run canonica `3cbaf57`

- Dataset: **29.535 righe**, sha256 `362c5f1e…c3c0d`, 2012-12-31..2025-11-16, 83 colonne.
- Opening odds **obbligatorie** di default: copertura **71,3%** (21.069 match modellati), 68,0% sui
  match 2017+. I match senza opening vengono esclusi, mai prezzati con le closing.
- Fallback closing solo etichettato: `TRADING_ALPHA_ALLOW_CLOSING_FALLBACK=1` → `Data mode:
  closing_fallback`, mai usato per i numeri pubblicati.
- Risultati (2017-2025, 1.825 bet): WR **42,2%**, avg odds **2,83**, staked €18.351, PnL StakeHead
  **-€958**, ROI **-5,22%**, CI 95% **[-12,72%, +2,28%]**, p=0,9127 → **nessun edge**.
- Brier modello **0,2373** vs mercato **0,2186** sulle stesse bet (delta +0,0186); ECE **0,1161**.
- Riferimento di mercato sul dataset (2018+): Brier **0,2037**.
- Flat va in bancarotta alla bet 1.562; Reference chiude a €986 (-€14).
- Provenienza: `walkforward_results.txt`, `walkforward_bets.csv`, `walkforward_manifest.json`,
  `benchmarks/validation_report.md`. Il manifest deve dire `data_mode: opening_odds`.

## Fonti dati (dettagli in `docs/data-sources.md`)

| Cosa | Dove |
|---|---|
| Match stats raw | `data/atp_matches_TML/` — TML-Database, 2000–2025, schema Jeff Sackmann |
| Closing odds | `data/match_odds/` — tennis-data.co.uk annuali 2013–2025 (xlsx). PSW/PSL = Pinnacle closing ~99%. **MaxW/AvgW NON sono opening** |
| Opening odds | `data/tennis_odds_data_cleaned.csv` (~1 GB, gitignored) — scraped da tennisexplorer.com. Obbligatorie per la run canonica |
| Scraper | `scraper/` — Selenium, tennisexplorer.com |

Nessun CSV grande in git. Il sidecar `tennis_features_v36.meta.json` è tracciato; il dataset no.
I raw si ri-scaricano con `python download_data.py`.

## Comandi

```bash
python download_data.py                        # popola data/ con i raw
python data_pipeline.py                        # rigenera tennis_features_v36.csv (~5 min)
python verify_leak_fix.py                      # verifica anti-leak, PRIMA DI OGNI NUMERO
python -c "import hydra_model as m; raise SystemExit(m.main())"  # ~20 min, scrive gli artefatti
python run_benchmarks.py                       # return + Brier + calibrazione + integrità
python run_benchmarks.py --quick               # salta solo il Test 5 (drift)
python make_figures.py                         # figure chiare del README da walkforward_bets.csv
python -m pytest tests -q                      # 34 test (requirements-dev.txt)
```

Su questa macchina Ubuntu tutto gira in Docker (`python:3.12` + pin di `requirements.txt`); il
Python di sistema è 3.14 e non ha TensorFlow. La run canonica va lanciata con
`TRADING_ALPHA_CODE_COMMIT=$(git rev-parse --short HEAD)` passato al container: senza git
nell'immagine il manifest scriverebbe `unknown`.

Config: costanti in testa a `hydra_model.py` (START_YEAR=2013, BETTING_START_YEAR=2017) e
`data_pipeline.py` (DATE_TOLERANCE_DAYS=7, range anni). Output: `benchmarks/`, `charts/`,
`model_state/`.

`run_benchmarks.py` rifiuta di pubblicare p-value se il sidecar manca o se `sha256`/`rows` non
coincidono con il dataset: report **VOID** ed exit 1.

## Cosa NON esiste nel codice (non documentarlo come se esistesse)

- Reinforcement learning / PPO / policy gradient: rimosso anche il PDF.
- CUSUM drift detector e calibratore isotonico: mai implementati.
- CLV (closing line value): non calcolato.

## Scraper

```bash
cd scraper && pip install -r ../requirements-scraping.txt
python test_proxies.py     # proxy da proxifly
python scraper.py          # input links_tennisexplorer_ATP_post2000_filtered.txt, resume-safe
```

Serve solo per odds fresche/aggiornamenti; il rebuild storico non lo richiede.

## Documentazione

- README = post-mortem + risultati onesti + figure. `BENCHMARKS.md` = tabelle misurate.
- `docs/index.md` è la mappa; `docs/methodology.md` la meccanica; `docs/dataset.md` la data card;
  `docs/RUNBOOK.md` l'operatività; `docs/reproducibility.md` come verificare un numero.
- Riferimenti nel vault Obsidian: `trading/wiki/log.md`, `6 - note principali/Creazione Dataset
  Tennis SOTA.md`, `1 - appunti grezzi/Guida al contenuto dei dati scommesse AI sul Tennis.md`.

*Regole*
- È mandatorio il continuo aggiornamento di un vault Obsidian interno per l'allineamento dei dati.
- Per capire il contesto è mandatorio interagire col vault Obsidian in caso di dubbi o riguardo a
  ciò che è stato fatto recentemente.
