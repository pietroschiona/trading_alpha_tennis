# CLAUDE.md — trading_alpha_tennis

Sistema ML per value betting tennistico (Hydra-Oracle: Siamese NN + XGBoost residual + RL staking).
Python ≥3.10, Windows. Dipendenze: `pip install -r requirements.txt` (+ `requirements-scraping.txt` per lo scraper).

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

## Stato al 2026-10-01

- ✅ Fix leak applicato e verificato: 24.494 righe, 0 duplicati, corr(h2h,outcome) **0.091**.
- ✅ Scraper, raw e fonti recuperati e documentati in `docs/data-sources.md`.
- ❌ **Prestazioni sul dataset riparato non ancora misurate**: serve `tensorflow`, non installato.
  Finché non gira `hydra_model.py`, **non esiste alcun edge dimostrato**.
- Riferimento da battere: **Brier di mercato 0.2035** (2018+), non 0.1974 (quello del dataset leakato).
- Attenzione: il dataset attuale ha **solo quote closing**, quindi la separazione mattina/sera
  dell'architettura non è alimentata da dati reali.

## Fonti dati (dettagli in `docs/data-sources.md`)

| Cosa | Dove |
|---|---|
| Match stats raw | `data/atp_matches_TML/` — TML-Database (github.com/Tennismylife/TML-Database), 2000–2025, schema Jeff Sackmann |
| Closing odds | `data/match_odds/` — tennis-data.co.uk annuali 2013–2025 (xlsx). PSW/PSL = Pinnacle closing ~99%. **MaxW/AvgW NON sono opening** |
| Opening odds | `data/tennis_odds_data_cleaned.csv` (~1 GB, gitignored) — scraped da tennisexplorer.com. Master copy: `D:/Betting/codice/modello2torri2/tennis_odds_data_cleaned.csv` |
| Scraper | `scraper/` — Selenium, tennisexplorer.com, recuperato da `D:/Betting/scraper_per_odds/prova3/` |

Nessun CSV grande in git. I raw si ri-scaricano con `python download_data.py` (URL e trappole in `docs/data-sources.md`).

## Comandi

```powershell
python download_data.py                        # popola data/ con i raw
python data_pipeline.py                        # rigenera tennis_features_v36.csv (~5 min)
python verify_leak_fix.py                      # verifica anti-leak, DA FARE PRIMA DI OGNI NUMERO
python -c "import hydra_model as m; m.main()"  # training + backtest (~8 min, richiede tensorflow)
python run_benchmarks.py                       # 6 test statistici + Test 6 duplicati
python run_benchmarks.py --quick               # salta solo il Test 5 (~30 s)
```

Config: costanti in testa a `hydra_model.py` (START_YEAR=2013, BETTING_START_YEAR=2017) e
`data_pipeline.py` (DATE_TOLERANCE_DAYS=7, YEAR range). Output: `benchmarks/`, `charts/`, `model_state/`.

`run_benchmarks.py` avvisa se `tennis_features_v36.csv` è più recente di `walkforward_results.txt`:
in quel caso i test 1-4 descrivono il vecchio dataset e i loro p-value non sono da usare.

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
