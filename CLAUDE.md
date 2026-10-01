# CLAUDE.md — trading_alpha_tennis

Sistema ML per value betting tennistico (Hydra-Oracle: Siamese NN + XGBoost residual + RL staking).
Python ≥3.10, Windows. Dipendenze: `pip install -r requirements.txt` (+ `requirements-scraping.txt` per lo scraper).

## ⚠️ Lezione fondamentale: il data leakage del 2026-10

Il claim originale "63.3% WR / +30% ROI" era un **leak**, non un edge (segnalato da u/1ma9nja su
r/algobetting, thread `1tviwt7`). Causa: `smart_merge` in `data_pipeline.py` faceva join odds↔match
su anno+giocatore con tolerance 15 giorni → match distinti fra gli stessi giocatori collassati su
una riga, `h2h_ratio` (shift/cumsum su `Date`) encodava l'esito del match corrente (corr 0.336 vs ~0.11 reale).
Fix: merge one-to-one, `DATE_TOLERANCE_DAYS = 3`, dedup su chiave reale. Non regredire.

**Regole anti-leakage (non negoziabili):**
- Mai merge su chiave anno+giocatore senza filtro data stretta; ogni riga match ↔ max una riga odds.
- Qualunque feature storica (EWMA, h2h, fatigue) usa solo date **strettamente precedenti** (`shift(1)`).
- Prima di credere a un edge: corr(feature, target) > 0.15 su una singola feature pre-match = sospetto leak.
- Test di regressione: `python run_benchmarks.py` — l'assert anti-leakage deve passare.

## Fonti dati (dettagli in `docs/data-sources.md`)

| Cosa | Dove |
|---|---|
| Match stats raw | `data/atp_matches_TML/` — TML-Database (github.com/Tennismylife/TML-Database), 2000–2025, schema Jeff Sackmann |
| Closing odds | `data/match_odds/` — tennis-data.co.uk annuali 2013–2025 (xlsx). PSW/PSL = Pinnacle closing ~99%. **MaxW/AvgW NON sono opening** |
| Opening odds | `data/tennis_odds_data_cleaned.csv` (~1 GB, gitignored) — scraped da tennisexplorer.com. Master copy: `D:/Betting/codice/modello2torri2/tennis_odds_data_cleaned.csv` |
| Scraper | `scraper/` — Selenium, tennisexplorer.com, recuperato da `D:/Betting/scraper_per_odds/prova3/` |

Nessun CSV grande in git. Le fonti raw si ri-scaricano con i comandi in `docs/data-sources.md`.

## Comandi

```powershell
python data_pipeline.py                        # rigenera tennis_features_v36.csv (~5 min)
python -c "import hydra_model as m; m.main()" # training + backtest (~8 min)
python run_benchmarks.py                       # 6 test statistici + anti-leakage
python run_benchmarks.py --quick               # solo test statistici (~30 s)
```

Config: costanti in testa a `hydra_model.py` (START_YEAR=2013, BETTING_START_YEAR=2017) e
`data_pipeline.py` (DATE_TOLERANCE_DAYS=3, YEAR range). Output: `benchmarks/`, `charts/`, `model_state/`.

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
