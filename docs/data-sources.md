# Fonti dati — confronto e scelta

Casò d'uso: value betting ATP 2013–2025; opening odds = feature (smart money bias),
closing odds = target/calibrazione. Confronto eseguito 2026-10-01.

## Tabella comparativa

| Criterio | Scraper tennisexplorer (`scraper/`) | OddsHarvester (OddsPortal) | tennis-data.co.uk | The Odds API |
|---|---|---|---|---|
| Opening odds storiche | Sì (Pinnacle ~88.7% fino dic 2025) | Sì (`--odds-history`) | **No** | Solo da giu 2020 |
| Closing odds | Parziale | Sì | **Eccellente** (PSW/PSL ~99%, MaxW/AvgW = aggregati Oddsportal al **closing**) | Sì |
| Costo | 0 | 0 | 0 | Free 500 cred/mese; storico paid |
| Sforzo | Già nel repo | Basso (pip) | Quasi nullo (xlsx annuali) | Basso (REST) |
| Rischio blocco | Medio | Medio-alto (anti-bot OddsPortal) | Nessuno | Nessuno |
| Update fresche | Da scrivere | Sì (`upcoming`) | Settimanale | Real-time |

Fatti verificati:
- tennis-data `MaxW`/`AvgW` **non sono opening** — notes.txt: aggregati Oddsportal "most recent
  before play starts", cioè closing. https://www.tennis-data.co.uk/notes.txt
- OddsHarvester attivo (owner jordantete, tennis supportato): https://github.com/jordantete/OddsHarvester
- The Odds API storico solo da 6 giu 2020, paid: https://the-odds-api.com/historical-odds-data/

## Scelta: stack a 3 livelli

1. **tennis-data.co.uk** — base closing 2013–2025 (file annuali in `data/match_odds/`, già scaricati).
   Zero sforzo, Pinnacle ~99%.
2. **Scraper tennisexplorer** (`scraper/`) — unica fonte free di **opening** Pinnacle pre-2020;
   dataset già prodotto: `data/tennis_odds_data_cleaned.csv` (~1 GB, non è in git).
   Per odds fresche: update settimanale.
3. **OddsHarvester** — backfill gap dell'88.7% + fallback se tennisexplorer cambia layout.
   Feature flag `opening_missing` + imputazione quando manca.

Opzionale: The Odds API free tier per le odds pre-match dei match di domani (500 cred/mese bastano).

## Pipeline dei dati

```
data/atp_matches_TML/   ← TML-Database (github.com/Tennismylife/TML-Database), 2000–2025
data/match_odds/        ← tennis-data.co.uk annuali 2013–2025 (xlsx)
data/tennis_odds_data_cleaned.csv ← scraped, ~1 GB (non è in git)
```

Scaricare i raw: `python download_data.py`.

URL da usare (verificati 2026-10-01, il path di tennis-data cambia nel tempo):

| Sorgente | Pattern | Note |
|---|---|---|
| TML-Database | `https://raw.githubusercontent.com/Tennismylife/TML-Database/master/<anno>.csv` | CSV **alla root** del repo, non in `csv/` |
| tennis-data.co.uk | `http://tennis-data.co.uk/<segmento_offuscato>/<anno>/<anno>.xlsx` | il segmento cambia: rileggilo su [data.php](https://www.tennis-data.co.uk/data.php). Serve un `User-Agent`, altrimenti 403. |
| odds 2000–2012 | `.../<anno>/<anno>.xls` | `.xls`, non `.xlsx`; il 2000 è solo risultati, senza quote |

Rigenerazione dataset: `python data_pipeline.py` → `tennis_features_v36.csv`
(con fix anti-leakage: merge one-to-one nearest-date, identity check sui set vinti dal
vincitore, dedup su chiave reale, `DATE_TOLERANCE_DAYS = 7`).

Verifica obbligatoria prima di credere a qualsiasi numero: `python verify_leak_fix.py`.
