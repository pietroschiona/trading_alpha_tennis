# scraper/ — tennisexplorer odds scraper

Scraper autonomo del repo. Ha prodotto
`tennis_odds_data_cleaned.csv` (~1 GB, 2005–2025, Pinnacle opening coverage 88.7%).
Fonte: tennisexplorer.com (match-detail pages), Selenium + Chrome + rotazione proxy gratuite.

## File

- `scraper.py` — scraper principale. Input: `links_tennisexplorer_ATP_post2000_filtered.txt`
  (55.811 match URL). Output: `tennis_odds_data.csv` (formato long: url, date_str, player_1/2,
  bookmaker, selection, closing_odd, opening_odd). Resume-safe: skip URL già in output/processed/skipped.
- `generate_links_ultra_fast.py` — genera la lista link da tennisexplorer.
- `test_proxies.py` — testa proxy da proxifly (GitHub) e scrive `verified_proxies.txt`.
- `orchestrator.py` — ciclo: scarica proxy → testa → lancia scraper → ripete; guard RAM 80%.
- `links_tennisexplorer_ATP_post2000_filtered.txt` — input link pre-generato.

## Uso

```powershell
pip install -r requirements-scraping.txt
cd scraper
python test_proxies.py          # opzionale ma consigliato
python scraper.py               # oppure python orchestrator.py per run autonomo
```

Chrome deve essere installato; il chromedriver viene scaricato via webdriver-manager.

Nota: serve solo per odds fresche (aggiornamento dataset). Il rebuild storico usa
`data/tennis_odds_data_cleaned.csv` (non è in git) e tennis-data.co.uk — vedi
`docs/data-sources.md`.
