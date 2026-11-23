#!/usr/bin/env python
"""
Popola data/ con i dati grezzi.

  1. data/atp_matches_TML/  - TML-Database, 26 CSV annuali 2000-2025 (schema Jeff Sackmann)
  2. data/match_odds/       - tennis-data.co.uk, 13 XLSX annuali 2013-2025 (PSW/PSL = Pinnacle closing)
  3. data/tennis_odds_data_cleaned.csv - ~1 GB di quote opening scrapate (opzionale, vedi sotto)

Uso:
    python download_data.py           # scarica 1 e 2; 3 resta opzionale
    python download_data.py --odds    # cerca anche il CSV da 1 GB in data/ o ~

Poi: python data_pipeline.py  ->  tennis_features_v36.csv
"""

import os
import sys
import urllib.request

# tennis-data.co.uk risponde 403 a urllib con il default UA
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) trading_alpha_tennis/1.0"}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TARGET_PATH = os.path.join(DATA_DIR, "tennis_odds_data_cleaned.csv")

TML_BASE = "https://raw.githubusercontent.com/Tennismylife/TML-Database/master"  # CSV at repo root
TML_YEARS = range(2000, 2026)  # 26 file
# NB: il path contiene un segmento offuscato, cambia nel tempo.
# Verificalo da https://www.tennis-data.co.uk/data.php se i download prendono 404.
TD_BASE = "http://tennis-data.co.uk/hrjk-85HytOjkhth76j_ygh4jf7/{year}/{year}.xlsx"
TD_YEARS = range(2013, 2026)  # 13 file


def download(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return "skip"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
            data = r.read()
        with open(dest, "wb") as f:
            f.write(data)
        return "ok"
    except Exception as e:  # noqa: BLE001 - one bad year must not kill the rest
        if os.path.exists(dest):
            os.remove(dest)
        return f"FAIL {e}"


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    tml_dir = os.path.join(DATA_DIR, "atp_matches_TML")
    odds_dir = os.path.join(DATA_DIR, "match_odds")
    os.makedirs(tml_dir, exist_ok=True)
    os.makedirs(odds_dir, exist_ok=True)

    print("== TML-Database (match stats, 2000-2025) ==")
    tally = {"ok": 0, "skip": 0}
    for y in TML_YEARS:
        r = download(f"{TML_BASE}/{y}.csv", os.path.join(tml_dir, f"{y}.csv"))
        tally["skip" if r == "skip" else "ok"] += 1
        print(f"  {y}: {r}")
    print(f"  -> {tally['ok']} scaricati, {tally['skip']} gia' presenti\n")

    print("== tennis-data.co.uk (quote closing, 2013-2025) ==")
    for y in TD_YEARS:
        r = download(TD_BASE.format(year=y), os.path.join(odds_dir, f"{y}.xlsx"))
        print(f"  {y}: {r}")
    print()

    # Optional: opening odds, ~1 GB, only obtainable by scraping (see scraper/)
    if "--odds" not in sys.argv[1:]:
        print("Quote opening (~1 GB) opzionali per il rebuild del dataset, ma OBBLIGATORIE")
        print("  per il run canonico di hydra_model (senza, main() esce con codice 1).")
        print("  Il dataset si ricostruisce con le sole closing odds (2013-2025).")
    else:
        for path in [TARGET_PATH, os.path.expanduser("~/tennis_odds_data_cleaned.csv")]:
            if not os.path.exists(path):
                continue
            if os.path.abspath(path) != os.path.abspath(TARGET_PATH):
                print(f"Quote opening trovate in {path}")
                try:
                    os.symlink(path, TARGET_PATH)
                    print(f"  -> symlink creato in {TARGET_PATH}")
                except (OSError, NotImplementedError):
                    print("  -> symlink non riuscito, copiarlo a mano")
            else:
                print(f"Quote opening gia' in {TARGET_PATH} ({os.path.getsize(path)/1e6:.0f} MB)")
            break
        else:
            print("Quote opening (~1 GB) non trovate. NON BLOCCANTE per il rebuild storico:")
            print("  il dataset si ricostruisce con le sole closing odds (2013-2025).")
            print("  Per rigenerarle serve lo scraper tennisexplorer + proxy: vedi scraper/README.md")
    print("\nProssimo passo: python data_pipeline.py")


if __name__ == "__main__":
    main()
