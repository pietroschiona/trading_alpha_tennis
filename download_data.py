#!/usr/bin/env python
"""
Download script for the Pinnacle opening odds dataset.

The full tennis dataset with Pinnacle opening odds is ~1GB and cannot be
stored in the git repository. Run this script before training to download
or locate the data.

## Data Sources

### Option 1: Already on disk (if you have the original project)
If you have access to the original Trading Alpha project structure,
symlink or copy the CSV:
  ln -s /path/to/tennis_odds_data_cleaned.csv data/

### Option 2: Download from Hugging Face (TODO)
The dataset will be uploaded to Hugging Face Datasets.
Once available:
  pip install huggingface-hub
  huggingface-cli download pietroschiona/trading-alpha-odds data/tennis_odds_data_cleaned.csv

### Option 3: Generate from scratch
Run data_pipeline.py with raw ATP match files in data/atp_matches_TML/ and data/match_odds/.
Note: this generates synthetic odds (not real Pinnacle opening odds).
For true results, you need real Pinnacle opening odds.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TARGET_PATH = os.path.join(DATA_DIR, "tennis_odds_data_cleaned.csv")

# Check common locations
POSSIBLE_PATHS = [
    TARGET_PATH,
    "D:/Betting/codice/modello2torri2/tennis_odds_data_cleaned.csv",
    os.path.expanduser("~/tennis_odds_data_cleaned.csv"),
]

for path in POSSIBLE_PATHS:
    if os.path.exists(path):
        if os.path.abspath(path) == os.path.abspath(TARGET_PATH):
            print(f"✅ Found at expected location: {path}")
            print(f"   Size: {os.path.getsize(path) / 1e6:.0f} MB")
            sys.exit(0)
        else:
            print(f"✅ Found at: {path}")
            print(f"   Size: {os.path.getsize(path) / 1e6:.0f} MB")
            print(f"   Creating symlink to {TARGET_PATH}...")
            try:
                os.symlink(path, TARGET_PATH)
                print("   ✅ Symlink created.")
                sys.exit(0)
            except (OSError, NotImplementedError):
                print("   ⚠️  Symlink failed. Copying file (this may take a while)...")
                import shutil
                shutil.copy2(path, TARGET_PATH)
                print("   ✅ File copied.")
                sys.exit(0)

print("❌ Pinnacle odds CSV not found.")
print()
print(f"Expected at: {TARGET_PATH}")
print("The model will still work without this file, but will fall back to")
print("closing odds (MaxW/MaxL) as morning odds, which introduces data leakage.")
print("Results without Pinnacle odds are NOT reliable.")
print()
print("To get the file:")
print("1. Download from Hugging Face: (link coming soon)")
print("2. Or run data_pipeline.py with ATP match files in data/ (synthetic odds only)")
print("3. Or copy your existing Pinnacle CSV to the path above")
sys.exit(1)
