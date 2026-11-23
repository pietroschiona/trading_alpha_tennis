import pandas as pd
import pytest

from data_pipeline import standardize_name_strict as s


@pytest.mark.parametrize("full,odds,expected", [
    ("Jannik Sinner", "Sinner J.", "sinner j"),
    ("Alex De Minaur", "De Minaur A.", "minaur a"),
    ("Pablo Carreno Busta", "Carreno Busta P.", "busta p"),
    ("Juan Martin Del Potro", "Del Potro J.M.", "potro j"),
    ("Yen-Hsun Lu", "Lu Y.", "lu y"),
    ("Yen-Hsun Lu", "Lu Y.H.", "lu y"),
    ("Duckhee Lee", "Lee D.H.", "lee d"),
    ("Tung-Lin Wu", "Wu T.L.", "wu t"),
    ("Alex Bogomolov Jr.", "Bogomolov A.", "bogomolov a"),
    ("Alex Bogomolov Jr.", "Bogomolov Jr. A.", "bogomolov a"),
    ("Felix Auger-Aliassime", "Auger-Aliassime F.", "aliassime f"),
    ("Botic van de Zandschulp", "Van De Zandschulp B.", "zandschulp b"),
])
def test_name_normalization_matches_both_sources(full, odds, expected):
    assert s(full) == expected
    assert s(odds) == expected

def test_metadata_sidecar_name(tmp_path):
    from data_pipeline import write_dataset_metadata
    csv = tmp_path / "tennis_features_v36.csv"
    csv.write_text("Date\n2020-01-01\n", encoding="utf-8")
    df = pd.DataFrame({"Date": ["2020-01-01"]})
    meta = write_dataset_metadata(str(csv), df)
    assert (tmp_path / "tennis_features_v36.meta.json").exists()
    assert meta["rows"] == 1
    assert meta["date_min"] == "2020-01-01"
