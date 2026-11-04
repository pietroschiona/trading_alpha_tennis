import pandas as pd
import pytest

from data_pipeline import standardize_name_strict as s


@pytest.mark.parametrize("full,odds,expected", [
    ("Jannik Sinner", "Sinner J.", "sinner j"),
    ("Alex De Minaur", "De Minaur A.", "minaur a"),
    ("Pablo Carreno Busta", "Carreno Busta P.", "busta p"),
    ("Juan Martin Del Potro", "Del Potro J.M.", "potro j"),
    ("Yen-Hsun Lu", "Lu Y.", "lu y"),
    ("Felix Auger-Aliassime", "Auger-Aliassime F.", "aliassime f"),
    ("Botic van de Zandschulp", "Van De Zandschulp B.", "zandschulp b"),
])
def test_name_normalization_matches_both_sources(full, odds, expected):
    assert s(full) == expected
    assert s(odds) == expected
