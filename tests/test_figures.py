import pandas as pd
import pytest

import make_figures


def _bets(n=400):
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    prob = [0.35, 0.55, 0.65, 0.75] * (n // 4)
    won = [False, True, True, False] * (n // 4)
    return pd.DataFrame({
        "date": dates,
        "prob": prob,
        "market_prob": [p + 0.03 for p in prob],
        "stake_head": [10.0] * n,
        "pnl_head": [5.0, -10.0, 3.0, -10.0] * (n // 4),
        "won": won,
    })


def test_main_writes_three_figures_and_reports_brier(tmp_path, capsys):
    bets = tmp_path / "bets.csv"
    _bets().to_csv(bets, index=False)
    out = tmp_path / "figures"

    rc = make_figures.main(["--bets", str(bets), "--out-dir", str(out)])

    assert rc == 0
    for name in ("equity.png", "calibration.png", "brier.png"):
        assert (out / name).stat().st_size > 0
    assert "model Brier" in capsys.readouterr().out


def test_load_bets_rejects_missing_columns(tmp_path):
    bets = tmp_path / "bets.csv"
    pd.DataFrame({"date": ["2020-01-01"], "prob": [0.5]}).to_csv(bets, index=False)
    with pytest.raises(SystemExit):
        make_figures.load_bets(bets)


def test_brier_figure_matches_manual_computation(tmp_path):
    df = _bets()
    model_brier, market_brier = make_figures.brier_figure(df, tmp_path / "b.png")
    won = df["won"].astype(int)
    assert model_brier == pytest.approx(((df["prob"] - won) ** 2).mean())
    assert market_brier == pytest.approx(((df["market_prob"] - won) ** 2).mean())
