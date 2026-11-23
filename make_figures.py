"""Build the light-theme figures used by the README.

Input: walkforward_bets.csv (the canonical per-bet log). Nothing is recomputed:
the equity curve is INITIAL_BANKROLL plus the recorded per-bet PnL, the calibration
and Brier figures use the recorded probabilities and outcomes.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

INITIAL_BANKROLL = 1000.0
MODEL_COLOR = "#2563eb"
MARKET_COLOR = "#64748b"
GRID_COLOR = "#e2e8f0"
TEXT_COLOR = "#0f172a"


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=13, fontweight="bold", color=TEXT_COLOR, loc="left")
    ax.set_xlabel(xlabel, fontsize=10, color=TEXT_COLOR)
    ax.set_ylabel(ylabel, fontsize=10, color=TEXT_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#cbd5e1")
    ax.tick_params(colors="#475569", labelsize=9)


def load_bets(path):
    bets = pd.read_csv(path)
    for col in ("date", "prob", "market_prob", "stake_head", "pnl_head", "won"):
        if col not in bets.columns:
            raise SystemExit(f"walkforward_bets.csv has no '{col}' column")
    bets["date"] = pd.to_datetime(bets["date"], errors="raise")
    return bets.sort_values("date").reset_index(drop=True)


def equity_figure(bets, out):
    equity = INITIAL_BANKROLL + bets["pnl_head"].cumsum()
    peak = equity.cummax()
    drawdown = (equity - peak).min()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(bets["date"], equity, color=MODEL_COLOR, linewidth=1.6,
            label="Stake head")
    ax.axhline(INITIAL_BANKROLL, color=MARKET_COLOR, linewidth=1.0,
               linestyle="--", label="Starting bankroll")
    ax.fill_between(bets["date"], equity, INITIAL_BANKROLL,
                    where=equity >= INITIAL_BANKROLL, color=MODEL_COLOR, alpha=0.08)
    ax.fill_between(bets["date"], equity, INITIAL_BANKROLL,
                    where=equity < INITIAL_BANKROLL, color="#dc2626", alpha=0.08)
    _style(ax, "Stake head: bankroll over out-of-sample bets",
           "Match date", "Bankroll (EUR)")
    ax.annotate(
        f"final EUR {equity.iloc[-1]:,.0f}\nmax drawdown EUR {drawdown:,.0f}",
        xy=(bets["date"].iloc[-1], equity.iloc[-1]),
        xytext=(-12, 12), textcoords="offset points",
        ha="right", fontsize=9, color=TEXT_COLOR)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _reliability(prob, won, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(prob, bins) - 1, 0, n_bins - 1)
    xs, ys = [], []
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() >= 20:
            xs.append(prob[mask].mean())
            ys.append(won[mask].mean())
    return np.array(xs), np.array(ys)


def calibration_figure(bets, out):
    prob = bets["prob"].to_numpy(dtype=float)
    won = bets["won"].to_numpy(dtype=float)
    market = bets["market_prob"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], color="#94a3b8", linewidth=1.0, linestyle="--",
            label="Perfect calibration")
    xm, ym = _reliability(market, won)
    ax.plot(xm, ym, color=MARKET_COLOR, linewidth=1.4, marker="o",
            markersize=4, label="Market")
    xp, yp = _reliability(prob, won)
    ax.plot(xp, yp, color=MODEL_COLOR, linewidth=1.6, marker="o",
            markersize=4, label="Model")
    _style(ax, "Reliability by probability bin", "Predicted probability",
           "Observed win rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def brier_figure(bets, out):
    prob = bets["prob"].to_numpy(dtype=float)
    market = bets["market_prob"].to_numpy(dtype=float)
    won = bets["won"].to_numpy(dtype=float)
    keep = np.isfinite(prob) & np.isfinite(market) & np.isfinite(won)
    prob, market, won = prob[keep], market[keep], won[keep]
    if prob.size == 0:
        raise SystemExit("no finite prob/market_prob/won rows for the Brier figure")
    model_brier = float(np.mean((prob - won) ** 2))
    market_brier = float(np.mean((market - won) ** 2))

    fig, ax = plt.subplots(figsize=(5, 4.2))
    bars = ax.bar(["Model", "Market"], [model_brier, market_brier],
                  color=[MODEL_COLOR, MARKET_COLOR], width=0.55)
    for bar, value in zip(bars, (model_brier, market_brier)):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.003,
                f"{value:.4f}", ha="center", fontsize=10, color=TEXT_COLOR)
    _style(ax, f"Brier score on the same {len(bets):,} bets", "",
           "Brier (lower is better)")
    ax.set_ylim(0, max(model_brier, market_brier) * 1.18)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return model_brier, market_brier


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bets", default="walkforward_bets.csv")
    parser.add_argument("--out-dir", default="docs/figures")
    args = parser.parse_args(argv)

    bets = load_bets(args.bets)
    if bets.empty:
        raise SystemExit("walkforward_bets.csv is empty")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    equity_figure(bets, out_dir / "equity.png")
    calibration_figure(bets, out_dir / "calibration.png")
    model_brier, market_brier = brier_figure(bets, out_dir / "brier.png")
    print(f"model Brier {model_brier:.4f} | market Brier {market_brier:.4f} "
          f"| delta {model_brier - market_brier:+.4f}")
    print(f"figures written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
