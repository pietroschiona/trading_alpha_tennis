#!/usr/bin/env python3
"""
Trading Alpha - Scientific Validation Framework
================================================
Validates the walk-forward results against the dataset that produced them.

The primary question is whether the staking strategy MADE MONEY, not whether
the model picked winners more than half the time. Win rate is descriptive: a
bet at odds 4.27 wins 23% of the time by chance alone.

Usage:
    python run_benchmarks.py          # full suite (Test 5 reads the ~1 GB odds CSV)
    python run_benchmarks.py --quick  # skip only Test 5 (~30 sec)

Output:
    - benchmarks/benchmark_results.json   # machine-readable results
    - benchmarks/validation_report.md     # human-readable report

Exit code: 0 only if the integrity checks pass, 1 otherwise (or if provenance
cannot be established, in which case the report is marked VOID and no p-value
is published).
"""

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import leak_checks

# Ensure we're in the right directory
BENCH_DIR = Path(__file__).parent.resolve()
os.chdir(BENCH_DIR)
sys.path.insert(0, str(BENCH_DIR))

DATASET_PATH = BENCH_DIR / "tennis_features_v36.csv"
META_PATH = BENCH_DIR / "tennis_features_v36.meta.json"
RESULTS_PATH = BENCH_DIR / "walkforward_results.txt"
BETS_PATH = BENCH_DIR / "walkforward_bets.csv"
SCRAPER_PATH = BENCH_DIR / "data" / "tennis_odds_data_cleaned.csv"

NOTE_WIN_RATE = "A win rate above 50% is not evidence of profit."


# --------------------------------------------------------------------------
# Results file parsing
# --------------------------------------------------------------------------

# Prefix-based, not `$`-anchored: hydra_model writes extra text on some lines
# (for example "Dataset sha256: abc  (regenerated 16:58)").
_RESULT_PATTERNS = {
    "total_bets": re.compile(r"^Bets:\s*(\d+)", re.MULTILINE),
    "wins": re.compile(r"^Wins:\s*(\d+)\s*/\s*(\d+)", re.MULTILINE),
    "dataset_rows": re.compile(r"^Dataset rows:\s*(\d+)", re.MULTILINE),
    "dataset_sha256": re.compile(r"^Dataset sha256:\s*(\S+)", re.MULTILINE),
    "opening_odds_coverage": re.compile(r"^Opening odds coverage:\s*([\d.]+)", re.MULTILINE),
    "code_commit": re.compile(r"^Code commit:\s*(\S+)", re.MULTILINE),
}


def load_results(path) -> dict:
    """Parse walk-forward results. Exits if a provenance line is missing.

    Without provenance there is no way to tell which dataset the numbers came
    from, and a number of unknown provenance is not evidence.
    """
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"VOID: {path.name} not found - run hydra_model.py first.")
    text = path.read_text(encoding="utf-8", errors="replace")

    missing = [k for k in ("total_bets", "wins", "dataset_rows",
                           "dataset_sha256", "opening_odds_coverage")
               if k not in _RESULT_PATTERNS or not _RESULT_PATTERNS[k].search(text)]
    if missing:
        raise SystemExit(
            f"VOID: {path.name} is missing provenance lines {sorted(missing)}. "
            "Regenerate it with a runner that writes Bets/Wins/Dataset rows/"
            "Dataset sha256/Opening odds coverage."
        )

    wins_match = _RESULT_PATTERNS["wins"].search(text)
    total_bets = int(_RESULT_PATTERNS["total_bets"].search(text).group(1))
    wins = int(wins_match.group(1))
    results = {
        "total_bets": total_bets,
        "wins": wins,
        "win_rate": wins / total_bets if total_bets else 0.0,
        "dataset_rows": int(_RESULT_PATTERNS["dataset_rows"].search(text).group(1)),
        "dataset_sha256": _RESULT_PATTERNS["dataset_sha256"].search(text).group(1),
        "opening_odds_coverage": float(
            _RESULT_PATTERNS["opening_odds_coverage"].search(text).group(1)),
    }
    commit = _RESULT_PATTERNS["code_commit"].search(text)
    results["code_commit"] = commit.group(1) if commit else "unknown"
    return results


# --------------------------------------------------------------------------
# Return test (the primary evidence)
# --------------------------------------------------------------------------

def bootstrap_roi(bets, n_boot: int = 10000, seed: int = 42) -> dict:
    """ROI = sum(pnl) / sum(staked), with a 95% bootstrap CI and a one-sided p-value.

    Returns {"available": False, ...} when the bets file is missing, empty, or
    lacks the stake/pnl columns, so the caller can report NOT MEASURED instead
    of crashing.
    """
    unavailable = {
        "available": False,
        "roi": None, "ci_95": [None, None], "p_value": None,
        "n_bets": 0, "total_staked": 0.0, "total_pnl": 0.0, "max_drawdown": None,
    }
    if bets is None or len(bets) == 0 or not {"stake_head", "pnl_head"} <= set(bets.columns):
        return unavailable

    pnl = pd.to_numeric(bets["pnl_head"], errors="coerce").to_numpy(dtype=float)
    stake = pd.to_numeric(bets["stake_head"], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(pnl) & np.isfinite(stake) & (stake > 0)
    pnl, stake = pnl[keep], stake[keep]
    if len(pnl) == 0 or stake.sum() <= 0:
        return unavailable

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pnl), size=(n_boot, len(pnl)))
    boot = pnl[idx].sum(axis=1) / stake[idx].sum(axis=1)
    cum = np.cumsum(pnl)
    return {
        "available": True,
        "roi": float(pnl.sum() / stake.sum()),
        "ci_95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "p_value": float((boot <= 0).mean()),
        "n_bets": int(len(pnl)),
        "total_staked": float(stake.sum()),
        "total_pnl": float(pnl.sum()),
        "max_drawdown": float((np.maximum.accumulate(cum) - cum).max()),
    }


def load_bets(path):
    """Load walkforward_bets.csv, or None if it is missing/empty."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        bets = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None
    return bets


def sha256_file(path, chunk_size: int = 1 << 20) -> str:
    """Stream the dataset hash: the guard must hash the file it is about to measure."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def benchmark_commit() -> str:
    """Commit of the code running the benchmarks (distinct from the model run commit)."""
    try:
        out = subprocess.run(
            ["git", "-C", str(BENCH_DIR), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=False,
        )
    except OSError:
        return "unknown"
    return out.stdout.strip() or "unknown"


# --------------------------------------------------------------------------
# Descriptive win-rate tests
# --------------------------------------------------------------------------

def test_1_binomial_significance(n_bets: int, wins: int) -> dict:
    """Descriptive Test 1: binomial test for H0: win_rate = 0.5"""
    win_rate = wins / n_bets
    binom_result = stats.binomtest(wins, n_bets, 0.5, alternative="greater")
    z_score = (win_rate - 0.5) / math.sqrt(0.5 * 0.5 / n_bets)
    ci_low, ci_high = stats.beta.interval(0.95, wins + 0.5, n_bets - wins + 0.5)

    return {
        "n_bets": n_bets,
        "wins": wins,
        "win_rate": round(win_rate, 4),
        "z_score": round(z_score, 2),
        "p_value": binom_result.pvalue,
        "ci_95": [round(ci_low, 4), round(ci_high, 4)],
    }


def test_2_monte_carlo(n_bets: int, wins: int, n_sim: int = 200000) -> dict:
    """Descriptive Test 2: Monte Carlo probability of a random model reaching this win rate"""
    win_rate = wins / n_bets
    rng = np.random.default_rng(42)
    random_wins = rng.binomial(n_bets, 0.5, n_sim)
    random_rates = random_wins / n_bets
    pct_better = (random_rates >= win_rate).mean()

    return {
        "n_simulations": n_sim,
        "observed_win_rate": round(win_rate, 4),
        "pct_random_models_achieving_this": pct_better,
        "p_value_approximation": max(pct_better, 1 / n_sim),
    }


def test_3_bayes_factor(n_bets: int, wins: int) -> dict:
    """Descriptive Test 3: Bayes Factor (BIC approximation) for H0 = 50% win rate"""
    win_rate = wins / n_bets
    if wins == 0 or wins == n_bets:
        return {"BF10": 0.0, "evidence": "undefined"}
    log_lik_H0 = wins * math.log(0.5) + (n_bets - wins) * math.log(0.5)
    log_lik_H1 = wins * math.log(win_rate) + (n_bets - wins) * math.log(1 - win_rate)
    BIC_H0 = -2 * log_lik_H0 + 1 * math.log(n_bets)
    BIC_H1 = -2 * log_lik_H1 + 1 * math.log(n_bets)
    BF10 = math.exp((BIC_H0 - BIC_H1) / 2)

    if BF10 > 100:
        evidence = "decisive"
    elif BF10 > 30:
        evidence = "very_strong"
    elif BF10 > 10:
        evidence = "strong"
    elif BF10 > 3:
        evidence = "moderate"
    else:
        evidence = "weak"

    return {"BF10": BF10, "evidence": evidence}


def test_4_effect_size(n_bets: int, wins: int) -> dict:
    """Descriptive Test 4: Cohen's h effect size on win rate"""
    win_rate = wins / n_bets
    h = 2 * math.asin(math.sqrt(win_rate)) - 2 * math.asin(math.sqrt(0.5))

    if abs(h) >= 0.5:
        magnitude = "large"
    elif abs(h) >= 0.2:
        magnitude = "medium"
    else:
        magnitude = "small"

    return {"cohens_h": round(abs(h), 4), "magnitude": magnitude}


def run_statistical_tests(results: dict) -> dict:
    """Run descriptive Tests 1-4 on the recorded bets and wins."""
    n_bets = results["total_bets"]
    wins = results["wins"]
    if n_bets <= 0:
        return None

    return {
        "test_1_binomial": test_1_binomial_significance(n_bets, wins),
        "test_2_monte_carlo": test_2_monte_carlo(n_bets, wins),
        "test_3_bayes_factor": test_3_bayes_factor(n_bets, wins),
        "test_4_effect_size": test_4_effect_size(n_bets, wins),
    }


# --------------------------------------------------------------------------
# Test 5: opening vs closing market drift (descriptive)
# --------------------------------------------------------------------------

def run_market_drift(csv_path, scraper_path) -> dict:
    """Test 5 (descriptive): how far closing implied probability sits from opening.

    Nothing here validates the model. It measures whether the dataset even has
    opening odds to reason about: with 0% coverage, the morning/evening split of
    the architecture is not fed by data.
    """
    import hydra_model as m

    usecols = ["Date", "MaxW", "MaxL", "winner_name", "loser_name"]
    raw_df = pd.read_csv(csv_path, low_memory=False, usecols=usecols)
    raw_df["Date"] = pd.to_datetime(raw_df["Date"], errors="coerce")
    raw_df = raw_df.dropna(subset=["Date", "winner_name", "loser_name"])
    if raw_df.empty:
        return {"total_betting_matches": 0, "with_opening_odds": 0,
                "coverage_pct": 0.0, "drift_mean": None, "drift_std": None}

    loader = m.HybridDataLoader()
    original_path = m.SCRAPER_CSV
    m.SCRAPER_CSV = str(scraper_path)
    try:
        raw_df = loader.enrich_with_opening_odds(raw_df)
    finally:
        m.SCRAPER_CSV = original_path

    betting = raw_df[raw_df["Date"].dt.year >= 2017].copy()
    out = {"total_betting_matches": int(len(betting)), "with_opening_odds": 0,
           "coverage_pct": 0.0, "drift_mean": None, "drift_std": None}
    if betting.empty or "has_pin_open" not in betting.columns:
        return out

    with_pin = betting[betting["has_pin_open"] == 1].copy()
    out["with_opening_odds"] = int(len(with_pin))
    out["coverage_pct"] = round(len(with_pin) / len(betting) * 100, 1)
    if with_pin.empty:
        return out

    close = (1 / with_pin["MaxW"]) / ((1 / with_pin["MaxW"]) + (1 / with_pin["MaxL"]))
    open_w = with_pin["Pin_Open_W"].fillna(with_pin["MaxW"])
    open_l = with_pin["Pin_Open_L"].fillna(with_pin["MaxL"])
    opening = (1 / open_w) / ((1 / open_w) + (1 / open_l))
    drift = (close - opening).dropna()
    if drift.empty:
        return out
    out["drift_mean"] = round(float(drift.mean()), 4)
    out["drift_std"] = round(float(drift.std()), 4)
    return out


# --------------------------------------------------------------------------
# Tests 6 and 7: integrity checks
# --------------------------------------------------------------------------

def run_duplicate_check(csv_path) -> dict:
    """Test 6: duplicate (Date, winner, loser) rows must stay at 0.

    The shipped v36 dataset contained 1,579 duplicates caused by a many-to-many
    year+player merge: distinct matches collapsed onto one Date, which leaked
    the outcome into h2h_ratio.
    """
    df = pd.read_csv(
        csv_path,
        low_memory=False,
        usecols=lambda c: c in ("Date", "winner_name", "loser_name", "surface",
                                "Winner", "Loser", "MaxW", "MaxL"),
    )
    if "winner_name" not in df.columns and "Winner" in df.columns:
        df = df.rename(columns={"Winner": "winner_name", "Loser": "loser_name"})
    return leak_checks.duplicate_groups(df)


def run_h2h_check(csv_path) -> dict:
    """Test 7: corr(h2h_ratio, outcome) must stay below the leak threshold."""
    df = pd.read_csv(
        csv_path,
        low_memory=False,
        usecols=lambda c: c in ("Date", "winner_name", "loser_name", "surface"),
    )
    corr = leak_checks.h2h_correlation(df)
    return {
        "corr": None if pd.isna(corr) else round(corr, 4),
        "threshold": leak_checks.H2H_CORR_MAX,
        "passed": bool(not pd.isna(corr) and abs(corr) < leak_checks.H2H_CORR_MAX),
    }


def run_market_brier(csv_path) -> float:
    """Market Brier index on 2018+ as the reference a model must not beat via leakage."""
    df = pd.read_csv(csv_path, low_memory=False,
                     usecols=lambda c: c in ("Date", "MaxW", "MaxL"))
    return leak_checks.market_brier(df)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def _integrity_passed(tests: dict) -> bool:
    prov = tests.get("provenance") or {}
    dup = tests.get("duplicate_check")
    h2h = tests.get("h2h")
    return bool(
        prov.get("passed")
        and dup is not None and dup["passed"]
        and h2h is not None and h2h["passed"]
    )


def _edge_verdict(ret) -> str:
    """YES needs a positive ROI and a 95% CI lower bound above zero.

    The one-sided p-value certifies the 90% interval, so it is not enough on its
    own: at p = 0.04 the 2.5th percentile can still be negative.
    """
    if not ret or not ret.get("available"):
        return "NOT MEASURED"
    lo = (ret.get("ci_95") or [None, None])[0]
    if lo is not None and lo > 0 and ret["roi"] > 0:
        return "YES"
    return "NO"


def format_report(tests: dict, results: dict) -> str:
    """Format the benchmark report. The return test leads; win rate is context."""
    prov = tests.get("provenance") or {}
    ret = tests.get("returns")
    desc = tests.get("descriptive")
    drift = tests.get("market_drift")
    dup = tests.get("duplicate_check")
    h2h = tests.get("h2h")
    brier = tests.get("market_brier")

    lines = [
        "# Trading Alpha - Validation Report",
        "",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Integrity checks:** {'PASS' if _integrity_passed(tests) else 'FAIL'}",
        f"**Edge demonstrated:** {_edge_verdict(ret)}",
        "",
        "## Provenance",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Model run commit (produced these bets) | `{results.get('code_commit', 'unknown')}` |",
        f"| Benchmark commit | `{prov.get('benchmark_commit', 'unknown')}` |",
        f"| Dataset rows | {results['dataset_rows']} |",
        f"| Dataset sha256 | `{results['dataset_sha256']}` |",
        f"| Meta rows | {prov.get('meta_rows')} |",
        f"| Meta sha256 | `{prov.get('meta_sha256')}` |",
        f"| Provenance match | {'yes' if prov.get('passed') else 'NO'} |",
        f"| Opening odds coverage | {results['opening_odds_coverage']}% |",
        f"| Bets | {results['total_bets']} |",
        f"| Wins | {results['wins']}/{results['total_bets']} ({results['win_rate']*100:.1f}%) |",
    ]
    problems = list(prov.get("reasons") or []) + list(prov.get("warnings") or [])
    if problems:
        lines.append(f"| Provenance problems | {'; '.join(problems)} |")

    lines.extend(["", "## Return test (primary evidence)", ""])
    if ret and ret.get("available"):
        lines.extend([
            "| Metric | Value |",
            "|---|---|",
            f"| Bets | {ret['n_bets']} |",
            f"| Total staked | {ret['total_staked']:.2f} |",
            f"| Total PnL | {ret['total_pnl']:.2f} |",
            f"| ROI | {ret['roi']*100:+.2f}% |",
            f"| 95% bootstrap CI | [{ret['ci_95'][0]*100:+.2f}%, {ret['ci_95'][1]*100:+.2f}%] |",
            f"| p-value (one-sided, ROI <= 0) | {ret['p_value']:.4f} |",
            f"| Max drawdown | {ret['max_drawdown']:.2f} |",
            "",
            "Stakes and PnL come from `walkforward_bets.csv` (bankroll = 1000, "
            "stake_head strategy). The confidence interval is the only claim "
            "that matters: if it contains zero, the strategy has not been shown "
            "to make money.",
        ])
    else:
        lines.extend([
            "`walkforward_bets.csv` is missing or has no usable stake/pnl rows.",
            "",
            "**Edge demonstrated: NOT MEASURED** - no per-bet returns to test.",
        ])

    if brier is not None and not (isinstance(brier, float) and math.isnan(brier)):
        lines.extend(["", f"Market Brier reference (2018+, MaxW/MaxL): **{brier:.4f}** "
                          "This is the bar a model must beat without leakage."])

    lines.extend(["", "## Descriptive: win-rate statistics", "",
                  f"**{NOTE_WIN_RATE}**"])
    if desc:
        t1 = desc["test_1_binomial"]
        t2 = desc["test_2_monte_carlo"]
        t3 = desc["test_3_bayes_factor"]
        t4 = desc["test_4_effect_size"]
        lines.extend([
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Win rate | {t1['win_rate']*100:.2f}% (95% CI "
            f"[{t1['ci_95'][0]*100:.2f}%, {t1['ci_95'][1]*100:.2f}%]) |",
            f"| Binomial p (H0: 50%) | {t1['p_value']:.2e} |",
            f"| Z-score | {t1['z_score']} |",
            f"| Monte Carlo p | {t2['p_value_approximation']:.2e} "
            f"({t2['n_simulations']:,} sims) |",
            f"| Bayes factor BF10 (50% vs observed rate) | "
            f"{t3['BF10']:.1e} ({t3['evidence']}) |",
            f"| Cohen's h | {t4['cohens_h']} ({t4['magnitude']}) |",
            "",
            "These describe how often the picker was right. They say nothing "
            "about whether the picks were priced correctly, which is where the "
            "money is made or lost.",
        ])
    else:
        lines.extend(["", "No bets recorded, so no win-rate statistics were computed."])

    lines.extend(["", "## Test 5: Opening vs closing market drift (descriptive)", ""])
    if drift is None:
        lines.append("Skipped (`--quick`, or the odds CSV is unavailable).")
    else:
        lines.extend([
            f"Betting matches (2017+): **{drift['total_betting_matches']}**",
            f"With opening odds: **{drift['with_opening_odds']}** "
            f"({drift['coverage_pct']}% coverage)",
        ])
        if drift["drift_mean"] is None:
            lines.append("No matched opening odds: drift not measurable.")
        else:
            lines.append(f"Mean (closing - opening) implied probability: "
                         f"{drift['drift_mean']:+.4f} (sd {drift['drift_std']:.4f})")

    lines.extend(["", "## Test 6: Duplicate-match integrity check", ""])
    if dup is None:
        lines.extend(["Dataset missing: not checked.",
                      "", "**Integrity: FAIL** (cannot verify without the dataset)."])
    else:
        lines.extend([
            f"Rows checked: **{dup['total_rows']}**",
            f"Duplicate (Date, winner, loser) rows: **{dup['duplicate_rows']}** "
            f"in {dup['duplicate_groups']} groups",
            "",
            "**Integrity:** " + ("PASS - no duplicates, the year+player merge leak is not back."
                                 if dup["passed"] else
                                 "FAIL - duplicates present, the many-to-many merge has returned."),
        ])

    lines.extend(["", "## Test 7: h2h leakage check", ""])
    if h2h is None or h2h["corr"] is None:
        lines.extend(["h2h correlation not computable (dataset missing or empty).",
                      "", "**Integrity: FAIL**"])
    else:
        lines.extend([
            f"corr(h2h_ratio, outcome) = **{h2h['corr']:+.3f}** "
            f"(threshold < {h2h['threshold']}; leaked dataset was +0.336)",
            "",
            "**Integrity:** " + ("PASS - no outcome information in the h2h feature."
                                 if h2h["passed"] else
                                 "FAIL - h2h carries the current match outcome."),
        ])

    lines.extend([
        "",
        "---",
        "## Summary",
        "",
        f"**Integrity checks: {'PASS' if _integrity_passed(tests) else 'FAIL'}**",
        f"**Edge demonstrated: {_edge_verdict(ret)}**",
        "",
        f"- Return test: " + (
            f"ROI {ret['roi']*100:+.2f}%, 95% CI "
            f"[{ret['ci_95'][0]*100:+.2f}%, {ret['ci_95'][1]*100:+.2f}%], "
            f"p = {ret['p_value']:.4f}" if ret and ret.get("available")
            else "not measured"),
        f"- Duplicates: {dup['duplicate_rows'] if dup else 'not checked'}",
        f"- h2h corr: {h2h['corr'] if h2h else 'not checked'}",
        f"- Provenance: {'verified' if prov.get('passed') else 'NOT verified'}",
    ])
    return "\n".join(lines)


def write_void_report(reasons: list) -> str:
    """Report used when provenance cannot be established: no p-values are written."""
    report = "\n".join([
        "# Trading Alpha - Validation Report",
        "",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## VOID",
        "",
        "Provenance could not be established, so no statistical test was run "
        "and no p-value is published.",
        "",
    ] + [f"- {r}" for r in reasons] + [
        "",
        "**Integrity checks: FAIL**",
        "**Edge demonstrated: NOT MEASURED**",
    ])
    os.makedirs(BENCH_DIR / "benchmarks", exist_ok=True)
    (BENCH_DIR / "benchmarks" / "validation_report.md").write_text(report, encoding="utf-8")
    (BENCH_DIR / "benchmarks" / "benchmark_results.json").write_text(
        json.dumps({"void": True, "reasons": reasons}, indent=2), encoding="utf-8")
    return report


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Trading Alpha - Scientific Validation Framework"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip Test 5 (opening vs closing drift, reads the ~1 GB odds CSV)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("TRADING ALPHA - SCIENTIFIC VALIDATION FRAMEWORK")
    print("=" * 60)

    # ---------------------------------------------------------------- provenance
    try:
        results = load_results(RESULTS_PATH)
    except SystemExit as exc:
        print(f"\n{exc}")
        write_void_report([str(exc)])
        print("Report saved to: benchmarks/validation_report.md (VOID)")
        return 1

    meta = {}
    if META_PATH.exists():
        try:
            meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}

    reasons = []
    warnings = []
    if not meta:
        reasons.append(f"{META_PATH.name} missing or unreadable")
    else:
        if str(meta.get("sha256")) != results["dataset_sha256"]:
            reasons.append(
                f"dataset sha256 mismatch: results {results['dataset_sha256']} "
                f"vs meta {meta.get('sha256')}")
        if meta.get("rows") != results["dataset_rows"]:
            reasons.append(
                f"dataset row count mismatch: results {results['dataset_rows']} "
                f"vs meta {meta.get('rows')}")

    if not DATASET_PATH.exists():
        reasons.append(f"{DATASET_PATH.name} missing")
    else:
        actual_sha = sha256_file(DATASET_PATH)
        if str(meta.get("sha256")) != actual_sha:
            reasons.append(
                f"dataset file sha256 mismatch: meta {meta.get('sha256')} "
                f"vs file {actual_sha}")

    if str(results.get("code_commit", "")).endswith("+dirty"):
        warnings.append(
            "results produced by uncommitted code - not reproducible from a commit")

    provenance = {
        "passed": not reasons,
        "reasons": reasons,
        "warnings": warnings,
        "meta_rows": meta.get("rows"),
        "meta_sha256": meta.get("sha256"),
        "benchmark_commit": benchmark_commit(),
    }

    if not provenance["passed"]:
        print("\nVOID - provenance not established:")
        for r in reasons:
            print(f"  - {r}")
        write_void_report(reasons)
        print("Report saved to: benchmarks/validation_report.md (VOID)")
        return 1

    print(f"\nProvenance verified: {results['dataset_rows']} rows, "
          f"sha256 {results['dataset_sha256'][:12]}..., commit {results['code_commit']}")
    for w in warnings:
        print(f"  warning: {w}")
    print(f"Recorded: {results['total_bets']} bets, {results['win_rate']*100:.1f}% WR, "
          f"opening odds coverage {results['opening_odds_coverage']}%\n")

    tests = {"provenance": provenance}

    # ---------------------------------------------------------------- returns
    print("Running return test (bootstrap ROI on walkforward_bets.csv)...")
    bets = load_bets(BETS_PATH)
    ret = bootstrap_roi(bets)
    tests["returns"] = ret
    if ret["available"]:
        print(f"  ROI {ret['roi']*100:+.2f}%  CI95 [{ret['ci_95'][0]*100:+.2f}%, "
              f"{ret['ci_95'][1]*100:+.2f}%]  p = {ret['p_value']:.4f}  "
              f"({ret['n_bets']} bets)")
    else:
        print(f"  NOT MEASURED - {BETS_PATH.name} missing or unusable")

    # -------------------------------------------------------------- integrity
    print("Running Tests 6-7 (duplicate and h2h integrity checks)...")
    dup = run_duplicate_check(DATASET_PATH)
    tests["duplicate_check"] = dup
    print(f"  duplicates: {dup['duplicate_rows']} in {dup['duplicate_groups']} groups "
          f"({dup['total_rows']} rows)")
    h2h = run_h2h_check(DATASET_PATH)
    tests["h2h"] = h2h
    print(f"  corr(h2h, outcome): {h2h['corr']} (threshold {h2h['threshold']})")
    try:
        tests["market_brier"] = run_market_brier(DATASET_PATH)
        print(f"  market Brier (2018+): {tests['market_brier']:.4f}")
    except Exception as exc:  # market_brier is a reference, not a gate
        tests["market_brier"] = None
        print(f"  market Brier unavailable: {exc}")

    # ------------------------------------------------------ descriptive stats
    print("Running descriptive win-rate tests...")
    tests["descriptive"] = run_statistical_tests(results)
    if tests["descriptive"]:
        t1 = tests["descriptive"]["test_1_binomial"]
        print(f"  win rate {t1['win_rate']*100:.2f}% (binomial p = {t1['p_value']:.2e})")
    print(f"  note: {NOTE_WIN_RATE}")

    # ----------------------------------------------------------- Test 5
    if args.quick:
        tests["market_drift"] = None
        print("Skipping Test 5 (--quick mode)")
    elif SCRAPER_PATH.exists():
        print("Running Test 5 (opening vs closing drift, descriptive)...")
        try:
            tests["market_drift"] = run_market_drift(DATASET_PATH, SCRAPER_PATH)
        except Exception as exc:  # Test 5 is descriptive and must never kill the report
            tests["market_drift"] = None
            print(f"  Skipping Test 5 - {type(exc).__name__}: {exc}")
        else:
            d = tests["market_drift"]
            print(f"  opening odds coverage: {d['coverage_pct']}% "
                  f"({d['with_opening_odds']}/{d['total_betting_matches']})")
    else:
        tests["market_drift"] = None
        print(f"Skipping Test 5 - {SCRAPER_PATH.name} not available")

    # -------------------------------------------------------------- report
    os.makedirs(BENCH_DIR / "benchmarks", exist_ok=True)
    report = format_report(tests, results)
    (BENCH_DIR / "benchmarks" / "validation_report.md").write_text(report, encoding="utf-8")
    with open(BENCH_DIR / "benchmarks" / "benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(tests, f, indent=2, default=str)

    print("\nReport saved to: benchmarks/validation_report.md")
    print("Data saved to:   benchmarks/benchmark_results.json")

    integrity_ok = _integrity_passed(tests)
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Integrity checks: {'PASS' if integrity_ok else 'FAIL'}")
    print(f"Edge demonstrated: {_edge_verdict(ret)}")
    if ret["available"]:
        print(f"  ROI {ret['roi']*100:+.2f}% (CI95 [{ret['ci_95'][0]*100:+.2f}%, "
              f"{ret['ci_95'][1]*100:+.2f}%]), p = {ret['p_value']:.4f}")
    print()
    return 0 if integrity_ok else 1


if __name__ == "__main__":
    sys.exit(main())
