from tqdm import tqdm
import re
import sys
import random
import xgboost as xgb
from collections import defaultdict, deque
from datetime import datetime, timezone
import json
import subprocess
from pathlib import Path
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import tensorflow as tf
keras = tf.keras  # pylint: disable=no-member
layers = keras.layers
K = keras.backend

# ==============================================================================
# 0. CONFIGURAZIONE CENTRALIZZATA - MODIFICABILE
# ==============================================================================

# ------------------------
# DATASET & PERCORSI
# ------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FILE = os.path.join(BASE_DIR, "tennis_features_v36.csv")
# Pinnacle Opening/Closing scraped CSV (~1 GB, gitignored; absent in clean checkouts)
SCRAPER_CSV = os.path.join(BASE_DIR, "data", "tennis_odds_data_cleaned.csv")
TML_PATH = os.path.join(BASE_DIR, "data", "atp_matches_TML")
TENNIS_DATA_PATH = os.path.join(BASE_DIR, "data", "match_odds")
GRAPH_DIR = os.path.join(BASE_DIR, "charts")
MODELS_DIR = os.path.join(BASE_DIR, "model_state")
META_FILE = os.path.join(BASE_DIR, "tennis_features_v36.meta.json")
RESULTS_FILE = os.path.join(BASE_DIR, "walkforward_results.txt")
BETS_FILE = os.path.join(BASE_DIR, "walkforward_bets.csv")
MANIFEST_FILE = os.path.join(BASE_DIR, "walkforward_manifest.json")
RNG_SEED = 42

# ------------------------
# ANNI & TRAINING (2013+ for Pinnacle coverage)
# ------------------------
START_YEAR = 2013  # Pinnacle coverage >90% from 2013+; pre-2013 data has minimal opening odds
BETTING_START_YEAR = 2017  # 4 years warmup for ELO/EWMA, 9 years out-of-sample betting

# ------------------------
# HYDRA-ORACLE: HARD CAP (Legal Constraint)
# ------------------------
MAX_STAKE_LEGAL_CAP = 5000.0  # €5000 Betclic/Eurobet hard limit

# Liquidity multipliers by tournament level
LIQUIDITY_MULT = {
    'G': 1.0,       # Grand Slam
    'M': 1.0,       # Masters 1000
    'A': 0.7,       # ATP 500
    '250': 0.4,     # ATP 250
    'C': 0.1,       # Challenger
    'F': 0.05,      # Futures/ITF
    'D': 0.05,      # Davis Cup
}
NUM_LOOPS = 1              # Numero di run per media risultati (1 for speed, 3+ for stability)
TEST_MODE = False

# ------------------------
# STRATEGIA BETTING (BASELINE)
# ------------------------
MIN_EDGE = 0.01            # Edge minimo per scommettere
CONFIDENCE_THR = 0.51      # Soglia confidenza minima
KELLY_FRACTION = 0.015     # Frazione conservativa Kelly (1.5%)
PERC_STAKE = 0.01          # Stake percentuale (1% del bankroll)
FLAT_STAKE = 10.0          # Stake fisso in €

# ------------------------
# IPERPARAMETRI STAKE HEAD (supervised PnL-shaped head, NOT policy-gradient RL)
# ------------------------
RL_STAKE_MIN = 0.00        # Stake minimo (frazione bankroll)
RL_STAKE_MAX = 0.05        # Stake massimo (frazione bankroll, 5%)
RL_LOSS_WEIGHT_VALUE = 1.0  # Peso loss Value Head
RL_LOSS_WEIGHT_POLICY = 0.3  # Peso loss Stake Head (supervised, PnL-shaped)
RL_SHARPE_PENALTY = 0.08   # Penalità per varianza (trade-off risk/return)

# ------------------------
# IPERPARAMETRI DEEP NETWORK
# ------------------------
BUFFER_SIZE = 8192
BATCH_SIZE = 256
LR_VALUE = 0.0001          # Learning rate Value Head
LR_POLICY = 0.0002         # Learning rate Policy Head
DROPOUT_RATE = 0.3
TOWER_DIM_1 = 64           # Dimensione primo layer Siamese Tower
TOWER_DIM_2 = 32           # Dimensione secondo layer
DENSE_DIM = 32             # Dimensione layer finale

# ------------------------
# IPERPARAMETRI XGBOOST
# ------------------------
XGB_WINDOW = 20000         # Finestra memoria (~7 anni)
XGB_MAX_DEPTH = 3
XGB_ETA = 0.03
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE = 0.8
XGB_VETO_THRESHOLD = -0.02  # Se il residuo di mercato < soglia, veto su stake

# ------------------------
# FEATURE ENGINEERING
# ------------------------
EWMA_COM = 20              # Center of mass per EWMA
ELO_K_GRAND_SLAM = 35      # K-factor ELO per Slam
ELO_K_NORMAL = 25          # K-factor ELO per tornei normali

# ------------------------
# ONLINE LEARNING
# ------------------------
DEBIASER_WINDOW = 4000     # Finestra Odds Debiaser
RETRAIN_DEEP_DAYS = 7      # Ritraining Deep Model ogni N giorni
RETRAIN_XGB_DAYS = 14      # Ritraining XGBoost ogni N giorni
RETRAIN_DEBIASER_DAYS = 30  # Ritraining Debiaser ogni N giorni

# ------------------------
# PORTFOLIO MANAGEMENT
# ------------------------
INITIAL_BANKROLL = 1000.0
MIN_STAKE_ABSOLUTE = 1.0   # Stake minimo assoluto in €
# Stake massimo realistico per simulare limiti di liquidità
MAX_REALISTIC_STAKE = 5000.0

# ------------------------
# MARKET REALISM (V35)
# ------------------------
EXECUTION_SLIPPAGE = 0.02  # 2% slippage

# Italian Players for Home Bias (Reference)
ITALIAN_PLAYERS = [
    'sinner j', 'berrettini m', 'musetti l', 'sonego l',
    'arnaldi m', 'cobolli f', 'fognini f', 'nardi l', 'passaro f', 'zeppieri g'
]

# ==============================================================================
# FINE CONFIGURAZIONE - NON MODIFICARE SOTTO QUESTA LINEA
# ==============================================================================

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ==============================================================================
# 1. UTILS: LOGGING
# ==============================================================================


class Log:
    H = '\033[95m'
    B = '\033[94m'
    G = '\033[92m'
    Y = '\033[93m'
    R = '\033[91m'
    E = '\033[0m'

    @staticmethod
    def banner(msg): print(
        f"\n{Log.H}╔{'═'*(len(msg)+4)}╗\n║ {Log.B}{msg}{Log.H} ║\n╚{'═'*(len(msg)+4)}╝{Log.E}")

    @staticmethod
    def info(step, msg): print(f"{Log.B}[{step}]{Log.E} {msg}")
    @staticmethod
    def ok(msg): print(f"{Log.G}[OK]{Log.E} {msg}")
    @staticmethod
    def err(msg): print(f"{Log.R}[ERROR]{Log.E} {msg}")
    @staticmethod
    def warn(msg): print(f"{Log.Y}[WARN]{Log.E} {msg}")


# ==============================================================================
# 1.1 HYBRID DATA LOADER (TML + tennis-data + scraped Pinnacle)
# ==============================================================================

class HybridDataLoader:
    """
    3-Layer Data Merge for Hydra-Oracle:
    Layer 1 (BASE): TML matches → ALL matches for ELO calculation
    Layer 2 (ODDS): tennis-data.co.uk → PSW/PSL as Pinnacle Closing
    Layer 3 (ENRICH): scraped CSV → Pin_Open where available

    Opening odds are the morning input; closing odds (MaxW/MaxL) are only the
    evening calibration target.  If opening odds are unavailable the caller must
    opt into the degraded closing-fallback mode explicitly (see main()).
    """

    BOOKMAKER_PRIORITY = [
        'Pinnacle',      # Most efficient, primary
        'bet365',        # Major, reliable
        'Betfair',       # Exchange prices
        'Unibet',        # Good coverage
        '10Bet',         # Good coverage
        'Betway',        # Backup
        'BetVictor',     # Backup
        'Betsafe',       # Backup
        'Betsson',       # Backup
    ]

    def __init__(self):
        self.scraper_df = None
        self.scraper_loaded = False

    def load_scraped_pinnacle_index(self):
        """Load scraped CSV with ALL bookmakers for cascade lookup."""
        if os.environ.get("TRADING_ALPHA_DISABLE_OPENING_ODDS") == "1":
            Log.warn(
                "TRADING_ALPHA_DISABLE_OPENING_ODDS=1 - skipping opening odds load")
            return None

        if not os.path.exists(SCRAPER_CSV):
            Log.warn(f"Scraped CSV not found: {SCRAPER_CSV}")
            return None

        Log.info(
            "LOADER", f"Loading ALL bookmaker opening odds from {SCRAPER_CSV}...")

        chunks = []
        try:
            # Only the priority bookmakers matter.  Filtering while reading keeps
            # the 5.8M-row CSV from becoming a multi-GB DataFrame.
            pattern = '|'.join(re.escape(b) for b in self.BOOKMAKER_PRIORITY)
            for chunk in pd.read_csv(SCRAPER_CSV,
                                     usecols=['date_str', 'player_1', 'player_2',
                                              'bookmaker', 'selection', 'opening_odd'],
                                     chunksize=500000,
                                     low_memory=False):
                chunk = chunk[chunk['opening_odd'].notna()]
                chunk = chunk[chunk['selection'].isin(
                    ['Selection_1', 'Selection_2'])]
                keep = chunk['bookmaker'].astype(str).str.contains(
                    pattern, case=False, regex=True, na=False)
                chunk_valid = chunk[keep]
                if not chunk_valid.empty:
                    chunks.append(chunk_valid)

            if chunks:
                self.scraper_df = pd.concat(chunks, ignore_index=True)
                self.scraper_loaded = True
                Log.ok(
                    f"Loaded {len(self.scraper_df)} opening odds rows from priority bookmakers")

                # Show bookmaker breakdown
                bm_counts = self.scraper_df.groupby(
                    'bookmaker').size().sort_values(ascending=False)
                Log.info(
                    "LOADER", f"Top bookmakers: {dict(bm_counts.head(5))}")

                # Pivot to get one row per match with odds for both players
                self._prepare_odds_index()
                return self.scraper_df
        except Exception as e:
            Log.err(f"Error loading scraped CSV: {e}")
        return None

    def _normalize_name(self, name):
        """
        Robust name normalization for fuzzy matching.
        Handles 'First Last' vs 'Last First', year suffixes, hyphens, middle names.
        """
        if pd.isna(name):
            return ''
        import re
        clean = re.sub(r'\s*\(\d{4}\)\s*$', '', str(name))
        clean = clean.replace('-', ' ')
        parts = clean.lower().strip().split()
        if len(parts) == 0:
            return ''
        if len(parts) == 1:
            return parts[0]
        key_parts = sorted(parts)
        return '_'.join(key_parts)

    def _prepare_odds_index(self):
        """Create per-bookmaker odds index with priority ranking."""
        if self.scraper_df is None:
            return

        # Parse date
        self.scraper_df['parsed_date'] = pd.to_datetime(
            self.scraper_df['date_str'], format='%d.%m.%Y', errors='coerce')
        self.scraper_df = self.scraper_df.dropna(
            subset=['parsed_date', 'player_1', 'player_2']).copy()

        # Normalise each unique name once, then map (the CSV has millions of
        # rows but only a few thousand distinct players).
        name_map = {}
        for col in ('player_1', 'player_2'):
            for name in self.scraper_df[col].astype(str).unique():
                if name not in name_map:
                    name_map[name] = self._normalize_name(name)

        p1 = self.scraper_df['player_1'].astype(str).map(name_map)
        p2 = self.scraper_df['player_2'].astype(str).map(name_map)
        first = p1.where(p1 <= p2, p2)
        second = p2.where(p1 <= p2, p1)
        dates = self.scraper_df['parsed_date'].dt.strftime('%Y-%m-%d')

        self.scraper_df['surname_p1'] = p1
        self.scraper_df['surname_p2'] = p2
        self.scraper_df['match_key'] = dates + '_' + first + '_' + second

        # Create per-bookmaker index
        self.odds_by_bookmaker = {}
        total_matches = set()

        for bm in self.BOOKMAKER_PRIORITY:
            bm_df = self.scraper_df[self.scraper_df['bookmaker'].str.contains(
                bm, case=False, na=False)]

            if len(bm_df) == 0:
                continue

            # Split by selection
            sel1 = bm_df[bm_df['selection'] == 'Selection_1'][
                ['match_key', 'player_1', 'player_2',
                    'surname_p1', 'surname_p2', 'opening_odd']
            ].rename(columns={'opening_odd': 'open_p1'})

            sel2 = bm_df[bm_df['selection'] == 'Selection_2'][
                ['match_key', 'opening_odd']
            ].rename(columns={'opening_odd': 'open_p2'})

            # Merge selections
            merged = pd.merge(sel1, sel2, on='match_key', how='outer')
            merged = merged.drop_duplicates(subset='match_key', keep='first')

            # Only keep complete pairs (both players)
            merged = merged[merged['open_p1'].notna() &
                            merged['open_p2'].notna()]

            self.odds_by_bookmaker[bm] = merged
            total_matches.update(merged['match_key'].tolist())

        Log.info(
            "LOADER", f"Created multi-bookmaker index: {len(total_matches)} unique matches across {len(self.odds_by_bookmaker)} bookmakers")

    def enrich_with_opening_odds(self, df):
        """
        Enrich dataset with Opening odds using multi-bookmaker CASCADE.
        Priority: Pinnacle → bet365 → Betfair → Unibet → etc.
        """
        if not self.scraper_loaded:
            self.load_scraped_pinnacle_index()

        if self.scraper_df is None or not hasattr(self, 'odds_by_bookmaker'):
            Log.warn("No scraped data available - proceeding without Opening odds")
            df['Pin_Open_W'] = np.nan
            df['Pin_Open_L'] = np.nan
            df['has_pin_open'] = 0.0
            df['opening_source'] = 'none'
            return df

        Log.info("LOADER", "Enriching with Opening odds (MULTI-BOOKMAKER CASCADE)...")

        # Prepare main df for merge
        df['_date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['_surname_w'] = df['winner_name'].apply(self._normalize_name)
        df['_surname_l'] = df['loser_name'].apply(self._normalize_name)

        def make_sota_key(row):
            d = str(row['_date'])[:10]
            names = sorted([row['_surname_w'], row['_surname_l']])
            return f"{d}_{names[0]}_{names[1]}"

        df['_match_key'] = df.apply(make_sota_key, axis=1)

        # Initialize result columns
        df['Pin_Open_W'] = np.nan
        df['Pin_Open_L'] = np.nan
        df['opening_source'] = 'none'

        # Cascade through bookmakers
        bookmaker_stats = {}

        for bm in self.BOOKMAKER_PRIORITY:
            if bm not in self.odds_by_bookmaker:
                continue

            bm_index = self.odds_by_bookmaker[bm]
            mask_missing = df['Pin_Open_W'].isna()

            if not mask_missing.any():
                break  # All matches have odds

            # Merge missing rows with this bookmaker
            missing_df = df.loc[mask_missing, [
                '_match_key', '_surname_w']].copy()
            merged = pd.merge(
                missing_df,
                bm_index[['match_key', 'surname_p1',
                          'surname_p2', 'open_p1', 'open_p2']],
                left_on='_match_key', right_on='match_key', how='left'
            )

            # Assign odds based on winner
            def assign_odds(row):
                if pd.isna(row['open_p1']):
                    return np.nan, np.nan
                w_surname = row['_surname_w']
                if row['surname_p1'] == w_surname:
                    return row['open_p1'], row['open_p2']
                elif row['surname_p2'] == w_surname:
                    return row['open_p2'], row['open_p1']
                else:
                    return row['open_p1'], row['open_p2']

            results = merged.apply(assign_odds, axis=1, result_type='expand')

            # Update df where we found matches
            found_mask = results[0].notna()
            found_indices = missing_df.index[found_mask]

            if len(found_indices) > 0:
                df.loc[found_indices,
                       'Pin_Open_W'] = results.loc[found_mask, 0].values
                df.loc[found_indices,
                       'Pin_Open_L'] = results.loc[found_mask, 1].values
                df.loc[found_indices, 'opening_source'] = bm
                bookmaker_stats[bm] = len(found_indices)

        # Set flag
        df['has_pin_open'] = df['Pin_Open_W'].notna().astype(float)

        # Cleanup
        df = df.drop(columns=['_date', '_surname_w',
                     '_surname_l', '_match_key'], errors='ignore')

        # Report coverage
        coverage = df['has_pin_open'].mean() * 100
        Log.ok(f"Opening odds coverage: {coverage:.1f}%")

        # Show breakdown by bookmaker
        Log.info("SOURCE", "Breakdown by bookmaker:")
        for bm, count in sorted(bookmaker_stats.items(), key=lambda x: -x[1]):
            print(f"    {bm}: {count} matches")
        print(
            f"    [No opening]: {int((df['has_pin_open'] == 0).sum())} matches")

        # Year-by-year coverage
        df['_year'] = pd.to_datetime(df['Date'], errors='coerce').dt.year
        yearly_cov = df.groupby('_year')['has_pin_open'].mean() * 100
        Log.info("COVERAGE", "By year (2012+):")
        for yr, cov in yearly_cov[yearly_cov.index >= 2012].items():
            marker = "✓" if cov > 80 else "✗" if cov < 50 else "~"
            print(f"    {yr}: {cov:.1f}% {marker}")
        df = df.drop(columns=['_year'], errors='ignore')

        return df




class OddsDebiaser:
    """
    Rolling Linear Regression to calibrate morning→closing odds gap.

    PARADIGMA MATTINA/SERA:
    - Input: fair_prob_A (probability from morning odds)
    - Target: closing_true_prob (probability from closing odds)
    - Output: Calibrated probability that accounts for market movement
    """

    def __init__(self, window_size=DEBIASER_WINDOW):
        self.window_size = window_size
        self.X_buffer = deque(maxlen=window_size)
        self.y_buffer = deque(maxlen=window_size)
        from sklearn.linear_model import Ridge
        self.model = Ridge(alpha=1.0)  # Ridge regression for stability
        self.is_fitted = False

    def update(self, raw_probs, targets):
        """
        raw_probs: morning probability (fair_prob_A)
        targets: closing probability (closing_true_prob)
        """
        for p, y in zip(raw_probs, targets):
            self.X_buffer.append([p])
            self.y_buffer.append(y)

    def fit(self):
        if len(self.X_buffer) > 500:
            X = np.array(self.X_buffer)
            y = np.array(self.y_buffer)
            self.model.fit(X, y)
            self.is_fitted = True

    def debias(self, raw_probs):
        """Return calibrated probabilities."""
        if not self.is_fitted:
            return raw_probs
        X = np.array(raw_probs).reshape(-1, 1)
        # Clip to valid probability range
        return np.clip(self.model.predict(X), 0.01, 0.99)











# ==============================================================================
# 2. FEATURE ENGINEERING (ROBUST)
# ==============================================================================


def calculate_surface_elo_optimized(df):
    print("[FE] Calculating ELO (Chronological)...")
    unique_surfaces = df['surface'].dropna().unique()
    surfaces_list = sorted(list(unique_surfaces))

    elo_global = defaultdict(lambda: 1500.0)
    elo_surface = {surf: defaultdict(lambda: 1500.0) for surf in surfaces_list}

    n_rows = len(df)
    out_global_w = np.zeros(n_rows, dtype=np.float32)
    out_global_l = np.zeros(n_rows, dtype=np.float32)
    out_surf_vals = {
        s: {'w': np.zeros(n_rows), 'l': np.zeros(n_rows)} for s in surfaces_list}

    winners = df['winner_name'].values
    losers = df['loser_name'].values
    surfs = df['surface'].values
    dates = df['Date'].dt.year.values
    levels = df['tourney_level'].values

    for i in range(n_rows):
        w, l, surf, yr, lvl = winners[i], losers[i], surfs[i], dates[i], levels[i]

        out_global_w[i] = elo_global[w]
        out_global_l[i] = elo_global[l]
        for s in surfaces_list:
            out_surf_vals[s]['w'][i] = elo_surface[s][w]
            out_surf_vals[s]['l'][i] = elo_surface[s][l]

        if pd.isna(surf) or surf not in surfaces_list:
            continue

        prob = 1.0 / (1.0 + 10.0 ** ((elo_global[l] - elo_global[w]) / 400.0))
        k = ELO_K_GRAND_SLAM if lvl == 'G' else ELO_K_NORMAL
        delta = k * (1.0 - prob)
        elo_global[w] += delta
        elo_global[l] -= delta

        prob_s = 1.0 / \
            (1.0 + 10.0 **
             ((elo_surface[surf][l] - elo_surface[surf][w]) / 400.0))
        delta_s = k * (1.0 - prob_s)
        elo_surface[surf][w] += delta_s
        elo_surface[surf][l] -= delta_s

    df['elo_global_w'] = out_global_w
    df['elo_global_l'] = out_global_l
    for s in surfaces_list:
        cln = str(s).replace(" ", "")
        df[f'elo_{cln}_w'] = out_surf_vals[s]['w']
        df[f'elo_{cln}_l'] = out_surf_vals[s]['l']
    return df


def compute_ewma_features(df, stat_cols, group_col='name', com=EWMA_COM):
    df = df.sort_values(['name', 'Date'])
    for col in stat_cols:
        if col in df.columns:
            df[f'ewma_{col}'] = (
                df.groupby(group_col)[col]
                .transform(lambda x: x.shift(1).ewm(com=com, adjust=False).mean())
            # No bfill: the first observation has no history and must not see the future.
            ).fillna(0.0)
    return df

def _swap_market_context(xmkt, cols_mkt):
    """Mirror the market features for the A/B-swapped training sample."""
    x = np.array(xmkt, dtype="float32", copy=True)
    if "fair_prob_A" in cols_mkt:
        i = cols_mkt.index("fair_prob_A")
        x[i] = 1.0 - x[i]
    if "implied_A" in cols_mkt and "implied_B" in cols_mkt:
        ia, ib = cols_mkt.index("implied_A"), cols_mkt.index("implied_B")
        x[ia], x[ib] = x[ib], x[ia]
    return x

def _market_prob_index(cols_mkt):
    """Index of the morning market probability inside the market context vector."""
    if "fair_prob_A" in cols_mkt:
        return cols_mkt.index("fair_prob_A")
    if "implied_A" in cols_mkt:
        return cols_mkt.index("implied_A")
    return 0

def _build_swapped_sample(xa, xb, xphy, xmkt, cols_mkt, y_true, odds_b):
    """Build the mirrored A/B training sample.

    Returns (xa_swap, xb_swap, xphy_swap, xmkt_swap, y_prob, y_stake, xgb_x, xgb_y).
    """
    xmkt_swap = _swap_market_context(xmkt, cols_mkt)
    mkt_val = float(xmkt[_market_prob_index(cols_mkt)])
    y_prob = 1.0 - float(y_true)
    y_stake = [y_prob, float(odds_b)]
    xgb_x = np.concatenate([xb, xa, xphy, xmkt_swap])
    xgb_y = (1.0 - float(y_true)) - (1.0 - mkt_val)
    return xb, xa, xphy, xmkt_swap, y_prob, y_stake, xgb_x, xgb_y


def calculate_fair_probability(odds_a, odds_b):
    """Standard Multiplicative Normalization to remove Vig."""
    ip_a = 1.0 / odds_a
    ip_b = 1.0 / odds_b
    margin = ip_a + ip_b
    return ip_a / margin


def enrich_features_residual(raw_df, scaler_fit=False, scaler=None):
    Log.info("PREP", "Feature Engineering Pipeline (V36 - LEAK FIX APPLIED)...")
    df = raw_df.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']).sort_values(
        ['Date', 'match_num'] if 'match_num' in df.columns else ['Date'])

    df = calculate_surface_elo_optimized(df)

    # --- CALCOLO PROBABILITÀ CLOSING (target del solo debiaser) ---
    # Usiamo le Closing Odds (MaxW/MaxL) come riferimento di mercato per calibrare
    # la probabilità mattina→sera. NON DEVONO MAI ESSERE USATE COME INPUT né come
    # target di training della rete.
    closing_margin = (1.0 / df['MaxW']) + (1.0 / df['MaxL'])
    df['closing_true_prob_W'] = (1.0 / df['MaxW']) / closing_margin
    df['closing_true_prob_L'] = (1.0 / df['MaxL']) / closing_margin

    # --- SELEZIONE QUOTE "MATTINA" (INPUT) ---
    # PARADIGMA MATTINA/SERA:
    # - Se esistono Opening Odds (Pin_Open_W/L) → usa quelle alla MATTINA
    # - Se NON esistono Opening → usa Closing (MaxW/MaxL) anche per MATTINA
    #
    # In entrambi i casi, le Closing (MaxW/MaxL) alimentano solo il debiaser.

    # Per-match logic: crea colonne morning_odds_W/L
    has_opening = (
        'Pin_Open_W' in df.columns and
        df['Pin_Open_W'].notna().any()
    )

    if has_opening:
        # Never substitute closing odds for a missing morning price: that is a
        # leak.  Restrict the modelling sample to matches with a real opening.
        missing = df['Pin_Open_W'].isna() | df['Pin_Open_L'].isna()
        n_missing = int(missing.sum())
        if n_missing:
            Log.warn(
                f"{n_missing} matches lack opening odds; excluding them from the "
                "modelling sample instead of substituting closing odds (leak)")
            df = df[~missing].copy()
        df['morning_odds_W'] = df['Pin_Open_W']
        df['morning_odds_L'] = df['Pin_Open_L']
        Log.info("ODDS", "Opening odds coverage: 100.0% after exclusion")
    else:
        # No opening odds available - use closing for both phases
        # (mattina == sera per le quote, nessuna distinzione possibile)
        df['morning_odds_W'] = df['MaxW']
        df['morning_odds_L'] = df['MaxL']
        Log.warn(
            "No Opening odds (Pin_Open_W/L) found. Using MaxW/MaxL for MATTINA. "
            "This means mattina==sera for odds (no temporal distinction)."
        )

    # Input columns per betting ("mattina")
    col_odds_W = 'morning_odds_W'
    col_odds_L = 'morning_odds_L'

    # Stack Rows - WINNER perspective
    df_w = df.copy()
    df_w['name'] = df_w['winner_name']
    df_w['opp_name'] = df_w['loser_name']
    df_w['target'] = 1
    df_w['odds_A'] = df_w[col_odds_W]  # Quota Apertura (Input/Betting)
    df_w['odds_B'] = df_w[col_odds_L]
    df_w['closing_true_prob'] = df_w['closing_true_prob_W']  # Debiaser target

    # Stack Rows - LOSER perspective
    df_l = df.copy()
    df_l['name'] = df_l['loser_name']
    df_l['opp_name'] = df_l['winner_name']
    df_l['target'] = 0
    df_l['odds_A'] = df_l[col_odds_L]  # Quota Apertura (Input/Betting)
    df_l['odds_B'] = df_l[col_odds_W]
    df_l['closing_true_prob'] = df_l['closing_true_prob_L']  # Debiaser target

    # Map Derived features (SSI, Fragility, Fatigue)
    sota_features = ['SSI', 'Fragility_Index',
                     'Fatigue_Load', 'Pythagorean', 'Clutch_Factor']
    for feat in sota_features:
        if f'w_{feat}' in df.columns:
            df_w[feat] = df_w[f'w_{feat}']
            df_l[feat] = df_l[f'l_{feat}']
        else:
            df_w[feat] = 0
            df_l[feat] = 0

    # sCPI
    if 'sCPI' in df.columns:
        df_w['sCPI'] = df_w['sCPI']
        df_l['sCPI'] = df_l['sCPI']
    else:
        df_w['sCPI'] = 0.05
        df_l['sCPI'] = 0.05

    # --- FIX LEAK: MARKET INEFFICIENCY ---
    # ERRORE ORIGINALE: np.abs(synth - MaxW). Questo forniva al modello la Closing Odd futura.
    # FIX: Impostiamo a 0.0 o calcoliamo spread interno (Implied vs Fair).
    # Per sicurezza scientifica assoluta, disabilitiamo il leak impostando a 0.
    df_w['Market_Inefficiency'] = 0.0
    df_l['Market_Inefficiency'] = 0.0

    # Tech Stats Mapping
    stat_keys = ['ace', 'df', 'svpt', '1stWon',
                 '2ndWon', 'bpSaved', 'bpFaced', 'rank']
    for k in stat_keys:
        df_w[k] = df_w[f'w_{k}'] if f'w_{k}' in df_w.columns else 0
        df_l[k] = df_l[f'l_{k}'] if f'l_{k}' in df_l.columns else 0
        df_w[f'opp_{k}'] = df_w[f'l_{k}'] if f'l_{k}' in df_w.columns else 0
        df_l[f'opp_{k}'] = df_l[f'w_{k}'] if f'w_{k}' in df_l.columns else 0

    df_w['elo_global'] = df_w['elo_global_w']
    df_l['elo_global'] = df_l['elo_global_l']

    surfaces = [c.replace('elo_', '').replace(
        '_w', '') for c in df.columns if 'elo_' in c and '_w' in c and 'global' not in c]
    for s in surfaces:
        df_w[f'elo_{s}'] = df_w[f'elo_{s}_w']
        df_l[f'elo_{s}'] = df_l[f'elo_{s}_l']

    df_full = pd.concat([df_w, df_l], axis=0).sort_values(
        ['Date', 'match_num'] if 'match_num' in df.columns else ['Date'])
    df_full['match_uid'] = df_full['Date'].astype(
        str) + "_" + df_full['winner_name'] + "_" + df_full['loser_name']

    # Derived Raw Stats & Filling
    df_full['srv_eff'] = (df_full['1stWon'] + df_full['2ndWon']
                          ) / df_full['svpt'].replace(0, np.nan)
    df_full['ret_eff'] = (df_full['opp_svpt'] - (df_full['opp_1stWon'] +
                          df_full['opp_2ndWon'])) / df_full['opp_svpt'].replace(0, np.nan)
    df_full = df_full.fillna({'srv_eff': 0.6, 'ret_eff': 0.35})

    # EWMA Logic
    stats_ewma = ['srv_eff', 'ret_eff', 'elo_global', 'rank',
                  'ace', 'df'] + sota_features + [f'elo_{s}' for s in surfaces]
    df_full = compute_ewma_features(df_full, stats_ewma, 'name', EWMA_COM)

    # H2H Logic
    df_full = df_full.sort_values(
        ['Date', 'match_num'] if 'match_num' in df_full.columns else ['Date'])
    df_full['h2h_wins'] = df_full.groupby(['name', 'opp_name', 'surface'])[
        'target'].transform(lambda x: x.shift(1).cumsum()).fillna(0)
    df_full['h2h_games'] = df_full.groupby(['name', 'opp_name', 'surface'])[
        'target'].transform(lambda x: x.shift(1).expanding().count()).fillna(0)
    df_full['h2h_ratio'] = (
        df_full['h2h_wins']/df_full['h2h_games'].replace(0, np.nan)).fillna(0.5)

    # Merge Opponent Features
    cols_to_merge = ['match_uid', 'name'] + [c for c in df_full.columns if 'ewma_' in c] + \
        ['h2h_ratio', 'sCPI', 'Market_Inefficiency']
    df_feats = df_full[cols_to_merge].copy()

    df_merged = pd.merge(
        df_full, df_feats,
        left_on=['match_uid', 'opp_name'],
        right_on=['match_uid', 'name'],
        suffixes=('', '_opp'),
        how='left'
    )

    # Fair Probability (Based on input odds)
    df_merged['fair_prob_A'] = list(
        map(calculate_fair_probability, df_merged['odds_A'], df_merged['odds_B']))

    # Final Feature Lists
    features_A = [c for c in df_merged.columns if c.startswith(
        'ewma_') and not c.endswith('_opp')] + ['h2h_ratio']
    features_B = [f"{c}_opp" for c in features_A if c !=
                  'h2h_ratio'] + ['h2h_ratio_opp']

    # Context - IMPORTANTE: implied_A/B derivano da odds_A/B che sono le quote MATTINA
    # (opening se disponibili, closing come fallback). Queste NON sono le closing.
    df_merged['implied_A'] = 1.0 / df_merged['odds_A']  # = 1/morning_odds
    df_merged['implied_B'] = 1.0 / df_merged['odds_B']  # = 1/morning_odds

    for s in surfaces:
        if f'is_{s}' not in df_merged.columns:
            df_merged[f'is_{s}'] = (df_merged['surface'] == s).astype(float)

    # Definizione Context
    cols_CTX = [f'is_{s}' for s in surfaces] + ['sCPI',
                                                'fair_prob_A', 'Market_Inefficiency', 'implied_A', 'implied_B']

    # Drop NaNs
    df_final = df_merged.dropna(
        subset=features_A + features_B + cols_CTX + ['odds_A', 'closing_true_prob']).copy()

    # SCALING
    if scaler_fit:
        scaler = StandardScaler()
        mask_train = df_final['Date'].dt.year < BETTING_START_YEAR
        if mask_train.sum() > 100:
            scaler.fit(df_final.loc[mask_train, features_A])
        else:
            Log.err("Not enough training data for scaler fit!")
            sys.exit(1)

    if scaler:
        df_final[features_A] = scaler.transform(df_final[features_A])
        df_final[features_B] = scaler.transform(df_final[features_B].values)

    # SPLIT CONTEXT
    cols_PHY = [c for c in cols_CTX if 'is_' in c or 'sCPI' in c]
    cols_MKT = [c for c in cols_CTX if c not in cols_PHY]

    Log.ok(
        f"Dataset Ready (LEAK FREE): {len(df_final)} rows. Feats: {len(features_A)}")
    return df_final, features_A, features_B, cols_PHY, cols_MKT, scaler


# ==============================================================================
# 3. MODELS: HYDRA DUAL-HEAD + XGBOOST WRAPPER
# ==============================================================================

def create_outcome_loss():
    """
    LEAK-FREE Loss for the Blind Analyst Head.

    PARADIGMA MATTINA/SERA:
    - MATTINA: Il modello predice probabilità basandosi su opening odds
    - SERA: Il modello impara dall'outcome reale (0/1)

    La closing_prob di mercato NON e' un target di training: viene usata solo
    dal debiaser per calibrare la probabilità mattina→sera.
    """
    bce = tf.keras.losses.BinaryCrossentropy()

    def outcome_loss(y_true, y_pred):
        # Target 1-D: solo l'outcome reale (0/1).
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        return bce(y_true, y_pred)

    return outcome_loss


def create_pnl_loss(rl_weight=RL_LOSS_WEIGHT_POLICY, sharpe_penalty=RL_SHARPE_PENALTY):
    """
    Supervised, PnL-shaped loss for the Stake Head (NOT policy-gradient RL).
    Minimizes negative PnL with a variance penalty for risk control.
    """
    def pnl_loss(y_true, y_pred):
        # y_true shape: (batch, 2) -> [outcome (0/1), odds]
        outcome = y_true[:, 0]
        odds = y_true[:, 1]
        stake_fraction = y_pred[:, 0]

        # PnL per bet: stake * ((outcome * odds) - 1)
        pnl = stake_fraction * ((outcome * (odds - 1)) - (1 - outcome))

        # Maximize mean PnL (minimize negative)
        mean_pnl = tf.reduce_mean(pnl)

        # Penalize variance (Sharpe-like)
        variance_pnl = tf.reduce_mean(tf.square(pnl - mean_pnl))

        # Combined loss
        loss = -mean_pnl + sharpe_penalty * variance_pnl

        return loss

    return pnl_loss


def build_hydra_model_blind_analyst(len_A, len_B, len_PHY, len_MKT):
    """
    Hydra V36: Blind Analyst + Aware Trader.

    HEAD 1 (Analyst): Sees only Physics (A, B, PHY). Blind to Market.
    HEAD 2 (Trader): Sees Analyst Opinion + Market Info (MKT). Decides Stake.
    """
    # Inputs
    in_A = layers.Input(shape=(len_A,), name="A")
    in_B = layers.Input(shape=(len_B,), name="B")
    in_PHY = layers.Input(shape=(len_PHY,), name="Ctx_Phy")
    in_MKT = layers.Input(shape=(len_MKT,), name="Ctx_Mkt")

    # --- BLIND ANALYST (Physics Only) ---
    # Siamese Tower for Players
    tower = keras.Sequential([
        layers.Dense(TOWER_DIM_1, activation='swish'),
        layers.Dropout(DROPOUT_RATE),
        layers.Dense(TOWER_DIM_2, activation='swish')
    ])
    emb_A = tower(in_A)
    emb_B = tower(in_B)

    diff = layers.Subtract()([emb_A, emb_B])
    abs_diff = layers.Lambda(lambda x: K.abs(x))(diff)

    # Analyst Concatenation (NO MARKET DATA)
    concat_analyst = layers.Concatenate()([diff, abs_diff, in_PHY])

    x_analyst = layers.Dense(DENSE_DIM, activation='swish')(concat_analyst)
    x_analyst = layers.Dropout(0.2)(x_analyst)

    # Analyst Output (Probability)
    out_prob = layers.Dense(1, activation='sigmoid',
                            name='win_prob')(x_analyst)

    # --- AWARE TRADER (Analyst + Market) ---
    # Trader sees the Analyst's probability AND the Market Context
    concat_trader = layers.Concatenate()([out_prob, in_MKT, concat_analyst])

    x_trader = layers.Dense(DENSE_DIM, activation='swish')(concat_trader)
    x_trader = layers.Dropout(0.2)(x_trader)

    # Trader Output (Stake)
    stake_raw = layers.Dense(1, activation='sigmoid',
                             name='stake_raw')(x_trader)
    out_stake = layers.Lambda(
        lambda x: x * RL_STAKE_MAX, name='stake_fraction')(stake_raw)

    model = keras.Model([in_A, in_B, in_PHY, in_MKT], [out_prob, out_stake])

    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR_VALUE),
        loss={
            'win_prob': create_outcome_loss(),  # Outcome-only loss (no market teacher)
            'stake_fraction': create_pnl_loss()
        },
        loss_weights={
            'win_prob': RL_LOSS_WEIGHT_VALUE,
            'stake_fraction': RL_LOSS_WEIGHT_POLICY
        },
        metrics={'win_prob': 'accuracy'}
    )
    return model


class EnsemblePredictor:
    """
    Gestisce il Deep Model (Hydra) e l'XGBoost market-residual censor.

    XGBoost non modella i residui della rete: impara il residuo di MERCATO
    (outcome - probabilità implicita mattina) e veta lo stake quando il mercato
    segnala che il modello sta sovrastimando.
    """

    def __init__(self, deep_model):
        self.deep_model = deep_model
        self.xgb_model = None
        self.xgb_buffer_X = []
        self.xgb_buffer_Y = []

    def train_xgb(self):
        if len(self.xgb_buffer_X) < 500:
            return

        X = np.array(self.xgb_buffer_X)
        y = np.array(self.xgb_buffer_Y)

        dtrain = xgb.DMatrix(X, label=y)
        params = {
            'objective': 'reg:squarederror',
            'max_depth': XGB_MAX_DEPTH,
            'eta': XGB_ETA,
            'subsample': XGB_SUBSAMPLE,
            'colsample_bytree': XGB_COLSAMPLE,
            'n_jobs': -1,
            'verbosity': 0
        }
        self.xgb_model = xgb.train(params, dtrain, num_boost_round=50)

    def predict(self, XA, XB, X_PHY, X_MKT):
        # Deep Prediction (2 outputs: prob, stake)
        preds = self.deep_model.predict([XA, XB, X_PHY, X_MKT], verbose=0)
        p_prob = preds[0].flatten()
        p_stake = preds[1].flatten()

        # XGB market-residual prediction (target: outcome - morning implied prob)
        # XGB sees everything flattened
        xgb_market_residual = np.zeros_like(p_prob)
        if self.xgb_model:
            X_flat = np.hstack([XA, XB, X_PHY, X_MKT])
            dtest = xgb.DMatrix(X_flat)
            xgb_market_residual = self.xgb_model.predict(dtest)

        return p_prob, p_stake, xgb_market_residual

    def apply_xgb_veto(self, stake_pred, xgb_market_residual):
        """
        XGBoost market-residual censor: if the market residual is negative
        (the model is overconfident relative to the market), veto the stake.
        """
        veto_mask = xgb_market_residual < XGB_VETO_THRESHOLD
        stake_censored = np.where(veto_mask, 0.0, stake_pred)
        return stake_censored

    def add_to_buffer(self, XA, XB, X_PHY, X_MKT, Y):
        flat = np.concatenate([XA, XB, X_PHY, X_MKT])
        self.xgb_buffer_X.append(flat)
        self.xgb_buffer_Y.append(Y)
        if len(self.xgb_buffer_X) > XGB_WINDOW:
            self.xgb_buffer_X = self.xgb_buffer_X[-8000:]
            self.xgb_buffer_Y = self.xgb_buffer_Y[-8000:]

# ==============================================================================
# 4. SIMULATION ENGINE (WALK-FORWARD WITH 5 STRATEGIES)
# ==============================================================================


def walk_forward_rl_hydra(df, cols_A, cols_B, cols_PHY, cols_MKT, target_col, deep_model):
    """
    Walk-Forward Loop V36 (Blind Analyst + supervised Stake Head).
    """
    # Buffers
    buf_A, buf_B, buf_PHY, buf_MKT = (
        deque(maxlen=BUFFER_SIZE) for _ in range(4))
    # entries: outcome only (0/1) - the market prob is NOT a training target
    buf_Y_prob = deque(maxlen=BUFFER_SIZE)
    buf_Y_stake = deque(maxlen=BUFFER_SIZE)  # Stake Head: [outcome, odds]
    xgb_buffer_X, xgb_buffer_Y = [], []

    # Components
    debiaser = OddsDebiaser()
    ensemble = EnsemblePredictor(deep_model)

    # Bankrolls for 5 strategies
    bankroll_flat = INITIAL_BANKROLL
    bankroll_perc = INITIAL_BANKROLL
    bankroll_kelly = INITIAL_BANKROLL
    bankroll_sota = INITIAL_BANKROLL
    bankroll_head = INITIAL_BANKROLL

    # Equity curves
    equity_flat = [INITIAL_BANKROLL]
    equity_perc = [INITIAL_BANKROLL]
    equity_kelly = [INITIAL_BANKROLL]
    equity_sota = [INITIAL_BANKROLL]
    equity_head = [INITIAL_BANKROLL]

    bets_list = []
    recorded_dates = []

    BETTING_START = pd.Timestamp(f'{BETTING_START_YEAR}-01-01')
    train_counter = 0

    # ==========================================================================
    # 1. VECTORIZED PRE-PROCESSING
    # ==========================================================================
    # Sort by Date
    df = df.sort_values('Date').reset_index(drop=True)

    # Global Numpy Arrays
    XA_full = df[cols_A].values.astype('float32')
    XB_full = df[cols_B].values.astype('float32')
    XPHY_full = df[cols_PHY].values.astype('float32')
    XMKT_full = df[cols_MKT].values.astype('float32')

    # Targets
    Y_outcome_full = df[target_col].values.astype('float32')
    Y_true_prob_full = df['closing_true_prob'].values.astype(
        'float32')  # Debiaser target only (never a training target)

    Odds_full = df['odds_A'].values.astype('float32')
    Odds_B_full = df['odds_B'].values.astype('float32')

    # Metadata
    Dates_full = df['Date'].values
    MatchNames = (df['name'] + " vs " + df['opp_name']).values

    # Date Map
    unique_dates, counts = np.unique(Dates_full, return_counts=True)
    date_indices = []
    current_idx = 0
    for d, count in zip(unique_dates, counts):
        date_indices.append((d, current_idx, current_idx + count))
        current_idx += count

    print(f"[RL-HYDRA V36] Walk-Forward: {len(date_indices)} days")

    # Single source of truth for the morning market probability column
    mkt_idx = _market_prob_index(cols_MKT)

    # ==========================================================================
    # 2. MAIN LOOP
    # ==========================================================================
    for d_ts, start_idx, end_idx in tqdm(date_indices, desc="Walk-Forward", unit="day"):
        d = pd.to_datetime(d_ts)

        # SLICING
        XA = XA_full[start_idx:end_idx]
        XB = XB_full[start_idx:end_idx]
        XPHY = XPHY_full[start_idx:end_idx]
        XMKT = XMKT_full[start_idx:end_idx]
        Y_true = Y_outcome_full[start_idx:end_idx]
        # Slice della probabilità closing (solo per il debiaser)
        true_prob_day = Y_true_prob_full[start_idx:end_idx]

        odds_day = Odds_full[start_idx:end_idx]
        names_day = MatchNames[start_idx:end_idx]

        # --- INFERENCE ---
        if d >= BETTING_START and len(buf_Y_prob) > 1000:

            # Batch Prediction
            # Head 1 vede solo XA, XB, XPHY. Head 2 vede tutto.
            preds_prob, preds_stake_raw, xgb_market_residual = ensemble.predict(
                XA, XB, XPHY, XMKT)

            # Apply XGBoost market-residual veto
            preds_stake_head = ensemble.apply_xgb_veto(
                preds_stake_raw, xgb_market_residual)

            # Debiased Market (per le strategie di confronto)
            raw_mkt_col = XMKT[:, mkt_idx]

            debiased_market = debiaser.debias(raw_mkt_col)

            # Vectorized Loop
            for i in range(len(XA)):
                odds = odds_day[i]
                if odds <= 1.01:
                    continue

                my_prob = preds_prob[i]
                mkt_prob = debiased_market[i]

                # EV Check (usando la prob della rete neurale)
                ev = (my_prob * odds) - 1.0

                # Filters
                signal_deep = (my_prob - mkt_prob) > MIN_EDGE
                # XGB market-residual: positive means the model under-prices
                signal_xgb = (xgb_market_residual[i] > 0.01)

                if signal_deep and signal_xgb and ev > MIN_EDGE:
                    won = (Y_true[i] == 1)
                    odds_executed = odds * (1.0 - EXECUTION_SLIPPAGE)

                    # --- STRATEGIES (ALL with €5000 cap + liquidity) ---
                    # Get tournament level for liquidity multiplier
                    tourney_level = df.iloc[start_idx + i].get(
                        'tourney_level', 'C') if 'tourney_level' in df.columns else 'C'
                    liq_mult = LIQUIDITY_MULT.get(tourney_level, 0.1)
                    effective_cap = MAX_STAKE_LEGAL_CAP * liq_mult

                    # Flat (capped, never more than the bankroll)
                    stake_flat = min(
                        FLAT_STAKE, effective_cap, max(bankroll_flat, 0.0))
                    pnl_flat = stake_flat * \
                        (odds_executed - 1) if won else -stake_flat
                    bankroll_flat += pnl_flat

                    # Perc (capped, never more than the bankroll)
                    stake_perc = min(
                        max(MIN_STAKE_ABSOLUTE, bankroll_perc * PERC_STAKE),
                        effective_cap, max(bankroll_perc, 0.0))
                    pnl_perc = stake_perc * \
                        (odds_executed - 1) if won else -stake_perc
                    bankroll_perc += pnl_perc

                    # Kelly (capped, never more than the bankroll)
                    b_k = odds - 1
                    f_star = (b_k * my_prob - (1 - my_prob)) / \
                        b_k if b_k > 0 else 0
                    stake_kelly = min(
                        max(MIN_STAKE_ABSOLUTE, bankroll_kelly * max(0, f_star) * 0.1),
                        effective_cap, max(bankroll_kelly, 0.0))
                    pnl_kelly = stake_kelly * \
                        (odds_executed - 1) if won else -stake_kelly
                    bankroll_kelly += pnl_kelly

                    # Reference (Adaptive, capped, never more than its own bankroll)
                    stake_sota = min(
                        stake_kelly * KELLY_FRACTION, effective_cap,
                        max(bankroll_sota, 0.0))
                    pnl_sota = stake_sota * \
                        (odds_executed - 1) if won else -stake_sota
                    bankroll_sota += pnl_sota

                    # STAKE HEAD (supervised PnL-shaped head + veto, capped, floored)
                    stake_fraction_head = preds_stake_head[i]
                    stake_head = min(
                        max(MIN_STAKE_ABSOLUTE,
                            bankroll_head * stake_fraction_head),
                        effective_cap, max(bankroll_head, 0.0))
                    pnl_head = stake_head * \
                        (odds_executed - 1) if won else -stake_head
                    bankroll_head += pnl_head

                    # A ruined StakeHead book is not a bet: do not pollute the log
                    if stake_head > 0.0:
                        bets_list.append({
                            'date': d,
                            'match': names_day[i],
                            'odds': odds,
                            'prob': my_prob,
                            'market_prob': float(mkt_prob),
                            'opening_odds': float(odds),
                            'stake_head': stake_head,
                            'pnl_head': pnl_head,
                            'won': won
                        })

            equity_flat.append(bankroll_flat)
            equity_perc.append(bankroll_perc)
            equity_kelly.append(bankroll_kelly)
            equity_sota.append(bankroll_sota)
            equity_head.append(bankroll_head)
            recorded_dates.append(d)

        # --- UPDATE BUFFERS (Evening) ---
        for k in range(len(XA)):
            # Value Head target: outcome only (0/1), 1-D
            # Direct
            buf_A.append(XA[k])
            buf_B.append(XB[k])
            buf_PHY.append(XPHY[k])
            buf_MKT.append(XMKT[k])
            buf_Y_prob.append(Y_true[k])

            # Swapped (Symmetry): features, market context and odds must mirror
            (xa_s, xb_s, xphy_s, xmkt_s, y_prob_s,
             y_stake_s, xgb_x_s, xgb_y_s) = _build_swapped_sample(
                XA[k], XB[k], XPHY[k], XMKT[k], cols_MKT,
                Y_true[k], Odds_B_full[start_idx + k])
            buf_A.append(xa_s)
            buf_B.append(xb_s)
            buf_PHY.append(xphy_s)
            buf_MKT.append(xmkt_s)
            buf_Y_prob.append(y_prob_s)

            # Stake Head targets: [Outcome, Odds] - swapped uses odds_B
            odds_k = odds_day[k]
            buf_Y_stake.append([Y_true[k], odds_k])
            buf_Y_stake.append(y_stake_s)

            # Debiaser Update - PARADIGMA SERA
            # Calibra le predizioni del modello rispetto alla closing_true_prob
            # Usa la probabilità mattina (fair_prob_A, o implied_A) come proxy
            mkt_val = XMKT[k, mkt_idx]
            closing_prob_k = true_prob_day[k]  # closing_true_prob

            # Debiaser impara dove la prob mattina differisce dalla closing (calibrazione mercato)
            debiaser.update([mkt_val], [closing_prob_k])
            debiaser.update([1.0 - mkt_val], [1.0 - closing_prob_k])

            # XGBoost market-residual buffer - FASE SERA (LEAK-FREE)
            # Impara il residuo di MERCATO: outcome reale - probabilità mattina.
            # NON usa closing_prob perché contiene info futura!
            flat = np.concatenate([XA[k], XB[k], XPHY[k], XMKT[k]])
            xgb_buffer_X.append(flat)
            # Residuo: Outcome Reale (sera) - Prob Mattina (mattina)
            # Positivo = il mercato sottostima, Negativo = il mercato sovrastima
            xgb_buffer_Y.append(Y_true[k] - mkt_val)

            # Swapped market-residual sample (same helper as the buffers)
            xgb_buffer_X.append(xgb_x_s)
            xgb_buffer_Y.append(xgb_y_s)

        # Periodic Retraining
        train_counter += 1

        if train_counter % RETRAIN_DEBIASER_DAYS == 0:
            debiaser.fit()

        if train_counter % RETRAIN_DEEP_DAYS == 0 and len(buf_A) > 2000:
            idx = np.random.choice(len(buf_A), min(
                BATCH_SIZE, len(buf_A)), replace=False)

            # Sample Weights (Recency Bias)
            current_year = d.year
            year_weight = ((current_year - START_YEAR) /
                           (2025 - START_YEAR)) ** 2
            year_weight = max(0.01, min(year_weight, 1.0))
            sample_weights = [
                np.ones(len(idx)) * year_weight, np.ones(len(idx)) * year_weight]

            deep_model.train_on_batch(
                [np.array([buf_A[i] for i in idx]),
                 np.array([buf_B[i] for i in idx]),
                 np.array([buf_PHY[i] for i in idx]),
                 np.array([buf_MKT[i] for i in idx])],
                {
                    'win_prob': np.array([buf_Y_prob[i] for i in idx]),
                    'stake_fraction': np.array([buf_Y_stake[i] for i in idx])
                },
                sample_weight=sample_weights
            )

        if train_counter % RETRAIN_XGB_DAYS == 0 and len(xgb_buffer_X) > 1000:
            ensemble.xgb_buffer_X = xgb_buffer_X[-XGB_WINDOW:]
            ensemble.xgb_buffer_Y = xgb_buffer_Y[-XGB_WINDOW:]
            ensemble.train_xgb()

    equity_dict = {
        'Flat': equity_flat,
        'Perc': equity_perc,
        'Kelly': equity_kelly,
        'Reference': equity_sota,
        'StakeHead': equity_head
    }

    return equity_dict, bets_list, recorded_dates
# ==============================================================================
# 5. CHARTING (5 STRATEGIES) - 5 strategies with Auto Log Scale
# ==============================================================================


def generate_charts(dates, hist_dict, run_id, total_bets, win_rate, avg_odds):
    """
    Charting with Automatic Log Scale and Separate Plots for Exponential Strategies.
    """
    os.makedirs(GRAPH_DIR, exist_ok=True)
    plt.style.use('dark_background')

    colors = {
        'Flat': 'cyan',
        'Perc': 'yellow',
        'Kelly': 'magenta',
        'Reference': '#00ff00',
        'StakeHead': '#FFA500'  # Orange/Gold
    }

    formulas_text = (
        "STRATEGY DEFINITIONS:\n\n"
        "────────────────────────\n"
        r"$\bf{Flat}$: $S_t = 10€$" "\n\n"
        r"$\bf{Perc}$: $S_t = B_{t-1} \times 1.0\%$" "\n\n"
        r"$\bf{Kelly}$: $S_t = B_{t-1} \times f^* \times 10\%_{cap}$" "\n\n"
        r"$\bf{Reference}$: $S_t = S_{Kelly} \times 1.5\%$" "\n\n"
        r"$\bf{StakeHead}$: $S_t = B_{t-1} \times p(s)$" "\n\n"
        "────────────────────────\n\n"
        "SYMBOL LEGEND:\n"
        r"$S_t$: Stake at time $t$" "\n"
        r"$B_{t-1}$: Bankroll at $t-1$" "\n"
        r"$f^*$: Kelly Fraction" "\n"
        r"$p(s)$: Supervised stake head output, $p \in [0, 5\%]$, XGB veto"
    )

    def create_plot(plot_dates, plot_hist, suffix, title_mod="", force_linear=False):
        """Create equity plot with automatic log scale detection."""
        if len(plot_dates) < 2:
            return

        # Pre-align all arrays to avoid mutation during iteration
        n_points = len(plot_dates)
        aligned_hist = {}
        max_equity = 0.0
        min_equity = float('inf')

        for strat, full_vals in plot_hist.items():
            if strat not in colors:
                continue

            if len(full_vals) >= n_points:
                vals = np.array(full_vals)[-n_points:]
            else:
                # Pad with initial values if shorter
                vals = np.array(full_vals)
                if len(vals) < n_points:
                    vals = np.concatenate(
                        [[vals[0]] * (n_points - len(vals)), vals])

            aligned_hist[strat] = vals
            max_equity = max(max_equity, np.max(vals))
            min_equity = min(min_equity, np.min(
                vals[vals > 0]) if np.any(vals > 0) else vals[0])

        # Detect if we need log scale
        use_log_scale = (max_equity > 1_000_000) and not force_linear

        # Detect exponential strategies (>10x discrepancy)
        final_values = {k: v[-1] for k, v in aligned_hist.items()}
        max_final = max(final_values.values())
        min_final = min(final_values.values())

        has_huge_discrepancy = (max_final / min_final >
                                10) if min_final > 0 else False

        # Sanity warnings
        if max_equity > 1e9:
            print(
                f"\\n[WARNING] Extremely high equity detected: {max_equity:.2e} €")
            print(f"           This suggests unrealistic compounding or a bug.")
            print(f"           Check stake caps and betting logic.\\n")

        # --- MAIN PLOT ---
        fig, ax = plt.subplots(figsize=(15, 8))
        plt.subplots_adjust(right=0.75)
        ax.axhline(INITIAL_BANKROLL, color='gray',
                   linestyle='--', alpha=0.5, lw=1)

        for strat, vals in aligned_hist.items():
            start_val = vals[0]
            final_val = vals[-1]
            profit = final_val - start_val
            peak = np.max(vals)
            dd_euro = peak - final_val

            # Format label based on magnitude
            if final_val > 1e6:
                label = f"{strat}: €{final_val:.2e} (P: {profit:+.2e}, DD: -{dd_euro:.2e})"
            else:
                label = f"{strat}: €{final_val:.0f} (P: {profit:+.0f}, DD: -{dd_euro:.0f})"

            ax.plot(plot_dates, vals,
                    color=colors[strat], label=label, linewidth=1.5)

        # Apply log scale if needed
        if use_log_scale:
            ax.set_yscale('log')
            ax.set_ylabel("Equity (€) - LOG SCALE",
                          fontsize=12, fontweight='bold')
            title_text = f"[LOG SCALE] {title_mod}Run {run_id} | Bets: {total_bets} | WR: {win_rate:.1f}% | AvgOdds: {avg_odds:.2f}"
        else:
            ax.set_ylabel("Equity (€)", fontsize=12)
            title_text = f"{title_mod}Run {run_id} | Bets: {total_bets} | WR: {win_rate:.1f}% | AvgOdds: {avg_odds:.2f}"

        ax.set_title(title_text, fontsize=14, color='white', pad=20)
        ax.grid(True, alpha=0.15, which='both')

        leg = ax.legend(loc='upper left', bbox_to_anchor=(
            1.02, 1.0), borderaxespad=0, fontsize=10, title="Performance Stats")
        leg.get_title().set_fontweight('bold')

        ax.text(1.02, 0.60, formulas_text, transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='#222222', alpha=0.5, edgecolor='gray'))

        plt.savefig(os.path.join(
            GRAPH_DIR, f"Run{run_id}_5Strategies{suffix}.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)

        # --- SEPARATE PLOTS FOR HUGE DISCREPANCY ---
        if has_huge_discrepancy and max_equity > 100_000:
            # Classify strategies
            exponential_strats = {
                k: v for k, v in aligned_hist.items() if final_values[k] > max_final / 5}
            normal_strats = {k: v for k, v in aligned_hist.items(
            ) if final_values[k] <= max_final / 5}

            # Plot Normal Strategies (Linear Scale)
            if normal_strats:
                fig_norm, ax_norm = plt.subplots(figsize=(15, 8))
                plt.subplots_adjust(right=0.75)
                ax_norm.axhline(INITIAL_BANKROLL, color='gray',
                                linestyle='--', alpha=0.5, lw=1)

                for strat, vals in normal_strats.items():
                    final_val = vals[-1]
                    profit = final_val - vals[0]
                    peak = np.max(vals)
                    dd = peak - final_val
                    label = f"{strat}: €{final_val:.0f} (P: {profit:+.0f}, DD: -{dd:.0f})"
                    ax_norm.plot(plot_dates, vals,
                                 color=colors[strat], label=label, linewidth=2)

                ax_norm.set_title(
                    f"[NORMAL STRATEGIES] {title_mod}Run {run_id} - Linear Scale", fontsize=14, color='white', pad=20)
                ax_norm.set_ylabel("Equity (€)", fontsize=12)
                ax_norm.grid(True, alpha=0.15)
                ax_norm.legend(loc='upper left', bbox_to_anchor=(
                    1.02, 1.0), fontsize=10)

                plt.savefig(os.path.join(
                    GRAPH_DIR, f"Run{run_id}_Normal{suffix}.png"), dpi=150, bbox_inches='tight')
                plt.close(fig_norm)

            # Plot Exponential Strategies (Log Scale)
            if exponential_strats:
                fig_exp, ax_exp = plt.subplots(figsize=(15, 8))
                plt.subplots_adjust(right=0.75)
                ax_exp.axhline(INITIAL_BANKROLL, color='gray',
                               linestyle='--', alpha=0.5, lw=1)

                for strat, vals in exponential_strats.items():
                    final_val = vals[-1]
                    profit = final_val - vals[0]
                    peak = np.max(vals)
                    dd = peak - final_val
                    label = f"{strat}: €{final_val:.2e} (P: {profit:+.2e})"
                    ax_exp.plot(plot_dates, vals,
                                color=colors[strat], label=label, linewidth=2)

                ax_exp.set_yscale('log')
                ax_exp.set_title(
                    f"[EXPONENTIAL STRATEGIES] {title_mod}Run {run_id} - Log Scale", fontsize=14, color='white', pad=20)
                ax_exp.set_ylabel("Equity (€) - LOG SCALE",
                                  fontsize=12, fontweight='bold')
                ax_exp.grid(True, alpha=0.15, which='both')
                ax_exp.legend(loc='upper left', bbox_to_anchor=(
                    1.02, 1.0), fontsize=10)

                plt.savefig(os.path.join(
                    GRAPH_DIR, f"Run{run_id}_Exponential{suffix}.png"), dpi=150, bbox_inches='tight')
                plt.close(fig_exp)

    # Full History
    create_plot(dates, hist_dict, "_Full")

    # 2025 Focus
    mask_2025 = [d.year == 2025 for d in dates]
    if any(mask_2025):
        dates_2025 = [d for d, m in zip(dates, mask_2025) if m]
        hist_2025 = {}
        for k, v in hist_dict.items():
            vals_aligned = np.array(v)[-len(dates):]
            hist_2025[k] = vals_aligned[mask_2025].tolist()

        if len(dates_2025) > 0:
            create_plot(dates_2025, hist_2025, "_2025",
                        title_mod="[2025 FOCUS] ")

# ==============================================================================
# 6. RESULTS PROVENANCE
# ==============================================================================

def write_bets_csv(path, bets):
    pd.DataFrame(bets).to_csv(path, index=False)


def write_results(path, equity_dict, bets, dataset_meta, opening_coverage, config):
    total_bets = len(bets)
    wins = sum(1 for b in bets if b["won"])
    win_rate = (wins / total_bets * 100) if total_bets else 0.0
    avg_odds = float(np.mean([b["odds"] for b in bets])) if bets else 0.0
    lines = [
        "TRADING ALPHA - Walk-Forward Results",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Code commit: {config.get('code_commit', 'unknown')}",
        f"Dataset rows: {dataset_meta.get('rows', 0)}",
        f"Dataset sha256: {dataset_meta.get('sha256', 'unknown')}",
        f"Dataset date range: {dataset_meta.get('date_min', '?')}..{dataset_meta.get('date_max', '?')}",
        f"Opening odds coverage: {opening_coverage:.1f}%",
        f"Data mode: {config.get('data_mode', 'unknown')}",
        f"Modeling rows: {config.get('modeling_rows', 0)}",
        f"Period: {config.get('betting_start_year', '?')}-{config.get('end_year', '?')}",
        f"Bets: {total_bets}",
        f"Wins: {wins}/{total_bets} ({win_rate:.1f}%)",
        f"Avg odds: {avg_odds:.2f}",
    ]
    for name in ["Flat", "Perc", "Kelly", "Reference", "StakeHead"]:
        raw_final = equity_dict[name][-1]
        # A bankroll cannot go below zero: report ruin as EUR 0, never a negative equity.
        final = max(raw_final, 0.0)
        suffix = " (bankrupt)" if raw_final <= 0.0 else ""
        lines.append(
            f"{name}: EUR {final:.0f} (P: {final - INITIAL_BANKROLL:+.0f}){suffix}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(path, results, dataset_meta, config):
    payload = {
        "config": config,
        "dataset": dataset_meta,
        "opening_odds_coverage": results.get("opening_odds_coverage"),
        "results": results,
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _code_commit():
    """Commit that produced the run.

    A container may not ship git, so the launcher can pass the value explicitly:
    TRADING_ALPHA_CODE_COMMIT=$(git rev-parse --short HEAD) on the host.
    """
    env_commit = os.environ.get("TRADING_ALPHA_CODE_COMMIT", "").strip()
    if env_commit:
        return env_commit
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=BASE_DIR,
            stderr=subprocess.DEVNULL, text=True).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=BASE_DIR,
            stderr=subprocess.DEVNULL, text=True).strip()
        return commit + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


# ==============================================================================
# 7. MAIN
# ==============================================================================


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(GRAPH_DIR, exist_ok=True)
    Log.banner("TENNIS HYDRA V37: BLIND ANALYST + SUPERVISED STAKE HEAD")

    random.seed(RNG_SEED)
    np.random.seed(RNG_SEED)
    tf.random.set_seed(RNG_SEED)

    if not os.path.exists(DATASET_FILE):
        Log.err(f"Dataset missing: {DATASET_FILE}")
        return 1

    try:
        raw_df = pd.read_csv(DATASET_FILE, low_memory=False)

        # STEP 1: Enrich with Opening odds from scraped CSV (~1 GB).
        # The default is the honest mode: opening odds are required, and matches
        # without them are dropped rather than silently priced with closing odds.
        allow_fallback = (
            os.environ.get("TRADING_ALPHA_ALLOW_CLOSING_FALLBACK") == "1"
            or os.environ.get("TRADING_ALPHA_DISABLE_OPENING_ODDS") == "1"
        )
        if allow_fallback:
            Log.warn(
                "DEGRADED MODE: closing odds used as morning odds. Results are "
                "NOT a test of the morning/evening architecture.")
            opening_coverage = 0.0
        else:
            if not os.path.exists(SCRAPER_CSV):
                Log.err(
                    f"Opening odds CSV missing: {SCRAPER_CSV}. Re-run with "
                    "TRADING_ALPHA_ALLOW_CLOSING_FALLBACK=1 only for a clearly "
                    "labelled degraded run.")
                return 1
            loader = HybridDataLoader()
            raw_df = loader.enrich_with_opening_odds(raw_df)
            opening_coverage = (
                float(raw_df["has_pin_open"].mean() * 100)
                if "has_pin_open" in raw_df.columns else 0.0)
            if opening_coverage <= 0.0:
                Log.err(
                    "Opening odds CSV loaded but no match was paired: refusing "
                    "to run a closing-only backtest silently.")
                return 1
            raw_df = raw_df[raw_df["has_pin_open"] == 1].copy()
            Log.ok(
                f"Modelling sample restricted to {len(raw_df)} matches with "
                f"opening odds (raw coverage {opening_coverage:.1f}%)")

        # Dataset provenance sidecar (written by data_pipeline.py)
        if os.path.exists(META_FILE):
            with open(META_FILE, encoding="utf-8") as f:
                dataset_meta = json.load(f)
        else:
            dataset_meta = {"rows": len(raw_df), "sha256": "unknown"}

        # STEP 2: Feature Engineering (now uses Pin_Open_W/L if available)
        df_data, cA, cB, cPHY, cMKT, scaler = enrich_features_residual(
            raw_df, scaler_fit=True)
        tgt = 'target'

        # Save Scaler
        with open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)
        with open(os.path.join(MODELS_DIR, "feature_columns.pkl"), "wb") as f:
            pickle.dump({'cols_A': cA, 'cols_B': cB,
                        'cols_PHY': cPHY, 'cols_MKT': cMKT}, f)

        config = {
            "code_commit": _code_commit(),
            "betting_start_year": BETTING_START_YEAR,
            "end_year": int(df_data['Date'].dt.year.max()),
            "seed": RNG_SEED,
            "num_loops": NUM_LOOPS,
            "data_mode": "closing_fallback" if allow_fallback else "opening_odds",
            "modeling_rows": int(len(raw_df)),
        }

        best_profit_head = -99999
        last_run = None

        for i in range(1, NUM_LOOPS + 1):
            K.clear_session()
            model = build_hydra_model_blind_analyst(
                len(cA), len(cB), len(cPHY), len(cMKT))

            # Run Simulation
            equity_dict, bets_list, dates = walk_forward_rl_hydra(
                df_data, cA, cB, cPHY, cMKT, tgt, model)

            total_bets = len(bets_list)

            if total_bets > 0:
                wins = sum(1 for b in bets_list if b['won'])
                win_rate = (wins / total_bets) * 100
                avg_odds = np.mean([b['odds'] for b in bets_list])
            else:
                win_rate = 0.0
                avg_odds = 0.0

            final_head = equity_dict['StakeHead'][-1]
            final_profit_head = final_head - INITIAL_BANKROLL

            Log.info(
                f"RES {i}", f"StakeHead: EUR {final_head:.0f} | Reference: EUR {equity_dict['Reference'][-1]:.0f} | Bets: {total_bets} | WR: {win_rate:.1f}%")

            # Generate Charts
            generate_charts(dates, equity_dict, i,
                            total_bets, win_rate, avg_odds)

            last_run = (equity_dict, bets_list, dates,
                        total_bets, wins, win_rate, avg_odds)

            if final_profit_head > best_profit_head:
                best_profit_head = final_profit_head
                model.save(os.path.join(
                    MODELS_DIR, "best_hydra_rl_model.keras"))

        equity_dict, bets_list, dates, total_bets, wins, win_rate, avg_odds = last_run

        write_results(Path(RESULTS_FILE), equity_dict, bets_list,
                      dataset_meta, opening_coverage, config)
        write_bets_csv(BETS_FILE, bets_list)
        write_manifest(Path(MANIFEST_FILE), {
            "opening_odds_coverage": opening_coverage,
            "bets": total_bets,
            "wins": wins,
            "win_rate": win_rate,
            "avg_odds": avg_odds,
            "final_equity": {k: v[-1] for k, v in equity_dict.items()},
        }, dataset_meta, config)
        Log.ok(f"Results written: {RESULTS_FILE}, {BETS_FILE}, {MANIFEST_FILE}")
        return 0

    except Exception as e:
        Log.err(f"Fatal Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
