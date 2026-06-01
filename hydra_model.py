from tqdm import tqdm
import sys
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb
from collections import defaultdict, deque
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
BASE_DIR = os.getcwd()
DATASET_FILE = os.path.join(BASE_DIR, "data", "tennis_features_v36.csv")
# Pinnacle Opening/Closing (86.86% coverage) — cerca in più path
SCRAPER_CSV = next(
    (p for p in [
        os.path.join(BASE_DIR, "data", "tennis_odds_data_cleaned.csv"),
        "D:/Betting/codice/modello2torri2/tennis_odds_data_cleaned.csv",
    ] if os.path.exists(p)),
    os.path.join(BASE_DIR, "data", "tennis_odds_data_cleaned.csv")  # fallback
)
TML_PATH = os.path.join(BASE_DIR, "data", "atp_matches")
TENNIS_DATA_PATH = os.path.join(BASE_DIR, "data", "match_odds")
GRAPH_DIR = os.path.join(BASE_DIR, "charts")
MODELS_DIR = os.path.join(BASE_DIR, "model_state")

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
# IPERPARAMETRI RL (POLICY HEAD)
# ------------------------
RL_STAKE_MIN = 0.00        # Stake minimo (frazione bankroll)
RL_STAKE_MAX = 0.05        # Stake massimo (frazione bankroll, 5%)
RL_LOSS_WEIGHT_VALUE = 1.0  # Peso loss Value Head
RL_LOSS_WEIGHT_POLICY = 0.3  # Peso loss Policy Head (stake)
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
XGB_VETO_THRESHOLD = -0.02  # Se residuo < -0.05, veto su stake

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
CALIBRATOR_WINDOW = 2000   # Finestra Probability Calibrator
META_LABELER_WINDOW = 4000  # Finestra Meta-Labeler
RETRAIN_DEEP_DAYS = 7      # Ritraining Deep Model ogni N giorni
RETRAIN_XGB_DAYS = 14      # Ritraining XGBoost ogni N giorni
RETRAIN_DEBIASER_DAYS = 30  # Ritraining Debiaser ogni N giorni

# ------------------------
# DRIFT DETECTION
# ------------------------
CUSUM_THRESHOLD = 5.0      # Soglia CUSUM per drift
CUSUM_DRIFT_PENALTY = 0.005
CUSUM_WARMUP = 100

# ------------------------
# PORTFOLIO MANAGEMENT
# ------------------------
INITIAL_BANKROLL = 1000.0
MIN_STAKE_ABSOLUTE = 1.0   # Stake minimo assoluto in €
MAX_STAKE_ABSOLUTE = 100.0  # Stake massimo assoluto in €
# Stake massimo realistico per simulare limiti di liquidità
MAX_REALISTIC_STAKE = 5000.0

# ------------------------
# MARKET REALISM (V35)
# ------------------------
EXECUTION_SLIPPAGE = 0.02  # 2% slippage
DISTILLATION_ALPHA = 0.5   # 0.5 BCE (Outcome) + 0.5 MSE (TrueProb)

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
# 1. UTILS: LOGGING & DRAWDOWN MANAGER
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
    Layer 3 (ENRICH): scraped CSV → Pin_Open where available (86.86%)

    NO SYNTHETIC ODDS - Model is hybrid (works with or without Opening)
    """

    def __init__(self):
        self.scraper_df = None
        self.scraper_loaded = False

    def load_scraped_pinnacle_index(self):
        """Load scraped CSV with ALL bookmakers for cascade lookup."""
        if not os.path.exists(SCRAPER_CSV):
            Log.warn(f"Scraped CSV not found: {SCRAPER_CSV}")
            return None

        Log.info(
            "LOADER", f"Loading ALL bookmaker opening odds from {SCRAPER_CSV}...")

        chunks = []
        try:
            for chunk in pd.read_csv(SCRAPER_CSV,
                                     usecols=['url', 'date_str', 'player_1', 'player_2',
                                              'bookmaker', 'selection', 'opening_odd', 'closing_odd'],
                                     chunksize=500000,
                                     low_memory=False):
                # Keep ALL bookmakers with opening odds
                chunk_valid = chunk[chunk['opening_odd'].notna()]
                chunks.append(chunk_valid)

            if chunks:
                self.scraper_df = pd.concat(chunks, ignore_index=True)
                self.scraper_loaded = True
                Log.ok(
                    f"Loaded {len(self.scraper_df)} opening odds rows from all bookmakers")

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

        # Bookmaker priority for opening odds (most accurate first)
        self.BOOKMAKER_PRIORITY = [
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

        # Parse date
        self.scraper_df['parsed_date'] = pd.to_datetime(
            self.scraper_df['date_str'], format='%d.%m.%Y', errors='coerce')

        # Extract normalized names
        self.scraper_df['surname_p1'] = self.scraper_df['player_1'].apply(
            self._normalize_name)
        self.scraper_df['surname_p2'] = self.scraper_df['player_2'].apply(
            self._normalize_name)

        # Create match key (date + sorted surnames)
        def make_key(row):
            d = str(row['parsed_date'])[:10]
            names = sorted([row['surname_p1'], row['surname_p2']])
            return f"{d}_{names[0]}_{names[1]}"

        self.scraper_df['match_key'] = self.scraper_df.apply(make_key, axis=1)

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


class DrawdownManager:
    def __init__(self, initial_balance=INITIAL_BANKROLL):
        self.balance = initial_balance
        self.equity_curve = [initial_balance]
        self.peak = initial_balance
        self.min_stake = MIN_STAKE_ABSOLUTE
        self.max_stake = MAX_STAKE_ABSOLUTE

    def update(self, profit):
        self.balance += profit
        if self.balance > self.peak:
            self.peak = self.balance
        self.equity_curve.append(self.balance)

    def get_dd_penalty(self):
        """Restituisce un moltiplicatore (0.0-1.0) basato sul drawdown attuale."""
        dd = (self.peak - self.balance) / self.peak
        if dd < 0.10:
            return 1.0
        if dd < 0.20:
            return 0.75
        if dd < 0.30:
            return 0.50
        return 0.25


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


class CUSUMDriftDetector:
    """CUSUM for drift detection."""

    def __init__(self, threshold=CUSUM_THRESHOLD, drift=CUSUM_DRIFT_PENALTY, warmup=CUSUM_WARMUP):
        self.threshold = threshold
        self.drift = drift
        self.warmup = warmup
        self.g_plus = 0.0
        self.g_minus = 0.0
        self.n = 0
        self.sum_error = 0.0
        self.sq_sum_error = 0.0
        self.detected = False
        self.current_score = 0.0

    def update(self, y_pred, y_true):
        error = np.abs(y_true - y_pred)
        self.n += 1

        self.sum_error += error
        self.sq_sum_error += error ** 2
        mean_error = self.sum_error / self.n
        std_error = np.sqrt(
            (self.sq_sum_error / self.n) - mean_error**2) + 1e-8

        if self.n < self.warmup:
            return 0.0, False

        std_norm_error = (error - mean_error) / std_error
        self.g_plus = max(0, self.g_plus + std_norm_error - self.drift)
        self.g_minus = max(0, self.g_minus - std_norm_error - self.drift)

        self.current_score = max(self.g_plus, self.g_minus)

        if self.current_score > self.threshold:
            self.detected = True
            self.g_plus *= 0.5
            self.g_minus *= 0.5
        else:
            self.detected = False

        return self.current_score, self.detected


class ProbabilityCalibrator:
    """Rolling Isotonic Regression wrapper."""

    def __init__(self, window_size=CALIBRATOR_WINDOW):
        self.window_size = window_size
        self.preds_buffer = deque(maxlen=window_size)
        self.label_buffer = deque(maxlen=window_size)
        self.iso_reg = IsotonicRegression(
            out_of_bounds='clip', y_min=0, y_max=1)
        self.is_fitted = False

    def update(self, raw_preds, labels):
        for p, l in zip(raw_preds, labels):
            self.preds_buffer.append(p)
            self.label_buffer.append(l)

    def fit(self):
        if len(self.preds_buffer) > 100:
            X = np.array(self.preds_buffer)
            y = np.array(self.label_buffer)
            X = X + np.random.uniform(0, 1e-6, size=X.shape)
            self.iso_reg.fit(X, y)
            self.is_fitted = True

    def calibrate(self, raw_probs):
        if not self.is_fitted:
            return raw_probs
        return self.iso_reg.transform(raw_probs)


class MetaLabelerXGB:
    """Meta-Model for bet confidence."""

    def __init__(self, window_size=META_LABELER_WINDOW, retrain_every=500):
        self.window_size = window_size
        self.retrain_every = retrain_every
        self.X_buffer = deque(maxlen=window_size)
        self.y_buffer = deque(maxlen=window_size)
        self.model = None
        self.counter = 0

        self.params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': XGB_MAX_DEPTH,
            'eta': 0.05,
            'subsample': XGB_SUBSAMPLE,
            'colsample_bytree': 1.0,
            'verbosity': 0,
            'n_jobs': -1
        }

    def extract_meta_features(self, prob, odds, ev, std_dev, drift_score):
        return np.array([prob, odds, ev, std_dev, drift_score])

    def update(self, meta_features, outcome_profit):
        target = 1 if outcome_profit > 0 else 0
        self.X_buffer.append(meta_features)
        self.y_buffer.append(target)
        self.counter += 1

    def fit(self):
        if len(self.X_buffer) < 200:
            return

        if self.counter >= self.retrain_every:
            X = np.array(self.X_buffer)
            y = np.array(self.y_buffer)
            dtrain = xgb.DMatrix(X, label=y)
            self.model = xgb.train(self.params, dtrain, num_boost_round=100)
            self.counter = 0

    def predict_confidence(self, meta_features_batch):
        if self.model is None:
            return np.ones(len(meta_features_batch)) * 0.5

        dtest = xgb.DMatrix(np.array(meta_features_batch))
        return self.model.predict(dtest)


class KellySizingAdaptive:
    """Adaptive Kelly Criterion."""

    def __init__(self, base_fraction=KELLY_FRACTION, max_cap=RL_STAKE_MAX):
        self.base_fraction = base_fraction
        self.max_cap = max_cap

    def get_stake(self, bankroll, prob, odds, meta_conf, drift_score):
        if prob <= 0 or odds <= 1:
            return 0.0

        b = odds - 1
        q = 1 - prob
        f_star = (b * prob - q) / b

        regime_multiplier = 1.0 / (1.0 + 0.5 * max(0, drift_score - 2.0))

        meta_multiplier = 1.0
        if meta_conf < 0.5:
            meta_multiplier = 0.0
        elif meta_conf < 0.6:
            meta_multiplier = 0.5
        elif meta_conf > 0.75:
            meta_multiplier = 1.2

        final_fraction = f_star * self.base_fraction * \
            regime_multiplier * meta_multiplier
        final_fraction = min(final_fraction, self.max_cap)
        final_fraction = max(0.0, final_fraction)

        return final_fraction * bankroll


class DriftMonitor:
    """Monitora la 'salute' del modello in tempo reale."""

    def __init__(self, window=100):
        self.window = window
        self.outcomes = deque(maxlen=window)
        self.roi_history = []

    def log_bet(self, stake, profit):
        self.outcomes.append((stake, profit))

    def get_regime_factor(self):
        if len(self.outcomes) < 20:
            return 1.0

        total_invested = sum(x[0] for x in self.outcomes)
        total_profit = sum(x[1] for x in self.outcomes)

        if total_invested == 0:
            return 1.0
        recent_roi = total_profit / total_invested

        if recent_roi > 0.05:
            return 1.2
        if recent_roi > -0.05:
            return 1.0
        if recent_roi > -0.15:
            return 0.5
        return 0.2

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
            ).bfill().fillna(0)
    return df


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
    df = df.dropna(subset=['Date']).sort_values('Date')

    df = calculate_surface_elo_optimized(df)

    # --- CALCOLO TARGET TEACHER (Distillation) ---
    # Usiamo le Closing Odds (MaxW/MaxL) SOLO come Target (Label) per insegnare la probabilità reale.
    # NON DEVONO MAI ESSERE USATE COME INPUT.
    closing_margin = (1.0 / df['MaxW']) + (1.0 / df['MaxL'])
    df['closing_true_prob_W'] = (1.0 / df['MaxW']) / closing_margin
    df['closing_true_prob_L'] = (1.0 / df['MaxL']) / closing_margin

    # --- SELEZIONE QUOTE "MATTINA" (INPUT) ---
    # PARADIGMA MATTINA/SERA:
    # - Se esistono Opening Odds (Pin_Open_W/L) → usa quelle alla MATTINA
    # - Se NON esistono Opening → usa Closing (MaxW/MaxL) anche per MATTINA
    #
    # In entrambi i casi, le Closing (MaxW/MaxL) sono usate SOLO come target
    # per il training nella fase "SERA" (Teacher Distillation).

    # Per-match logic: crea colonne morning_odds_W/L
    has_opening = (
        'Pin_Open_W' in df.columns and
        df['Pin_Open_W'].notna().any()
    )

    if has_opening:
        # Usa Opening dove disponibile, altrimenti fallback a Closing
        df['morning_odds_W'] = df['Pin_Open_W'].fillna(df['MaxW'])
        df['morning_odds_L'] = df['Pin_Open_L'].fillna(df['MaxL'])

        # Track coverage
        opening_coverage = df['Pin_Open_W'].notna().mean() * 100
        Log.info(
            "ODDS", f"Opening odds coverage: {opening_coverage:.1f}% (fallback to Closing for rest)")
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
    df_w['closing_true_prob'] = df_w['closing_true_prob_W']  # Teacher Target

    # Stack Rows - LOSER perspective
    df_l = df.copy()
    df_l['name'] = df_l['loser_name']
    df_l['opp_name'] = df_l['winner_name']
    df_l['target'] = 0
    df_l['odds_A'] = df_l[col_odds_L]  # Quota Apertura (Input/Betting)
    df_l['odds_B'] = df_l[col_odds_W]
    df_l['closing_true_prob'] = df_l['closing_true_prob_L']  # Teacher Target

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
    df_full = df_full.sort_values('Date')
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

def create_distillation_loss(alpha=DISTILLATION_ALPHA):
    """
    LEAK-FREE Loss for Blind Analyst Head.

    PARADIGMA MATTINA/SERA:
    - MATTINA: Il modello predice probabilità basandosi su opening odds
    - SERA: Il modello impara dall'outcome reale (0/1)

    La closing_true_prob NON viene usata come training target!
    Viene usata solo per calibrazione post-predizione (XGBoost, Debiaser).
    """
    bce = tf.keras.losses.BinaryCrossentropy()

    def calibrated_loss(y_true_combined, y_pred):
        # y_true_combined shape: (batch, 2) -> [outcome (0/1), closing_true_prob]
        # MA usiamo SOLO outcome per training!
        outcome = y_true_combined[:, 0]
        # closing_true_prob = y_true_combined[:, 1]  # NON USATO per loss!

        # Loss basata SOLO su outcome reale
        loss = bce(outcome, y_pred)

        return loss

    return calibrated_loss


def create_pnl_loss(rl_weight=RL_LOSS_WEIGHT_POLICY, sharpe_penalty=RL_SHARPE_PENALTY):
    """
    Custom Loss for Policy Head (Stake Optimization).
    Minimizes negative PnL with Sharpe penalty for risk control.
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
            'win_prob': create_distillation_loss(),  # Custom Distillation
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
    Gestisce il Deep Model (Hydra) e l'XGBoost Censor.
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

        # XGB Residual Prediction
        # XGB sees everything flattened
        xgb_residual = np.zeros_like(p_prob)
        if self.xgb_model:
            X_flat = np.hstack([XA, XB, X_PHY, X_MKT])
            dtest = xgb.DMatrix(X_flat)
            xgb_residual = self.xgb_model.predict(dtest)

        return p_prob, p_stake, xgb_residual

    def apply_xgb_veto(self, stake_pred, xgb_residual):
        """
        XGBoost Censor: If XGB predicts negative residual (model overconfident),
        veto the stake.
        """
        veto_mask = xgb_residual < XGB_VETO_THRESHOLD
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
    Walk-Forward Loop V36 (Blind Analyst + Distillation).
    """
    # Buffers
    buf_A, buf_B, buf_PHY, buf_MKT = (
        deque(maxlen=BUFFER_SIZE) for _ in range(4))
    # entries: [outcome, closing_true_prob] for Distillation
    buf_Y_prob = deque(maxlen=BUFFER_SIZE)
    buf_Y_stake = deque(maxlen=BUFFER_SIZE)  # For policy head: [outcome, odds]
    xgb_buffer_X, xgb_buffer_Y = [], []

    # Components
    debiaser = OddsDebiaser()
    ensemble = EnsemblePredictor(deep_model)

    # Bankrolls for 5 strategies
    bankroll_flat = INITIAL_BANKROLL
    bankroll_perc = INITIAL_BANKROLL
    bankroll_kelly = INITIAL_BANKROLL
    bankroll_sota = INITIAL_BANKROLL
    bankroll_rl = INITIAL_BANKROLL

    # Equity curves
    equity_flat = [INITIAL_BANKROLL]
    equity_perc = [INITIAL_BANKROLL]
    equity_kelly = [INITIAL_BANKROLL]
    equity_sota = [INITIAL_BANKROLL]
    equity_rl = [INITIAL_BANKROLL]

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
        'float32')  # Teacher Target

    Odds_full = df['odds_A'].values.astype('float32')

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
        # Slice del Teacher Target per il giorno corrente
        true_prob_day = Y_true_prob_full[start_idx:end_idx]

        odds_day = Odds_full[start_idx:end_idx]
        names_day = MatchNames[start_idx:end_idx]

        # --- INFERENCE ---
        if d >= BETTING_START and len(buf_Y_prob) > 1000:

            # Batch Prediction
            # Head 1 vede solo XA, XB, XPHY. Head 2 vede tutto.
            preds_prob, preds_stake_raw, xgb_residual = ensemble.predict(
                XA, XB, XPHY, XMKT)

            # Apply XGBoost Veto
            preds_stake_rl = ensemble.apply_xgb_veto(
                preds_stake_raw, xgb_residual)

            # Debiased Market (per le strategie di confronto)
            # Fallback se colonna non trovata
            if 'fair_prob_A' in cols_MKT:
                raw_mkt_col = XMKT[:, cols_MKT.index('fair_prob_A')]
            else:
                raw_mkt_col = XMKT[:, 0]

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
                # XGB conferma che non stiamo sovrastimando
                signal_xgb = (xgb_residual[i] > 0.01)

                if signal_deep and signal_xgb and ev > MIN_EDGE:
                    won = (Y_true[i] == 1)
                    odds_executed = odds * (1.0 - EXECUTION_SLIPPAGE)

                    # --- STRATEGIES (ALL with €5000 cap + liquidity) ---
                    # Get tournament level for liquidity multiplier
                    tourney_level = df.iloc[start_idx + i].get(
                        'tourney_level', 'C') if 'tourney_level' in df.columns else 'C'
                    liq_mult = LIQUIDITY_MULT.get(tourney_level, 0.1)
                    effective_cap = MAX_STAKE_LEGAL_CAP * liq_mult

                    # Flat (capped)
                    stake_flat = min(FLAT_STAKE, effective_cap)
                    pnl_flat = stake_flat * \
                        (odds_executed - 1) if won else -stake_flat
                    bankroll_flat += pnl_flat

                    # Perc (capped)
                    stake_perc = min(
                        max(MIN_STAKE_ABSOLUTE, bankroll_perc * PERC_STAKE), effective_cap)
                    pnl_perc = stake_perc * \
                        (odds_executed - 1) if won else -stake_perc
                    bankroll_perc += pnl_perc

                    # Kelly (capped)
                    b_k = odds - 1
                    f_star = (b_k * my_prob - (1 - my_prob)) / \
                        b_k if b_k > 0 else 0
                    stake_kelly = min(
                        max(MIN_STAKE_ABSOLUTE, bankroll_kelly * max(0, f_star) * 0.1), effective_cap)
                    pnl_kelly = stake_kelly * \
                        (odds_executed - 1) if won else -stake_kelly
                    bankroll_kelly += pnl_kelly

                    # Reference (Adaptive, capped)
                    stake_sota = min(
                        stake_kelly * KELLY_FRACTION, effective_cap)
                    pnl_sota = stake_sota * \
                        (odds_executed - 1) if won else -stake_sota
                    bankroll_sota += pnl_sota

                    # RL AGENT (Policy Head + Veto, capped)
                    stake_fraction_rl = preds_stake_rl[i]
                    stake_rl = bankroll_rl * stake_fraction_rl
                    stake_rl = min(
                        max(MIN_STAKE_ABSOLUTE, stake_rl), effective_cap)
                    pnl_rl = stake_rl * \
                        (odds_executed - 1) if won else -stake_rl
                    bankroll_rl += pnl_rl

                    bets_list.append({
                        'date': d,
                        'match': names_day[i],
                        'odds': odds,
                        'prob': my_prob,
                        'stake_rl': stake_rl,
                        'pnl_rl': pnl_rl,
                        'won': won
                    })

            equity_flat.append(bankroll_flat)
            equity_perc.append(bankroll_perc)
            equity_kelly.append(bankroll_kelly)
            equity_sota.append(bankroll_sota)
            equity_rl.append(bankroll_rl)
            recorded_dates.append(d)

        # --- UPDATE BUFFERS (Evening) ---
        for k in range(len(XA)):
            # Value Head Targets: [Outcome, Teacher_Prob]
            # Direct
            buf_A.append(XA[k])
            buf_B.append(XB[k])
            buf_PHY.append(XPHY[k])
            buf_MKT.append(XMKT[k])
            buf_Y_prob.append([Y_true[k], true_prob_day[k]])

            # Swapped (Symmetry)
            buf_A.append(XB[k])
            buf_B.append(XA[k])
            buf_PHY.append(XPHY[k])
            buf_MKT.append(XMKT[k])
            # Se A vince (1), B perde (0). Se A ha prob Teacher P, B ha 1-P.
            buf_Y_prob.append([1.0 - Y_true[k], 1.0 - true_prob_day[k]])

            # Policy Head Targets: [Outcome, Odds]
            odds_k = odds_day[k]
            buf_Y_stake.append([Y_true[k], odds_k])
            # Loss logic handles outcome
            buf_Y_stake.append([1.0 - Y_true[k], odds_k])

            # Debiaser Update - PARADIGMA SERA
            # Calibra le predizioni del modello rispetto alla closing_true_prob
            # Usa mkt_val (fair_prob_A da morning odds) come proxy se predizioni non disponibili
            mkt_val = XMKT[k, 0]  # fair_prob_A from morning odds
            closing_prob_k = true_prob_day[k]  # closing_true_prob

            # Debiaser impara dove la prob mattina differisce dalla closing (calibrazione mercato)
            debiaser.update([mkt_val], [closing_prob_k])
            debiaser.update([1.0 - mkt_val], [1.0 - closing_prob_k])

            # XGBoost Buffer (Residual) - FASE SERA (LEAK-FREE)
            # XGB impara dove la probabilità MATTINA (fair_prob_A) sbaglia rispetto all'OUTCOME REALE
            # NON usa closing_prob perché contiene info futura!
            flat = np.concatenate([XA[k], XB[k], XPHY[k], XMKT[k]])
            xgb_buffer_X.append(flat)
            # Residuo: Outcome Reale (sera) - Prob Mattina (mattina)
            # Positivo = modello sottostima, Negativo = modello sovrastima
            xgb_buffer_Y.append(Y_true[k] - mkt_val)

            # Swapped XGB
            flat_swap = np.concatenate([XB[k], XA[k], XPHY[k], XMKT[k]])
            xgb_buffer_X.append(flat_swap)
            xgb_buffer_Y.append((1.0 - Y_true[k]) - (1.0 - mkt_val))

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
        'RL_Agent': equity_rl
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
        'RL_Agent': '#FFA500'  # Orange/Gold
    }

    formulas_text = (
        "STRATEGY DEFINITIONS:\n\n"
        "────────────────────────\n"
        r"$\bf{Flat}$: $S_t = 10€$" "\n\n"
        r"$\bf{Perc}$: $S_t = B_{t-1} \times 1.0\%$" "\n\n"
        r"$\bf{Kelly}$: $S_t = B_{t-1} \times f^* \times 5\%_{cap}$" "\n\n"
        r"$\bf{Reference}$: $S_t = S_{Kelly} \times \phi(Drift)$" "\n\n"
        r"$\bf{RL\_Agent}$: $S_t = B_{t-1} \times \pi(s) \times \mathbb{1}_{XGB}$" "\n\n"
        "────────────────────────\n\n"
        "SYMBOL LEGEND:\n"
        r"$S_t$: Stake at time $t$" "\n"
        r"$B_{t-1}$: Bankroll at $t-1$" "\n"
        r"$f^*$: Kelly Fraction" "\n"
        r"$\phi$: Regime Penalty" "\n"
        r"$\pi(s)$: RL Policy Output" "\n"
        r"$\mathbb{1}_{XGB}$: XGBoost Veto"
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
# 6. MAIN
# ==============================================================================


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(GRAPH_DIR, exist_ok=True)
    Log.banner("TENNIS HYDRA V37: BLIND ANALYST + SOFT BOOKIE")

    if not os.path.exists(DATASET_FILE):
        Log.err(f"Dataset missing: {DATASET_FILE}")
        return

    try:
        raw_df = pd.read_csv(DATASET_FILE, low_memory=False)

        # STEP 1: Enrich with Opening odds from scraped CSV
        loader = HybridDataLoader()
        raw_df = loader.enrich_with_opening_odds(raw_df)

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

        best_profit_rl = -99999

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

            final_rl = equity_dict['RL_Agent'][-1]
            final_profit_rl = final_rl - INITIAL_BANKROLL

            Log.info(
                f"RES {i}", f"RL_Agent: €{final_rl:.0f} | Reference: €{equity_dict['Reference'][-1]:.0f} | Bets: {total_bets} | WR: {win_rate:.1f}%")

            # Generate Charts
            generate_charts(dates, equity_dict, i,
                            total_bets, win_rate, avg_odds)

            if final_profit_rl > best_profit_rl:
                best_profit_rl = final_profit_rl
                model.save(os.path.join(
                    MODELS_DIR, "best_hydra_rl_model.keras"))

    except Exception as e:
        Log.err(f"Fatal Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
