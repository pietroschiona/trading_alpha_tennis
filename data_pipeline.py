import pandas as pd
import numpy as np
import os
import glob
import re
import logging
import warnings
from datetime import timedelta
from unidecode import unidecode
from tqdm import tqdm

# ==============================================================================
# DATASET GENERATION CONTROL PANEL
# ==============================================================================
# All variables that control dataset generation are grouped here for easy access

# --- FILE PATHS ---
MATCH_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "atp_matches_TML")
ODDS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "match_odds")
OUTPUT_FILE = "tennis_features_v36.csv"

# --- YEAR RANGE ---
START_YEAR = 2000
END_YEAR = 2025

# --- BOOKMAKERS ---
KNOWN_BOOKIES = ['B365', 'CB', 'GB', 'IW', 'SB', 'B&W', 'EX', 'PS', 'SJ', 'UB', 'LB']

# --- FINANCIAL FEATURES (SYNTHETIC ODDS GENERATION) ---
# Favorite-Longshot Bias (FLB) Parameters
FLB_GAMMA_START = 0.90        # Initial Prelec gamma (year 2000) - Strong bias
FLB_GAMMA_END = 0.995         # Final Prelec gamma (year 2025) - Market efficiency cap
FLB_GAMMA_YEARLY_GAIN = 0.0035  # Yearly efficiency improvement

# Sentiment Bias Parameters (Tijmensen)
SENTIMENT_FACTOR = 0.95       # Odds adjustment for popular favorites
RANK_GAP_THRESHOLD = 50       # Minimum rank gap to trigger sentiment bias

# Market Evolution Noise Parameters
NOISE_BASE_SCALE = 0.05       # Base noise level (5% of odds)
NOISE_DECAY_YEARS = 25.0      # Years over which noise decays
NOISE_MAX_MULTIPLIER = 1.2    # Maximum noise multiplier for old data

# Odds Rounding (Real-world simulation)
ODDS_DECIMAL_PLACES = 2       # Match bookmaker precision

# --- CONTEXTUAL METRICS (sCPI) ---
# sCPI = Synthetic Court Pace Index (Aces / Service Points)
SCPI_DEFAULT = 0.05           # Default sCPI when no historical data available

# --- HISTORICAL FEATURES (EWMA & FATIGUE) ---
# Exponential Weighted Moving Average
EWMA_SPAN = 20                # Span for EWMA calculation (20 matches)

# Fatigue Calculation
FATIGUE_WINDOW = 14           # Days for rolling fatigue calculation

# Cold Start Defaults (for players with no history)
DEFAULT_FRAGILITY_INDEX = 0.0
DEFAULT_EWMA_1ST_WON = 0.70
DEFAULT_EWMA_ACE = 0.05
DEFAULT_EWMA_DF = 0.03
DEFAULT_EWMA_BP_SAVED = 0.60
DEFAULT_EWMA_GMS_WON = 4.0
DEFAULT_EWMA_GMS_LOST = 4.0
DEFAULT_FATIGUE_LOAD = 0.0
DEFAULT_MINUTES = 90          # Default match duration when missing

# Derived Metrics Weights
SSI_FIRST_WON_WEIGHT = 1.0    # Service Strength Index: 1st serve won weight
SSI_ACE_WEIGHT = 3.5          # Service Strength Index: Ace weight
SSI_DF_WEIGHT = 2.5           # Service Strength Index: Double fault penalty

PYTHAGOREAN_EXPONENT = 1.5    # Exponent for Pythagorean expectation formula

# --- ODDS VALIDATION ---
MIN_VALID_ODDS = 1.01         # Minimum valid odds threshold
MAX_VALID_ODDS = 100.0        # Maximum valid odds threshold

# --- MERGE PARAMETERS ---
DATE_TOLERANCE_DAYS = 15      # Maximum days difference for match-odds alignment

# --- LOGGING ---
LOG_LEVEL = logging.INFO

# ==============================================================================
# LOGGING & SETUP
# ==============================================================================
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("CRETA_17_SCIAUG")
warnings.filterwarnings('ignore')

print("=== CRETA 17: ZERO-LEAKAGE SCIENTIFIC DATA AUGMENTATION ===")
print("Target: Generating Synthetic Latent Variables & Market Inefficiencies")
print("✨ All generation parameters grouped at file top for full control")

# ==============================================================================
# UTILITIES
# ==============================================================================

def standardize_name_strict(name):
    """Standardizza i nomi per il merging (es. 'Jannik Sinner' -> 'sinner j')."""
    if not isinstance(name, str) or pd.isna(name): return ""
    name = unidecode(name).lower().strip()
    name = re.sub(r'[^a-z\s,]', '', name)
    parts = re.split(r'[\s,]+', name)
    parts = [p for p in parts if p]
    if not parts: return ""
    if len(parts) == 1: return parts[0]
    if len(parts[-1]) == 1: return f"{parts[0]} {parts[-1]}"
    return f"{parts[-1]} {parts[0][0]}"

def parse_date_robust(date_val):
    """Gestisce formati data misti (YYYYMMDD, Excel, ISO)."""
    try:
        if pd.isna(date_val): return pd.NaT
        str_val = str(date_val).strip()
        if len(str_val) == 8 and str_val.isdigit():
            return pd.to_datetime(str_val, format='%Y%m%d', errors='coerce')
        if str_val.replace('.', '', 1).isdigit() and 30000 < float(str_val) < 60000:
             return pd.to_datetime(float(str_val), unit='D', origin='1899-12-30')
        return pd.to_datetime(date_val, dayfirst=True, errors='coerce')
    except:
        return pd.NaT

def sanitize_numeric_columns(df):
    """Converte forzatamente in float tutte le colonne statistiche."""
    numeric_cols = [
        'winner_ht', 'loser_ht', 'winner_age', 'loser_age', 
        'winner_rank', 'loser_rank', 'winner_rank_points', 'loser_rank_points',
        'best_of', 'draw_size', 'match_num', 'minutes',
        'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 
        'w_SvGms', 'w_bpSaved', 'w_bpFaced',
        'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon', 
        'l_SvGms', 'l_bpSaved', 'l_bpFaced'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# ==============================================================================
# STEP 1: LOAD MATCHES (TML SOURCE)
# ==============================================================================

def load_matches_tml():
    logger.info(f"[1/5] Loading Raw Matches from {MATCH_PATH}...")
    
    # Search for both CSV and Excel files
    all_files = []
    for ext in ['*.csv', '*.xls', '*.xlsx']:
        all_files.extend(glob.glob(os.path.join(MATCH_PATH, ext)))
    
    logger.info(f"Found {len(all_files)} total files in matches directory")
    
    valid_files = []
    for f in all_files:
        match = re.search(r'(\d{4})', os.path.basename(f))
        if match:
            yr = int(match.group(1))
            if START_YEAR <= yr <= END_YEAR:
                valid_files.append(f)
    
    logger.info(f"Filtered to {len(valid_files)} files matching year range {START_YEAR}-{END_YEAR}")
    
    if not valid_files:
        logger.error(f"No valid files found in {MATCH_PATH}")
        return pd.DataFrame()
    
    dfs = []
    min_cols = ['tourney_date', 'winner_name', 'loser_name', 'surface']

    for f in tqdm(valid_files, desc="Reading Match Files"):
        try:
            # Handle both CSV and Excel
            if f.endswith('.csv'):
                df = pd.read_csv(f, encoding='utf-8', on_bad_lines='skip', low_memory=False)
            else:
                df = pd.read_excel(f)
            
            df.columns = [c.strip() for c in df.columns]
            
            if not all(c in df.columns for c in min_cols): 
                logger.warning(f"Skipping {os.path.basename(f)}: missing required columns")
                continue
            
            df['Date'] = df['tourney_date'].apply(parse_date_robust)
            df = df.dropna(subset=['Date'])
            df = sanitize_numeric_columns(df)
            
            df['winner_std'] = df['winner_name'].apply(standardize_name_strict)
            df['loser_std'] = df['loser_name'].apply(standardize_name_strict)
            
            if 'surface' in df.columns:
                df['surface'] = df['surface'].astype(str).str.title().str.strip()
            
            dfs.append(df)
            logger.debug(f"Loaded {len(df)} rows from {os.path.basename(f)}")
            
        except Exception as e:
            logger.error(f"Error processing {os.path.basename(f)}: {e}")

    if not dfs:
        logger.error("No dataframes were successfully loaded from match files")
        return pd.DataFrame()
    
    full_match = pd.concat(dfs, ignore_index=True)
    full_match = full_match.sort_values('Date').reset_index(drop=True)
    logger.info(f"✅ Matches Loaded: {len(full_match)} total rows from {len(dfs)} files")
    return full_match

# ==============================================================================
# STEP 2: LOAD ODDS (WITH COMMENTS)
# ==============================================================================

def load_odds():
    logger.info("[2/5] Loading Odds Data (including Comments)...")
    
    files = []
    for ext in ['*.csv', '*.xls', '*.xlsx']:
        files.extend(glob.glob(os.path.join(ODDS_PATH, "**", ext), recursive=True))
        
    dfs = []
    
    for f in tqdm(files, desc="Processing Odds"):
        try:
            if "non usare" in f.lower(): continue
            
            if f.endswith('.csv'):
                df = pd.read_csv(f, encoding='latin1', on_bad_lines='skip', low_memory=False)
            else:
                df = pd.read_excel(f)
            
            df.columns = [c.strip() for c in df.columns]
            
            if not {'Date', 'Winner', 'Loser'}.issubset(df.columns): continue
            
            df['Date'] = df['Date'].apply(parse_date_robust)
            df = df.dropna(subset=['Date'])
            df = df[(df['Date'].dt.year >= START_YEAR) & (df['Date'].dt.year <= END_YEAR)]
            
            if df.empty: continue

            # --- AUTO-CALC AVG/MAX ---
            exist_w, exist_l = [], []
            for b in KNOWN_BOOKIES:
                cw = next((c for c in [f"{b}W", f"{b} W"] if c in df.columns), None)
                cl = next((c for c in [f"{b}L", f"{b} L"] if c in df.columns), None)
                if cw and cl:
                    df[f"{b}W"] = pd.to_numeric(df[cw], errors='coerce')
                    df[f"{b}L"] = pd.to_numeric(df[cl], errors='coerce')
                    exist_w.append(f"{b}W"); exist_l.append(f"{b}L")

            if 'AvgW' not in df.columns: df['AvgW'] = np.nan
            if 'AvgL' not in df.columns: df['AvgL'] = np.nan
            if exist_w:
                df['AvgW'] = df['AvgW'].fillna(df[exist_w].mean(axis=1))
                df['AvgL'] = df['AvgL'].fillna(df[exist_l].mean(axis=1))

            if 'MaxW' not in df.columns: df['MaxW'] = np.nan
            if 'MaxL' not in df.columns: df['MaxL'] = np.nan
            if exist_w:
                df['MaxW'] = df['MaxW'].fillna(df[exist_w].max(axis=1))
                df['MaxL'] = df['MaxL'].fillna(df[exist_l].max(axis=1))
            
            valid = df[(df['AvgW'] > MIN_VALID_ODDS) & (df['AvgL'] > MIN_VALID_ODDS)].copy()
            if valid.empty: continue
            
            valid['winner_std'] = valid['Winner'].apply(standardize_name_strict)
            valid['loser_std'] = valid['Loser'].apply(standardize_name_strict)
            
            # Keep Comment for Fragility Index
            keep_cols = ['Date', 'winner_std', 'loser_std', 'AvgW', 'AvgL', 'MaxW', 'MaxL']
            if 'Comment' in valid.columns:
                keep_cols.append('Comment')
            else:
                valid['Comment'] = 'Completed' # Default if missing
                keep_cols.append('Comment')

            dfs.append(valid[keep_cols])
            
        except Exception: pass

    if not dfs: return pd.DataFrame()
    full_odds = pd.concat(dfs, ignore_index=True)
    
    # --- STRICT DEDUPLICATION (Fix Data Duplication) ---
    logger.info(f"Odds Lines Before Deduplication: {len(full_odds)}")
    
    # Create unique key: Date + Winner + Loser
    # We use string conversion for safety
    full_odds['dedup_key'] = (
        full_odds['Date'].astype(str) + "_" + 
        full_odds['winner_std'] + "_" + 
        full_odds['loser_std']
    )
    
    # Drop duplicates, keeping FIRST occurrence
    full_odds = full_odds.drop_duplicates(subset=['dedup_key'], keep='first')
    full_odds = full_odds.drop(columns=['dedup_key'])
    
    logger.info(f"Odds Lines After Deduplication: {len(full_odds)}")
    return full_odds

# ==============================================================================
# STEP 3: SCIENTIFIC FEATURE ENGINEERING (AUGMENTATION)
# ==============================================================================

def calculate_financial_features(df):
    """
    Implements Financial ML features based on Snowberg & Wolfers (2010), Penn et al. (2025).
    UPDATE: Dynamic FLB Gamma (Evolutionary Market Efficiency).
    FIX APPLIED: Added Noise Floor to prevent 0-variance in recent years (Look-Ahead Fix).
    """
    logger.info("   -> Generating Financial Features (True Prob, Synthetic Odds)...")
    
    # 1. True Probability Extraction (Removing Vig)
    df['implied_W'] = 1 / df['MaxW']
    df['implied_L'] = 1 / df['MaxL']
    df['overround'] = df['implied_W'] + df['implied_L']
    df['true_prob_W'] = df['implied_W'] / df['overround']
    df['true_prob_L'] = df['implied_L'] / df['overround']
    
    # 2. Synthetic Open Odds (Dynamic Favorite-Longshot Bias)
    # FIX: Cap maximum efficiency. Even in 2025, markets are not perfectly efficient (Gamma < 1.0)
    # Previous logic allowed Gamma -> 0.995 which is practically identity function.
    MAX_GAMMA_CAP = 0.97 
    
    years_from_start = (df['Date'].dt.year - START_YEAR).clip(0)
    dynamic_gamma = FLB_GAMMA_START + (years_from_start * FLB_GAMMA_YEARLY_GAIN) 
    dynamic_gamma = dynamic_gamma.clip(upper=MAX_GAMMA_CAP)

    def apply_prelec_vectorized(p, g):
        # Avoid log(0) and ensure numerical stability
        p = np.clip(p, 0.0001, 0.9999)
        return np.exp(-((-np.log(p)) ** g))

    df['subj_prob_W'] = apply_prelec_vectorized(df['true_prob_W'], dynamic_gamma)
    df['subj_prob_L'] = apply_prelec_vectorized(df['true_prob_L'], dynamic_gamma)
    
    # Synthetic Odds (Pure FLB)
    df['synth_odds_W_FLB'] = 1 / df['subj_prob_W']
    df['synth_odds_L_FLB'] = 1 / df['subj_prob_L']
    
    # 3. Sentiment Bias (Tijmensen)
    if 'winner_rank' not in df.columns: df['winner_rank'] = 100
    if 'loser_rank' not in df.columns: df['loser_rank'] = 100
    
    df['rank_gap'] = (df['winner_rank'] - df['loser_rank']).abs()
    
    mask_w_fav = df['MaxW'] < df['MaxL']
    mask_gap = df['rank_gap'] > RANK_GAP_THRESHOLD
    
    df['synth_odds_W'] = df['synth_odds_W_FLB']
    df['synth_odds_L'] = df['synth_odds_L_FLB']
    
    df.loc[mask_w_fav & mask_gap, 'synth_odds_W'] *= SENTIMENT_FACTOR
    df.loc[(~mask_w_fav) & mask_gap, 'synth_odds_L'] *= SENTIMENT_FACTOR
    
    # 4. Market Evolution Noise (CRITICAL FIX)
    df['year'] = df['Date'].dt.year
    
    # FIX: Introduce a MIN_NOISE_FLOOR.
    # Real opening odds ALWAYS differ from closing odds by at least 2-3% (volatility).
    # Without this floor, in 2025 Input == Target (Leak).
    MIN_NOISE_FLOOR = 0.03 # 3% minimum volatility strictly enforced
    
    decay_factor = (END_YEAR - df['year']) / NOISE_DECAY_YEARS
    decay_component = decay_factor.clip(0, NOISE_MAX_MULTIPLIER) * NOISE_BASE_SCALE
    
    # The noise scale is the greater of the decayed historical noise OR the minimum floor
    df['noise_scale'] = np.maximum(decay_component, MIN_NOISE_FLOOR)
    
    # Add Gaussian Noise
    n_rows = len(df)
    # Generate noise centered at 0 but with guaranteed volatility
    noise_w = np.random.normal(0, 1, n_rows) * df['noise_scale'].values * df['synth_odds_W'].values
    noise_l = np.random.normal(0, 1, n_rows) * df['noise_scale'].values * df['synth_odds_L'].values
    
    df['synth_odds_W'] += noise_w
    df['synth_odds_L'] += noise_l
    
    # Clip to safety
    df['synth_odds_W'] = df['synth_odds_W'].clip(MIN_VALID_ODDS, MAX_VALID_ODDS)
    df['synth_odds_L'] = df['synth_odds_L'].clip(MIN_VALID_ODDS, MAX_VALID_ODDS)
    
    # Rounding (Simulate Bookmaker Display)
    df['synth_odds_W'] = df['synth_odds_W'].round(ODDS_DECIMAL_PLACES)
    df['synth_odds_L'] = df['synth_odds_L'].round(ODDS_DECIMAL_PLACES)
    
    # Final Sanity Check: Ensure Synth != MaxW exactly (Floating point strictness)
    # Add a tiny epsilon if they accidentally match to force model to learn abstraction
    epsilon = 0.01
    mask_equal_w = np.isclose(df['synth_odds_W'], df['MaxW'])
    df.loc[mask_equal_w, 'synth_odds_W'] += np.random.choice([-epsilon, epsilon], size=mask_equal_w.sum())

    return df
def calculate_contextual_metrics(df):
    """
    Calculates sCPI (Synthetic Court Pace Index) and other tournament-level stats.
    Uses the sCPI of the SAME tournament from the PREVIOUS year.
    All parameters controlled from top of file.
    """
    logger.info("   -> Generating Contextual Metrics (sCPI - Historical)...")
    
    # sCPI: Aces / Service Points per Tournament-Year
    df['total_serv_points'] = df['w_svpt'] + df['l_svpt']
    df['total_aces'] = df['w_ace'] + df['l_ace']
    
    df['total_serv_points'] = df['total_serv_points'].replace(0, np.nan)
    
    if 'tourney_name' in df.columns:
        # 1. Calculate RAW sCPI for every tournament-year (including current)
        tourney_stats = df.groupby(['tourney_name', 'year']).agg({
            'total_aces': 'sum',
            'total_serv_points': 'sum'
        }).reset_index()
        
        tourney_stats['raw_sCPI'] = tourney_stats['total_aces'] / tourney_stats['total_serv_points']
        
        # 2. SHIFT sCPI to get "Previous Year sCPI" (The Leak Fix)
        tourney_stats = tourney_stats.sort_values(['tourney_name', 'year'])
        tourney_stats['prev_sCPI'] = tourney_stats.groupby('tourney_name')['raw_sCPI'].shift(1)
        
        # 3. Merge back using 'prev_sCPI' as the actual feature
        df = pd.merge(df, tourney_stats[['tourney_name', 'year', 'prev_sCPI']], on=['tourney_name', 'year'], how='left')
        
        # 4. Fill NaNs (First year of tournament or missing history)
        # Use global average for that surface if possible, else global mean
        global_avg = df['prev_sCPI'].mean()
        df['sCPI'] = df['prev_sCPI'].fillna(global_avg if not pd.isna(global_avg) else SCPI_DEFAULT)
        
        # Cleanup
        df = df.drop(columns=['prev_sCPI', 'total_serv_points', 'total_aces'])
    else:
        df['sCPI'] = SCPI_DEFAULT
        
    return df

def calculate_historical_features(df):
    """
    Calculates Walk-Forward metrics (Physical, Mental, Technical) using EWMA.
    CRITICAL FIX: Double shift on Fatigue_Load to eliminate ANY leakage.
    All parameters controlled from top of file.
    """
    logger.info("   -> Generating Historical Features (EWMA, Fatigue, Fragility)...")
    
    # Prepare Long Format for Player-Centric Calculation
    # We need a dataframe where each row is a player-match
    
    # Include Opponent for unique key (Date + Player + Opponent)
    cols_w = ['Date', 'winner_std', 'loser_std', 'minutes', 'w_ace', 'w_svpt', 'w_1stWon', 'w_df', 'w_bpSaved', 'w_bpFaced', 'Comment', 'w_SvGms', 'l_SvGms']
    cols_l = ['Date', 'loser_std', 'winner_std', 'minutes', 'l_ace', 'l_svpt', 'l_1stIn', 'l_df', 'l_bpSaved', 'l_bpFaced', 'Comment', 'l_SvGms', 'w_SvGms']
    
    # Rename for unification
    w_df = df[cols_w].copy()
    w_df.columns = ['Date', 'Player', 'Opponent', 'minutes', 'ace', 'svpt', '1stWon', 'df', 'bpSaved', 'bpFaced', 'Comment', 'GmsWon', 'GmsLost']
    w_df['Won'] = 1
    
    l_df = df[cols_l].copy()
    l_df.columns = ['Date', 'Player', 'Opponent', 'minutes', 'ace', 'svpt', '1stWon', 'df', 'bpSaved', 'bpFaced', 'Comment', 'GmsWon', 'GmsLost']
    l_df['1stWon'] = df['l_1stWon'] # Explicit map
    l_df['Won'] = 0
    
    # Concat
    player_df = pd.concat([w_df, l_df], ignore_index=True)
    player_df = player_df.sort_values(['Player', 'Date'])
    
    # --- 1. Physical: Fatigue Load ---
    player_df = player_df.set_index('Date')
    player_df['minutes'] = player_df['minutes'].fillna(DEFAULT_MINUTES)
    
    # --- 2. Fragility Index ---
    def is_fragile(comment):
        if pd.isna(comment): return 0
        c = str(comment).lower()
        if 'ret' in c or 'w.o' in c or 'walkover' in c: return 1
        return 0
    
    player_df['is_fragile_event'] = player_df['Comment'].apply(is_fragile)

    # --- 3. Fatigue Load (FATIGUE_WINDOW-day rolling) - CRITICAL FIX ---
    logger.info(f"      ... Calculating Fatigue Load ({FATIGUE_WINDOW}D) with DOUBLE SHIFT")
    
    # ZERO-LEAKAGE FIX: Apply explicit .shift(1) AFTER rolling
    # This ensures the value used is ALWAYS from BEFORE the current match
    player_df['Fatigue_Load_raw'] = player_df.groupby('Player')['minutes'].transform(
        lambda x: x.rolling(f'{FATIGUE_WINDOW}D', closed='left').sum()
    )
    # EXPLICIT SHIFT to guarantee no same-day leakage
    player_df['Fatigue_Load'] = player_df.groupby('Player')['Fatigue_Load_raw'].shift(1)

    # --- 4. Technical Stats (Raw for EWMA) ---
    player_df['pct_1stWon'] = player_df['1stWon'] / player_df['svpt']
    player_df['pct_Ace'] = player_df['ace'] / player_df['svpt']
    player_df['pct_DF'] = player_df['df'] / player_df['svpt']
    
    player_df['pct_bpSaved'] = player_df['bpSaved'] / player_df['bpFaced']
    player_df['pct_bpSaved'] = player_df['pct_bpSaved'].fillna(DEFAULT_EWMA_BP_SAVED)
    
    player_df['GmsWon'] = player_df['GmsWon'].fillna(0)
    player_df['GmsLost'] = player_df['GmsLost'].fillna(0)
    
    # --- APPLY EWMA (Walk-Forward) ---
    logger.info(f"      ... Calculating EWMA Metrics (span={EWMA_SPAN})")
    
    features_to_smooth = {
        'is_fragile_event': 'Fragility_Index',
        'pct_1stWon': 'EWMA_1stWon',
        'pct_Ace': 'EWMA_Ace',
        'pct_DF': 'EWMA_DF',
        'pct_bpSaved': 'EWMA_bpSaved',
        'GmsWon': 'EWMA_GmsWon',
        'GmsLost': 'EWMA_GmsLost'
    }
    
    player_df = player_df.reset_index()
    grouped = player_df.groupby('Player')
    
    for col, new_name in features_to_smooth.items():
        player_df[new_name] = grouped[col].transform(lambda x: x.ewm(span=EWMA_SPAN).mean().shift(1))
    
    # Fatigue_Load already shifted above - just fillna
    player_df['Fatigue_Load'] = player_df['Fatigue_Load'].fillna(DEFAULT_FATIGUE_LOAD)
    
    # --- COLD START HANDLING ---
    defaults = {
        'Fragility_Index': DEFAULT_FRAGILITY_INDEX,
        'EWMA_1stWon': DEFAULT_EWMA_1ST_WON,
        'EWMA_Ace': DEFAULT_EWMA_ACE,
        'EWMA_DF': DEFAULT_EWMA_DF,
        'EWMA_bpSaved': DEFAULT_EWMA_BP_SAVED,
        'EWMA_GmsWon': DEFAULT_EWMA_GMS_WON,
        'EWMA_GmsLost': DEFAULT_EWMA_GMS_LOST
    }
    player_df = player_df.fillna(defaults)
    
    # --- DERIVED METRICS ---
    player_df['SSI'] = (player_df['EWMA_1stWon'] * SSI_FIRST_WON_WEIGHT) + \
                       (player_df['EWMA_Ace'] * SSI_ACE_WEIGHT) - \
                       (player_df['EWMA_DF'] * SSI_DF_WEIGHT)
    
    w_exp = player_df['EWMA_GmsWon'] ** PYTHAGOREAN_EXPONENT
    l_exp = player_df['EWMA_GmsLost'] ** PYTHAGOREAN_EXPONENT
    player_df['Pythagorean'] = w_exp / (w_exp + l_exp)
    player_df['Pythagorean'] = player_df['Pythagorean'].fillna(0.5)
    
    player_df['Clutch_Factor'] = player_df['EWMA_bpSaved'] - player_df['EWMA_1stWon']
    
    # --- MERGE BACK TO MAIN DF ---
    # Create lookup keys with Opponent to ensure uniqueness
    player_df['key'] = player_df['Date'].astype(str) + "_" + player_df['Player'] + "_" + player_df['Opponent']
    df['key_w'] = df['Date'].astype(str) + "_" + df['winner_std'] + "_" + df['loser_std']
    df['key_l'] = df['Date'].astype(str) + "_" + df['loser_std'] + "_" + df['winner_std']
    
    cols_to_merge = ['key', 'Fatigue_Load', 'Fragility_Index', 'SSI', 'Pythagorean', 'Clutch_Factor']
    
    # Merge for Winner
    w_stats = player_df[cols_to_merge].copy().add_prefix('w_')
    df = pd.merge(df, w_stats, left_on='key_w', right_on='w_key', how='left')
    
    # Merge for Loser
    l_stats = player_df[cols_to_merge].copy().add_prefix('l_')
    df = pd.merge(df, l_stats, left_on='key_l', right_on='l_key', how='left')
    
    df = df.drop(columns=['key_w', 'key_l', 'w_key', 'l_key'])
    
    return df

# ==============================================================================
# STEP 4: MERGE & EXPORT
# ==============================================================================

def smart_merge(df_m, df_o):
    logger.info("[4/5] Merging TML Data with Market Odds (Optimized Alignment)...")
    
    # Pre-cleaning
    df_m = df_m.drop_duplicates(subset=['Date', 'winner_std', 'loser_std'])
    
    # Create join keys (Year + Players is usually unique enough for ATP level)
    # Using 'year' allows faster join than strict date, we filter later
    df_m['join_year'] = df_m['Date'].dt.year
    df_o['join_year'] = df_o['Date'].dt.year
    
    df_m['key'] = df_m['join_year'].astype(str) + "_" + df_m['winner_std'] + "_" + df_m['loser_std']
    df_o['key'] = df_o['join_year'].astype(str) + "_" + df_o['winner_std'] + "_" + df_o['loser_std']
    
    # Inner Join on loose key
    merged = pd.merge(df_m, df_o, on='key', how='inner', suffixes=('_match', '_odds'))
    
    # Restore names from match source (usually cleaner)
    merged['winner_std'] = merged['winner_std_match']
    merged['loser_std'] = merged['loser_std_match']
    
    # --- DATE PROXIMITY FILTER ---
    # Date_match: Tournament Start Date (often Monday)
    # Date_odds: Actual Match Date (accurate)
    # We allow a wide window (DATE_TOLERANCE_DAYS) because TML dates are Tourney-Start based
    merged['diff_days'] = (merged['Date_match'] - merged['Date_odds']).dt.days.abs()
    
    valid_merge = merged[merged['diff_days'] <= DATE_TOLERANCE_DAYS].copy()
    
    # --- CRITICAL: BEST MATCH SELECTION ---
    # If duplicates exist (same players played twice in same tourney/year? rare but possible),
    # or multiple odds lines found, pick the one with closest date match.
    valid_merge = valid_merge.sort_values('diff_days')
    
    # Unique Key: Date_match (Tourney) + Players. 
    # This ensures we attach odds to the correct tournament round.
    final = valid_merge.drop_duplicates(subset=['Date_match', 'winner_std', 'loser_std'], keep='first')
    
    # --- FINAL DATE FIX ---
    # We MUST use Date_odds (Actual Match Day) for time-series strictness.
    # Date_match is just the Tourney Monday.
    final['Date_Tourney'] = final['Date_match']
    final['Date'] = final['Date_odds']
    
    # Cleanup auxiliary columns
    drop_cols = ['key', 'join_year_match', 'join_year_odds', 'diff_days', 
                 'Date_match', 'Date_odds', 'winner_std_match', 'loser_std_match', 
                 'winner_std_odds', 'loser_std_odds']
    final = final.drop(columns=drop_cols, errors='ignore')
    
    # Re-Sort strictly by Date (Chronological)
    sort_cols = ['Date']
    if 'match_num' in final.columns:
        sort_cols.append('match_num')
    final = final.sort_values(sort_cols)
    
    # Basic Odds Validity Filter
    final['MaxW'] = final['MaxW'].fillna(final['AvgW'])
    final['MaxL'] = final['MaxL'].fillna(final['AvgL'])
    final = final[(final['MaxW'] > MIN_VALID_ODDS) & (final['MaxL'] > MIN_VALID_ODDS)]
    
    # --- AUGMENTATION PIPELINE ---
    final = calculate_financial_features(final) # Noise & Synth Odds
    final = calculate_contextual_metrics(final) # sCPI
    final = calculate_historical_features(final) # EWMA / Fatigue
    
    logger.info(f"Final Deep Dataset Size: {len(final)} rows.")
    return final
# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    matches = load_matches_tml()
    odds = load_odds()
    
    if not matches.empty and not odds.empty:
        dataset = smart_merge(matches, odds)
        
        if not dataset.empty:
            dataset.to_csv(OUTPUT_FILE, index=False)
            logger.info(f"✅ DONE. Dataset saved to: {OUTPUT_FILE}")
            
            print("\n--- SCIENTIFIC FEATURE CHECK ---")
            check_cols = ['true_prob_W', 'synth_odds_W', 'sCPI', 'w_Fatigue_Load', 'w_Fragility_Index', 'w_SSI', 'w_Pythagorean']
            print(dataset[[c for c in check_cols if c in dataset.columns]].tail(5))
            print("--------------------------------\n")
        else:
            logger.error("Dataset empty after merge.")
    else:
        logger.error("Source data missing.")
