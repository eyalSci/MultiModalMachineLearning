"""
VAR-Style Granger Causality Analysis
=====================================
Tests whether each physiological signal Granger-causes reported_stress.

Model structure (lag p, signal x):
  Restricted  : stress(t) = c + Σ a_l * stress(t-l)              + ε  [l=1..p]
  Unrestricted: stress(t) = c + Σ a_l * stress(t-l)
                              + Σ b_l * signal(t-l) + b0 * signal(t) + ε  [l=1..p]

Lag selection : AIC on pooled training data (per LOOCV fold)
Loss          : MSE (Mean Squared Error)
Evaluation    : Leave-One-Participant-Out Cross-Validation (18 folds)
Granger test  : F-test on training residuals (RSS_R - RSS_U) + out-of-sample MSE drop
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from typing import Optional, Tuple, List, Dict

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DATA_PATH    = 'aggregated_df1.csv'
TARGET       = 'reported_stress'
MAX_LAG      = 1
NOISE_LEVELS = [0]
BASE_SIGNALS    = ['ACC_x', 'ACC_y', 'ACC_z', 'BVP', 'EDA', 'TEMP', 'HR']
SIGNALS_TYPE    = ['_mean', '_sd']

SIGNALS = []
for s in SIGNALS_TYPE:
    SIGNALS += [e+s for e in BASE_SIGNALS]

# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.sort_values(['participant_id', 'timestamp']).reset_index(drop=True)
    return df

# ─────────────────────────────────────────────
# Feature building
# ─────────────────────────────────────────────
def build_features(
    participant_data: pd.DataFrame,
    lag: int,
    signal: Optional[str] = None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str]]:
    """
    Build lagged feature matrix for one participant.
    """
    d = participant_data.reset_index(drop=True)
    n = len(d)

    if n <= lag:
        return None, None, []

    rows, feat_names_set = [], None

    for t in range(lag, n):
        row = {}
        # ── lagged stress ──────────────────────────────
        for l in range(1, lag + 1):
            row[f'stress_lag{l}'] = d.loc[t - l, TARGET]

        # ── lagged signal + current signal ─────────────
        if signal is not None:
            for l in range(1, lag + 1):
                row[f'signal_lag{l}'] = d.loc[t - l, signal]
            row['signal_t0'] = d.loc[t, signal]   

        row['__y__'] = d.loc[t, TARGET]
        rows.append(row)

        if feat_names_set is None:
            feat_names_set = [k for k in row if not k.startswith('__')]

    if not rows:
        return None, None, []

    df_r = pd.DataFrame(rows)
    feat_cols = [c for c in df_r.columns if not c.startswith('__')]
    X = df_r[feat_cols].values.astype(float)
    y = df_r['__y__'].values.astype(float)
    return X, y, feat_cols

def pool_participants(
    df: pd.DataFrame,
    participants: List[str],
    lag: int,
    signal: Optional[str],
    noise_std: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Pool feature matrices across multiple participants and inject noise."""
    Xs, ys, names = [], [], []
    for pid in participants:
        pid_df = df[df['participant_id'] == pid]
        X, y, feat_names = build_features(pid_df, lag, signal)
        if X is not None and len(X) > 0:
            Xs.append(X); ys.append(y)
            names = feat_names   
    if not Xs:
        return None, None, []
        
    X_pooled = np.vstack(Xs)
    
    # ── Inject Noise N(0, noise_std) into training features ─────────────
    if noise_std > 0.0:
        noise = np.random.normal(0, noise_std, size=X_pooled.shape)
        X_pooled += noise
        
    return X_pooled, np.concatenate(ys), names

# ─────────────────────────────────────────────
# OLS fitting
# ─────────────────────────────────────────────
def fit_ols(X: np.ndarray, y: np.ndarray) -> sm.regression.linear_model.RegressionResultsWrapper:
    Xc = sm.add_constant(X, has_constant='add')
    return sm.OLS(y, Xc).fit()

def predict_ols(model, X: np.ndarray) -> np.ndarray:
    Xc = sm.add_constant(X, has_constant='add')
    if Xc.shape[1] != model.params.shape[0]:
        Xc = np.column_stack([np.ones(len(X)), X])
    return model.predict(Xc)

# ─────────────────────────────────────────────
# AIC-based lag selection
# ─────────────────────────────────────────────
def select_lag_aic(
    df: pd.DataFrame,
    train_pids: List[str],
    signal: Optional[str],
    max_lag: int,
    noise_std: float = 0.0
) -> Tuple[int, Dict[int, float]]:
    aic_by_lag = {}
    for lag in range(1, max_lag + 1):
        X, y, _ = pool_participants(df, train_pids, lag, signal, noise_std=noise_std)
        if X is None or len(X) < X.shape[1] + 5:   
            continue
        model = fit_ols(X, y)
        aic_by_lag[lag] = model.aic

    if not aic_by_lag:
        return 1, {}
    best_lag = min(aic_by_lag, key=aic_by_lag.get)
    return best_lag, aic_by_lag

# ─────────────────────────────────────────────
# MSE
# ─────────────────────────────────────────────
def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))

# ─────────────────────────────────────────────
# F-test for Granger causality
# ─────────────────────────────────────────────
def granger_f_test(
    model_r, model_u,
    X_r, y_r,
    X_u, y_u
) -> Tuple[float, float]:
    resid_r = y_r - predict_ols(model_r, X_r)
    resid_u = y_u - predict_ols(model_u, X_u)

    rss_r = float(np.sum(resid_r ** 2))
    rss_u = float(np.sum(resid_u ** 2))

    n     = len(y_u)
    k_r   = X_r.shape[1] + 1   
    k_u   = X_u.shape[1] + 1
    q     = k_u - k_r            

    if q <= 0 or rss_u <= 0 or (n - k_u) <= 0:
        return np.nan, np.nan

    F_stat = ((rss_r - rss_u) / q) / (rss_u / (n - k_u))
    p_val  = float(1 - stats.f.cdf(F_stat, dfn=q, dfd=n - k_u))
    return float(F_stat), p_val

# ─────────────────────────────────────────────
# LOOCV
# ─────────────────────────────────────────────
def run_loocv(
    df: pd.DataFrame,
    participants: List[str],
    signal: Optional[str],
    noise_std: float = 0.0
) -> List[Dict]:
    fold_results = []

    for i, test_pid in enumerate(participants):
        train_pids = [p for p in participants if p != test_pid]
        
        # Keep print output minimal to avoid terminal flooding during multi-noise runs
        print(f"  Fold {i+1:2d}/18", end='\r')

        best_lag, aic_dict = select_lag_aic(df, train_pids, signal, MAX_LAG, noise_std=noise_std)

        X_u_tr, y_u_tr, feat_u = pool_participants(df, train_pids, best_lag, signal, noise_std=noise_std)
        X_r_tr, y_r_tr, feat_r = pool_participants(df, train_pids, best_lag, None, noise_std=noise_std)

        if X_u_tr is None or X_r_tr is None:
            continue

        model_u_tr = fit_ols(X_u_tr, y_u_tr)
        model_r_tr = fit_ols(X_r_tr, y_r_tr)

        if signal is not None:
            F_stat, p_val = granger_f_test(
                model_r_tr, model_u_tr,
                X_r_tr, y_r_tr,
                X_u_tr, y_u_tr
            )
        else:
            F_stat, p_val = np.nan, np.nan

        # Test evaluation (ALWAYS NOISE-FREE)
        test_df_pid = df[df['participant_id'] == test_pid]
        X_u_te, y_te, _ = build_features(test_df_pid, best_lag, signal)
        X_r_te, _,    _ = build_features(test_df_pid, best_lag, None)

        if X_u_te is None or len(X_u_te) == 0:
            continue

        y_pred_u = predict_ols(model_u_tr, X_u_te)
        y_pred_r = predict_ols(model_r_tr, X_r_te)

        mse_u = mse(y_te, y_pred_u)
        mse_r = mse(y_te, y_pred_r)

        fold_results.append({
            'noise_std'       : noise_std,
            'test_participant': test_pid,
            'signal'          : signal if signal else 'stress_only',
            'best_lag'        : best_lag,
            'aic_by_lag'      : aic_dict,
            'F_stat_train'    : F_stat,
            'p_val_train'     : p_val,
            'mse_unrestricted': mse_u,
            'mse_restricted'  : mse_r,
            'mse_improvement' : mse_r - mse_u,
            'mse_improvement_%': 100 * (mse_r - mse_u) / mse_r if mse_r > 0 else np.nan,
            'n_test'          : len(y_te)
        })

    print(f"  Processed 18 folds successfully.            ")
    return fold_results

# ─────────────────────────────────────────────
# Summary table builder
# ─────────────────────────────────────────────
def summarise(all_folds: List[Dict]) -> pd.DataFrame:
    df_folds = pd.DataFrame(all_folds)
    rows = []
    
    for (noise_val, sig), grp in df_folds.groupby(['noise_std', 'signal']):
        lag_mode = grp['best_lag'].mode().iloc[0]
        lag_counts = grp['best_lag'].value_counts().to_dict()

        row = {
            'noise_std'             : noise_val,
            'signal'                : sig,
            'n_folds'               : len(grp),
            'lag_most_common'       : lag_mode,
            'lag_distribution'      : str(lag_counts),
            'mean_F_train'          : grp['F_stat_train'].mean(),
            'mean_p_train'          : grp['p_val_train'].mean(),
            'folds_p<0.05'          : (grp['p_val_train'] < 0.05).sum(),
            'mean_mse_restricted'   : grp['mse_restricted'].mean(),
            'mean_mse_unrestricted' : grp['mse_unrestricted'].mean(),
            'mean_mse_improvement'  : grp['mse_improvement'].mean(),
            'mean_mse_improvement_%': grp['mse_improvement_%'].mean(),
            'folds_u_better'        : (grp['mse_improvement'] > 0).sum(),
        }
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values(by=['noise_std', 'mean_mse_improvement_%'], ascending=[True, False])
    return summary

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 65)
    print("VAR-Style Granger Causality — Robustness Testing")
    print("=" * 65)

    df   = load_data(DATA_PATH)
    pids = list(df['participant_id'].unique())
    print(f"Participants: {len(pids)}  |  Total rows: {len(df)}")
    print(f"Noise levels to test: {NOISE_LEVELS}\n")

    all_folds = []

    for noise in NOISE_LEVELS:
        print(f"\n{'━'*65}")
        print(f"▶ RUNNING EXPERIMENTS WITH NOISE STD = {noise}")
        print(f"{'━'*65}")
        
        # ── Restricted model (baseline: stress only) ──────────────────
        print(f"MODEL: stress_only (restricted baseline)")
        folds_r = run_loocv(df, pids, signal=None, noise_std=noise)
        all_folds.extend(folds_r)

        # ── Unrestricted models (stress + each signal) ────────────────
        for sig in SIGNALS:
            print(f"MODEL: stress + {sig}")
            folds_u = run_loocv(df, pids, signal=sig, noise_std=noise)
            all_folds.extend(folds_u)

    # ── Save fold-level results ───────────────────────────────────
    df_folds = pd.DataFrame(all_folds)
    df_folds.to_csv('granger_fold_results.csv', index=False)

    # ── Summary table ─────────────────────────────────────────────
    summary = summarise(all_folds)
    summary.to_csv('granger_summary.csv', index=False)

    # ── Print summary ─────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("FINAL SUMMARY (grouped by noise_std, sorted by out-of-sample improvement %)")
    print("=" * 80)

    display_cols = [
        'noise_std', 'signal', 'lag_most_common', 'mean_F_train', 'mean_p_train',
        'folds_p<0.05', 'mean_mse_restricted', 'mean_mse_unrestricted',
        'mean_mse_improvement_%', 'folds_u_better'
    ]
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 200)
    pd.set_option('display.float_format', '{:.4f}'.format)
    print(summary[display_cols].to_string(index=False))

    print(f"\nAll results saved to pwd")
