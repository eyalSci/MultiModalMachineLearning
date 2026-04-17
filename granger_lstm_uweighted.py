"""
LSTM Granger Causality
==============================
Restricted  : input = [stress(t)]            → predict stress(t+1)
Unrestricted: input = [stress(t), signal(t+1)] → predict stress(t+1)

Loss          : MSE (Mean Squared Error)
Regularisation: Gaussian noise (training only), dropout, L2 weight decay
HP search     : grid over hidden_size x noise_sigma via single inner fold
Evaluation    : Leave-One-Participant-Out (18 folds)
"""

import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from itertools import product
import random, copy

# ── Reproducibility ──────────────────────────────────────
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

# ── Config ───────────────────────────────────────────────
DATA_PATH   = 'aggregated_df1_no_windows.csv'
TARGET      = 'reported_stress'

MAX_EPOCHS  = 300
PATIENCE    = 30
DROPOUT     = 0.3
L2          = 1e-3
LR          = 1e-3

HP_GRID = {
    'hidden_size': [4],
    'noise_sigma': [0],
}

BASE_SIGNALS    = ['ACC_x', 'ACC_y', 'ACC_z', 'BVP', 'EDA', 'TEMP', 'HR']
SIGNALS_TYPE    = ['_mean', '_sd']

SIGNALS = []
for s in SIGNALS_TYPE:
    SIGNALS += [e+s for e in BASE_SIGNALS]

DEVICE = torch.device('cpu')

# ── Data loading ─────────────────────────────────────────
def load_sequences(path):
    df = pd.read_csv(path)
    df = df.sort_values(['participant_id','timestamp']).reset_index(drop=True)
    return df

def make_sequence(pid_df, signal=None):
    """
    x : (T, input_dim)  input at each step
        restricted  : [stress(t-1)]
        unrestricted: [stress(t-1), signal_mean(t), signal_sd(t)]
    y : (T,)  target stress(t) - raw scale, normalised via fit_scaler_y
    T = len(pid_df) - 1
    """
    d      = pid_df.reset_index(drop=True)
    stress = d[TARGET].values.astype(np.float32)

    # Inputs from t-1, Targets from t
    x_stress = stress[:-1].reshape(-1, 1)  # stress(t-1)
    y        = stress[1:]                  # stress(t)

    if signal is not None:
        # Get column names (assuming signal passed is like 'EDA_mean')
        signal_mean_col = signal
        signal_sd_col   = signal.replace('_mean', '_sd')
        
        # Extract values
        sig_mean = d[signal_mean_col].values.astype(np.float32)
        sig_sd   = d[signal_sd_col].values.astype(np.float32)
        
        # We want signal_mean(t) and signal_sd(t), which align with y (stress(t))
        x_sig_mean = sig_mean[1:].reshape(-1, 1)
        x_sig_sd   = sig_sd[1:].reshape(-1, 1)
        
        # Concatenate stress(t-1), signal_mean(t), and signal_sd(t)
        x = np.concatenate([x_stress, x_sig_mean, x_sig_sd], axis=1)
    else:
        x = x_stress

    return x, y
    
# ── Normalisation ─────────────────────────────────────────
def fit_scaler(seqs):
    all_x = np.concatenate([s[0] for s in seqs], axis=0)
    mu    = all_x.mean(axis=0)
    std   = all_x.std(axis=0) + 1e-8
    return mu, std

def apply_scaler(seqs, mu, std):
    return [((x - mu) / std, y) for x, y in seqs]

def fit_scaler_y(seqs):
    """Fit mean/std on training targets y (stress values)."""
    all_y = np.concatenate([s[1] for s in seqs])
    mu_y  = all_y.mean()
    std_y = all_y.std() + 1e-8
    return float(mu_y), float(std_y)

def apply_scaler_y(seqs, mu_y, std_y):
    """Normalise y; leave x unchanged."""
    return [(x, (y - mu_y) / std_y) for x, y in seqs]

# ── Model ────────────────────────────────────────────────
class GrangerLSTM(nn.Module):
    def __init__(self, input_dim, hidden_size, dropout, noise_sigma):
        super().__init__()
        self.noise_sigma = noise_sigma
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers=1, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden_size, 1)

    def forward(self, x, training=False):
        if training and self.noise_sigma > 0:
            x = x + torch.randn_like(x) * self.noise_sigma
        out, _ = self.lstm(x)
        out    = self.drop(out)
        return self.fc(out).squeeze(-1)

# ── MSE ──────────────────────────────────────────────────
def mse(pred, target):
    loss = nn.MSELoss()
    return loss(pred, target)

# ── Train ────────────────────────────────────────────────
def train_model(train_seqs, val_seqs, hidden_size, noise_sigma):
    input_dim = train_seqs[0][0].shape[1]
    model     = GrangerLSTM(input_dim, hidden_size, DROPOUT, noise_sigma).to(DEVICE)
    opt       = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=L2)

    best_val, best_state, wait = np.inf, None, 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        random.shuffle(train_seqs)
        for x, y in train_seqs:
            xt = torch.tensor(x).unsqueeze(0)
            yt = torch.tensor(y).unsqueeze(0)
            opt.zero_grad()
            mse(model(xt, training=True), yt).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = np.mean([
                mse(model(torch.tensor(x).unsqueeze(0)),
                    torch.tensor(y).unsqueeze(0)).item()
                for x, y in val_seqs
            ])

        if val_loss < best_val - 1e-6:
            best_val, best_state, wait = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, best_val

# ── Evaluate ──────────────────────────────────────────────
def eval_model(model, seqs, mu_y=0.0, std_y=1.0):
    """
    Compute MSE in the *original* stress scale by inverse-transforming
    both predictions and targets before computing the loss.
    """
    model.eval()
    with torch.no_grad():
        losses = []
        for x, y in seqs:
            xt   = torch.tensor(x).unsqueeze(0)
            pred = model(xt).squeeze(0).numpy() * std_y + mu_y   # inverse transform predictions
            y_orig = y   # y is already in original scale (not normalised in test_sc_x)
            pt   = torch.tensor(pred)
            yt   = torch.tensor(y_orig)
            losses.append(mse(pt, yt).item())
        return np.mean(losses)

# ── HP search (mini 3-fold over training participants) ───
def hp_search(train_seqs, k_inner=3):
    """
    Split train_seqs into 3 inner folds (participant-level).
    Average val MSE across folds for each HP combo.
    """
    rng    = random.Random(SEED)
    seqs   = train_seqs[:]
    rng.shuffle(seqs)
    folds  = [seqs[i::k_inner] for i in range(k_inner)]

    best_hp, best_score = None, np.inf
    for hidden_size, noise_sigma in product(HP_GRID['hidden_size'], HP_GRID['noise_sigma']):
        fold_scores = []
        for fi in range(k_inner):
            i_val   = folds[fi]
            i_train = [s for j, f in enumerate(folds) if j != fi for s in f]
            if not i_train or not i_val:
                continue
            mu, std   = fit_scaler(i_train)
            mu_y, std_y = fit_scaler_y(i_train)
            itr = apply_scaler_y(apply_scaler(i_train, mu, std), mu_y, std_y)
            iva = apply_scaler_y(apply_scaler(i_val,   mu, std), mu_y, std_y)
            _, val_loss = train_model(itr, iva, hidden_size, noise_sigma)
            fold_scores.append(val_loss)
        mean_score = np.mean(fold_scores)
        if mean_score < best_score:
            best_score, best_hp = mean_score, (hidden_size, noise_sigma)
    return best_hp

# ── Leave-One-Participant-Out CV ──────────────────────────
def run_loocv(df, participants, signal):
    label        = signal if signal else 'stress_only'
    fold_results = []
    n_folds      = len(participants)

    for fold_idx, test_pid in enumerate(participants):
        test_pids  = [test_pid]
        train_pids = [p for p in participants if p != test_pid]
        
        print(f"  Fold {fold_idx+1}/{n_folds} | test={test_pid}", end='  ')

        train_seqs = [make_sequence(df[df['participant_id']==p], signal) for p in train_pids]
        test_seqs  = [make_sequence(df[df['participant_id']==p], signal) for p in test_pids]

        best_hidden, best_sigma = hp_search(train_seqs)

        mu, std     = fit_scaler(train_seqs)
        mu_y, std_y = fit_scaler_y(train_seqs)
        train_sc    = apply_scaler_y(apply_scaler(train_seqs, mu, std), mu_y, std_y)
        test_sc_x   = apply_scaler(test_seqs, mu, std)   # x scaled; y kept raw for eval

        val_idx     = random.randint(0, len(train_sc)-1)
        val_sc      = [train_sc[val_idx]]
        train_sc_es = [s for j, s in enumerate(train_sc) if j != val_idx]

        model, _  = train_model(train_sc_es, val_sc, best_hidden, best_sigma)
        test_mse = eval_model(model, test_sc_x, mu_y, std_y)

        print(f"hidden={best_hidden} σ={best_sigma:.2f} MSE={test_mse:.4f}")
        fold_results.append({
            'fold'            : fold_idx + 1,
            'test_participant': test_pid,
            'signal'          : label,
            'hidden_size'     : best_hidden,
            'noise_sigma'     : best_sigma,
            'test_mse'        : test_mse,
        })

    return fold_results

# ── Main ──────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("LSTM Granger Causality")
    print("=" * 60)

    df   = load_sequences(DATA_PATH)
    pids = list(df['participant_id'].unique())

    all_folds = []

    print("\n── stress_only (restricted) ──")
    all_folds += run_loocv(df, pids, signal=None)

    for sig in SIGNALS:
        print(f"\n── stress + {sig} ──")
        all_folds += run_loocv(df, pids, signal=sig)

    df_folds = pd.DataFrame(all_folds)
    df_folds.to_csv('lstm_fold_results.csv', index=False)

    # Summary
    baseline = (df_folds[df_folds['signal']=='stress_only']
                [['fold','test_mse']]
                .rename(columns={'test_mse':'mse_restricted'}))

    rows = []
    for sig in SIGNALS:
        sub = (df_folds[df_folds['signal']==sig]
               [['fold','test_mse','hidden_size','noise_sigma']]
               .merge(baseline, on='fold'))
        sub['impr_%'] = 100 * (sub['mse_restricted'] - sub['test_mse']) / sub['mse_restricted']
        rows.append({
            'signal'            : sig,
            'mean_mse_restr'    : sub['mse_restricted'].mean(),
            'mean_mse_unrest'   : sub['test_mse'].mean(),
            'mean_impr_%'       : sub['impr_%'].mean(),
            'folds_u_better'    : (sub['impr_%'] > 0).sum(),
            'hidden_mode'       : sub['hidden_size'].mode().iloc[0],
            'sigma_mode'        : sub['noise_sigma'].mode().iloc[0],
        })

    summary = pd.DataFrame(rows).sort_values('mean_impr_%', ascending=False)
    summary.to_csv('lstm_summary.csv', index=False)

    print("\n" + "=" * 60)
    print("SUMMARY (sorted by out-of-sample MSE improvement %)")
    print("=" * 60)
    pd.set_option('display.float_format', '{:.4f}'.format)
    print(summary.to_string(index=False))