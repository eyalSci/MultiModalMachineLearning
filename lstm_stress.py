"""
  Input  = body signals only (no stress in the input)
  Target = stress score at the end of the 30s window

  Experiment structure:
    Model 1 (baseline) : ALL 6 signals together
    Models 2-7         : ONE signal at a time

  This directly answers: which signals, on their own, can predict stress?
  If a single signal model matches or beats the full model, that signal
  carries most of the useful information.

  Train on V1 (merged_df1.csv, 18 participants)
  Test  on V2 (merged_df2.csv, 17 participants)

Overfitting fixes:
  - 50 sequences per participant (900 train, 850 test)
  - Smaller model: 32 hidden units, 1 layer
  - Dropout 0.5
  - Weight decay 1e-4
  - Early stopping (patience=7)
"""

import warnings
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
TRAIN_CSV    = "merged_df1.csv"
TEST_CSV     = "merged_df2.csv"

SEQ_LEN      = 960     # 30 s x 32 Hz
BATCH_SIZE   = 64
EPOCHS       = 50
LR           = 0.0005
HIDDEN_SIZE  = 32
NUM_LAYERS   = 1
DROPOUT      = 0.5
WEIGHT_DECAY = 1e-4
PATIENCE     = 7

SEQS_PER_PARTICIPANT = 50

# All 6 physiological signals
PHYS_SIGNALS = ["EDA", "BVP", "TEMP", "HR", "IBI", "ACC_x"]
TARGET       = "reported_stress"
ALL_COLS     = PHYS_SIGNALS + [TARGET]

DEVICE = torch.device(
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()          else
    "cpu"
)
print(f"Using device: {DEVICE}\n")


# ──────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────

def load_data(csv_path, sig_scalers=None, fit=False):
    """
    Load merged CSV. Forward-fill stress within participant x phase.
    Normalise physiological signals only (stress stays as 0-10).
    Returns data grouped by participant.
    """
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])

    # Forward-fill stress labels within each participant x phase block
    df[TARGET] = (
        df.groupby(["participant_id", "phase"], sort=False)[TARGET]
        .transform(lambda s: s.ffill().bfill())
    )
    df = df.dropna(subset=ALL_COLS)

    print(f"  Rows after cleaning : {len(df):>10,}")
    print(f"  Stress range        : {df[TARGET].min():.1f} - "
          f"{df[TARGET].max():.1f}  (mean {df[TARGET].mean():.2f})")
    print(f"  Participants        : {df['participant_id'].nunique()}\n")

    # Normalise physiological signals (fit on train, apply on test)
    if fit:
        sig_scalers = {}
        for sig in PHYS_SIGNALS:
            sc = StandardScaler()
            df[sig] = sc.fit_transform(df[[sig]]).flatten()
            sig_scalers[sig] = sc
    else:
        for sig in PHYS_SIGNALS:
            df[sig] = sig_scalers[sig].transform(df[[sig]]).flatten()

    # Group by participant so sequences never cross person boundaries
    participants = {}
    for pid, group in df.groupby("participant_id", sort=False):
        group = group.sort_values("timestamp").reset_index(drop=True)
        participants[pid] = {
            "stress":  group[TARGET].values.astype(np.float32),
            "signals": {sig: group[sig].values.astype(np.float32)
                        for sig in PHYS_SIGNALS},
        }
    return participants, sig_scalers


# ──────────────────────────────────────────────
# SEQUENCE CREATION
# ──────────────────────────────────────────────

def make_sequences(participants, signal_names, seq_len, seqs_per_person):
    """
    Create sequences using ONLY the listed signal_names as input.
    The target (y) is the stress score at the END of each window.

    signal_names : list of signal column names to use as input
                   stress is NEVER included in the input.
    """
    all_X, all_y = [], []

    for pid, data in participants.items():
        # Build input array from selected signals only
        cols = [data["signals"][s] for s in signal_names]
        input_arr = np.stack(cols, axis=1)   # (n_rows, n_signals)
        stress    = data["stress"]           # (n_rows,) — target only

        total = len(input_arr) - seq_len
        if total <= 0:
            continue

        n_local = min(seqs_per_person, total)
        idx = np.linspace(0, total - 1, n_local, dtype=int)

        for i in idx:
            all_X.append(input_arr[i : i + seq_len])
            all_y.append(stress[i + seq_len - 1])   # stress at window end

    return (np.array(all_X, dtype=np.float32),
            np.array(all_y, dtype=np.float32))


def make_loader(X, y, batch_size, shuffle=False):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      pin_memory=(DEVICE.type == "cuda"))


# ──────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────

class LSTMRegressor(nn.Module):
    def __init__(self, n_features,
                 hidden_size=HIDDEN_SIZE,
                 num_layers=NUM_LAYERS,
                 dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = n_features,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(self.drop(h[-1])).squeeze(-1)


# ──────────────────────────────────────────────
# TRAINING WITH EARLY STOPPING
# ──────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total = 0.0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for Xb, yb in loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            pred   = model(Xb)
            loss   = criterion(pred, yb)
            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total += loss.item() * Xb.size(0)
    return total / len(loader.dataset)


def train_and_evaluate(train_loader, test_loader, n_features, label):
    """Train with early stopping. Returns best test MSE."""
    model     = LSTMRegressor(n_features).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=LR, weight_decay=WEIGHT_DECAY)

    print(f"\n{'─'*56}")
    print(f"  {label}")
    print(f"{'─'*56}")

    best_mse          = float("inf")
    best_weights      = None
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        tr = run_epoch(model, train_loader, criterion, optimizer)
        te = run_epoch(model, test_loader,  criterion)

        improved = te < best_mse
        if improved:
            best_mse          = te
            best_weights      = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        marker = "  <- best" if improved else ""
        print(f"  Epoch {epoch:2d}/{EPOCHS}  "
              f"Train: {tr:.4f}  Test MSE: {te:.4f}  "
              f"RMSE≈{te**0.5:.2f}{marker}")

        if epochs_no_improve >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(no improvement for {PATIENCE} epochs)")
            break

    model.load_state_dict(best_weights)
    print(f"\n  Best Test MSE : {best_mse:.4f}  "
          f"(RMSE ≈ {best_mse**0.5:.2f} stress points on 0-10 scale)")
    return best_mse


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    np.random.seed(42)
    torch.manual_seed(42)

    # Load data
    train_data, sig_scalers = load_data(TRAIN_CSV, fit=True)
    test_data,  _           = load_data(TEST_CSV,
                                        sig_scalers=sig_scalers,
                                        fit=False)

    results = {}

    # ── MODEL 1: ALL signals together (full baseline) ─────────────────────────
    # This is the ceiling — the best possible result using all signals.
    print("\n" + "=" * 56)
    print("  MODEL 1: all 6 signals (full model — ceiling)")
    print("=" * 56)

    Xtr, ytr = make_sequences(train_data, PHYS_SIGNALS,
                               SEQ_LEN, SEQS_PER_PARTICIPANT)
    Xte, yte = make_sequences(test_data,  PHYS_SIGNALS,
                               SEQ_LEN, SEQS_PER_PARTICIPANT)
    print(f"  Sequences — train: {len(Xtr)}  test: {len(Xte)}")

    results["All signals"] = train_and_evaluate(
        make_loader(Xtr, ytr, BATCH_SIZE, shuffle=True),
        make_loader(Xte, yte, BATCH_SIZE),
        n_features=len(PHYS_SIGNALS),
        label=f"All {len(PHYS_SIGNALS)} signals"
    )

    # ── MODELS 2-7: one signal at a time ──────────────────────────────────────
    # Each model uses ONLY one body signal to predict stress.
    # Signals closest to the full model MSE are the most informative.
    print("\n" + "=" * 56)
    print("  MODELS 2-7: one signal at a time")
    print("=" * 56)

    for sig in PHYS_SIGNALS:
        Xtr, ytr = make_sequences(train_data, [sig],
                                   SEQ_LEN, SEQS_PER_PARTICIPANT)
        Xte, yte = make_sequences(test_data,  [sig],
                                   SEQ_LEN, SEQS_PER_PARTICIPANT)

        results[sig] = train_and_evaluate(
            make_loader(Xtr, ytr, BATCH_SIZE, shuffle=True),
            make_loader(Xte, yte, BATCH_SIZE),
            n_features=1,
            label=f"{sig} only  (1 feature)"
        )

    # ── Results table ──────────────────────────────────────────────────────────
    full_mse = results["All signals"]

    print("\n" + "=" * 68)
    print("  FINAL RESULTS  (stress scale 0-10, lower MSE = better)")
    print("=" * 68)
    print(f"  {'Model':<18} {'MSE':>7}  {'RMSE':>6}  {'vs full model':>14}  {'verdict':>10}")
    print(f"  {'─'*18} {'─'*7}  {'─'*6}  {'─'*14}  {'─'*10}")
    print(f"  {'All signals':<18} {full_mse:>7.4f}  "
          f"{full_mse**0.5:>6.3f}  {'—':>14}  {'ceiling':>10}")

    sorted_signals = sorted(
        [(k, v) for k, v in results.items() if k != "All signals"],
        key=lambda x: x[1]
    )
    for sig, mse in sorted_signals:
        delta = mse - full_mse
        sign  = "+" if delta >= 0 else ""
        if delta <= 0.3:
            verdict = "STRONG"
        elif delta <= 1.0:
            verdict = "moderate"
        else:
            verdict = "weak"
        print(f"  {sig:<18} {mse:>7.4f}  {mse**0.5:>6.3f}  "
              f"{sign}{delta:>13.4f}  {verdict:>10}")

    print()
    print("  STRONG   = this signal alone is nearly as good as all signals")
    print("  moderate = useful but not dominant")
    print("  weak     = this signal contributes little to stress prediction")
    print("=" * 68)


if __name__ == "__main__":
    main()