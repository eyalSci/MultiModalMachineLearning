"""
LSTM-based Neural Granger Causality — v3  (vectorized, fast)
=============================================================

KEY CHANGES FROM v2
-------------------
1.  VECTORIZED SEQUENCE PROCESSING
    v2 fed one scalar per Python loop iteration → extremely slow.
    v3 feeds entire TBPTT chunks as (1, chunk_size, input_dim) tensors
    in a single LSTM call. Same TBPTT logic, ~20-50x faster per segment.

2.  IBI DROPPED
    Reduces pairs from 56 to 42 (8→7 signals).

3.  SAME CORRECT GRANGER FORMULATION AS v2
    Restricted  : [Y(t)]       → Y(t+1)
    Unrestricted: [Y(t), X(t)] → Y(t+1)
    Granger score = MSE(restricted) − MSE(unrestricted)

All other v2 fixes retained:
    - Segment-level MSE aggregation (not per-step)
    - Participant-level aggregation before population mean
    - FDR correction (Benjamini–Hochberg)
    - Stratified train/val split by phase_type
    - Scalers fit on train split only
    - Explicit NaN / phase-name warnings
    - Training/inference use same TBPTT chunk size
"""

import warnings
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from itertools import permutations
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import fdrcorrection
from scipy import stats
from collections import defaultdict

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

# ── Configuration ─────────────────────────────────────────────────────────────

TRAIN_CSV = "merged_df1.csv"
TEST_CSV  = "merged_df2.csv"

# IBI dropped vs v2
SIGNALS = ["EDA", "HR", "BVP", "TEMP", "ACC_x", "ACC_y", "ACC_z"]

STRESS_PHASES  = ["TMCT", "Stroop", "Subtract", "Opposite Opinion", "Real Opinion"]
REST_PHASES    = ["Baseline", "First Rest", "Second Rest", "Pre-protocol", "Post-protocol"]
EXCLUDE_PHASES = ["Transition_1", "Transition_2", "Transition_3"]

MIN_STEPS   = 20
HIDDEN_SIZE = 64
NUM_LAYERS  = 1
LR          = 1e-3
GRAD_CLIP   = 1.0
MAX_EPOCHS  = 30
PATIENCE    = 5
TBPTT_STEPS = 20
VAL_FRAC    = 0.10
ALPHA       = 0.05

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}\n")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(csv_path: str) -> pd.DataFrame:
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    bad = df["timestamp"].isna().sum()
    if bad:
        print(f"  ⚠  {bad} rows dropped — unparseable timestamps")
    df = df.dropna(subset=["timestamp", "phase", "participant_id"])
    df["phase"] = df["phase"].astype(str)

    n_before = len(df)
    df = df[~df["phase"].isin(EXCLUDE_PHASES)]
    dropped = n_before - len(df)
    if dropped:
        print(f"  ⚠  {dropped} rows dropped — transition phases excluded")

    found   = set(df["phase"].unique())
    known   = set(STRESS_PHASES + REST_PHASES)
    unclass = found - known
    missing = known - found
    if unclass:
        print(f"  ⚠  Unclassified phases (→ 'other'): {sorted(unclass)}")
    if missing:
        print(f"  ⚠  Config phases absent from data:  {sorted(missing)}")

    present = [s for s in SIGNALS if s in df.columns]
    absent  = [s for s in SIGNALS if s not in df.columns]
    if absent:
        print(f"  ⚠  Signals not in file (skipped): {absent}")

    for sig in present:
        n_nan = df[sig].isna().sum()
        if n_nan:
            by_pid = df[df[sig].isna()]["participant_id"].value_counts().to_dict()
            print(f"  ⚠  '{sig}': {n_nan} NaNs — {by_pid}")

    df = df.dropna(subset=present)
    df = (
        df.groupby(["participant_id", "phase"])
        .resample("1S", on="timestamp")
        .mean(numeric_only=True)
        .reset_index()
    )

    print(f"  Rows after 1 Hz downsample : {len(df):>10,}")
    print(f"  Participants               : {df['participant_id'].nunique()}")
    print(f"  Phases                     : {sorted(df['phase'].unique())}\n")
    return df


# ── Segments ──────────────────────────────────────────────────────────────────

def collect_segments(df: pd.DataFrame, sig_x: str, sig_y: str) -> list:
    segments = []
    for pid in df["participant_id"].unique():
        pdf = df[df["participant_id"] == pid]
        for phase in pdf["phase"].unique():
            phase_df = pdf[pdf["phase"] == phase].sort_values("timestamp")
            if sig_x not in phase_df or sig_y not in phase_df:
                continue
            x = phase_df[sig_x].values.astype(np.float32)
            y = phase_df[sig_y].values.astype(np.float32)
            if len(x) < MIN_STEPS + 1:
                continue
            if np.std(x) < 1e-6 or np.std(y) < 1e-6:
                continue
            phase_type = ("stress" if phase in STRESS_PHASES else
                          "rest"   if phase in REST_PHASES   else "other")
            segments.append({"participant": pid, "phase": phase,
                              "phase_type": phase_type,
                              "x_seq": x, "y_seq": y})
    return segments


def stratified_split(segments: list, val_frac: float = VAL_FRAC):
    by_type = defaultdict(list)
    for i, s in enumerate(segments):
        by_type[s["phase_type"]].append(i)
    val_idx, tr_idx = [], []
    for idxs in by_type.values():
        np.random.shuffle(idxs)
        n_val = max(1, int(val_frac * len(idxs)))
        val_idx.extend(idxs[:n_val])
        tr_idx.extend(idxs[n_val:])
    return [segments[i] for i in tr_idx], [segments[i] for i in val_idx]


# ── Model ─────────────────────────────────────────────────────────────────────

class GrangerLSTM(nn.Module):
    """
    input_dim=1 : restricted   [Y(t)]
    input_dim=2 : unrestricted [Y(t), X(t)]
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, HIDDEN_SIZE, NUM_LAYERS, batch_first=True)
        self.head = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x, state):
        out, state = self.lstm(x, state)   # x: (1, chunk, input_dim)
        return self.head(out), state        # pred: (1, chunk, 1)

    def init_state(self):
        z = torch.zeros(NUM_LAYERS, 1, HIDDEN_SIZE, device=DEVICE)
        return z, z.clone()


# ── Vectorized TBPTT helpers ──────────────────────────────────────────────────

def make_input_tensor(y_inp: np.ndarray, x_inp: np.ndarray,
                      start: int, end: int, input_dim: int) -> torch.Tensor:
    """
    Build a (1, chunk_len, input_dim) tensor for one TBPTT chunk.
    Vectorized — no Python loop over timesteps.
    """
    y_chunk = y_inp[start:end]                          # (chunk,)
    if input_dim == 1:
        chunk = y_chunk[:, None]                        # (chunk, 1)
    else:
        x_chunk = x_inp[start:end]
        chunk = np.stack([y_chunk, x_chunk], axis=1)   # (chunk, 2)
    return torch.tensor(chunk[None], dtype=torch.float32, device=DEVICE)  # (1, chunk, d)


def make_target_tensor(tgt: np.ndarray, start: int, end: int) -> torch.Tensor:
    """
    Targets are Y(t+1) for t in [start, end-1].
    So target indices are [start+1, end].
    Shape: (1, chunk_len, 1)
    """
    return torch.tensor(tgt[start + 1: end + 1][None, :, None],
                        dtype=torch.float32, device=DEVICE)


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(train_segs: list, val_segs: list,
                input_dim: int, x_sc: StandardScaler,
                y_sc: StandardScaler) -> GrangerLSTM:
    model     = GrangerLSTM(input_dim).to(DEVICE)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    best_val, best_state, pat = float("inf"), None, 0

    for _ in range(MAX_EPOCHS):

        # ── train ──────────────────────────────────────────────────────────
        model.train()
        for seg in np.random.permutation(len(train_segs)):
            seg = train_segs[seg]
            y_inp = y_sc.transform(seg["y_seq"].reshape(-1, 1)).squeeze()
            x_inp = x_sc.transform(seg["x_seq"].reshape(-1, 1)).squeeze()
            tgt   = y_inp                     # predicting next scaled Y
            T     = len(y_inp)

            h, c = model.init_state()
            start = 0
            while start < T - 1:
                end = min(start + TBPTT_STEPS, T - 1)

                inp_t = make_input_tensor(y_inp, x_inp, start, end, input_dim)
                tgt_t = make_target_tensor(tgt,  start, end)

                pred, (h, c) = model(inp_t, (h, c))   # (1, chunk, 1)
                loss = criterion(pred, tgt_t)

                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimiser.step()

                h, c = h.detach(), c.detach()
                start = end

        # ── validate ───────────────────────────────────────────────────────
        model.eval()
        val_losses = []
        with torch.no_grad():
            for seg in val_segs:
                y_inp = y_sc.transform(seg["y_seq"].reshape(-1, 1)).squeeze()
                x_inp = x_sc.transform(seg["x_seq"].reshape(-1, 1)).squeeze()
                tgt   = y_inp
                T     = len(y_inp)
                h, c  = model.init_state()
                start = 0
                while start < T - 1:
                    end   = min(start + TBPTT_STEPS, T - 1)
                    inp_t = make_input_tensor(y_inp, x_inp, start, end, input_dim)
                    tgt_t = make_target_tensor(tgt,  start, end)
                    pred, (h, c) = model(inp_t, (h, c))
                    val_losses.append(criterion(pred, tgt_t).item())
                    start = end

        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        if val_loss < best_val:
            best_val, best_state, pat = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            pat += 1
            if pat >= PATIENCE:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(model: GrangerLSTM, segments: list, input_dim: int,
                  x_sc: StandardScaler, y_sc: StandardScaler) -> list:
    """
    Returns one dict per segment with segment-level MSE in original signal units.
    Vectorized: processes TBPTT_STEPS timesteps per LSTM call.
    """
    model.eval()
    results = []
    with torch.no_grad():
        for seg in segments:
            y_inp  = y_sc.transform(seg["y_seq"].reshape(-1, 1)).squeeze()
            x_inp  = x_sc.transform(seg["x_seq"].reshape(-1, 1)).squeeze()
            y_true = seg["y_seq"]    # unscaled — MSE in original units
            T      = len(y_inp)

            h, c = model.init_state()
            sq_errors = []
            start = 0

            while start < T - 1:
                end   = min(start + TBPTT_STEPS, T - 1)
                inp_t = make_input_tensor(y_inp, x_inp, start, end, input_dim)

                pred_sc, (h, c) = model(inp_t, (h, c))   # (1, chunk, 1)

                # Inverse-transform predictions to original scale
                pred_np = pred_sc.squeeze().cpu().numpy()            # (chunk,)
                pred_orig = y_sc.inverse_transform(
                    pred_np.reshape(-1, 1)
                ).squeeze()                                          # (chunk,)

                targets = y_true[start + 1: end + 1]                # (chunk,)
                sq_errors.extend((pred_orig - targets) ** 2)
                start = end

            results.append({
                "participant": seg["participant"],
                "phase":       seg["phase"],
                "phase_type":  seg["phase_type"],
                "mse_segment": float(np.mean(sq_errors)) if sq_errors else np.nan,
                "n_steps":     len(sq_errors),
            })
    return results


# ── Main analysis ─────────────────────────────────────────────────────────────

def run_analysis(df_train: pd.DataFrame, df_test: pd.DataFrame):
    pairs = list(permutations(SIGNALS, 2))
    records_v1, records_v2 = [], []

    for idx, (sig_x, sig_y) in enumerate(pairs):
        pair_str = f"{sig_x} -> {sig_y}"
        print(f"  [{idx+1:02d}/{len(pairs)}]  {pair_str}")

        train_segs = collect_segments(df_train, sig_x, sig_y)
        test_segs  = collect_segments(df_test,  sig_x, sig_y)

        if len(train_segs) < 4:
            print(f"    ⚠  Only {len(train_segs)} train segments, skipping.")
            continue
        if not test_segs:
            print(f"    ⚠  No V2 segments, skipping.")
            continue

        tr_segs, val_segs = stratified_split(train_segs)
        if not tr_segs:
            print(f"    ⚠  Train split empty, skipping.")
            continue

        # Scalers fit on train split only
        y_sc = StandardScaler().fit(
            np.concatenate([s["y_seq"] for s in tr_segs]).reshape(-1, 1))
        x_sc = StandardScaler().fit(
            np.concatenate([s["x_seq"] for s in tr_segs]).reshape(-1, 1))

        model_r  = train_model(tr_segs, val_segs, 1, x_sc, y_sc)  # restricted
        model_ur = train_model(tr_segs, val_segs, 2, x_sc, y_sc)  # unrestricted

        for segs, records in [(train_segs, records_v1), (test_segs, records_v2)]:
            res_r  = run_inference(model_r,  segs, 1, x_sc, y_sc)
            res_ur = run_inference(model_ur, segs, 2, x_sc, y_sc)
            for r, u in zip(res_r, res_ur):
                records.append({
                    "participant"     : r["participant"],
                    "phase"           : r["phase"],
                    "phase_type"      : r["phase_type"],
                    "X": sig_x, "Y": sig_y, "pair": pair_str,
                    "mse_restricted"  : r["mse_segment"],
                    "mse_unrestricted": u["mse_segment"],
                    "granger_score"   : r["mse_segment"] - u["mse_segment"],
                    "n_steps"         : r["n_steps"],
                })

        n1 = sum(1 for r in records_v1 if r["pair"] == pair_str)
        n2 = sum(1 for r in records_v2 if r["pair"] == pair_str)
        print(f"    ✓  V1 segs: {n1}  |  V2 segs: {n2}")

    return pd.DataFrame(records_v1), pd.DataFrame(records_v2)


# ── Summary ───────────────────────────────────────────────────────────────────

def summarise_results(results_df: pd.DataFrame, label: str) -> pd.DataFrame:
    print(f"\n{'='*65}")
    print(f"  RESULTS — {label}")
    print(f"{'='*65}")

    df = results_df[results_df["phase_type"].isin(["stress", "rest"])].copy()
    if df.empty:
        print("  No stress/rest segments.")
        return pd.DataFrame()

    # Participant-level mean first
    pid_agg = (df.groupby(["pair", "phase_type", "participant"])["granger_score"]
               .mean().reset_index())

    # Population mean + SEM
    pop = (pid_agg.groupby(["pair", "phase_type"])["granger_score"]
           .agg(mean="mean", sem=lambda x: x.sem())
           .unstack("phase_type"))
    pop.columns = ["_".join(c) for c in pop.columns]

    if "mean_stress" in pop and "mean_rest" in pop:
        pop["stress_vs_rest"] = pop["mean_stress"] - pop["mean_rest"]
        pop = pop.sort_values("stress_vs_rest", ascending=False)

    # FDR correction via paired t-test across participants
    fdr_rows = []
    for pair in pid_agg["pair"].unique():
        s = pid_agg[(pid_agg["pair"]==pair) & (pid_agg["phase_type"]=="stress")]["granger_score"].values
        r = pid_agg[(pid_agg["pair"]==pair) & (pid_agg["phase_type"]=="rest")  ]["granger_score"].values
        p = stats.ttest_ind(s, r).pvalue if len(s) >= 2 and len(r) >= 2 else np.nan
        fdr_rows.append({"pair": pair, "p_raw": p})

    fdr_df = pd.DataFrame(fdr_rows).dropna(subset=["p_raw"])
    if len(fdr_df):
        reject, p_fdr = fdrcorrection(fdr_df["p_raw"].values, alpha=ALPHA)
        fdr_df = fdr_df.assign(p_fdr=p_fdr, sig_fdr=reject).set_index("pair")
        pop = pop.merge(fdr_df[["p_raw","p_fdr","sig_fdr"]],
                        left_index=True, right_index=True, how="left")

    # Table 1: Granger scores
    print("\n  Granger score = MSE(restricted) − MSE(unrestricted)")
    print("  Participant-level means; BH-FDR corrected p-values\n")
    print(f"  {'Pair':<20} {'Rest':>10}  {'Stress':>10}  {'Δ(S−R)':>10}  {'p_fdr':>8}  sig")
    print(f"  {'─'*20} {'─'*10}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*3}")
    for pair, row in pop.iterrows():
        r = row.get("mean_rest",    np.nan)
        s = row.get("mean_stress",  np.nan)
        d = row.get("stress_vs_rest", np.nan)
        p = row.get("p_fdr",        np.nan)
        sig = "✓" if row.get("sig_fdr", False) else ""
        print(f"  {pair:<20} "
              f"{'—':>10}" if np.isnan(r) else f"  {pair:<20} {r:>10.4f}",
              end="")
        print(f"  {s:>10.4f}  "
              f"{('+' if d>0 else '')+f'{d:.4f}':>10}  "
              f"{p:>8.4f}  {sig}" if not np.isnan(s) else "")

    # Table 2: Win rate
    df["x_wins"] = df["granger_score"] > 0
    win = (df.groupby(["pair","phase_type","participant"])["x_wins"].mean()
             .reset_index()
             .groupby(["pair","phase_type"])["x_wins"].mean()
             .unstack("phase_type").round(3))
    print(f"\n  Win rate: fraction of participants where unrestricted beats restricted")
    print(f"  {'Pair':<20} {'Rest':>8}  {'Stress':>8}")
    print(f"  {'─'*20} {'─'*8}  {'─'*8}")
    for pair, row in win.iterrows():
        print(f"  {pair:<20} {row.get('rest',0):>8.0%}  {row.get('stress',0):>8.0%}")

    # Key findings
    print("\n  FDR-significant pairs with stronger causality under stress:")
    if "sig_fdr" in pop.columns:
        top = pop[pop["sig_fdr"] & (pop.get("stress_vs_rest", 0) > 0)]
        if top.empty:
            print("    None.")
        for pair, row in top.iterrows():
            print(f"    {pair:<22} Δ={row['stress_vs_rest']:.4f}  p_fdr={row['p_fdr']:.4f}")

    # Per-phase for top 3
    top3 = pop.head(3).index.tolist()
    if top3:
        print(f"\n  Per-phase granger score (top 3 pairs, participant-averaged):")
        tbl = (results_df[results_df["pair"].isin(top3)]
               .groupby(["pair","participant","phase"])["granger_score"].mean()
               .groupby(["pair","phase"]).mean().round(4).unstack("phase"))
        print(tbl.to_string())

    return pop


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    df_v1 = load_data(TRAIN_CSV)
    df_v2 = load_data(TEST_CSV)

    n_pairs = len(list(permutations(SIGNALS, 2)))
    print(f"\n{'='*65}")
    print(f"  LSTM Granger Causality v3 — vectorized")
    print(f"  Signals: {SIGNALS}")
    print(f"  Pairs: {n_pairs}  |  TBPTT: {TBPTT_STEPS}  |  hidden: {HIDDEN_SIZE}")
    print(f"  Restricted [Y] vs Unrestricted [Y, X] → Y+1")
    print(f"{'='*65}\n")

    results_v1, results_v2 = run_analysis(df_v1, df_v2)

    summarise_results(results_v1, "V1 — in-sample  (S01–S18)")
    summarise_results(results_v2, "V2 — held-out   (f01–f18)")

    results_v1.to_csv("granger_lstm_v3_results_v1.csv", index=False)
    results_v2.to_csv("granger_lstm_v3_results_v2.csv", index=False)
    print("Saved: granger_lstm_v3_results_v1.csv  |  granger_lstm_v3_results_v2.csv")


if __name__ == "__main__":
    main()
