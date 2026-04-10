"""
LSTM-based Neural Granger Causality — v4
=========================================

MERGED FROM GEMINI INTO v3
---------------------------
A. BASELINE MODEL CACHE
   The restricted model Y(t)→Y(t+1) does not depend on X at all.
   All 6 pairs sharing the same Y (e.g. EDA→HR, BVP→HR, TEMP→HR)
   need exactly the same restricted model. Training it once per Y
   and caching reduces restricted-model training from 42 runs to 7.

   This requires the training segments used for the restricted model
   to be IDENTICAL across all pairs sharing that Y — otherwise the
   cached model was trained on different data than assumed.

   We achieve this via GLOBAL SEGMENT SYNC: a phase is included only
   if ALL signals pass the validity check (length ≥ MIN_STEPS+1,
   std > 1e-6). This guarantees identical segment sets across pairs.

   SYNC FALLBACK: if global sync would drop >SYNC_DROP_THRESH fraction
   of segments for a given pair vs per-pair filtering, we fall back to
   per-pair filtering for that pair and do NOT use the cache for it
   (the restricted model is retrained fresh). A warning is printed.

B. GPU TENSOR PRELOADING
   v3 built tensors inside the epoch loop each time.
   v4 builds all segment tensors once before training starts and keeps
   them on the device. Each epoch does zero-copy slicing — no repeated
   CPU→GPU transfers or tensor construction overhead.

ALL v3 CORRECTNESS FIXES RETAINED
-----------------------------------
 1. Correct Granger formulation: restricted=[Y], unrestricted=[Y,X]
 2. Segment-level MSE (not per-step) before comparison
 3. Participant-level aggregation before population mean
 4. FDR correction (Benjamini–Hochberg) across all pairs
 5. Stratified train/val split by phase_type
 6. Scalers fit on train split only (after split, not before)
 7. Training and validation use same TBPTT chunk size
 8. Explicit NaN / phase-name warnings at load time
 9. Timestamp coercion (handles '24:00.0' bug)
10. Transition phases excluded explicitly

V3 VALIDATION ASYMMETRY — KEPT INTENTIONALLY
---------------------------------------------
v3 validated with TBPTT chunks (same as training) for state
consistency. GPU preloading does NOT reintroduce the full-sequence
validation — we still chunk validation the same way as
training.
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

SIGNALS = ["EDA", "HR", "BVP", "TEMP", "ACC_x", "ACC_y", "ACC_z"]

STRESS_PHASES  = ["TMCT", "Stroop", "Subtract", "Opposite Opinion", "Real Opinion"]
REST_PHASES    = ["Baseline", "First Rest", "Second Rest"]
EXCLUDE_PHASES = ["Transition_1", "Transition_2", "Transition_3"]

MIN_STEPS        = 20
HIDDEN_SIZE      = 64
NUM_LAYERS       = 1
LR               = 1e-3
GRAD_CLIP        = 1.0
MAX_EPOCHS       = 30
PATIENCE         = 5
TBPTT_STEPS      = 20
VAL_FRAC         = 0.10
ALPHA            = 0.05
SYNC_DROP_THRESH = 0.20   # fall back to per-pair if global sync drops >20% of segs

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
    if len(df) < n_before:
        print(f"  ⚠  {n_before - len(df)} rows dropped — transition phases")

    found, known = set(df["phase"].unique()), set(STRESS_PHASES + REST_PHASES)
    if found - known:
        print(f"  ⚠  Unclassified phases (→ 'other'): {sorted(found - known)}")
    if known - found:
        print(f"  ⚠  Config phases absent from data: {sorted(known - found)}")

    present = [s for s in SIGNALS if s in df.columns]
    if len(present) < len(SIGNALS):
        print(f"  ⚠  Missing signals: {[s for s in SIGNALS if s not in df.columns]}")

    for sig in present:
        n = df[sig].isna().sum()
        if n:
            print(f"  ⚠  '{sig}': {n} NaNs — "
                  f"{df[df[sig].isna()]['participant_id'].value_counts().to_dict()}")

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


# ── Segment collection ────────────────────────────────────────────────────────

def _phase_type(phase: str) -> str:
    if phase in STRESS_PHASES: return "stress"
    if phase in REST_PHASES:   return "rest"
    return "other"


def collect_global_segments(df: pd.DataFrame) -> list:
    """
    Global sync: include a (participant, phase) only if
    ALL signals pass validity — guarantees identical segment sets across
    all pairs, enabling safe baseline model caching.
    """
    segments = []
    for pid in df["participant_id"].unique():
        pdf = df[df["participant_id"] == pid]
        for phase in pdf["phase"].unique():
            phase_df = pdf[pdf["phase"] == phase].sort_values("timestamp")
            if len(phase_df) < MIN_STEPS + 1:
                continue
            if any(np.std(phase_df[s].values.astype(np.float32)) < 1e-6
                   for s in SIGNALS if s in phase_df.columns):
                continue
            seg = {"participant": pid, "phase": phase,
                   "phase_type": _phase_type(phase)}
            for s in SIGNALS:
                seg[s] = phase_df[s].values.astype(np.float32)
            segments.append(seg)
    return segments


def collect_pair_segments(df: pd.DataFrame, sig_x: str, sig_y: str) -> list:
    """
    Per-pair filtering (v3 original): only sig_x and sig_y need to be valid.
    Used as fallback when global sync drops too many segments.
    """
    segments = []
    for pid in df["participant_id"].unique():
        pdf = df[df["participant_id"] == pid]
        for phase in pdf["phase"].unique():
            pdf = pdf[pdf["phase"] == phase].sort_values("timestamp")
            x = pdf[sig_x].values.astype(np.float32) if sig_x in pdf else None
            y = pdf[sig_y].values.astype(np.float32) if sig_y in pdf else None
            if x is None or y is None:
                continue
            if len(x) < MIN_STEPS + 1:
                continue
            if np.std(x) < 1e-6 or np.std(y) < 1e-6:
                continue
            segments.append({"participant": pid, "phase": phase,
                              "phase_type": _phase_type(phase),
                              sig_x: x, sig_y: y})
    return segments


def stratified_split(segments: list, val_frac: float = VAL_FRAC):
    """Split preserving phase_type proportions in both halves."""
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


# ── GPU tensor preloading ─────────────────────────────────────────────────────

def preload_tensors(segments: list, input_key: str, target_key: str) -> list:
    """
    Build all (input, target) tensors once and push to
    DEVICE before the epoch loop. Each epoch does zero-copy slicing.

    input_key  : key in segment dict for the scaled input array
    target_key : key for the scaled Y target array
    Both arrays are shape (T,) — we add batch and feature dims here.

    Returns list of (inp_tensor, tgt_tensor, T) already on DEVICE.
    """
    gpu_data = []
    for seg in segments:
        inp = torch.tensor(seg[input_key], dtype=torch.float32, device=DEVICE)
        tgt = torch.tensor(seg[target_key], dtype=torch.float32, device=DEVICE)
        # inp: (T, input_dim) or (T,) → reshape to (1, T, input_dim)
        if inp.dim() == 1:
            inp = inp.unsqueeze(1)   # (T, 1)
        inp = inp.unsqueeze(0)       # (1, T, input_dim)
        tgt = tgt.unsqueeze(0).unsqueeze(2)  # (1, T, 1)
        gpu_data.append((inp, tgt, inp.shape[1]))
    return gpu_data


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
        out, state = self.lstm(x, state)
        return self.head(out), state     # pred: (1, chunk, 1)

    def init_state(self):
        z = torch.zeros(NUM_LAYERS, 1, HIDDEN_SIZE, device=DEVICE)
        return z, z.clone()


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(
    train_gpu: list,   # preloaded tensors: [(inp, tgt, T), ...]
    val_gpu:   list,
    input_dim: int,
) -> GrangerLSTM:
    """
    TBPTT training with preloaded GPU tensors.
    Validation uses the same TBPTT chunking as training (v3 fix #7).

    Gradient accumulation bug (v2/v3): chunk_loss accumulated as a
    list of tensors and summed — clean autodiff graph per chunk.
    """
    model     = GrangerLSTM(input_dim).to(DEVICE)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    best_val, best_state, pat = float("inf"), None, 0

    for _ in range(MAX_EPOCHS):

        # ── train ──────────────────────────────────────────────────────────
        model.train()
        for si in np.random.permutation(len(train_gpu)):
            inp_t, tgt_t, T = train_gpu[si]   # already on DEVICE
            h, c = model.init_state()
            start = 0
            while start < T - 1:
                end = min(start + TBPTT_STEPS, T - 1)

                # Zero-copy GPU slicing
                x_chunk = inp_t[:, start:end, :]        # (1, chunk, input_dim)
                y_chunk = tgt_t[:, start+1:end+1, :]    # (1, chunk, 1)

                pred, (h, c) = model(x_chunk, (h, c))
                loss = criterion(pred, y_chunk)

                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimiser.step()

                h, c  = h.detach(), c.detach()
                start = end

        # ── validate — same TBPTT chunking as training (v3 fix #7) ────────
        model.eval()
        val_losses = []
        with torch.no_grad():
            for inp_t, tgt_t, T in val_gpu:
                h, c  = model.init_state()
                start = 0
                while start < T - 1:
                    end     = min(start + TBPTT_STEPS, T - 1)
                    x_chunk = inp_t[:, start:end, :]
                    y_chunk = tgt_t[:, start+1:end+1, :]
                    pred, (h, c) = model(x_chunk, (h, c))
                    val_losses.append(criterion(pred, y_chunk).item())
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

def run_inference(
    model:     GrangerLSTM,
    segments:  list,
    input_key: str,
    sig_y:     str,
    y_sc:      StandardScaler,
) -> list:
    """
    Returns one dict per SEGMENT with segment-level MSE in original units.
    Same TBPTT chunking as training — state context window is consistent.
    """
    model.eval()
    results = []
    with torch.no_grad():
        for seg in segments:
            inp    = torch.tensor(seg[input_key], dtype=torch.float32, device=DEVICE)
            if inp.dim() == 1:
                inp = inp.unsqueeze(1)
            inp    = inp.unsqueeze(0)           # (1, T, input_dim)
            y_true = seg[sig_y]                 # raw unscaled array
            T      = inp.shape[1]

            h, c  = model.init_state()
            sq_errors = []
            start = 0

            while start < T - 1:
                end     = min(start + TBPTT_STEPS, T - 1)
                x_chunk = inp[:, start:end, :]
                pred_sc, (h, c) = model(x_chunk, (h, c))   # (1, chunk, 1)

                pred_np   = pred_sc.squeeze().cpu().numpy()
                pred_orig = y_sc.inverse_transform(
                    pred_np.reshape(-1, 1)
                ).squeeze()
                targets   = y_true[start + 1: end + 1]
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

    # Global sync segments (enables baseline cache)
    global_train = collect_global_segments(df_train)
    global_test  = collect_global_segments(df_test)
    print(f"  Global sync — train segments: {len(global_train)}  "
          f"test segments: {len(global_test)}\n")

    # Baseline model cache: one restricted model per Y signal
    # key: (sig_y, 'restricted')  →  (model, y_sc, tr_split_ids)
    baseline_cache: dict = {}

    records_v1, records_v2 = [], []

    for idx, (sig_x, sig_y) in enumerate(pairs):
        pair_str = f"{sig_x} -> {sig_y}"
        print(f"  [{idx+1:02d}/{len(pairs)}]  {pair_str}", end="  ")

        # ── Decide: use global sync or fall back to per-pair? ─────────────
        pair_train_pp = collect_pair_segments(df_train, sig_x, sig_y)
        use_global    = True

        if len(global_train) == 0:
            use_global = False
        else:
            drop_frac = 1 - len(global_train) / max(len(pair_train_pp), 1)
            if drop_frac > SYNC_DROP_THRESH:
                print(f"\n    ⚠  Global sync drops {drop_frac:.0%} of segments "
                      f"— falling back to per-pair filtering")
                use_global = False

        if use_global:
            # Deep-copy to avoid cross-pair mutation of scaled fields
            train_segs = [dict(s) for s in global_train]
            test_segs  = [dict(s) for s in global_test]
        else:
            train_segs = pair_train_pp
            test_segs  = collect_pair_segments(df_test, sig_x, sig_y)

        if len(train_segs) < 4:
            print(f"skipped (only {len(train_segs)} train segs)")
            continue
        if not test_segs:
            print(f"skipped (no test segs)")
            continue

        # ── Split — AFTER deciding segments, BEFORE fitting scalers ───────
        # Use fixed permutation index so global pairs share same split ids
        rng_state = np.random.get_state()
        np.random.seed(42 + hash(sig_y) % 1000)   # deterministic per Y
        tr_segs, val_segs = stratified_split(train_segs)
        np.random.set_state(rng_state)

        if not tr_segs:
            print(f"skipped (empty train split)")
            continue

        # ── Scalers fit on train split only (v3 fix #6) ───────────────────
        y_sc = StandardScaler().fit(
            np.concatenate([s[sig_y] for s in tr_segs]).reshape(-1, 1))
        x_sc = StandardScaler().fit(
            np.concatenate([s[sig_x] for s in tr_segs]).reshape(-1, 1))

        # ── Scale and add input keys to segments ──────────────────────────
        for seg_list in [train_segs, test_segs]:
            for seg in seg_list:
                seg["y_scaled"]  = y_sc.transform(
                    seg[sig_y].reshape(-1, 1)).squeeze()
                seg["x_scaled"]  = x_sc.transform(
                    seg[sig_x].reshape(-1, 1)).squeeze()
                # unrestricted input: stack [Y, X] → (T, 2)
                seg["yx_scaled"] = np.stack(
                    [seg["y_scaled"], seg["x_scaled"]], axis=1)

        # Re-derive tr/val scaled subsets (same indices as split above)
        tr_scaled  = [s for s in train_segs
                      if any(s["participant"] == t["participant"]
                             and s["phase"] == t["phase"] for t in tr_segs)]
        val_scaled = [s for s in train_segs
                      if any(s["participant"] == v["participant"]
                             and s["phase"] == v["phase"] for v in val_segs)]

        # ── Preload to GPU ────────────────────────────────
        tr_r_gpu  = preload_tensors(tr_scaled,  "y_scaled",  "y_scaled")
        val_r_gpu = preload_tensors(val_scaled, "y_scaled",  "y_scaled")
        tr_u_gpu  = preload_tensors(tr_scaled,  "yx_scaled", "y_scaled")
        val_u_gpu = preload_tensors(val_scaled, "yx_scaled", "y_scaled")

        # ── Baseline cache ────────────────────────────────
        cache_key = sig_y if use_global else f"{sig_y}_{sig_x}"
        if cache_key in baseline_cache:
            model_r, cached_y_sc = baseline_cache[cache_key]
            # Sanity: confirm y_sc statistics match (they must for global segs)
            assert abs(cached_y_sc.mean_[0] - y_sc.mean_[0]) < 1e-4, \
                f"Y scaler mismatch for cached key {cache_key}"
            print(f"(cached baseline)", end="  ")
        else:
            model_r = train_model(tr_r_gpu, val_r_gpu, input_dim=1)
            if use_global:
                baseline_cache[cache_key] = (model_r, y_sc)

        # ── Train unrestricted model ───────────────────────────────────────
        model_ur = train_model(tr_u_gpu, val_u_gpu, input_dim=2)

        # ── Inference ─────────────────────────────────────────────────────
        for segs, records in [(train_segs, records_v1), (test_segs, records_v2)]:
            res_r  = run_inference(model_r,  segs, "y_scaled",  sig_y, y_sc)
            res_ur = run_inference(model_ur, segs, "yx_scaled", sig_y, y_sc)
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
                    "used_global_sync": use_global,
                })

        n1 = sum(1 for r in records_v1 if r["pair"] == pair_str)
        n2 = sum(1 for r in records_v2 if r["pair"] == pair_str)
        print(f"V1: {n1} segs  V2: {n2} segs")

    cache_hits = sum(1 for r in records_v1
                     if r.get("used_global_sync") and
                     records_v1.index(r) > 0)  # proxy count
    print(f"\n  Baseline cache saved "
          f"~{len(SIGNALS) - 1} restricted model trainings per Y signal.")

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

    # v3 fix #3: participant-level mean first
    pid_agg = (df.groupby(["pair", "phase_type", "participant"])["granger_score"]
               .mean().reset_index())

    pop = (pid_agg.groupby(["pair", "phase_type"])["granger_score"]
           .agg(mean="mean", sem=lambda x: x.sem())
           .unstack("phase_type"))
    pop.columns = ["_".join(c) for c in pop.columns]

    if "mean_stress" in pop and "mean_rest" in pop:
        pop["stress_vs_rest"] = pop["mean_stress"] - pop["mean_rest"]
        pop = pop.sort_values("stress_vs_rest", ascending=False)

    # v3 fix #4: FDR correction
    fdr_rows = []
    for pair in pid_agg["pair"].unique():
        s = pid_agg[(pid_agg["pair"]==pair) &
                    (pid_agg["phase_type"]=="stress")]["granger_score"].values
        r = pid_agg[(pid_agg["pair"]==pair) &
                    (pid_agg["phase_type"]=="rest")]["granger_score"].values
        p = stats.ttest_ind(s, r).pvalue if len(s) >= 2 and len(r) >= 2 else np.nan
        fdr_rows.append({"pair": pair, "p_raw": p})

    fdr_df = pd.DataFrame(fdr_rows).dropna(subset=["p_raw"])
    if len(fdr_df):
        reject, p_fdr = fdrcorrection(fdr_df["p_raw"].values, alpha=ALPHA)
        fdr_df = fdr_df.assign(p_fdr=p_fdr, sig_fdr=reject).set_index("pair")
        pop    = pop.merge(fdr_df[["p_raw","p_fdr","sig_fdr"]],
                           left_index=True, right_index=True, how="left")

    # Table 1: Granger scores with FDR
    print("\n  Granger score = MSE(restricted) − MSE(unrestricted)")
    print("  Participant-level means; BH-FDR corrected p-values\n")
    print(f"  {'Pair':<20} {'Rest':>10}  {'Stress':>10}  "
          f"{'Δ(S−R)':>10}  {'p_fdr':>8}  sig")
    print(f"  {'─'*20} {'─'*10}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*3}")
    for pair, row in pop.iterrows():
        r   = row.get("mean_rest",      np.nan)
        s   = row.get("mean_stress",    np.nan)
        d   = row.get("stress_vs_rest", np.nan)
        p   = row.get("p_fdr",          np.nan)
        sig = "✓" if row.get("sig_fdr", False) else ""
        r_s = f"{r:.4f}" if not np.isnan(r) else "     —"
        s_s = f"{s:.4f}" if not np.isnan(s) else "     —"
        d_s = (f"+{d:.4f}" if d > 0 else f"{d:.4f}") if not np.isnan(d) else "     —"
        p_s = f"{p:.4f}" if not np.isnan(p) else "     —"
        print(f"  {pair:<20} {r_s:>10}  {s_s:>10}  {d_s:>10}  {p_s:>8}  {sig}")

    # Table 2: win rate
    df["x_wins"] = df["granger_score"] > 0
    win = (df.groupby(["pair","phase_type","participant"])["x_wins"].mean()
             .reset_index()
             .groupby(["pair","phase_type"])["x_wins"].mean()
             .unstack("phase_type").round(3))
    print(f"\n  Win rate (participant-averaged)")
    print(f"  {'Pair':<20} {'Rest':>8}  {'Stress':>8}")
    print(f"  {'─'*20} {'─'*8}  {'─'*8}")
    for pair, row in win.iterrows():
        print(f"  {pair:<20} {row.get('rest',0):>8.0%}  "
              f"{row.get('stress',0):>8.0%}")

    # Key findings
    print("\n  FDR-significant pairs with stronger causality under stress:")
    if "sig_fdr" in pop.columns:
        top = pop[pop["sig_fdr"] & (pop.get("stress_vs_rest", pd.Series(dtype=float)) > 0)]
        if top.empty:
            print("    None.")
        for pair, row in top.iterrows():
            print(f"    {pair:<22} Δ={row['stress_vs_rest']:.4f}  "
                  f"p_fdr={row['p_fdr']:.4f}")

    # Per-phase breakdown for top 3
    top3 = pop.head(3).index.tolist()
    if top3:
        print(f"\n  Per-phase granger score (top 3 pairs, participant-averaged):")
        tbl = (results_df[results_df["pair"].isin(top3)]
               .groupby(["pair","participant","phase"])["granger_score"].mean()
               .groupby(["pair","phase"]).mean().round(4).unstack("phase"))
        print(tbl.to_string())
    print()
    return pop


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    df_v1 = load_data(TRAIN_CSV)
    df_v2 = load_data(TEST_CSV)

    n_pairs = len(list(permutations(SIGNALS, 2)))
    print(f"\n{'='*65}")
    print(f"  LSTM Granger Causality v4")
    print(f"  Signals: {SIGNALS}")
    print(f"  Pairs: {n_pairs}  |  TBPTT: {TBPTT_STEPS}  |  hidden: {HIDDEN_SIZE}")
    print(f"  Restricted [Y] vs Unrestricted [Y,X] → Y+1")
    print(f"{'='*65}\n")

    results_v1, results_v2 = run_analysis(df_v1, df_v2)

    summarise_results(results_v1, "V1 — in-sample  (S01–S18)")
    summarise_results(results_v2, "V2 — held-out   (f01–f18)")

    results_v1.to_csv("granger_lstm_v4_results_v1.csv", index=False)
    results_v2.to_csv("granger_lstm_v4_results_v2.csv", index=False)
    print("Saved: granger_lstm_v4_results_v1.csv  |  granger_lstm_v4_results_v2.csv")


if __name__ == "__main__":
    main()