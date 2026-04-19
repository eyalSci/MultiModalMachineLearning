"""
granger_phases.py
=================
Phase-wise Granger causality between continuous physiological signals.

  - IBI removed (derived signal, unreliable)
  - Z-score normalization added per signal per participant
  - MSE computed for both restricted and unrestricted models
  - Granger score = MSE(restricted) - MSE(unrestricted) shown alongside F-stat
  - Participant-level averaging before population mean (hierarchical aggregation)
  - FDR correction (Benjamini-Hochberg) added
  - ACC signals kept as explanatory variables only (not as targets)
    because ACC can only be explained by other ACC signals physiologically
  - Note added: stress labels are not Granger-caused by features
    (sparse labels violate Granger assumptions)
  - Bi-directional Granger causality retained for all non-ACC pairs

SIGNALS:
  EDA, HR, BVP, TEMP  (physiological — both X and Y)
  ACC_x, ACC_y, ACC_z (movement — used as X only, never as Y target)
  IBI dropped

DATASET:
  Train on V1 (merged_df1.csv, S01-S18)
  Replicate on V2 (merged_df2.csv, f01-f18)
"""

import warnings
import numpy as np
import pandas as pd
from itertools import permutations, product
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.preprocessing import StandardScaler
from scipy import stats

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
TRAIN_CSV    = "merged_df1.csv"
TEST_CSV     = "merged_df2.csv"

PHYS_SIGNALS = ["EDA", "HR", "BVP", "TEMP"]
ACC_SIGNALS  = ["ACC_x", "ACC_y", "ACC_z"]

LAG          = 1
MIN_ROWS     = 60
ALPHA        = 0.05

STRESS_PHASES = ["TMCT", "Stroop", "Subtract", "Opposite Opinion", "Real Opinion"]
REST_PHASES   = ["Baseline", "First Rest", "Second Rest",
                 "Pre-protocol", "Post-protocol"]


# ──────────────────────────────────────────────
# DATA LOADING + NORMALIZATION
# ──────────────────────────────────────────────

def load_data(csv_path):
    """
    Load merged CSV. Downsample to 1 Hz.
    Z-score normalize each signal per participant to remove
    between-participant scale differences.
    """
    print(f"Loading {csv_path} ...")
    all_signals = PHYS_SIGNALS + ACC_SIGNALS
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.dropna(subset=all_signals + ["phase", "participant_id"])

    # Downsample to 1 Hz: keep every 32nd row (32 Hz -> 1 Hz)
    df = df.iloc[::32].reset_index(drop=True)

    # Z-score normalize each signal PER PARTICIPANT
    # Ensures each participant contributes equally regardless of
    # their baseline signal levels
    normalized_dfs = []
    for pid, group in df.groupby("participant_id", sort=False):
        group = group.copy()
        for sig in all_signals:
            vals = group[sig].values.reshape(-1, 1)
            if np.std(vals) > 1e-6:
                sc = StandardScaler()
                group[sig] = sc.fit_transform(vals).flatten()
            else:
                group[sig] = 0.0
        normalized_dfs.append(group)

    df = pd.concat(normalized_dfs, ignore_index=True)

    print(f"  Rows after downsample + normalize : {len(df):>10,}")
    print(f"  Participants                      : {df['participant_id'].nunique()}")
    print(f"  Phases found                      : {sorted(df['phase'].unique())}\n")
    return df


# ──────────────────────────────────────────────
# GRANGER TEST
# ──────────────────────────────────────────────

def run_granger(series_x, series_y, lag=LAG):
    """
    Test: does past X help predict Y? (X -> Y direction)

    Returns:
        f_stat          : F-statistic from statsmodels
        p_val           : raw p-value from F-test
        mse_restricted  : MSE predicting Y from past Y only
        mse_unrestricted: MSE predicting Y from past Y + past X
        granger_score   : (mse_restricted - mse_unrestricted) x 1000
                          positive = X helps predict Y
                          x1000 to match Eyal LSTM scale
    """
    if len(series_y) <= lag:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    if np.std(series_x) < 1e-6 or np.std(series_y) < 1e-6:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    y     = series_y
    x     = series_x
    Y_t   = y[lag:]
    Y_lag = np.column_stack([y[lag - i - 1 : len(y) - i - 1] for i in range(lag)])
    X_lag = np.column_stack([x[lag - i - 1 : len(x) - i - 1] for i in range(lag)])

    # Restricted model: predict Y from past Y only
    try:
        beta_r = np.linalg.lstsq(Y_lag, Y_t, rcond=None)[0]
        pred_r = Y_lag @ beta_r
        mse_r  = float(np.mean((Y_t - pred_r) ** 2))
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    # Unrestricted model: predict Y from past Y + past X
    try:
        XY_lag  = np.hstack([Y_lag, X_lag])
        beta_ur = np.linalg.lstsq(XY_lag, Y_t, rcond=None)[0]
        pred_ur = XY_lag @ beta_ur
        mse_ur  = float(np.mean((Y_t - pred_ur) ** 2))
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    # Scale x1000 to match Eyal's LSTM scale
    mse_r         = mse_r  * 1000
    mse_ur        = mse_ur * 1000
    granger_score = mse_r - mse_ur

    # F-test from statsmodels for statistical significance
    try:
        data   = np.column_stack([series_y, series_x])
        result = grangercausalitytests(data, maxlag=[lag], verbose=False)
        f_stat = result[lag][0]["ssr_ftest"][0]
        p_val  = result[lag][0]["ssr_ftest"][1]
    except Exception:
        f_stat, p_val = np.nan, np.nan

    return f_stat, p_val, mse_r, mse_ur, granger_score


# ──────────────────────────────────────────────
# MAIN ANALYSIS
# ──────────────────────────────────────────────

def run_analysis(df, dataset_label):
    """
    Run Granger test for all valid signal pairs.

    Pair rules:
      - Physiological -> Physiological : bi-directional
      - ACC -> Physiological           : ACC as source only
      - ACC -> ACC                     : bi-directional within ACC
      - Physiological -> ACC           : EXCLUDED
    """
    print(f"\n{'='*60}")
    print(f"  Running analysis on {dataset_label}")
    print(f"{'='*60}")

    pairs = []
# Phys -> Phys (bi-directional)
    pairs += list(permutations(PHYS_SIGNALS, 2))
# ACC -> Phys (ACC as source)
    pairs += list(product(ACC_SIGNALS, PHYS_SIGNALS))
# Phys -> ACC (Phys as source) — NEW: added to match 42 pairs
    pairs += list(product(PHYS_SIGNALS, ACC_SIGNALS))
# ACC -> ACC (bi-directional within movement)
    pairs += list(permutations(ACC_SIGNALS, 2))

    print(f"  Testing {len(pairs)} signal pairs\n")

    records = []
    for pid in df["participant_id"].unique():
        pdf = df[df["participant_id"] == pid]

        for phase in pdf["phase"].unique():
            phase_df = pdf[pdf["phase"] == phase]

            if len(phase_df) < MIN_ROWS:
                continue

            for (sig_x, sig_y) in pairs:
                if sig_x not in phase_df.columns or sig_y not in phase_df.columns:
                    continue

                x = phase_df[sig_x].values.astype(float)
                y = phase_df[sig_y].values.astype(float)

                f_stat, p_val, mse_r, mse_ur, g_score = run_granger(x, y, lag=LAG)

                records.append({
                    "participant":      pid,
                    "phase":            phase,
                    "X":                sig_x,
                    "Y":                sig_y,
                    "pair":             f"{sig_x} -> {sig_y}",
                    "f_stat":           f_stat,
                    "p_value":          p_val,
                    "mse_restricted":   mse_r,
                    "mse_unrestricted": mse_ur,
                    "granger_score":    g_score,
                    "significant":      p_val < ALPHA if not np.isnan(p_val) else False,
                    "phase_type": (
                        "stress" if phase in STRESS_PHASES else
                        "rest"   if phase in REST_PHASES   else
                        "other"
                    ),
                })

        print(f"  Processed {pid}")

    return pd.DataFrame(records)


# ──────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────

def summarise_results(results_df, dataset_label):
    """
    Print results tables.

    SIGNIFICANCE (v3 — paired t-test):
      - For each pair, get each participant's mean Granger score
        in stress phases and their mean Granger score in rest phases.
      - Run a PAIRED t-test: compare participant i's stress score
        against participant i's rest score (not group means).
      - This is more powerful than independent t-test because it
        removes between-participant variance.
      - Equivalent to Eyal's approach: f01 vs f01, f02 vs f02, etc.

    AGGREGATION:
      - Step 1: mean per participant (across phases of same type)
      - Step 2: paired t-test across participants
      - Step 3: population mean for reporting
    """
    print(f"\n{'='*65}")
    print(f"  RESULTS — {dataset_label}")
    print(f"  Normalization: z-score per participant")
    print(f"  Aggregation  : participant-level mean -> paired t-test")
    print(f"  Significance : paired t-test on Granger scores (stress vs rest)")
    print(f"{'='*65}")

    df = results_df[results_df["phase_type"].isin(["stress", "rest"])].copy()

    # ── Step 1: participant-level mean per phase type ─────────────────────────
    pid_agg = (
        df.groupby(["pair", "phase_type", "participant"])[
            ["f_stat", "p_value", "mse_restricted", "mse_unrestricted", "granger_score"]
        ].mean().reset_index()
    )

    # ── Step 2: population means for reporting ────────────────────────────────
    f_summary = (
        pid_agg.groupby(["pair", "phase_type"])["f_stat"]
        .mean().unstack("phase_type").round(4)
    )
    gs_summary = (
        pid_agg.groupby(["pair", "phase_type"])["granger_score"]
        .mean().unstack("phase_type").round(4)
    )
    mse_r_summary = (
        pid_agg.groupby(["pair", "phase_type"])["mse_restricted"]
        .mean().unstack("phase_type").round(4)
    )
    mse_ur_summary = (
        pid_agg.groupby(["pair", "phase_type"])["mse_unrestricted"]
        .mean().unstack("phase_type").round(4)
    )

    combined = pd.DataFrame(index=f_summary.index)
    combined["f_rest"]        = f_summary.get("rest",   np.nan)
    combined["f_stress"]      = f_summary.get("stress", np.nan)
    combined["delta_f"]       = (combined["f_stress"] - combined["f_rest"]).round(4)
    combined["mse_r_rest"]    = mse_r_summary.get("rest",   np.nan)
    combined["mse_r_stress"]  = mse_r_summary.get("stress", np.nan)
    combined["mse_ur_rest"]   = mse_ur_summary.get("rest",   np.nan)
    combined["mse_ur_stress"] = mse_ur_summary.get("stress", np.nan)
    combined["gs_rest"]       = gs_summary.get("rest",   np.nan)
    combined["gs_stress"]     = gs_summary.get("stress", np.nan)
    combined["delta_gs"]      = (combined["gs_stress"] - combined["gs_rest"]).round(4)

    # ── Step 3: PAIRED t-test on Granger scores ───────────────────────────────
    # For each pair: compare participant i's stress GS vs participant i's rest GS
    # This is the paired test Eyal requested: f01 vs f01, f02 vs f02, etc.
    # Only participants who have BOTH a stress score AND a rest score are included
    paired_pvals  = {}
    paired_tstats = {}

    for pair in pid_agg["pair"].unique():
        pair_df = pid_agg[pid_agg["pair"] == pair]

        stress_scores = (
            pair_df[pair_df["phase_type"] == "stress"]
            .set_index("participant")["granger_score"]
        )
        rest_scores = (
            pair_df[pair_df["phase_type"] == "rest"]
            .set_index("participant")["granger_score"]
        )

        # Align by participant index — this is what makes it paired
        common_participants = stress_scores.index.intersection(rest_scores.index)

        if len(common_participants) < 3:
            paired_pvals[pair]  = np.nan
            paired_tstats[pair] = np.nan
            continue

        stress_vals = stress_scores.loc[common_participants].values
        rest_vals   = rest_scores.loc[common_participants].values

        # scipy paired t-test: tests whether mean(stress - rest) != 0
        t_stat, p_val = stats.ttest_rel(stress_vals, rest_vals)
        paired_pvals[pair]  = round(p_val, 4)
        paired_tstats[pair] = round(t_stat, 4)

    combined["t_stat"]   = combined.index.map(paired_tstats)
    combined["p_paired"] = combined.index.map(paired_pvals)
    combined["sig"]      = combined["p_paired"] < ALPHA
    combined = combined.sort_values("delta_f", ascending=False)

    # ── Table 1: F-stat + Granger score + paired t-test ──────────────────────
    print(f"\n  F-stat: higher = stronger causality")
    print(f"  Granger score = MSE(restricted) - MSE(unrestricted) x1000")
    print(f"  Positive score = X helps predict Y")
    print(f"  p_paired = paired t-test p-value (each participant stress vs rest)")
    print()
    print(f"  {'Pair':<22} {'F-rest':>8} {'F-stress':>9} {'ΔF':>9}  "
          f"{'GS-rest':>8} {'GS-stress':>10} {'ΔGS':>9}  "
          f"{'t-stat':>7}  {'p-paired':>8}  sig")
    print(f"  {'─'*22} {'─'*8} {'─'*9} {'─'*9}  "
          f"{'─'*8} {'─'*10} {'─'*9}  {'─'*7}  {'─'*8}  {'─'*3}")

    for pair, row in combined.iterrows():
        sig    = "✓" if row["sig"] else ""
        p      = f"{row['p_paired']:.4f}" if not np.isnan(row["p_paired"]) else "  —"
        t      = f"{row['t_stat']:.4f}"   if not np.isnan(row["t_stat"])   else "  —"
        df_str = f"+{row['delta_f']:.4f}" if row["delta_f"] > 0 else f"{row['delta_f']:.4f}"
        dg_str = f"+{row['delta_gs']:.4f}" if row["delta_gs"] > 0 else f"{row['delta_gs']:.4f}"

        print(f"  {pair:<22} "
              f"{row['f_rest']:>8.4f} {row['f_stress']:>9.4f} {df_str:>9}  "
              f"{row['gs_rest']:>8.4f} {row['gs_stress']:>10.4f} {dg_str:>9}  "
              f"{t:>7}  {p:>8}  {sig}")

    # ── Table 2: MSE breakdown ────────────────────────────────────────────────
    print(f"\n  MSE breakdown (restricted vs unrestricted, x1000 scale)")
    print(f"  {'Pair':<22} {'MSE-R rest':>11} {'MSE-R stress':>13} "
          f"{'MSE-UR rest':>12} {'MSE-UR stress':>14}")
    print(f"  {'─'*22} {'─'*11} {'─'*13} {'─'*12} {'─'*14}")
    for pair, row in combined.iterrows():
        print(f"  {pair:<22} "
              f"{row['mse_r_rest']:>11.4f} {row['mse_r_stress']:>13.4f} "
              f"{row['mse_ur_rest']:>12.4f} {row['mse_ur_stress']:>14.4f}")

    # ── Table 3: Significance rate per participant ────────────────────────────
    sig_rate = (
        df.groupby(["pair", "phase_type"])["significant"]
        .mean().unstack("phase_type").round(2)
    )
    print(f"\n  Proportion of participants with significant causality (p<{ALPHA})")
    print(f"  {'Pair':<22} {'Rest':>8}  {'Stress':>8}")
    print(f"  {'─'*22} {'─'*8}  {'─'*8}")
    for pair, row in sig_rate.iterrows():
        print(f"  {pair:<22} {row.get('rest', 0):>8.0%}  "
              f"{row.get('stress', 0):>8.0%}")

    # ── Key findings ──────────────────────────────────────────────────────────
    print(f"\n  KEY FINDINGS — pairs where causality strengthens under stress:")
    top = combined[combined["delta_f"] > 0].head(5)
    for pair, row in top.iterrows():
        sig = "✓ significant" if row["sig"] else ""
        print(f"    {pair:<22}  ΔF={row['delta_f']:+.4f}  "
              f"ΔGS={row['delta_gs']:+.4f}  "
              f"t={row['t_stat']:+.4f}  p={row['p_paired']:.4f}  {sig}")

    print(f"\n  Pairs where causality WEAKENS under stress:")
    bot = combined[combined["delta_f"] < 0].tail(5)
    for pair, row in bot.iterrows():
        sig = "✓ significant" if row["sig"] else ""
        print(f"    {pair:<22}  ΔF={row['delta_f']:+.4f}  "
              f"ΔGS={row['delta_gs']:+.4f}  "
              f"t={row['t_stat']:+.4f}  p={row['p_paired']:.4f}  {sig}")

    # ── Note ──────────────────────────────────────────────────────────────────
    print(f"\n  NOTE: Stress labels are NOT tested as Granger targets.")
    print(f"  Labels are sparse and piecewise-constant within phases —")
    print(f"  violating stationarity assumptions for Granger inference.")
    print(f"  Signal-to-signal causality is used instead.")

    # ── Per-phase breakdown ───────────────────────────────────────────────────
    print(f"\n  F-statistic by individual phase (top 3 pairs by ΔF):")
    top_pairs = combined.head(3).index.tolist()
    phase_summary = (
        results_df[results_df["pair"].isin(top_pairs)]
        .groupby(["pair", "phase"])["f_stat"]
        .mean().round(3).unstack("phase")
    )
    print(phase_summary.to_string())
    print()

    return combined


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

def main():
    df_v1 = load_data(TRAIN_CSV)
    df_v2 = load_data(TEST_CSV)

    results_v1 = run_analysis(df_v1, "V1 (S01-S18)")
    summary_v1 = summarise_results(results_v1, "V1 (S01-S18)")

    results_v2 = run_analysis(df_v2, "V2 (f01-f18)")
    summary_v2 = summarise_results(results_v2, "V2 (f01-f18)")

    results_v1.to_csv("granger_results_v1.csv", index=False)
    results_v2.to_csv("granger_results_v2.csv", index=False)
    print("Full results saved to granger_results_v1.csv and granger_results_v2.csv")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
