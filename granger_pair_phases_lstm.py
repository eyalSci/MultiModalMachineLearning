"""
=================
Phase-wise Granger causality between continuous physiological signals.

IDEA:
  Instead of predicting sparse stress labels directly, we test causal
  relationships between dense continuous signals (EDA, HR, BVP, TEMP, IBI)
  and compare how those relationships change across phases.

  If EDA -> HR causality is stronger during stress phases than baseline,
  that tells us stress is changing how the body's signals interact —
  without ever needing the sparse stress column.

WHAT THIS DOES:
  For each participant:
    For each phase (Baseline, TMCT, Stroop, First Rest, etc.):
      For each signal pair (EDA->HR, HR->EDA, EDA->BVP, etc.):
        Run Granger causality test with lag=1 (1 second)
        Record F-statistic and p-value

  Then average across participants and compare phases.

OUTPUT:
  - Table of F-statistics per signal pair per phase
  - Which pairs show significantly stronger causality under stress
  - Which direction (e.g. EDA->HR vs HR->EDA) is dominant

GRANGER TEST USED:
  statsmodels grangercausalitytests
  lag = [1] (1 second at 1 Hz — we downsample to 1 Hz for speed)
  p < 0.05 = significant causality

Train on V1 (merged_df1.csv) only — V2 used as replication check.
"""

import warnings
import numpy as np
import pandas as pd
from itertools import permutations
from statsmodels.tsa.stattools import grangercausalitytests

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
TRAIN_CSV   = "merged_df1.csv"
TEST_CSV    = "merged_df2.csv"

# Signals to test (must be continuous and reasonably variable)
SIGNALS     = ["EDA", "HR", "BVP", "TEMP", "IBI"]

# Granger lag in seconds (after downsampling to 1 Hz)
# lag=1 means: does signal X 1 second ago help predict signal Y now?
LAG         = 1

# Minimum rows needed in a phase to run the test
MIN_ROWS    = 60   # 60 seconds minimum

# Phases to compare — stress phases vs rest phases
STRESS_PHASES  = ["TMCT", "Stroop", "Subtract", "Opposite Opinion", "Real Opinion"]
REST_PHASES    = ["Baseline", "First Rest", "Second Rest"]

# Significance threshold
ALPHA = 0.05


# ──────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────

def load_data(csv_path):
    """
    Load merged CSV. Downsample to 1 Hz by taking 1 row per second.
    This makes Granger tests fast and avoids autocorrelation issues
    from the original 32 Hz data.
    """
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.dropna(subset=SIGNALS + ["phase", "participant_id"])

    # Downsample to 1 Hz: keep every 32nd row (32 Hz -> 1 Hz)
    df = df.iloc[::32].reset_index(drop=True)

    print(f"  Rows after downsample : {len(df):>10,}")
    print(f"  Participants          : {df['participant_id'].nunique()}")
    print(f"  Phases found          : {sorted(df['phase'].unique())}\n")
    return df


# ──────────────────────────────────────────────
# GRANGER TEST
# ──────────────────────────────────────────────

def run_granger(series_x, series_y, lag=LAG):
    """
    Test: does past X help predict Y?
    (X -> Y direction)

    Returns (f_stat, p_value) or (NaN, NaN) if test fails.

    statsmodels grangercausalitytests takes a 2-column array [Y, X]
    and tests whether X Granger-causes Y.
    """
    data = np.column_stack([series_y, series_x])

    # Need variance in both signals
    if np.std(series_x) < 1e-6 or np.std(series_y) < 1e-6:
        return np.nan, np.nan

    try:
        result = grangercausalitytests(data, maxlag=[lag], verbose=False)
        # Extract F-test result for the specified lag
        f_stat = result[lag][0]["ssr_ftest"][0]
        p_val  = result[lag][0]["ssr_ftest"][1]
        return f_stat, p_val
    except Exception:
        return np.nan, np.nan


# ──────────────────────────────────────────────
# MAIN ANALYSIS
# ──────────────────────────────────────────────

def run_analysis(df, dataset_label):
    """
    For each participant x phase x signal pair, run Granger test.
    Returns a DataFrame of results.
    """
    print(f"\n{'='*60}")
    print(f"  Running analysis on {dataset_label}")
    print(f"{'='*60}")

    records = []
    pairs   = list(permutations(SIGNALS, 2))   # all directed pairs X->Y

    participants = df["participant_id"].unique()

    for pid in participants:
        pdf = df[df["participant_id"] == pid]

        for phase in pdf["phase"].unique():
            phase_df = pdf[pdf["phase"] == phase]

            if len(phase_df) < MIN_ROWS:
                continue

            for (sig_x, sig_y) in pairs:
                x = phase_df[sig_x].values.astype(float)
                y = phase_df[sig_y].values.astype(float)

                f_stat, p_val = run_granger(x, y, lag=LAG)

                records.append({
                    "participant": pid,
                    "phase":       phase,
                    "X":           sig_x,
                    "Y":           sig_y,
                    "pair":        f"{sig_x} -> {sig_y}",
                    "f_stat":      f_stat,
                    "p_value":     p_val,
                    "significant": p_val < ALPHA if not np.isnan(p_val) else False,
                    "phase_type":  ("stress"   if phase in STRESS_PHASES else
                                    "rest"     if phase in REST_PHASES   else
                                    "other"),
                })

        print(f"  Processed {pid}")

    return pd.DataFrame(records)


def summarise_results(results_df, dataset_label):
    """
    Print two summary tables:
      1. Mean F-statistic per signal pair per phase type (stress vs rest)
      2. Which pairs are significantly stronger under stress
    """
    print(f"\n{'='*60}")
    print(f"  RESULTS — {dataset_label}")
    print(f"{'='*60}")

    # Filter to stress and rest only
    df = results_df[results_df["phase_type"].isin(["stress", "rest"])].copy()

    # ── Table 1: mean F-stat per pair per phase type ──────────────────────────
    summary = (
        df.groupby(["pair", "phase_type"])["f_stat"]
        .mean()
        .unstack("phase_type")
        .round(3)
    )

    # Add a column: how much stronger is causality under stress vs rest
    if "stress" in summary.columns and "rest" in summary.columns:
        summary["stress_vs_rest"] = (summary["stress"] - summary["rest"]).round(3)
        summary = summary.sort_values("stress_vs_rest", ascending=False)

    print("\n  Mean F-statistic per signal pair (higher = stronger causality)")
    print(f"  {'Pair':<18} {'Rest':>8}  {'Stress':>8}  {'Stress-Rest':>12}")
    print(f"  {'─'*18} {'─'*8}  {'─'*8}  {'─'*12}")
    for pair, row in summary.iterrows():
        rest_val   = f"{row.get('rest',   np.nan):.3f}" if not np.isnan(row.get('rest',   np.nan)) else "  —"
        stress_val = f"{row.get('stress', np.nan):.3f}" if not np.isnan(row.get('stress', np.nan)) else "  —"
        delta      = row.get("stress_vs_rest", np.nan)
        delta_str  = f"+{delta:.3f}" if delta > 0 else f"{delta:.3f}" if not np.isnan(delta) else "  —"
        print(f"  {pair:<18} {rest_val:>8}  {stress_val:>8}  {delta_str:>12}")

    # ── Table 2: significance rate per pair per phase type ────────────────────
    sig_rate = (
        df.groupby(["pair", "phase_type"])["significant"]
        .mean()
        .unstack("phase_type")
        .round(3)
    )

    print("\n  Proportion of participants showing significant causality (p<0.05)")
    print(f"  {'Pair':<18} {'Rest':>8}  {'Stress':>8}")
    print(f"  {'─'*18} {'─'*8}  {'─'*8}")
    for pair, row in sig_rate.iterrows():
        rest_val   = f"{row.get('rest',   0):.0%}"
        stress_val = f"{row.get('stress', 0):.0%}"
        print(f"  {pair:<18} {rest_val:>8}  {stress_val:>8}")

    # ── Key findings ──────────────────────────────────────────────────────────
    print("\n  KEY FINDINGS:")
    print("  Pairs with strongest increase in causality under stress:")

    if "stress_vs_rest" in summary.columns:
        top = summary[summary["stress_vs_rest"] > 0].head(3)
        for pair, row in top.iterrows():
            delta = row["stress_vs_rest"]
            print(f"    {pair:<20}  stress F = {row.get('stress', 0):.3f}  "
                  f"rest F = {row.get('rest', 0):.3f}  "
                  f"(+{delta:.3f} stronger under stress)")

    # ── Per-phase breakdown ───────────────────────────────────────────────────
    print("\n  F-statistic by individual phase (top 3 pairs only):")
    top_pairs = summary.head(3).index.tolist() if "stress_vs_rest" in summary.columns else []

    phase_summary = (
        results_df[results_df["pair"].isin(top_pairs)]
        .groupby(["pair", "phase"])["f_stat"]
        .mean()
        .round(3)
        .unstack("phase")
    )

    print(phase_summary.to_string())
    print()

    return summary


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

def main():
    # Load both datasets
    df_v1 = load_data(TRAIN_CSV)
    df_v2 = load_data(TEST_CSV)

    # Run analysis on V1 (main results)
    results_v1 = run_analysis(df_v1, "V1 (S01-S18)")
    summary_v1 = summarise_results(results_v1, "V1 (S01-S18)")

    # Run analysis on V2 (replication check)
    results_v2 = run_analysis(df_v2, "V2 (f01-f18)")
    summary_v2 = summarise_results(results_v2, "V2 (f01-f18)")

    # Save full results to CSV for further analysis or report tables
    results_v1.to_csv("granger_results_v1.csv", index=False)
    results_v2.to_csv("granger_results_v2.csv", index=False)
    print("Full results saved to granger_results_v1.csv and granger_results_v2.csv")


if __name__ == "__main__":
    main()