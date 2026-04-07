"""
granger_pair_phases_v2.py
=========================

Phase-wise Granger causality between continuous physiological signals,
with proper 1 Hz resampling and multiple lag testing.

MAIN IMPROVEMENTS OVER THE PREVIOUS VERSION
-------------------------------------------
1. Proper 1 Hz resampling:
   Instead of taking every 32nd row, signals are averaged within each 1-second bin.
   This reduces noise and avoids alignment sensitivity.

2. Multiple lag testing:
   Instead of forcing lag=1 only, we test several lags and keep the best lag
   (lowest p-value; ties broken by higher F-statistic).

3. Same phase-wise signal-pair idea:
   We still test whether causal interactions between physiological signals
   become stronger during stress phases than rest phases.

WHAT THIS DOES
--------------
For each participant:
  For each phase:
    For each directed signal pair X -> Y:
      Test lags in LAGS
      Keep the best lag
      Record best F-statistic and p-value

Then:
  - Compare stress vs rest
  - Report strongest increases under stress
  - Save full results for V1 and V2

OUTPUT FILES
------------
- granger_results_v1_multilag.csv
- granger_results_v2_multilag.csv
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
TRAIN_CSV = "merged_df1.csv"
TEST_CSV  = "merged_df2.csv"

SIGNALS = ["EDA", "HR", "BVP", "TEMP", "IBI"]

# Test several lags (in seconds after 1 Hz resampling)
LAGS = [1, 2, 3, 5]

# Minimum rows needed in a phase to run the test.
# Must be comfortably larger than max lag.
MIN_ROWS = 60

STRESS_PHASES = ["TMCT", "Stroop", "Subtract", "Opposite Opinion", "Real Opinion"]
REST_PHASES   = ["Baseline", "First Rest", "Second Rest", "Pre-protocol", "Post-protocol"]

ALPHA = 0.05

# Optional: participant-wise z-score normalization before Granger
NORMALIZE_WITHIN_PARTICIPANT = True


# ──────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────

def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load merged CSV, keep needed columns, and resample properly to 1 Hz.

    Why this is better than df.iloc[::32]:
      - uses all 32 samples per second
      - reduces noise
      - avoids dependence on arbitrary alignment

    Returns
    -------
    DataFrame with one row per participant × phase × second.
    """
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])

    # Keep only rows with all required fields present
    df = df.dropna(subset=SIGNALS + ["phase", "participant_id", "timestamp"]).copy()

    # Sort first
    df = df.sort_values(["participant_id", "timestamp"]).reset_index(drop=True)

    # Proper 1 Hz resampling inside each participant × phase block
    resampled_parts = []

    for (pid, phase), grp in df.groupby(["participant_id", "phase"], sort=False):
        if grp.empty:
            continue

        g = grp.set_index("timestamp")[SIGNALS].resample("1s").mean()

        # Keep metadata
        g["participant_id"] = pid
        g["phase"] = phase
        g = g.reset_index()

        # Drop rows where resampling created fully-missing signals
        g = g.dropna(subset=SIGNALS, how="any")

        if not g.empty:
            resampled_parts.append(g)

    if not resampled_parts:
        return pd.DataFrame(columns=["timestamp", "participant_id", "phase"] + SIGNALS)

    df_1hz = pd.concat(resampled_parts, ignore_index=True)

    if NORMALIZE_WITHIN_PARTICIPANT:
        df_1hz[SIGNALS] = (
            df_1hz.groupby("participant_id", sort=False)[SIGNALS]
            .transform(lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-8))
        )

    print(f"  Rows after 1 Hz resample : {len(df_1hz):>10,}")
    print(f"  Participants             : {df_1hz['participant_id'].nunique()}")
    print(f"  Phases found             : {sorted(df_1hz['phase'].dropna().unique())}\n")

    return df_1hz


# ──────────────────────────────────────────────
# GRANGER TEST
# ──────────────────────────────────────────────

def run_granger_for_lag(series_x: np.ndarray, series_y: np.ndarray, lag: int):
    """
    Test X -> Y for a single lag.

    Returns
    -------
    (f_stat, p_value) or (np.nan, np.nan) if test fails.
    """
    if len(series_x) <= lag or len(series_y) <= lag:
        return np.nan, np.nan

    if np.std(series_x) < 1e-8 or np.std(series_y) < 1e-8:
        return np.nan, np.nan

    data = np.column_stack([series_y, series_x])

    try:
        result = grangercausalitytests(data, maxlag=[lag], verbose=False)
        f_stat = result[lag][0]["ssr_ftest"][0]
        p_val  = result[lag][0]["ssr_ftest"][1]
        return f_stat, p_val
    except Exception:
        return np.nan, np.nan


def run_granger_multilag(series_x: np.ndarray, series_y: np.ndarray, lags=LAGS):
    """
    Test multiple lags and keep the best result.

    Best lag rule:
      1. smallest p-value
      2. if tied, larger F-statistic

    Returns
    -------
    best_lag, best_f, best_p, all_results_dict
    """
    results = {}

    for lag in lags:
        f_stat, p_val = run_granger_for_lag(series_x, series_y, lag)
        results[lag] = {"f_stat": f_stat, "p_value": p_val}

    valid = [
        (lag, vals["f_stat"], vals["p_value"])
        for lag, vals in results.items()
        if not np.isnan(vals["f_stat"]) and not np.isnan(vals["p_value"])
    ]

    if not valid:
        return np.nan, np.nan, np.nan, results

    valid_sorted = sorted(valid, key=lambda t: (t[2], -t[1]))
    best_lag, best_f, best_p = valid_sorted[0]
    return best_lag, best_f, best_p, results


# ──────────────────────────────────────────────
# MAIN ANALYSIS
# ──────────────────────────────────────────────

def run_analysis(df: pd.DataFrame, dataset_label: str) -> pd.DataFrame:
    """
    For each participant × phase × directed signal pair:
      - test multiple lags
      - keep best lag
      - store result
    """
    print(f"\n{'='*64}")
    print(f"Running analysis on {dataset_label}")
    print(f"{'='*64}")

    records = []
    pairs = list(permutations(SIGNALS, 2))

    participants = df["participant_id"].dropna().unique()

    for pid in participants:
        pdf = df[df["participant_id"] == pid]

        for phase in pdf["phase"].dropna().unique():
            phase_df = pdf[pdf["phase"] == phase].sort_values("timestamp")

            if len(phase_df) < MIN_ROWS:
                continue

            for sig_x, sig_y in pairs:
                x = phase_df[sig_x].values.astype(float)
                y = phase_df[sig_y].values.astype(float)

                best_lag, best_f, best_p, lag_results = run_granger_multilag(x, y, lags=LAGS)

                records.append({
                    "participant": pid,
                    "phase": phase,
                    "X": sig_x,
                    "Y": sig_y,
                    "pair": f"{sig_x} -> {sig_y}",
                    "best_lag": best_lag,
                    "f_stat": best_f,
                    "p_value": best_p,
                    "significant": (best_p < ALPHA) if not np.isnan(best_p) else False,
                    "phase_type": (
                        "stress" if phase in STRESS_PHASES else
                        "rest"   if phase in REST_PHASES else
                        "other"
                    ),
                    "n_rows": len(phase_df),
                    "lag_results": str(lag_results),
                })

        print(f"  Processed {pid}")

    return pd.DataFrame(records)


# ──────────────────────────────────────────────
# SUMMARIES
# ──────────────────────────────────────────────

def summarise_results(results_df: pd.DataFrame, dataset_label: str) -> pd.DataFrame:
    """
    Print summary tables:
      1. Mean F-statistic per pair in rest vs stress
      2. Significance rate per pair in rest vs stress
      3. Mean chosen lag per pair in rest vs stress
    """
    print(f"\n{'='*64}")
    print(f"RESULTS — {dataset_label}")
    print(f"{'='*64}")

    df = results_df[results_df["phase_type"].isin(["stress", "rest"])].copy()

    if df.empty:
        print("No valid stress/rest results found.")
        return pd.DataFrame()

    # Mean F-stat
    summary = (
        df.groupby(["pair", "phase_type"])["f_stat"]
        .mean()
        .unstack("phase_type")
        .round(3)
    )

    if "stress" in summary.columns and "rest" in summary.columns:
        summary["stress_vs_rest"] = (summary["stress"] - summary["rest"]).round(3)
        summary = summary.sort_values("stress_vs_rest", ascending=False)

    print("\nMean F-statistic per signal pair")
    print(f"{'Pair':<18} {'Rest':>8}  {'Stress':>8}  {'Stress-Rest':>12}")
    print(f"{'─'*18} {'─'*8}  {'─'*8}  {'─'*12}")
    for pair, row in summary.iterrows():
        rest_val = row.get("rest", np.nan)
        stress_val = row.get("stress", np.nan)
        delta = row.get("stress_vs_rest", np.nan)

        rest_str = f"{rest_val:.3f}" if pd.notna(rest_val) else "—"
        stress_str = f"{stress_val:.3f}" if pd.notna(stress_val) else "—"
        delta_str = f"{delta:+.3f}" if pd.notna(delta) else "—"

        print(f"{pair:<18} {rest_str:>8}  {stress_str:>8}  {delta_str:>12}")

    # Significance rate
    sig_rate = (
        df.groupby(["pair", "phase_type"])["significant"]
        .mean()
        .unstack("phase_type")
        .round(3)
    )

    print("\nProportion significant (p < 0.05)")
    print(f"{'Pair':<18} {'Rest':>8}  {'Stress':>8}")
    print(f"{'─'*18} {'─'*8}  {'─'*8}")
    for pair, row in sig_rate.iterrows():
        rest_val = f"{row.get('rest', 0):.0%}"
        stress_val = f"{row.get('stress', 0):.0%}"
        print(f"{pair:<18} {rest_val:>8}  {stress_val:>8}")

    # Mean chosen lag
    lag_summary = (
        df.groupby(["pair", "phase_type"])["best_lag"]
        .mean()
        .unstack("phase_type")
        .round(2)
    )

    print("\nMean selected lag")
    print(f"{'Pair':<18} {'Rest':>8}  {'Stress':>8}")
    print(f"{'─'*18} {'─'*8}  {'─'*8}")
    for pair, row in lag_summary.iterrows():
        rest_val = row.get("rest", np.nan)
        stress_val = row.get("stress", np.nan)
        rest_str = f"{rest_val:.2f}" if pd.notna(rest_val) else "—"
        stress_str = f"{stress_val:.2f}" if pd.notna(stress_val) else "—"
        print(f"{pair:<18} {rest_str:>8}  {stress_str:>8}")

    # Top findings
    print("\nKEY FINDINGS:")
    if "stress_vs_rest" in summary.columns:
        top = summary[summary["stress_vs_rest"] > 0].head(5)
        if len(top) == 0:
            print("  No pairs showed stronger average causality under stress.")
        else:
            for pair, row in top.iterrows():
                print(
                    f"  {pair:<18} rest F = {row.get('rest', np.nan):.3f}, "
                    f"stress F = {row.get('stress', np.nan):.3f}, "
                    f"delta = +{row['stress_vs_rest']:.3f}"
                )

    return summary


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

def main():
    df_v1 = load_data(TRAIN_CSV)
    df_v2 = load_data(TEST_CSV)

    results_v1 = run_analysis(df_v1, "V1 (S01-S18)")
    summarise_results(results_v1, "V1 (S01-S18)")

    results_v2 = run_analysis(df_v2, "V2 (f01-f18)")
    summarise_results(results_v2, "V2 (f01-f18)")

    results_v1.to_csv("granger_results_v1_multilag.csv", index=False)
    results_v2.to_csv("granger_results_v2_multilag.csv", index=False)
    print("\nSaved:")
    print(" - granger_results_v1_multilag.csv")
    print(" - granger_results_v2_multilag.csv")


if __name__ == "__main__":
    main()