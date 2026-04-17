#!/usr/bin/env python3
"""
merge_wearable_stress.py
========================
Merges Empatica E4 wristband recordings from the STRESS protocol of the
"Wearable Device Dataset" (PhysioNet, DOI 10.13026/3tkz-7j93) into two
aggregated DataFrames — one per trial version.

DATASET OVERVIEW
----------------
36 participants wore an Empatica E4 wristband through a cognitive-stress
induction protocol.  The dataset is split in two trial versions:

  v1  →  participants S01 … S18  (folders named "Sxx")
  v2  →  participants f01 … f18  (folders named "fxx")

Each participant folder contains up to seven CSV files recorded by the E4:

  ACC.csv   3-axis accelerometer        32 Hz   units: 1/64 g
  BVP.csv   Blood Volume Pulse (PPG)    64 Hz   units: adimensional
  EDA.csv   Electrodermal Activity       4 Hz   units: µS
  HR.csv    Heart Rate (from BVP)        1 Hz   units: bpm
  IBI.csv   Inter-Beat Interval         event   units: s (irregular)
  TEMP.csv  Skin Temperature             4 Hz   units: °C
  tags.csv  Button-press timestamps     event   UTC datetime strings

File format (fixed-rate signals: ACC, BVP, EDA, HR, TEMP)
----------------------------------------------------------
  Line 0  :  UTC session-start timestamp (repeated N times for N-column files)
  Line 1  :  Sampling frequency in Hz   (repeated N times for N-column files)
  Lines 2+:  Numeric data

File format (IBI)
-----------------
  Line 0  :  UTC session-start timestamp (repeated twice, one per column)
  Lines 1+:  col0 = seconds from session start to the detected beat
             col1 = duration of the inter-beat interval (s since prev beat)
  NOTE: IBI has only ONE header line (no sample-rate line).

OUTPUT
------
Two aggregated DataFrames (df_v1, df_v2) — one row per phase per participant —
with columns:

  participant_id   str            e.g. "S01", "f07"
  phase            str            protocol phase label (Transitions, Pre- and
                                  Post-protocol are excluded entirely)
  timestamp        datetime64[ns] last ACC timestamp within the phase
  <signal>_mean    float          mean of the signal across the entire phase
  <signal>_sd      float          sample std dev (ddof=1); NaN when n < 2
  <signal>_max     float          maximum value within the phase
  <signal>_min     float          minimum value within the phase
                   signals: ACC_x, ACC_y, ACC_z, BVP, EDA, TEMP, HR, IBI
  reported_stress  float          self-reported stress score from
                                  Stress_Level_v1/2.csv

AGGREGATION STRATEGY
--------------------
Each non-Transition phase is aggregated as a single unit over its full
duration from phase_start to phase_end.  Signals are aggregated at their
native sampling rates — no resampling is performed.

PHASE LABELS
------------
Notebook-confirmed stress task boundaries (from graph_multiple inline comments):
  V1: Stroop [3→4], TMCT [5→6], Real Opinion [7→8],
      Opposite Opinion [9→10], Subtract [11→12]
  V2: TMCT [2→3], Real Opinion [4→5],
      Opposite Opinion [6→7], Subtract [8→9]
Gap phases (Baseline, First/Second Rest) are inferred from the ordering of
Stress_Level_v1/2.csv columns.  Transitions are labelled _1/_2/_3.

SPECIAL CASES HANDLED
---------------------
S02  — Duplicated signals (E4 Connect download artefact):
         Known truncation rows per file (see S02_MAX_DATA_ROWS).
         HR and IBI exact boundaries estimated from BVP valid duration.

f07  — Protection dock never removed; PPG and TEMP sensors blocked.
         BVP, HR, IBI, TEMP aggregates are NaN; rows are kept.

f14  — Bluetooth lost mid-session; only f14_b (9 button presses) is used.
         f14_a (no button presses) is discarded.

USAGE
-----
  python merge_wearable_stress.py

  Or as a module:
      from merge_wearable_stress import main
      df_v1, df_v2 = main()

  Set BASE_DIR to the folder containing the unzipped dataset files.
"""

import os
import warnings
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR   = "."
STRESS_DIR = os.path.join(BASE_DIR, "Wearable_Dataset", "STRESS")


# ============================================================
# SIGNAL METADATA
# ============================================================
SIGNAL_UNITS = {
    "ACC_x": "1/64g",        "ACC_y": "1/64g",       "ACC_z": "1/64g",
    "BVP":   "adimensional", "EDA":   "µS",
    "HR":    "bpm",          "IBI":   "seconds",      "TEMP":  "°C",
}

# Output signal names in column order (ACC already expanded to 3 axes)
ALL_SIGNALS  = ["ACC_x", "ACC_y", "ACC_z", "BVP", "EDA", "TEMP", "HR", "IBI"]
AGGREGATIONS = ["mean", "sd", "max", "min"]


# ============================================================
# PHASE DEFINITIONS
# ============================================================
# tags[0] = session-start timestamp (prepended from signal-file header)
# tags[1..N] = button-press timestamps from tags.csv
# Each tuple: (start_tag_index, end_tag_index, phase_label)
#
# V1 (S01–S18): 13 button presses → 14 tag values (indices 0–13)
V1_PHASE_MAP = [
    ( 0,  1, "Pre-protocol"),
    ( 1,  2, "Baseline"),
    ( 2,  3, "Transition_1"),
    ( 3,  4, "Stroop"),
    ( 4,  5, "First Rest"),
    ( 5,  6, "TMCT"),
    ( 6,  7, "Second Rest"),
    ( 7,  8, "Real Opinion"),
    ( 8,  9, "Transition_2"),
    ( 9, 10, "Opposite Opinion"),
    (10, 11, "Transition_3"),
    (11, 12, "Subtract"),
    (12, 13, "Post-protocol"),
]

# V2 (f01–f18): 9 button presses → 10 tag values (indices 0–9)
V2_PHASE_MAP = [
    (0, 1, "Pre-protocol"),
    (1, 2, "Baseline"),
    (2, 3, "TMCT"),
    (3, 4, "First Rest"),
    (4, 5, "Real Opinion"),
    (5, 6, "Transition_1"),
    (6, 7, "Opposite Opinion"),
    (7, 8, "Second Rest"),
    (8, 9, "Subtract"),
]

# Phases omitted from the output entirely
SKIP_PHASES = frozenset({
    "Pre-protocol", "Post-protocol",
    "Transition_1", "Transition_2", "Transition_3",
})


# ============================================================
# S02 TRUNCATION BOUNDARIES
# ============================================================
# Duplicated raw values start at (1-indexed file lines):
#   ACC row 49,545 | BVP row 99,091 | EDA & TEMP row 6,195
# Each fixed-rate file has 2 header lines → pandas row 0 = file line 3.
# Data rows to keep = file_row - 3.
S02_BVP_VALID_SECONDS = 99088 / 64      # ≈ 1548.25 s  (used for HR/IBI estimate)

S02_MAX_DATA_ROWS = {
    "ACC":  49542,
    "BVP":  99088,
    "EDA":   6192,
    "TEMP":  6192,
    "HR":    int(S02_BVP_VALID_SECONDS),  # 1548 — estimated
}

# ============================================================
# f07 BLOCKED SENSORS
# ============================================================
F07_INVALID_SIGNALS = frozenset({"BVP", "HR", "IBI", "TEMP"})


# ============================================================
# LOW-LEVEL FILE READERS
# ============================================================

def _parse_start_time(filepath: str) -> pd.Timestamp:
    """
    Read the UTC session-start timestamp from line 0 of an E4 CSV file.
    Multi-column files repeat the timestamp; we take the first value.
    """
    with open(filepath, "r", encoding="utf-8") as fh:
        first_line = fh.readline().strip()
    return pd.Timestamp(first_line.split(",")[0].strip())


def _parse_sample_rate(filepath: str) -> float:
    """
    Read the sampling frequency (Hz) from line 1 of a fixed-rate E4 file.
    Do not call this for IBI (which has no sample-rate line).
    """
    with open(filepath, "r", encoding="utf-8") as fh:
        fh.readline()
        second_line = fh.readline().strip()
    return float(second_line.split(",")[0].strip())


def read_fixed_rate_signal(
    filepath: str,
    signal_name: str,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """
    Load one fixed-rate E4 signal file at its native sampling rate.

    Timestamps are reconstructed arithmetically:
        timestamp[i] = session_start + i / sample_rate  (seconds)

    Parameters
    ----------
    signal_name : "ACC", "BVP", "EDA", "HR", or "TEMP"
    max_rows    : keep only this many data rows (S02 truncation)

    Returns
    -------
    DataFrame with column 'timestamp' plus:
        ACC   → ACC_x, ACC_y, ACC_z
        other → one column named after the signal
    """
    start_time  = _parse_start_time(filepath)
    sample_rate = _parse_sample_rate(filepath)

    if signal_name == "ACC":
        data = pd.read_csv(filepath, skiprows=2, header=None,
                           names=["ACC_x", "ACC_y", "ACC_z"])
    else:
        data = pd.read_csv(filepath, skiprows=2, header=None,
                           names=[signal_name])

    if max_rows is not None:
        data = data.iloc[:max_rows].copy()

    n_samples = len(data)
    offsets   = pd.to_timedelta(np.arange(n_samples) / sample_rate, unit="s")
    data.insert(0, "timestamp", start_time + offsets)
    return data.reset_index(drop=True)


def read_ibi_signal(filepath: str) -> pd.DataFrame:
    """
    Load the IBI E4 file (event-driven, irregular timestamps).

    IBI file format:
      Line 0  : session-start timestamp (repeated twice)
      Lines 1+: col0 = seconds from session start to the detected beat
                col1 = duration of the inter-beat interval (s since prev beat)

    Returns DataFrame with columns: timestamp, IBI.
    Returns empty DataFrame if file is absent, empty, or unreadable.
    """
    start_time = _parse_start_time(filepath)
    try:
        data = pd.read_csv(filepath, skiprows=1, header=None,
                           names=["offset_s", "IBI"])
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=["timestamp", "IBI"])

    if data.empty:
        return pd.DataFrame(columns=["timestamp", "IBI"])

    data["offset_s"] = pd.to_numeric(data["offset_s"], errors="coerce")
    data["IBI"]      = pd.to_numeric(data["IBI"],      errors="coerce")
    data = data.dropna(subset=["offset_s"])

    timestamps = start_time + pd.to_timedelta(data["offset_s"], unit="s")
    data.insert(0, "timestamp", timestamps.values)
    return data.drop(columns=["offset_s"]).reset_index(drop=True)


def read_tags(folder_path: str) -> list[pd.Timestamp]:
    """
    Read button-press timestamps from tags.csv.
    Returns empty list if file is absent or empty.
    """
    tags_path = os.path.join(folder_path, "tags.csv")
    if not os.path.exists(tags_path):
        warnings.warn(f"tags.csv not found in: {folder_path}")
        return []
    try:
        df = pd.read_csv(tags_path, header=None, names=["ts"])
        if df.empty:
            return []
        return [pd.Timestamp(t) for t in df["ts"].dropna()]
    except pd.errors.EmptyDataError:
        return []


# ============================================================
# PHASE BOUNDARIES
# ============================================================

def build_phase_boundaries(
    session_start: pd.Timestamp,
    tag_timestamps: list[pd.Timestamp],
    phase_map: list[tuple],
) -> list[tuple]:
    """
    Combine session_start + button presses into (start_dt, end_dt, label) tuples.

    tags[0] = session_start; tags[1]+ = button presses from tags.csv.
    Phases whose required tag index is out of range are silently omitted.
    """
    all_tags = [session_start] + list(tag_timestamps)
    n        = len(all_tags)
    return [
        (all_tags[si], all_tags[ei], label)
        for si, ei, label in phase_map
        if si < n and ei < n
    ]


# ============================================================
# SIGNAL READING  (all signals for one participant, native rates)
# ============================================================

def read_all_signals(
    folder_path: str,
    participant_id: str,
    is_s02: bool,
    is_f07: bool,
) -> dict[str, pd.DataFrame]:
    """
    Read every signal file for one participant at its native sampling rate.

    Returns
    -------
    dict  key → DataFrame
      "ACC"  → timestamp, ACC_x, ACC_y, ACC_z  (32 Hz)
      "BVP"  → timestamp, BVP                  (64 Hz)
      "EDA"  → timestamp, EDA                  ( 4 Hz)
      "HR"   → timestamp, HR                   ( 1 Hz)
      "TEMP" → timestamp, TEMP                 ( 4 Hz)
      "IBI"  → timestamp, IBI                  (event)
    Missing files are omitted with a warning.
    For f07, value columns of blocked sensors are NaN.
    """
    signals: dict[str, pd.DataFrame] = {}

    for sig in ["ACC", "BVP", "EDA", "HR", "TEMP"]:
        fpath = os.path.join(folder_path, f"{sig}.csv")
        if not os.path.exists(fpath):
            warnings.warn(f"{participant_id}: {sig}.csv not found — skipping.")
            continue

        max_rows = S02_MAX_DATA_ROWS.get(sig) if is_s02 else None
        if is_s02 and sig == "HR":
            warnings.warn(
                f"S02 HR: truncating at estimated row {max_rows} "
                f"(derived from BVP valid duration ≈ {S02_BVP_VALID_SECONDS:.1f} s). "
                "Exact boundary unknown — see data_constraints.txt."
            )

        df = read_fixed_rate_signal(fpath, sig, max_rows=max_rows)

        # f07: zero out value columns for physically blocked sensors
        if is_f07 and sig in F07_INVALID_SIGNALS:
            val_cols = ["ACC_x", "ACC_y", "ACC_z"] if sig == "ACC" else [sig]
            for col in val_cols:
                df[col] = np.nan

        signals[sig] = df

    # IBI (event-driven)
    ibi_path = os.path.join(folder_path, "IBI.csv")
    if os.path.exists(ibi_path):
        ibi_df = read_ibi_signal(ibi_path)

        if is_f07:
            ibi_df["IBI"] = np.nan

        elif is_s02 and not ibi_df.empty:
            ibi_start = _parse_start_time(ibi_path)
            # S02's IBI file has a different de-identification time-shift than
            # the other signal files (timestamps land in 1959 while EDA/ACC/BVP
            # are in 2013).  Check whether the IBI timestamps are plausibly
            # aligned with the rest of the recording by comparing the IBI file's
            # session-start to EDA's session-start.  If they differ by more than
            # one hour the files cannot be aligned and IBI must be treated as NaN.
            eda_start = _parse_start_time(os.path.join(folder_path, "EDA.csv"))
            shift_hours = abs((ibi_start - eda_start).total_seconds()) / 3600
            if shift_hours > 1:
                warnings.warn(
                    f"S02 IBI: session-start timestamp in IBI.csv ({ibi_start}) "
                    f"differs from EDA.csv ({eda_start}) by {shift_hours:,.0f} hours. "
                    "This is a known de-identification mismatch for S02 — IBI timestamps "
                    "cannot be aligned with phase boundaries. Setting IBI to NaN."
                )
                ibi_df["IBI"] = np.nan
            else:
                # Timestamps are aligned — apply normal S02 truncation
                cutoff_dt = ibi_start + pd.Timedelta(seconds=S02_BVP_VALID_SECONDS)
                before    = len(ibi_df)
                ibi_df    = ibi_df[ibi_df["timestamp"] < cutoff_dt].copy()
                ibi_df    = ibi_df[~ibi_df["timestamp"].duplicated(keep="last")]
                warnings.warn(
                    f"S02 IBI: kept {len(ibi_df)}/{before} events < "
                    f"session_start + {S02_BVP_VALID_SECONDS:.1f} s "
                    "(estimated; see data_constraints.txt)."
                )

        signals["IBI"] = ibi_df

    return signals


# ============================================================
# SIGNAL QUALITY CHECK
# ============================================================

def _check_signal_quality(
    signals: dict[str, pd.DataFrame],
    participant_id: str,
) -> None:
    """Warn if any PPG-derived or temperature signal is entirely NaN."""
    for sig, col in {"BVP": "BVP", "HR": "HR", "IBI": "IBI", "TEMP": "TEMP"}.items():
        if sig in signals and col in signals[sig].columns:
            if signals[sig][col].isna().all():
                warnings.warn(
                    f"{participant_id}: '{col}' is entirely NaN. "
                    "Check data_constraints.txt for known sensor issues."
                )


# ============================================================
# AGGREGATION
# ============================================================

def _slice_signal(
    df: pd.DataFrame,
    phase_start: pd.Timestamp,
    phase_end:   pd.Timestamp,
) -> pd.DataFrame:
    """
    Return rows where phase_start <= timestamp < phase_end.
    Half-open interval keeps adjacent phases non-overlapping.
    """
    ts = df["timestamp"]
    return df.loc[(ts >= phase_start) & (ts < phase_end)]


def _agg_series(vals: pd.Series, name: str) -> dict:
    """
    Compute mean / sd / max / min for a numeric Series.
    sd uses ddof=1 (sample standard deviation); NaN when fewer than 2 values.
    All stats are NaN when the Series is empty after dropping NaN.
    """
    clean = vals.dropna()
    n     = len(clean)
    if n == 0:
        return {f"{name}_mean": np.nan, f"{name}_sd": np.nan,
                f"{name}_max":  np.nan, f"{name}_min": np.nan}
    return {
        f"{name}_mean": float(clean.mean()),
        f"{name}_sd":   float(clean.std(ddof=1)) if n > 1 else np.nan,
        f"{name}_max":  float(clean.max()),
        f"{name}_min":  float(clean.min()),
    }


def aggregate_phase(
    signals:     dict[str, pd.DataFrame],
    phase_start: pd.Timestamp,
    phase_end:   pd.Timestamp,
) -> dict:
    """
    Aggregate all signals across the full phase duration [phase_start, phase_end).

    Each signal is sliced at its native rate (no resampling).
    The representative timestamp is the last ACC timestamp within the phase,
    falling back to phase_end if ACC is unavailable.

    Returns dict with keys <signal>_mean/sd/max/min and 'timestamp'.
    """
    row: dict = {}

    # ACC: three axes aggregated independently
    if "ACC" in signals and not signals["ACC"].empty:
        acc_slice = _slice_signal(signals["ACC"], phase_start, phase_end)
        for axis in ("ACC_x", "ACC_y", "ACC_z"):
            row.update(_agg_series(acc_slice[axis], axis))
        row["timestamp"] = (
            acc_slice["timestamp"].max() if not acc_slice.empty else phase_end
        )
    else:
        for axis in ("ACC_x", "ACC_y", "ACC_z"):
            row.update(_agg_series(pd.Series([], dtype=float), axis))
        row["timestamp"] = phase_end

    # Single-column fixed-rate signals
    for sig in ("BVP", "EDA", "TEMP", "HR"):
        if sig in signals and not signals[sig].empty:
            sliced = _slice_signal(signals[sig], phase_start, phase_end)
            row.update(_agg_series(sliced[sig], sig))
        else:
            row.update(_agg_series(pd.Series([], dtype=float), sig))

    # IBI (event-driven)
    if "IBI" in signals and not signals["IBI"].empty:
        ibi_slice = _slice_signal(signals["IBI"], phase_start, phase_end)
        row.update(_agg_series(ibi_slice["IBI"], "IBI"))
    else:
        row.update(_agg_series(pd.Series([], dtype=float), "IBI"))

    return row


# ============================================================
# PER-PARTICIPANT ROW BUILDER
# ============================================================

def build_aggregated_rows(
    signals:        dict[str, pd.DataFrame],
    boundaries:     list[tuple],
    participant_id: str,
    stress_lut:     dict[str, float],
) -> list[dict]:
    """
    For each non-skipped phase, aggregate all signals across the full phase
    duration and return one row per phase.

    Parameters
    ----------
    signals        : from read_all_signals()
    boundaries     : from build_phase_boundaries()
    participant_id : written into every output row
    stress_lut     : {phase_label: reported_stress_score}

    Returns
    -------
    list of row dicts, one per phase
    """
    rows: list[dict] = []

    for phase_start, phase_end, phase_label in boundaries:

        if phase_label in SKIP_PHASES:
            continue

        row = aggregate_phase(signals, phase_start, phase_end)
        row["participant_id"]  = participant_id
        row["phase"]           = phase_label
        row["reported_stress"] = stress_lut.get(phase_label, np.nan)
        rows.append(row)

    return rows


# ============================================================
# PER-PARTICIPANT ENTRY POINTS
# ============================================================

def process_participant(
    folder_path:    str,
    participant_id: str,
    trial:          str,
    stress_df:      pd.DataFrame,
) -> pd.DataFrame:
    """
    Read signals, build phase boundaries, aggregate windows.

    Parameters
    ----------
    trial      : "v1" or "v2" — selects phase map; not stored as a column
    stress_df  : index=participant_id, columns=phase names
    """
    is_s02 = (participant_id == "S02")
    is_f07 = (participant_id == "f07")

    signals = read_all_signals(folder_path, participant_id, is_s02, is_f07)
    _check_signal_quality(signals, participant_id)

    phase_map     = V1_PHASE_MAP if trial == "v1" else V2_PHASE_MAP
    session_start = _parse_start_time(os.path.join(folder_path, "EDA.csv"))
    tag_ts        = read_tags(folder_path)
    boundaries    = build_phase_boundaries(session_start, tag_ts, phase_map)

    if participant_id in stress_df.index:
        stress_lut: dict[str, float] = stress_df.loc[participant_id].to_dict()
    else:
        warnings.warn(f"{participant_id}: not found in stress-level CSV.")
        stress_lut = {}

    rows = build_aggregated_rows(signals, boundaries, participant_id, stress_lut)
    return pd.DataFrame(rows)


def process_f14(stress_df: pd.DataFrame) -> pd.DataFrame:
    """
    Special case: participant f14 (v2 trial).
    Only f14_b is used; f14_a (no button presses) is discarded.
    """
    folder_b = os.path.join(STRESS_DIR, "f14_b")

    signals = read_all_signals(folder_b, "f14_b", is_s02=False, is_f07=False)
    _check_signal_quality(signals, "f14")

    session_start = _parse_start_time(os.path.join(folder_b, "EDA.csv"))
    tag_ts        = read_tags(folder_b)
    boundaries    = build_phase_boundaries(session_start, tag_ts, V2_PHASE_MAP)

    stress_lut: dict[str, float] = (
        stress_df.loc["f14"].to_dict()
        if "f14" in stress_df.index
        else {}
    )
    if not stress_lut:
        warnings.warn("f14: not found in stress-level CSV.")

    rows = build_aggregated_rows(signals, boundaries, "f14", stress_lut)
    return pd.DataFrame(rows)


# ============================================================
# MAIN ASSEMBLY
# ============================================================

def build_trial_df(
    trial:            str,
    participant_list: list[tuple[str, str]],
    stress_df:        pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate all participants for one trial and concatenate."""
    dfs = []
    for folder_path, pid in participant_list:
        print(f"    [{trial.upper()}] Processing {pid} …")
        try:
            df = process_participant(folder_path, pid, trial, stress_df)
            if not df.empty:
                dfs.append(df)
        except Exception as exc:
            warnings.warn(f"Failed to process {pid}: {exc}")

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def main() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build and return the two aggregated DataFrames.

    Returns
    -------
    df_v1 : pd.DataFrame — trial v1 (S01–S18)
    df_v2 : pd.DataFrame — trial v2 (f01–f18)

    Column order:
        participant_id, phase, timestamp,
        ACC_x_mean … IBI_min  (8 signals × 4 aggregations = 32 columns),
        reported_stress
    """
    print("=" * 60)
    print("Wearable Stress Dataset — Merge Script")
    print("=" * 60)

    print("\nLoading stress level scores …")
    sl_v1 = pd.read_csv(
        os.path.join(BASE_DIR, "Stress_Level_v1.csv"),
        index_col=0, encoding="latin-1",
    )
    sl_v2 = pd.read_csv(
        os.path.join(BASE_DIR, "Stress_Level_v2.csv"),
        index_col=0, encoding="latin-1",
    )

    all_entries = sorted(os.listdir(STRESS_DIR))
    v1_participants: list[tuple[str, str]] = []
    v2_participants: list[tuple[str, str]] = []

    for entry in all_entries:
        full_path = os.path.join(STRESS_DIR, entry)
        if not os.path.isdir(full_path):
            continue
        if entry in ("f14_a", "f14_b"):
            continue        # f14 handled separately below
        if entry.startswith("S"):
            v1_participants.append((full_path, entry))
        elif entry.startswith("f"):
            v2_participants.append((full_path, entry))

    # V1
    print(f"\nProcessing V1 ({len(v1_participants)} participants) …")
    df_v1 = build_trial_df("v1", v1_participants, sl_v1)

    # V2
    print(f"\nProcessing V2 ({len(v2_participants) + 1} participants, incl. f14) …")
    df_v2_base = build_trial_df("v2", v2_participants, sl_v2)

    print("    [V2] Processing f14 (f14_b only) …")
    df_f14 = process_f14(sl_v2)

    df_v2 = (
        pd.concat([df_v2_base, df_f14], ignore_index=True)
        .sort_values(["participant_id", "timestamp"])
        .reset_index(drop=True)
    )

    # Enforce column order
    meta_cols = ["participant_id", "phase", "timestamp"]
    sig_cols  = [f"{s}_{a}" for s in ALL_SIGNALS for a in AGGREGATIONS]
    tail_cols = ["reported_stress"]
    col_order = meta_cols + sig_cols + tail_cols

    df_v1 = df_v1[[c for c in col_order if c in df_v1.columns]].reset_index(drop=True)
    df_v2 = df_v2[[c for c in col_order if c in df_v2.columns]].reset_index(drop=True)

    # Summary
    print("\n" + "=" * 60)
    for label, df in [("df_v1", df_v1), ("df_v2", df_v2)]:
        print(f"  {label} : {len(df):>6,} rows × {df.shape[1]} columns")
        print(f"           {df['participant_id'].nunique()} participants, "
              f"{df['phase'].nunique()} phases each")
        print(f"           Phases: {sorted(df['phase'].unique())}")
    print("=" * 60)

    return df_v1, df_v2


if __name__ == "__main__":
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        df_v1, df_v2 = main()

    if caught_warnings:
        print("\n── Warnings raised during processing ──")
        for w in caught_warnings:
            print(f"  {w.category.__name__}: {w.message}")

    print("\ndf_v1 sample (first 5 rows):")
    print(df_v1[["participant_id", "phase", "timestamp",
                 "ACC_x_mean", "EDA_mean", "HR_mean",
                 "reported_stress"]].head())
    print(f"\nTotal columns: {df_v1.shape[1]}  {list(df_v1.columns)}")

    df_v1.to_csv("aggregated_df1_no_windows.csv", index=False)
    df_v2.to_csv("aggregated_df2_no_windows.csv", index=False)
