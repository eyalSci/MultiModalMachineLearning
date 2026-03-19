#!/usr/bin/env python3
"""
merge_wearable_stress.py
========================
Merges Empatica E4 wristband recordings from the STRESS protocol of the
"Wearable Device Dataset" (PhysioNet, DOI 10.13026/3tkz-7j93) into two
long-format DataFrames — one per trial version.

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
  Line 0  :  UTC session-start timestamp  (repeated N times for N-column files)
  Line 1  :  Sampling frequency in Hz     (repeated N times for N-column files)
  Lines 2+:  Numeric data

File format (IBI)
-----------------
  Line 0  :  UTC session-start timestamp  (repeated twice — one per column)
  Lines 1+:  col1 = seconds from session start to the detected beat
             col2 = duration of that inter-beat interval (seconds from prev beat)
  NOTE: IBI has only ONE header line, not two.

OUTPUT
------
Two DataFrames (df_v1, df_v2) in "wide" format at a uniform 32 Hz, with columns:

  participant_id  str            e.g. "S01", "f07"
  timestamp       datetime64[ns] UTC time of each 32 Hz tick (from ACC grid)
  phase           str            protocol phase label. Pre-protocol, Post-protocol,
                                 and trailing NaN rows (recording tail after the last
                                 phase) are dropped.  Any unexpected mid-session NaN
                                 rows emit a UserWarning and are also dropped.
  ACC_x           float          accelerometer x-axis   (1/64 g)
  ACC_y           float          accelerometer y-axis   (1/64 g)
  ACC_z           float          accelerometer z-axis   (1/64 g)
  BVP             float          blood volume pulse     (adimensional; NaN for f07)
  EDA             float          electrodermal activity (µS)
  TEMP            float          skin temperature       (°C; NaN for f07)
  HR              float          heart rate             (bpm; NaN for f07)
  IBI             float          inter-beat interval    (s; NaN for f07)
  reported_stress float          self-reported stress score (0–10 scale) from
                                 Stress_Level_v1/2.csv.  Placed at the last
                                 timestamp of each non-Transition phase, then
                                 propagated backward in 30-second steps while
                                 the candidate timestamp is ≥ 30 s from the
                                 phase start.  NaN everywhere else.

RESAMPLING STRATEGY
-------------------
All signals are aligned to the ACC 32 Hz master grid using index arithmetic
(not floating-point timestamp comparisons) to avoid rounding-error drift over
long sessions.  All fixed-rate signals share the same session_start timestamp,
so sample i of any signal maps to ACC step i by a known integer ratio.

  ACC   (32 Hz → 32 Hz)  used as-is; defines the master grid
  BVP   (64 Hz → 32 Hz)  downsampled: average of each consecutive pair [2i, 2i+1]
  EDA   ( 4 Hz → 32 Hz)  forward-fill: each EDA sample held for 8 ACC steps  (32÷4)
  TEMP  ( 4 Hz → 32 Hz)  forward-fill: each TEMP sample held for 8 ACC steps
  HR    ( 1 Hz → 32 Hz)  forward-fill: each HR sample held for 32 ACC steps  (32÷1)
  IBI   (event → 32 Hz)  forward-fill by timestamp; leading rows before the first
                          detected beat are back-filled with the first beat value

SPECIAL CASES HANDLED
---------------------
S02  — Duplicated signals (E4 Connect download artefact):
         Known truncation rows per file (see S02_MAX_DATA_ROWS below).
         HR and IBI boundaries are not precisely known; they are estimated
         from the validated BVP recording duration and flagged with a warning.

f07  — Protection dock never removed from wristband; PPG and TEMP sensors
         were blocked.  Signals BVP, HR, IBI, and TEMP are set to NaN.

f14  — Bluetooth connection lost mid-session; data split across two folders:
         f14_a (Baseline only, no button presses) and f14_b (rest of protocol).
         Only f14_b is used; it is renamed to "f14".  The Baseline data from
         f14_a is discarded because it has no tags to anchor phase boundaries.

USAGE
-----
  python merge_wearable_stress.py

  Or, as a module:
      from merge_wearable_stress import main
      df_v1, df_v2 = main()

  Adjust BASE_DIR (below) to point to the folder containing the unzipped
  dataset files (Wearable_Dataset/ subfolder must be present).
"""

import os
import warnings
import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION — set BASE_DIR to wherever you unzipped the dataset
# ============================================================
BASE_DIR   = "."                                    # parent of Wearable_Dataset/
STRESS_DIR = os.path.join(BASE_DIR, "Wearable_Dataset", "STRESS")


# ============================================================
# SIGNAL METADATA
# ============================================================
# Fixed-rate signals are sampled at a constant rate stored in their file header.
# IBI is event-driven (variable rate) and requires separate parsing logic.
FIXED_RATE_SIGNALS = ["ACC", "BVP", "EDA", "HR", "TEMP"]

# ACC stores three axes in three columns; all others are single-column.
ACC_COMPONENTS = ["ACC_x", "ACC_y", "ACC_z"]

SIGNAL_UNITS = {
    "ACC_x": "1/64g",        "ACC_y": "1/64g",       "ACC_z": "1/64g",
    "BVP":   "adimensional", "EDA":   "µS",
    "HR":    "bpm",          "IBI":   "seconds",      "TEMP":  "°C",
}


# ============================================================
# PHASE DEFINITIONS — V1 and V2 STRESS PROTOCOLS
# ============================================================
# The Empatica E4 records physical button presses into tags.csv.
# Each press marks the start or end of a protocol phase.
#
# Phase mapping convention used here (derived from the reference notebook):
#   tags[0] = UTC session-start (the timestamp in the signal file header,
#              prepended programmatically before the tags.csv values)
#   tags[1], tags[2], … = actual button-press timestamps from tags.csv
#
# Each tuple: (start_tag_index, end_tag_index, phase_label)
# Phase label matches the column names in Stress_Level_v1/2.csv so the join works.
#
# ── V1 protocol (S01–S18): 13 button presses in tags.csv ─────────────────────
# Tags[0] prepended → 14 total tag values (indices 0–13).
#
#  Index interval   Duration (approx.)   Phase
#  ─────────────────────────────────────────────────────────────────────────────
#  [0] → [1]        ~6 min               Pre-protocol  (sensor settling / setup)
#  [1] → [2]        ~4 min               Baseline      (resting, eyes open)
#  [2] → [3]        ~3 min               Transition_1  (pre-Stroop instructions)
#  [3] → [4]        ~2 min               Stroop        (cognitive stress task)
#  [4] → [5]        ~7 min               First Rest    (recovery)
#  [5] → [6]        ~2 min               TMCT          (Trail Making / cognitive)
#  [6] → [7]        ~5 min               Second Rest   (recovery)
#  [7] → [8]        ~0.5 min             Real Opinion  (opinion elicitation)
#  [8] → [9]        ~0.5 min             Transition_2  (brief prep)
#  [9] → [10]       ~0.5 min             Opposite Opinion
#  [10] → [11]      ~0.5 min             Transition_3  (brief prep)
#  [11] → [12]      ~0.5 min             Subtract      (mental arithmetic)
#  [12] → [13]      ~0.25 min            Post-protocol (session end marker)
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

# ── V2 protocol (f01–f18): 9 button presses in tags.csv ──────────────────────
# Tags[0] prepended → 10 total tag values (indices 0–9).
#
#  Index interval   Duration (approx.)   Phase
#  ─────────────────────────────────────────────────────────────────────────────
#  [0] → [1]        ~5 min               Pre-protocol  (sensor settling / setup)
#  [1] → [2]        ~2 min               Baseline
#  [2] → [3]        ~7.5 min             TMCT
#  [3] → [4]        ~14 min              First Rest
#  [4] → [5]        ~0.5 min             Real Opinion
#  [5] → [6]        ~0.35 min            Transition_1  (brief pause)
#  [6] → [7]        ~0.5 min             Opposite Opinion
#  [7] → [8]        ~19 min              Second Rest
#  [8] → [9]        ~0.5 min             Subtract
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


# ============================================================
# S02 TRUNCATION BOUNDARIES
# ============================================================
# Data constraint (from data_constraints.txt):
#   "Files downloaded from E4 Connect have duplicated signals.
#    Duplicated raw values start in:
#      ACC.csv  : row 49,545
#      BVP.csv  : row 99,091
#      EDA.csv and TEMP.csv: row 6,195
#    As IBI and HR files are obtained from the BVP signal through Empatica's
#    algorithm, it is not evident where the duplicated data start."
#
# Row numbers are 1-indexed file line numbers.
# Each fixed-rate file has 2 header lines (line 1 = UTC start, line 2 = Hz).
# IBI has 1 header line (line 1 = UTC start).
# First data line = file line 3 → pandas row index 0.
#
# Conversion: keep rows 0 … (file_row - 3) exclusive = file_row - 3 rows total.
#   ACC  : keep 49545 - 3 = 49542 data rows
#   BVP  : keep 99091 - 3 = 99088 data rows
#   EDA  : keep  6195 - 3 =  6192 data rows
#   TEMP : keep  6195 - 3 =  6192 data rows
#
# HR and IBI: exact boundary unknown. We estimate from BVP:
#   Valid BVP duration = 99088 samples / 64 Hz = 1548.25 seconds
#   HR at 1 Hz        → keep first 1548 rows  (≈ 1548 s)
#   IBI               → keep events with offset_s < 1548.25 s
#
# ⚠️  The HR/IBI cutoffs are approximate. Adjust S02_BVP_VALID_SECONDS
#    if you have better information about the true duplication boundary.
S02_BVP_VALID_SECONDS = 99088 / 64        # ≈ 1548.25 s — used for HR & IBI estimates

S02_MAX_DATA_ROWS = {
    "ACC":  49542,
    "BVP":  99088,
    "EDA":   6192,
    "TEMP":  6192,
    "HR":    int(S02_BVP_VALID_SECONDS),   # 1548 — estimated
}

# ============================================================
# f07 INVALID SIGNALS
# ============================================================
# "The protection dock was never removed from the wristband, covering the
#  PPG and TEMPERATURE sensors."
# → PPG sensor drives: BVP (raw), HR (derived), IBI (derived)
# → TEMP sensor drives: TEMP
# All four are set to NaN for f07; ACC and EDA remain valid.
F07_INVALID_SIGNALS = {"BVP", "HR", "IBI", "TEMP"}


# ============================================================
# LOW-LEVEL FILE READERS
# ============================================================

def _parse_start_time(filepath: str) -> pd.Timestamp:
    """
    Read the UTC session-start timestamp from line 0 of an E4 CSV file.

    For multi-column files (ACC, IBI) the timestamp is repeated once per
    column — we just take the first comma-separated value.

    Returns
    -------
    pd.Timestamp  (timezone-naive, but values are UTC by dataset convention)
    """
    with open(filepath, "r", encoding="utf-8") as fh:
        first_line = fh.readline().strip()
    return pd.Timestamp(first_line.split(",")[0].strip())


def _parse_sample_rate(filepath: str) -> float:
    """
    Read the sampling frequency (Hz) from line 1 of a fixed-rate E4 CSV file.

    For multi-column files (ACC) the rate is repeated — we take the first value.
    IBI does NOT have a sample-rate line; do not call this function for IBI.

    Returns
    -------
    float — sampling frequency in Hz
    """
    with open(filepath, "r", encoding="utf-8") as fh:
        fh.readline()                          # skip line 0 (timestamp)
        second_line = fh.readline().strip()
    return float(second_line.split(",")[0].strip())


def read_fixed_rate_signal(
    filepath: str,
    signal_name: str,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """
    Load one fixed-rate Empatica E4 signal file (ACC, BVP, EDA, HR, or TEMP).

    Timestamps are reconstructed arithmetically from the session-start time
    and the sample index:
        timestamp[i] = session_start + i / sample_rate   (seconds)

    Parameters
    ----------
    filepath    : path to the signal CSV file
    signal_name : one of "ACC", "BVP", "EDA", "HR", "TEMP"
    max_rows    : if given, only keep this many data rows (used for S02 truncation)

    Returns
    -------
    pd.DataFrame with columns:
        timestamp            pd.Timestamp
        ACC_x, ACC_y, ACC_z  (only when signal_name == "ACC")
        <signal_name>        (for all other signals)
    """
    start_time  = _parse_start_time(filepath)
    sample_rate = _parse_sample_rate(filepath)

    if signal_name == "ACC":
        # ACC stores x, y, z in three columns
        data = pd.read_csv(
            filepath, skiprows=2, header=None,
            names=["ACC_x", "ACC_y", "ACC_z"],
        )
    else:
        data = pd.read_csv(
            filepath, skiprows=2, header=None,
            names=[signal_name],
        )

    # Apply truncation limit before generating timestamps (avoids wasted work)
    if max_rows is not None:
        data = data.iloc[:max_rows].copy()

    # Build a DatetimeIndex from the arithmetic sequence of offsets
    n_samples = len(data)
    offsets   = pd.to_timedelta(np.arange(n_samples) / sample_rate, unit="s")
    data.insert(0, "timestamp", start_time + offsets)

    return data.reset_index(drop=True)


def read_ibi_signal(filepath: str) -> pd.DataFrame:
    """
    Load the IBI (Inter-Beat Interval) Empatica E4 file.

    IBI differs from fixed-rate signals in two ways:
      1. Only ONE header line (the start timestamp, repeated for each column).
      2. Each row is an event, not a regularly-spaced sample.
         Column 0: seconds elapsed from session start to the detected beat.
         Column 1: duration of the inter-beat interval (seconds since prev beat).

    The absolute timestamp for each event is:
        timestamp = session_start + column_0_offset

    Returns
    -------
    pd.DataFrame with columns:
        timestamp  pd.Timestamp — when the beat was detected
        IBI        float        — inter-beat interval duration (seconds)

    Returns an empty DataFrame if the file is missing, empty, or unreadable.
    """
    start_time = _parse_start_time(filepath)

    try:
        data = pd.read_csv(
            filepath, skiprows=1, header=None,
            names=["offset_s", "IBI"],
        )
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=["timestamp", "IBI"])

    if data.empty:
        return pd.DataFrame(columns=["timestamp", "IBI"])

    # Coerce to numeric (guards against stray whitespace / malformed rows)
    data["offset_s"] = pd.to_numeric(data["offset_s"], errors="coerce")
    data["IBI"]      = pd.to_numeric(data["IBI"],      errors="coerce")
    data = data.dropna(subset=["offset_s"])

    # Convert offset (seconds) to absolute timestamp
    timestamps = start_time + pd.to_timedelta(data["offset_s"], unit="s")
    data.insert(0, "timestamp", timestamps.values)
    data = data.drop(columns=["offset_s"])

    return data.reset_index(drop=True)


def read_tags(folder_path: str) -> list[pd.Timestamp]:
    """
    Read all button-press timestamps from a participant's tags.csv.

    Each row in tags.csv is one UTC datetime string (one button press).
    Returns a list of pd.Timestamp objects in file order (chronological).

    Returns an empty list if tags.csv is absent or empty.
    """
    tags_path = os.path.join(folder_path, "tags.csv")
    if not os.path.exists(tags_path):
        warnings.warn(f"tags.csv not found in: {folder_path}")
        return []
    try:
        df = pd.read_csv(tags_path, header=None, names=["ts"])
        if df.empty:
            return []
        return [pd.Timestamp(t) for t in df["ts"].dropna().tolist()]
    except pd.errors.EmptyDataError:
        return []


# ============================================================
# PHASE ASSIGNMENT
# ============================================================

def build_phase_boundaries(
    session_start: pd.Timestamp,
    tag_timestamps: list[pd.Timestamp],
    phase_map: list[tuple],
) -> list[tuple]:
    """
    Translate a phase map (index pairs) into concrete (start_dt, end_dt, label)
    tuples using the session-start timestamp and button-press tags.

    The session_start is prepended as tags[0]; the actual button presses from
    tags.csv become tags[1], tags[2], …

    Parameters
    ----------
    session_start   : UTC session-start from the signal file header
    tag_timestamps  : list of button-press timestamps from tags.csv
    phase_map       : list of (start_idx, end_idx, label) tuples (see module top)

    Returns
    -------
    list of (start_dt, end_dt, label) — concrete datetime boundaries, in order.
    Intervals are excluded when the required tag index is out of range.
    """
    # Prepend session start so that index 0 always refers to the recording start
    all_tags = [session_start] + list(tag_timestamps)
    n_tags   = len(all_tags)

    boundaries = []
    for start_idx, end_idx, label in phase_map:
        if start_idx < n_tags and end_idx < n_tags:
            boundaries.append((all_tags[start_idx], all_tags[end_idx], label))
        else:
            # Graceful degradation: skip phases we cannot bound
            pass

    return boundaries


def assign_phases(
    timestamps: pd.Series,
    phase_boundaries: list[tuple],
) -> pd.Series:
    """
    Assign a phase label to every timestamp using vectorised interval lookups.

    A timestamp t belongs to phase P if:
        phase_start <= t < phase_end

    Timestamps outside any defined phase interval receive pd.NA.

    Parameters
    ----------
    timestamps       : pd.Series of pd.Timestamp
    phase_boundaries : list of (start_dt, end_dt, label) — from build_phase_boundaries()

    Returns
    -------
    pd.Series[str]   — same length as ``timestamps``, pd.NA where unassigned
    """
    phases = pd.array([pd.NA] * len(timestamps), dtype="object")

    for start_dt, end_dt, label in phase_boundaries:
        mask = (timestamps >= start_dt) & (timestamps < end_dt)
        phases[mask.values] = label

    return pd.Series(phases, index=timestamps.index, dtype="object")


# ============================================================
# PER-PARTICIPANT PROCESSING
# ============================================================

def _signals_to_wide(
    folder_path: str,
    participant_id: str,
    is_s02: bool,
    is_f07: bool,
) -> pd.DataFrame:
    """
    Read all signal files from one participant folder and return them
    aligned to a common 32 Hz time grid in wide format.

    The 32 Hz master grid is defined by the ACC timestamps (already recorded
    at exactly 32 Hz). Every other signal is resampled onto this grid:

      Signal   Native Hz   Strategy
      ─────────────────────────────────────────────────────────────────────────
      ACC      32          Used as-is — defines the master grid
      BVP      64          Downsampled: average of each consecutive sample pair.
                           BVP pair i → grid row i = (BVP[2i] + BVP[2i+1]) / 2
      EDA       4          Upsampled: forward-fill. Each EDA sample held for
                           8 ACC steps  (32 Hz ÷ 4 Hz = 8)
      TEMP      4          Same forward-fill strategy as EDA
      HR        1          Upsampled: forward-fill. Each HR sample held for
                           32 ACC steps (32 Hz ÷ 1 Hz = 32)
      IBI      event       Forward-fill by timestamp. The IBI value at each
                           detected beat is carried forward until the next beat.
                           Rows before the first beat are back-filled with the
                           first available IBI value to avoid leading NaN.

    Index-based resampling (ACC / BVP / EDA / TEMP / HR)
    ─────────────────────────────────────────────────────
    All fixed-rate files share the same session_start timestamp, so sample i
    of any signal maps to ACC step i by a fixed integer ratio — no floating-
    point timestamp comparison needed.  This prevents drift over long sessions.

    Parameters
    ----------
    folder_path    : absolute path to the participant's data folder
    participant_id : used only for warning messages
    is_s02         : apply S02-specific row-truncation limits
    is_f07         : set BVP / HR / IBI / TEMP to NaN (sensor physically blocked)

    Returns
    -------
    pd.DataFrame with columns:
        timestamp, ACC_x, ACC_y, ACC_z, BVP, EDA, TEMP, HR, IBI
    One row per 32 Hz tick.  BVP/HR/IBI/TEMP are NaN for f07.
    """

    # ── 1.  ACC — define the master 32 Hz grid ─────────────────────────────
    acc_path = os.path.join(folder_path, "ACC.csv")
    if not os.path.exists(acc_path):
        warnings.warn(f"{participant_id}: ACC.csv not found — cannot build grid.")
        return pd.DataFrame()

    acc_max = S02_MAX_DATA_ROWS["ACC"] if is_s02 else None
    acc_df  = read_fixed_rate_signal(acc_path, "ACC", max_rows=acc_max)
    n          = len(acc_df)                      # master grid length
    timestamps = acc_df["timestamp"].values        # numpy datetime64 array

    # ── 2.  BVP — downsample 64 Hz → 32 Hz by averaging consecutive pairs ──
    bvp_path = os.path.join(folder_path, "BVP.csv")
    if os.path.exists(bvp_path) and not is_f07:
        bvp_max = S02_MAX_DATA_ROWS["BVP"] if is_s02 else None
        bvp_df  = read_fixed_rate_signal(bvp_path, "BVP", max_rows=bvp_max)
        bvp_raw = bvp_df["BVP"].values.astype(float)

        # Drop a trailing odd sample so the array can be cleanly split into pairs
        n_even     = (len(bvp_raw) // 2) * 2
        bvp_paired = bvp_raw[:n_even].reshape(-1, 2).mean(axis=1)

        # Align to the ACC grid length: truncate if longer, forward-fill if shorter
        n_pairs = len(bvp_paired)
        bvp_32  = np.empty(n, dtype=float)
        fill_to = min(n_pairs, n)
        bvp_32[:fill_to] = bvp_paired[:fill_to]
        if fill_to < n:
            # Forward-fill: repeat the last averaged BVP value for the tail
            bvp_32[fill_to:] = bvp_paired[fill_to - 1] if fill_to > 0 else np.nan
    else:
        # f07: PPG sensor was physically blocked — BVP is meaningless
        bvp_32 = np.full(n, np.nan)
        if not os.path.exists(bvp_path):
            warnings.warn(f"{participant_id}: BVP.csv not found.")


    # ── 3.  EDA — upsample 4 Hz → 32 Hz via forward-fill ──────────────────
    eda_path = os.path.join(folder_path, "EDA.csv")
    if os.path.exists(eda_path):
        eda_max = S02_MAX_DATA_ROWS["EDA"] if is_s02 else None
        eda_df  = read_fixed_rate_signal(eda_path, "EDA", max_rows=eda_max)
        eda_raw = eda_df["EDA"].values.astype(float)
        # 32 Hz ÷ 4 Hz = 8: ACC step i maps to EDA sample i // 8
        idx_eda = np.minimum(np.arange(n) // 8, len(eda_raw) - 1)
        eda_32  = eda_raw[idx_eda]
    else:
        warnings.warn(f"{participant_id}: EDA.csv not found.")
        eda_32 = np.full(n, np.nan)

    # ── 4.  TEMP — upsample 4 Hz → 32 Hz via forward-fill ─────────────────
    temp_path = os.path.join(folder_path, "TEMP.csv")
    if os.path.exists(temp_path) and not is_f07:
        temp_max = S02_MAX_DATA_ROWS["TEMP"] if is_s02 else None
        temp_df  = read_fixed_rate_signal(temp_path, "TEMP", max_rows=temp_max)
        temp_raw = temp_df["TEMP"].values.astype(float)
        # Same 8× ratio as EDA
        idx_temp = np.minimum(np.arange(n) // 8, len(temp_raw) - 1)
        temp_32  = temp_raw[idx_temp]
    else:
        # f07: TEMP sensor was physically blocked
        temp_32 = np.full(n, np.nan)
        if not os.path.exists(temp_path):
            warnings.warn(f"{participant_id}: TEMP.csv not found.")

    # ── 5.  HR — upsample 1 Hz → 32 Hz via forward-fill ───────────────────
    hr_path = os.path.join(folder_path, "HR.csv")
    if os.path.exists(hr_path) and not is_f07:
        hr_max = S02_MAX_DATA_ROWS["HR"] if is_s02 else None
        if is_s02:
            warnings.warn(
                f"S02 HR: truncating at estimated row {hr_max} "
                f"(derived from BVP valid duration ≈ {S02_BVP_VALID_SECONDS:.1f} s). "
                "Exact duplication boundary is unknown — see data_constraints.txt."
            )
        hr_df  = read_fixed_rate_signal(hr_path, "HR", max_rows=hr_max)
        hr_raw = hr_df["HR"].values.astype(float)
        # 32 Hz ÷ 1 Hz = 32: ACC step i maps to HR sample i // 32
        idx_hr = np.minimum(np.arange(n) // 32, len(hr_raw) - 1)
        hr_32  = hr_raw[idx_hr]
    else:
        # f07: HR is derived from the blocked PPG — physically invalid
        hr_32 = np.full(n, np.nan)
        if not os.path.exists(hr_path):
            warnings.warn(f"{participant_id}: HR.csv not found.")


    # ── 6.  IBI — forward-fill irregular events onto the 32 Hz grid ────────
    ibi_path = os.path.join(folder_path, "IBI.csv")
    if os.path.exists(ibi_path) and not is_f07:
        ibi_df = read_ibi_signal(ibi_path)

        if not ibi_df.empty:
            # S02: drop IBI events beyond the estimated valid BVP duration
            if is_s02:
                ibi_start = _parse_start_time(ibi_path)
                cutoff_dt = ibi_start + pd.Timedelta(seconds=S02_BVP_VALID_SECONDS)
                before    = len(ibi_df)
                ibi_df    = ibi_df[ibi_df["timestamp"] < cutoff_dt].copy()
                warnings.warn(
                    f"S02 IBI: kept {len(ibi_df)}/{before} events with timestamp < "
                    f"session_start + {S02_BVP_VALID_SECONDS:.1f} s "
                    "(estimated cutoff from BVP valid duration; see data_constraints.txt)."
                )

            # Build a timestamp-indexed Series of IBI values, then reindex to
            # the 32 Hz grid using forward-fill.
            ibi_series = ibi_df.set_index("timestamp")["IBI"].sort_index()
            # Guard against duplicate timestamps in the IBI file (observed in S02
            # after truncation): keep the last recorded value for any timestamp
            # that appears more than once.
            ibi_series = ibi_series[~ibi_series.index.duplicated(keep="last")]
            grid_index = pd.DatetimeIndex(timestamps)

            # Union the event timestamps and the grid, forward-fill, then
            # project back down to grid timestamps only.
            ibi_reindexed = (
                ibi_series
                .reindex(ibi_series.index.union(grid_index))
                .ffill()
                .reindex(grid_index)
            )
            # Back-fill leading NaN: rows before the very first beat carry the
            # first available IBI value rather than being left as NaN.
            ibi_32 = ibi_reindexed.bfill().values.astype(float)
        else:
            # No beats detected in this session — cannot produce meaningful IBI
            ibi_32 = np.full(n, np.nan)
    else:
        # f07: IBI is derived from the blocked PPG — physically invalid
        ibi_32 = np.full(n, np.nan)
        if not os.path.exists(ibi_path):
            warnings.warn(f"{participant_id}: IBI.csv not found.")


    # ── 7.  Assemble the wide 32 Hz DataFrame ───────────────────────────────
    return pd.DataFrame({
        "timestamp": timestamps,
        "ACC_x":     acc_df["ACC_x"].values.astype(float),
        "ACC_y":     acc_df["ACC_y"].values.astype(float),
        "ACC_z":     acc_df["ACC_z"].values.astype(float),
        "BVP":       bvp_32,
        "EDA":       eda_32,
        "TEMP":      temp_32,
        "HR":        hr_32,
        "IBI":       ibi_32,
    })



# ── Phase trimming ─────────────────────────────────────────────────────────

# Phases that exist solely as recording bookends and carry no experimental
# content — dropped after phase assignment.
BOOKEND_PHASES = {"Pre-protocol", "Post-protocol"}


def _trim_phases(df: pd.DataFrame, participant_id: str) -> pd.DataFrame:
    """
    Remove bookend and NaN phase rows from a per-participant wide DataFrame.

    Three operations are applied in order:

    1. Drop bookend phases ("Pre-protocol", "Post-protocol").
       These are recording artefacts, not experimental conditions.

    2. Identify the trailing NaN block — all rows after the last row whose
       phase is not NaN.  This corresponds to the E4 continuing to record
       after the final button press.  These rows are dropped silently.

    3. Check whether any NaN-phase rows remain after step 2 (i.e. mid-session
       gaps).  These are unexpected: a NaN in the middle of the protocol means
       a timestamp fell outside every defined phase boundary, which could
       indicate a missing tag or a phase-map alignment error.  A UserWarning
       is emitted for each such participant, and those rows are also dropped.

    Parameters
    ----------
    df             : per-participant wide DataFrame with a 'phase' column
    participant_id : used in warning messages

    Returns
    -------
    pd.DataFrame with bookend and NaN rows removed, index reset.
    """
    # ── 1. Drop bookend phases ───────────────────────────────────────────────
    df = df[~df["phase"].isin(BOOKEND_PHASES)].copy()

    # ── 2. Drop trailing NaN tail ────────────────────────────────────────────
    # Find the integer position of the last non-NaN phase row.
    valid_mask   = df["phase"].notna()
    if not valid_mask.any():
        # Entire DataFrame has no valid phase — nothing to keep
        warnings.warn(
            f"{participant_id}: no valid phase rows remain after dropping "
            "bookend phases. Returning empty DataFrame."
        )
        return pd.DataFrame(columns=df.columns)

    last_valid_pos = valid_mask.values.nonzero()[0][-1]   # last True index
    # Rows strictly after the last valid phase are the trailing NaN tail
    trailing_nan_count = len(df) - (last_valid_pos + 1)
    df = df.iloc[: last_valid_pos + 1].copy()

    # ── 3. Warn about any remaining mid-session NaN rows ─────────────────────
    mid_nan_mask  = df["phase"].isna()
    mid_nan_count = mid_nan_mask.sum()
    if mid_nan_count > 0:
        # Find which timestamps are affected, to help with debugging
        mid_nan_ts = df.loc[mid_nan_mask, "timestamp"]
        warnings.warn(
            f"{participant_id}: {mid_nan_count} unexpected mid-session NaN-phase "
            f"rows found (not part of the trailing tail). "
            f"First occurrence: {mid_nan_ts.iloc[0]}. "
            "This may indicate a missing tag or a phase-map misalignment. "
            "These rows are dropped."
        )
        df = df[~mid_nan_mask].copy()

    return df.reset_index(drop=True)


def process_participant(
    folder_path: str,
    participant_id: str,
    trial: str,
) -> pd.DataFrame:
    """
    Load and return all signal data for one participant as a wide-format
    32 Hz DataFrame, with phase labels assigned from tags.csv.

    This function handles S02 and f07 special cases.
    f14 is handled separately by process_f14().

    Parameters
    ----------
    folder_path    : absolute path to the participant's data folder
    participant_id : e.g. "S01", "f07"
    trial          : "v1" (S-participants) or "v2" (f-participants); used only
                     to select the correct phase map — not stored as a column.

    Returns
    -------
    pd.DataFrame with columns:
        participant_id, timestamp, phase,
        ACC_x, ACC_y, ACC_z, BVP, EDA, TEMP, HR, IBI
    """
    is_s02 = (participant_id == "S02")
    is_f07 = (participant_id == "f07")

    # ── Read and resample all signals onto the 32 Hz ACC grid ───────────────
    wide_df = _signals_to_wide(folder_path, participant_id, is_s02, is_f07)

    if wide_df.empty:
        warnings.warn(f"No data loaded for participant {participant_id}.")
        return pd.DataFrame()

    # ── Build phase boundaries from tags ────────────────────────────────────
    # EDA.csv is used as the canonical session-start reference (always present).
    phase_map     = V1_PHASE_MAP if trial == "v1" else V2_PHASE_MAP
    session_start = _parse_start_time(os.path.join(folder_path, "EDA.csv"))
    tag_ts        = read_tags(folder_path)
    boundaries    = build_phase_boundaries(session_start, tag_ts, phase_map)

    # ── Prepend metadata columns  ────────────────────────
    wide_df.insert(0, "participant_id", participant_id)
    wide_df.insert(2, "phase",          assign_phases(wide_df["timestamp"], boundaries))

    # ── Drop trailing tail, bookend phases, and warn on mid-session NaN ──────
    return _trim_phases(wide_df, participant_id)


def process_f14() -> pd.DataFrame:
    """
    Special case: participant f14 (v2 trial).

    The Bluetooth connection was lost mid-session:
      - f14_a : Baseline only — no button presses, so phase boundaries cannot
                be established from this half alone.  It is discarded.
      - f14_b : Contains the rest of the protocol with all 9 button presses.
                This is the only half used; it is relabelled as "f14".

    Returns
    -------
    pd.DataFrame — same schema as process_participant()
    """
    folder_b = os.path.join(STRESS_DIR, "f14_b")

    wide_df = _signals_to_wide(folder_b, "f14_b", is_s02=False, is_f07=False)

    if wide_df.empty:
        warnings.warn("No data found for f14_b.")
        return pd.DataFrame()

    # ── Phase boundaries from f14_b tags and its own session start ──────────
    session_start = _parse_start_time(os.path.join(folder_b, "EDA.csv"))
    tag_ts        = read_tags(folder_b)
    boundaries    = build_phase_boundaries(session_start, tag_ts, V2_PHASE_MAP)

    # ── Prepend metadata columns ────────────────────────
    wide_df.insert(0, "participant_id", "f14")
    wide_df.insert(2, "phase",          assign_phases(wide_df["timestamp"], boundaries))

    # ── Drop trailing tail, bookend phases, and warn on mid-session NaN ──────
    return _trim_phases(wide_df, "f14")




# ============================================================
# STRESS SCORE LOADING & ASSIGNMENT
# ============================================================

def load_stress_levels(filepath: str) -> pd.DataFrame:
    """
    Load a Stress_Level_v*.csv file and return it in long format.

    Raw file layout: one row per participant, one column per phase, values
    are self-reported stress scores on a 0–10 scale.

    Returns
    -------
    pd.DataFrame with columns:
        participant_id  str
        phase           str   — matches the phase labels used in the DataFrame
        reported_stress float
    """
    df = pd.read_csv(filepath, index_col=0, encoding="latin-1")
    df.index.name = "participant_id"
    df = df.reset_index()
    melted = df.melt(
        id_vars="participant_id",
        var_name="phase",
        value_name="reported_stress",
    )
    melted["participant_id"] = melted["participant_id"].str.strip()
    return melted


def assign_reported_stress(
    df: pd.DataFrame,
    stress_lut: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add a 'reported_stress' column to a wide 32 Hz DataFrame.

    Placement rule (applied per participant × phase, skipping Transition phases):

      1. Always place the stress score at the very last timestamp of the phase.
      2. Step backward in 30-second intervals: place the score at
         last_ts − 30 s, last_ts − 60 s, … as long as the candidate
         timestamp is still at least 30 seconds after the phase start.
         (Rule: candidate_ts − phase_start ≥ 30 s)
      3. All other rows in the phase remain NaN.

    At 32 Hz, 30 seconds = exactly 960 rows.  Backward steps are therefore
    taken as row-index offsets of 960 rather than floating-point timestamp
    arithmetic, which avoids any sub-millisecond rounding issues.

    Parameters
    ----------
    df         : merged wide DataFrame (all participants, one trial version)
    stress_lut : long-format stress scores from load_stress_levels()

    Returns
    -------
    pd.DataFrame — same as input with 'reported_stress' column appended.
    NaN for Transition phases, for rows outside the placement grid, and for
    any participant/phase pair not found in stress_lut.
    """
    STEP_ROWS   = 30 * 32          # 960 rows = 30 s at 32 Hz
    MIN_SECS    = 30.0             # minimum seconds from phase start to place a score

    # Initialise the column as all-NaN float
    df = df.copy()
    df["reported_stress"] = np.nan

    # Build a fast lookup dict: (participant_id, phase) → score
    score_map: dict[tuple, float] = {
        (row.participant_id, row.phase): row.reported_stress
        for row in stress_lut.itertuples(index=False)
    }

    # Process each (participant, phase) group independently
    for (pid, phase), grp in df.groupby(
        ["participant_id", "phase"], sort=False, observed=True
    ):
        # Skip Transition phases — no stress score is recorded for them
        if isinstance(phase, str) and phase.startswith("Transition"):
            continue

        score = score_map.get((pid, phase))
        if score is None or (isinstance(score, float) and np.isnan(score)):
            # No score available for this participant/phase — leave as NaN
            warnings.warn(f"Reported stress on stage {phase} for {pid} not found.")
            continue

        # Integer row positions within this group (0 = first row of phase)
        n          = len(grp)
        phase_start_ts = grp["timestamp"].iloc[0]

        # Collect row positions (within grp) where the score will be placed.
        # Position n-1 is always included (last timestamp of the phase).
        positions = []
        pos = n - 1                           # start at the last row
        while pos >= 0:
            ts = grp["timestamp"].iloc[pos]
            secs_from_start = (ts - phase_start_ts).total_seconds()

            if pos == n - 1:
                # Last timestamp: always place regardless of phase duration
                positions.append(pos)
            elif secs_from_start >= MIN_SECS:
                # Subsequent backward steps: only if ≥ 30 s from phase start
                positions.append(pos)
            else:
                # Too close to the phase start — stop stepping backward
                break

            pos -= STEP_ROWS                  # step back 30 s (960 rows)

        # Write the score into the main DataFrame at the resolved positions
        target_indices = grp.index[positions]
        df.loc[target_indices, "reported_stress"] = score

    return df


# ============================================================
# MAIN ASSEMBLY
# ============================================================

def build_trial_df(
    trial: str,
    participant_list: list[tuple[str, str]],
) -> pd.DataFrame:
    """
    Process all participants for one trial version and concatenate results.

    Parameters
    ----------
    trial            : "v1" or "v2"
    participant_list : list of (folder_path, participant_id) tuples

    Returns
    -------
    pd.DataFrame — all participants stacked in wide 32 Hz format
    """
    dfs = []
    for folder_path, pid in participant_list:
        print(f"    [{trial.upper()}] Processing {pid} …")
        try:
            df = process_participant(folder_path, pid, trial)
            if not df.empty:
                dfs.append(df)
        except Exception as exc:
            warnings.warn(f"Failed to process {pid}: {exc}")

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


# ============================================================
# ENTRY POINT
# ============================================================

def main() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build and return the two merged DataFrames.

    Returns
    -------
    df_v1 : pd.DataFrame — trial v1 (participants S01–S18), wide 32 Hz format
    df_v2 : pd.DataFrame — trial v2 (participants f01–f18), wide 32 Hz format

    Both DataFrames share the same column schema (see module docstring).
    No 'trial' column is included; v1 vs v2 membership is implicit in the
    participant ID prefix (S = v1, f = v2).
    """
    print("=" * 60)
    print("Wearable Stress Dataset — Merge Script")
    print("=" * 60)

    # ── Load self-reported stress scores ─────────────────────────────────────
    print("\nLoading stress level scores …")
    sl_v1 = load_stress_levels(os.path.join(BASE_DIR, "Stress_Level_v1.csv"))
    sl_v2 = load_stress_levels(os.path.join(BASE_DIR, "Stress_Level_v2.csv"))

    # ── Discover participant folders ──────────────────────────────────────────
    all_entries = sorted(os.listdir(STRESS_DIR))

    v1_participants: list[tuple[str, str]] = []
    v2_participants: list[tuple[str, str]] = []

    for entry in all_entries:
        full_path = os.path.join(STRESS_DIR, entry)
        if not os.path.isdir(full_path):
            continue
        if entry in ("f14_a", "f14_b"):
            # f14 is handled as a special case — only f14_b is used, skip both here
            continue
        if entry.startswith("S"):
            v1_participants.append((full_path, entry))
        elif entry.startswith("f"):
            v2_participants.append((full_path, entry))

    # ── V1 (S01–S18) ─────────────────────────────────────────────────────────
    print(f"\nProcessing V1 ({len(v1_participants)} participants) …")
    df_v1 = build_trial_df("v1", v1_participants)

    # ── V2 (f01–f18, including f14 from f14_b only) ───────────────────────────
    print(f"\nProcessing V2 ({len(v2_participants) + 1} participants, incl. f14) …")
    df_v2_base = build_trial_df("v2", v2_participants)

    print("    [V2] Processing f14 (f14_b only) …")
    df_f14 = process_f14()

    # Combine regular V2 participants with the f14 record
    df_v2 = (
        pd.concat([df_v2_base, df_f14], ignore_index=True)
        .sort_values(["participant_id", "timestamp"])
        .reset_index(drop=True)
    )

    # ── Assign self-reported stress scores ───────────────────────────────────
    print("\nAssigning reported stress scores …")
    df_v1 = assign_reported_stress(df_v1, sl_v1)
    df_v2 = assign_reported_stress(df_v2, sl_v2)

    # ── Summary ───────────────────────────────────────────────────────────────
    signal_cols = ["ACC_x", "ACC_y", "ACC_z", "BVP", "EDA", "TEMP", "HR", "IBI"]
    print("\n" + "=" * 60)
    print(f"  df_v1 : {len(df_v1):>12,} rows  ×  {df_v1.shape[1]} columns")
    print(f"          {df_v1['participant_id'].nunique()} participants | "
          f"32 Hz | signals: {signal_cols}")
    print(f"          Phases : {sorted(df_v1['phase'].dropna().unique())}")
    print()
    print(f"  df_v2 : {len(df_v2):>12,} rows  ×  {df_v2.shape[1]} columns")
    print(f"          {df_v2['participant_id'].nunique()} participants | "
          f"32 Hz | signals: {signal_cols}")
    print(f"          Phases : {sorted(df_v2['phase'].dropna().unique())}")
    print("=" * 60)

    return df_v1, df_v2


if __name__ == "__main__":
    # When run directly, execute and show the first few rows of each DataFrame.
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        df_v1, df_v2 = main()

    if caught_warnings:
        print("\n── Warnings raised during processing ──")
        for w in caught_warnings:
            print(f"  {w.category.__name__}: {w.message}")

    print("\ndf_v1 sample (first 5 rows):")
    print(df_v1.head())
    print("\ndf_v1 dtypes:")
    print(df_v1.dtypes)
    print("\ndf_v2 sample (first 5 rows):")
    print(df_v2.head())
    df_v1.to_csv("merged_df1.csv", index=False)
    df_v2.to_csv("merged_df2.csv", index=False)