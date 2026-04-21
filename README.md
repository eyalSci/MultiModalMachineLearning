# Conditional Granger Causality for Physiological Stress Detection

> Identifying causal relationships between wearable sensor signals under cognitive stress using LSTMs and Vector Autoregressive models.

---

## Overview

This project investigates two causal questions using physiological signals captured by wearable devices:

1. **Do past values of physiological signals Granger-cause self-reported stress?** That is, does adding a signal's history to the stress model's own past values reduce prediction error?
2. **How do causal relationships between physiological signals change between stress and rest states?**

Both questions are addressed using **Granger causality analysis** — a framework for testing predictive causality in time series — comparing a classical statistical approach (Vector Autoregressive model) against a deep learning approach (Long Short-Term Memory network). The Granger score is defined as `GS = (MSE_restricted − MSE_unrestricted) × 1000`, where a positive value means the added signal improves prediction beyond the target's own history.

This work has direct relevance to:

- **Mental health tech** — real-time stress monitoring in consumer and clinical wearables
- **Occupational health & safety** — detecting cognitive overload in high-stakes environments (healthcare, aviation, logistics)
- **HR & workforce analytics** — objective physiological indicators of employee wellbeing
- **Neurotech & biofeedback** — understanding autonomic coupling for adaptive systems

---

## Key Findings

| Finding | Method | Signal(s) | Effect |
|---|---|---|---|
| Signal *variability* Granger-causes stress more than mean level | Linear (VAR) | `ACC.z (sd)`, `BVP (sd)`, `ACC.x (sd)` | +10.4%, +6.7%, +1.2% MSE improvement |
| Depth motion Granger-causes vertical motion more during stress | LSTM | `ACC.z → ACC.y` | ΔGS = +3.21, p=0.01 |
| EDA predicts movement less effectively under stress | LSTM | `EDA → ACC.y` | ΔGS = −2.68, p=0.04 |
| Vertical acceleration Granger-causes heart rate more strongly under stress | Linear (VAR) | `ACC.y → HR` | ΔF = +17.68 (replicated in V2) |
| EDA becomes more informative for blood volume pulse under stress | Linear (VAR) | `EDA → BVP` | ΔGS = +11.30 (replicated in V2) |
| EDA–HR temporal coupling *weakens* under stress | Linear (VAR) | `EDA → HR` | ΔF = −12.42 |

**Bottom line:** Stress does not uniformly amplify all physiological couplings — it selectively strengthens some signal relationships while disrupting others. This selectivity is a meaningful signal for downstream stress classification models.

---

## Dataset

**Source:** [PhysioNet — Wearable Device Dataset from Induced Stress and Structured Exercise Sessions](https://physionet.org/content/wearable-device-dataset/1.0.1/)

- **Device:** Empatica E4 wristband
- **Participants:** 36 healthy adults (V1: S01–S18, V2: f01–f18)
- **Signals:** EDA, BVP, HR, IBI, TEMP, ACC (x/y/z)
- **Stress tasks:** Stroop, TMCT, Mental Subtraction, Real/Opposite Opinion tasks
- **Sampling rates:** 1–64 Hz (signal-dependent), downsampled to 1 Hz for analysis

---

## Architecture (RQ2)

The pipeline below applies to the conditional Granger causality between signals. For RQ1, signals are instead aggregated into phase-level windows and evaluated against self-reported stress labels using LOO-CV on V1.

```
Multimodal Wearable Input (HR, EDA, TEMP, BVP, ACC)
          │
          ▼
  Downsample to 1 Hz + z-score normalisation (per participant)
          │
    ┌─────┴─────┐
    │           │
  LSTM        VAR (Linear)
  approach    approach
    │           │
  20s phase   Phase segmentation
  segments    (stress vs. rest ≥60s)
    │           │
  MSE-based   F-statistic
  Granger     per signal pair
  score           │
    │           Paired t-test
  LME model   (significance)
  (significance)
    └─────┬─────┘
          ▼
  Physiological coupling under stress
```

**Training:** V1 cohort (18 participants)  
**Testing:** V2 cohort (17 participants)  
**Granger score:** `GS = (MSE_restricted − MSE_unrestricted) × 1000`  
Positive = signal X adds predictive power for Y beyond Y's own history.

---

## Models

### Vector Autoregressive (VAR) — Linear Granger Causality
- Lag: 1 second; unrestricted model: `Y(t) = c + aY(t−1) + bX(t−1)`
- F-statistic computed per participant per signal pair
- Significance: paired t-test at participant level (α = 0.05)

### Long Short-Term Memory (LSTM) — Non-linear Granger Causality

**RQ1:** 1 hidden layer, 4 units (kept small to avoid overfitting on sparse labels), 300 epochs, patience=30, dropout=0.3, L2=1e-3, lr=1e-3, Adam optimizer

**RQ2:** 1 hidden layer, 64 units, 20-second non-overlapping phase segments, stateful across segments within a phase, 30 epochs, patience=5, gradient clipping=1, lr=1e-3, Adam optimizer. Significance: Linear Mixed-Effects model (phase and participant as random effects, stage type as fixed effect)

---

## Results Summary

### RQ1 — Granger Causality on Stress (V1, LOO-CV, N=18)

Both models test whether a signal's past values Granger-cause self-reported stress — i.e., whether the unrestricted model (past stress + past signal) outperforms the restricted model (past stress alone).

**Linear model:** Only `ACC.z (sd)`, `ACC.x (sd)`, and `BVP (sd)` improved on the restricted baseline. Signal *variability* is more informative than mean level for Granger-causing stress.

**LSTM model:** Most features reduced MSE relative to the restricted baseline, with `HR (sd)` showing the largest improvement (+42%). However, results lacked statistical power due to sparse labels (8 observations per participant).

### RQ2 — Conditional Granger Causality Between Signals (V2, N=17)

Directional patterns from V1 replicated on the held-out V2 cohort. Statistically significant pairs:

| Pair | Direction | Model | p-value |
|---|---|---|---|
| `BVP → TEMP` | Stronger under stress | VAR | 0.026 (V1 only) |
| `ACC.z → ACC.y` | Stronger under stress | LSTM | 0.01 |
| `EDA → ACC.y` | Stronger at rest | LSTM | 0.04 |
| `ACC.z → BVP` | Stronger at rest | LSTM | 0.03 |

---

## Repository Structure

```
├── Wearable_Dataset/STRESS/          # Raw wearable data (per participant)
├── granger_network_outputs/          # Saved model outputs and results
├── aggregate_wearable_stress.py      # Aggregate signals into phase-level windows
├── merge_wearable_stress.py          # Merge signal streams with stress labels
├── granger_var.py                    # VAR Granger causality on stress (RQ1)
├── granger_lstm.py                   # LSTM Granger causality on stress (RQ1)
├── granger_pair_phases.py            # VAR conditional Granger causality between signals (RQ2)
├── granger_lstm_pair_phases.py       # LSTM conditional Granger causality between signals (RQ2)
├── Data_Dictionary.csv               # Signal and feature descriptions
├── Stress_Level_v1.csv               # Self-reported stress labels (V1)
└── Stress_Level_v2.csv               # Self-reported stress labels (V2)
```

---

## Getting Started

### Prerequisites

```bash
pip install numpy pandas scikit-learn torch statsmodels scipy
```

### Data

Download the dataset from [PhysioNet](https://physionet.org/content/wearable-device-dataset/1.0.1/) and place it under `Wearable_Dataset/STRESS/`.

### Run

```bash
# Preprocess: aggregate and merge signals with stress labels
python aggregate_wearable_stress.py
python merge_wearable_stress.py

# RQ1: Granger causality on stress
python granger_var.py       # Linear model
python granger_lstm.py      # LSTM model

# RQ2: Conditional Granger causality between signals
python granger_pair_phases.py       # Linear model
python granger_lstm_pair_phases.py  # LSTM model
```

---

## Limitations & Future Work

- **Data scarcity:** Only 8 self-reported stress observations per participant limits statistical power for RQ1.
- **Negative Granger scores:** Artifacts from high-variance signals (notably BVP) inflate delta values. Future work should cap or zero negative scores before computing conditionality deltas.
- **Directionality:** Granger causality is statistical, not mechanistic. Domain expertise is required to interpret physiological direction of effect.

**Suggested next steps (RQ2):**
- Segment-level significance testing using Generalized Estimating Equations (GEE)
- LOO-CV applied to the signal-to-signal causality question — V1 is used entirely for training and V2 for testing, leaving no validation set for hyperparameter tuning without leaking test data.

**Suggested next steps (general):**
- Testing on naturalistic (non-lab) stress datasets for ecological validity

---

## References

1. Can et al. (2019). Continuous stress detection using wearable sensors. *Sensors*, 19(8).
2. Garg et al. (2021). Stress detection by machine learning and wearable sensors. *IUI '21 Companion*.
3. Bose et al. (2017). Vector autoregressive models and Granger causality in nursing research. *Nursing Research*, 66(1).
4. Hongn et al. (2025). Wearable device dataset from induced stress and structured exercise sessions. *PhysioNet*.

---

## License

This project was developed for academic research purposes. Dataset usage is subject to the [PhysioNet Credentialed Health Data License](https://physionet.org/content/wearable-device-dataset/1.0.1/).
