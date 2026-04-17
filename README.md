# Automated Sleep EEG Analysis — Group 28
### BITS Pilani, Goa Campus | EEE Department | Signals & Systems Project

---

## Project Overview

A complete MATLAB pipeline for automated polysomnographic analysis:

1. **Load** — reads PhysioNet Sleep-EDF `.edf` files + annotations
2. **Pre-process** — FIR band-pass, Chebyshev notch, epoch segmentation
3. **Feature Extraction** — PSD, STFT, DWT, Hilbert envelope (23 features/epoch)
4. **Classification** — KNN + Decision Tree + SVM ensemble, 5-fold CV
5. **Transient Detection** — spindles (sigma-band Hilbert) + K-complexes
6. **Export** — 7 figures + 4 CSVs per recording

---

## Folder Structure

```
sleep_eeg_project/
├── code/
│   ├── main.m               ← Run this for real EDF files
│   ├── demo_synthetic.m     ← Run this first to verify setup
│   ├── load_edf.m           ← Module 1: EDF reader + annotation parser
│   ├── preprocess.m         ← Module 2: Filtering + epoching
│   ├── extract_features.m   ← Module 3: Feature computation
│   ├── classify_stages.m    ← Module 4: ML classifier
│   ├── detect_transients.m  ← Module 5: Spindle + K-complex detector
│   └── export_results.m     ← Module 6: Figures + CSV export
├── data/                    ← Place .edf files here
├── outputs/
│   ├── figures/             ← PNG plots auto-saved here
│   └── *.csv                ← Results CSVs auto-saved here
└── README.md
```

---

## Quick Start

### Step 1 — Verify setup (no data needed)
```matlab
cd sleep_eeg_project/code
demo_synthetic
```
This generates synthetic EEG, runs the full pipeline, and saves outputs.

### Step 2 — Download real data
1. Go to: https://physionet.org/content/sleep-edfx/
2. Download any `*PSG.edf` + matching `*Hypnogram.edf` pairs
3. Place both files in `sleep_eeg_project/data/`

### Step 3 — Run on real data
```matlab
cd sleep_eeg_project/code
main
```

---

## Dependencies

| Toolbox | Used For |
|---------|----------|
| Signal Processing Toolbox | `fir1`, `filtfilt`, `pwelch`, `spectrogram`, `hilbert` |
| Wavelet Toolbox | `wavedec`, `detcoef` |
| Statistics & ML Toolbox | `fitcknn`, `fitctree`, `fitcecoc`, `cvpartition` |

MATLAB R2020b or later recommended (`edfread` built-in).

---

## Outputs Per Recording

| File | Description |
|------|-------------|
| `*_fig1_hypnogram.png`   | True vs predicted sleep stages over night |
| `*_fig2_confusion.png`   | Normalised confusion matrix heatmap |
| `*_fig3_psd.png`         | PSD per sleep stage (0–35 Hz) |
| `*_fig4_spectrogram.png` | STFT spectrogram with hypnogram |
| `*_fig5_spindles.png`    | Gallery of detected spindles |
| `*_fig6_kcomplexes.png`  | Gallery of detected K-complexes |
| `*_fig7_features.png`    | Feature box-plots by stage |
| `*_epoch_labels.csv`     | Epoch-by-epoch true vs predicted labels |
| `*_spindles.csv`         | Spindle onset/offset/duration/freq table |
| `*_kcomplexes.csv`       | K-complex event table |
| `*_features.csv`         | Full 23-feature matrix with stage labels |

---

## Group Members & Modules

| Member | ID | Module |
|--------|----|--------|
| Adarsh | 2023B1A31167G | Pre-processing & Filtering (`preprocess.m`) |
| Mehul Tiwari | 2023B1A80700G | Feature Extraction (`extract_features.m`) |
| Kavish Kumar | 2023B1A30716G | Wavelet & Hilbert Analysis (inside `extract_features.m`) |
| Munish Jain | 2023B1A80706G | Classifier & Validation (`classify_stages.m`) |
| Adarsh | 2023B1A31167G | Transient Detection & Integration (`detect_transients.m`, `main.m`) |
