# Predictive Maintenance — TensorFlow

A **machine learning (ML) project** that learns to detect **bearing faults** from vibration data. You give it a vibration signal (from a motor, pump, or fan), and it tells you: *Is the bearing healthy, or does it have a fault? If faulty, what kind?* This helps plan maintenance **before** machines break down.

---

## Table of Contents

1. [What is this project? (Simple overview)](#1-what-is-this-project-simple-overview)
2. [Key concepts — Glossary for beginners](#2-key-concepts--glossary-for-beginners)
3. [Project structure — What each file does](#3-project-structure--what-each-file-does)
4. [Setup — Step by step](#4-setup--step-by-step)
5. [How to run everything](#5-how-to-run-everything)
6. [Understanding the two models](#6-understanding-the-two-models)
7. [How to interpret results](#7-how-to-interpret-results)
8. [Troubleshooting](#8-troubleshooting)
9. [Learning resources](#9-learning-resources)

---

## 1. What is this project? (Simple overview)

### The problem
Machines with rotating parts (motors, pumps, fans) use **bearings**. When a bearing starts to fail, it often vibrates differently. If we can detect that change early, we can repair it before a costly breakdown.

### The solution
We use **machine learning** to train a computer program on real vibration data from healthy and faulty bearings. Once trained, the program can:

1. **Classify** vibration as: *normal*, *inner race fault*, *ball fault*, or *outer race fault*
2. **Give a health score** (0–100%)
3. **Recommend** what to do: no action, monitor, or inspect immediately

### What makes this project good for learning
- Uses **real data** from Case Western Reserve University (CWRU) — industry-standard benchmark
- **Two approaches** shown: hand-crafted features vs. raw signal (neural networks)
- **End-to-end pipeline**: download → load → features → train → predict → demo
- Step-by-step plan in **[PLAN.md](PLAN.md)**

---

## 2. Key concepts — Glossary for beginners

| Term | Plain English |
|------|---------------|
| **Machine learning (ML)** | A program that learns patterns from data instead of being told every rule. We show it examples of “healthy” and “faulty” vibrations; it learns to tell them apart. |
| **Predictive maintenance** | Fixing things *before* they fail, based on signs (like unusual vibration). Contrast with “fix it when it breaks.” |
| **Dataset** | A collection of labeled examples. Here: vibration files + labels (normal, inner_race, ball, outer_race). |
| **Training** | The process of showing the model many examples so it learns. |
| **Model** | The learned program. We save it so we can use it later without re-training. |
| **Inference / prediction** | Using the trained model to classify new vibration data. |
| **Feature** | A number we compute from the raw signal (e.g. RMS, peak). Features summarize the signal in a way the model can use. |
| **Window** | A short segment of the signal (e.g. 1024 samples). We split long signals into windows and predict per window. |
| **Validation / val set** | Data we hold back during training to check how well the model generalizes (not used to update weights). |
| **Accuracy** | Fraction of predictions that are correct (e.g. 98% = 98 out of 100 right). |
| **Health score** | Our 0–100% measure: higher = healthier. 100% = confidently normal; 0% = confidently faulty. |
| **RUL (Remaining Useful Life)** | Number of cycles until failure. Regression task (NASA C-MAPSS). |

### Vibration / signal terms

| Term | Meaning |
|------|---------|
| **Vibration signal** | A sequence of numbers: amplitude at each time step. Recorded by an accelerometer. |
| **Sample rate (Hz)** | How many numbers per second (e.g. 12,000 Hz = 12,000 samples/sec). |
| **RMS (Root Mean Square)** | A measure of signal strength / energy. |
| **FFT (Fast Fourier Transform)** | Converts time signal → frequency spectrum. Faults often show up as extra frequencies. |
| **Wavelet** | A way to decompose a signal into different frequency bands (like a multi-scale FFT). |

### Model terms

| Term | Meaning |
|------|---------|
| **Feature-based model** | We compute features (RMS, peak, etc.) by hand, then a simple neural net classifies from those numbers. |
| **Raw model (1D-CNN / LSTM)** | The neural net sees the raw vibration window directly and learns its own internal features. |
| **Dense layer** | A layer where every input connects to every output (fully connected). |
| **1D-CNN** | Convolutional Neural Network for 1D (time-series) data. Good at finding local patterns. |
| **LSTM** | Recurrent network that can remember patterns over time. |

---

## 3. Project structure — What each file does

```
ml/
├── data/                    # Datasets
│   ├── *.mat                # CWRU bearing vibration (Normal_0, IR007_0, etc.)
│   └── cmapss/              # NASA C-MAPSS FD001 (train_FD001.txt, test_FD001.txt, RUL_FD001.txt)
│
├── models/                  # Saved trained models (create after training)
│   ├── fault_classifier.keras      # Feature-based bearing model
│   ├── fault_classifier.npz        # Metadata for feature model
│   ├── fault_classifier_raw.keras  # Raw-signal (1D-CNN) bearing model
│   ├── fault_classifier_raw.npz   # Metadata for raw model
│   ├── rul_predictor.keras        # RUL (Remaining Useful Life) LSTM
│   └── rul_predictor.npz          # Metadata for RUL model
│
├── notebooks/               # Jupyter notebooks for exploration
│   ├── exploration.ipynb   # Data exploration, plots, FFT, windowing
│   └── training_curves.png  # Loss/accuracy curves (generated by training)
│
├── src/                     # Core Python code
│   ├── load_data.py         # Load .mat files, return (signal, sample_rate, rpm)
│   ├── feature_engineering.py  # Extract features (RMS, peak, FFT, wavelet), build datasets
│   ├── train_model.py       # Feature-based Dense model training
│   ├── raw_model.py         # 1D-CNN and LSTM for raw bearing windows
│   ├── load_cmapss.py       # NASA C-MAPSS FD001 loading, RUL labels, sequences
│   ├── rul_model.py         # LSTM for RUL regression
│   └── predict.py           # Inference: bearing fault + RUL
│
├── scripts/                 # Run these from the command line
│   ├── download_cwru.py     # Download CWRU .mat files into data/
│   ├── verify_data.py       # Plot a sample signal, sanity check
│   ├── train.py             # Train feature-based model
│   ├── train_raw.py         # Train 1D-CNN or LSTM on raw windows
│   ├── demo.py              # Quick demo: predict from sample, print health score
│   ├── demo_raw.py          # Demo with raw model (pass .mat path)
│   ├── run_predict.py       # Run prediction from features or .mat file
│   ├── run_features.py      # Run feature engineering, print dataset stats
│   ├── download_cmapss.py   # Download NASA C-MAPSS FD001
│   ├── train_rul.py         # Train RUL predictor (LSTM)
│   ├── demo_rul.py          # Demo RUL prediction on test engines
│   └── dashboard.py         # Streamlit web UI: bearing fault + RUL
│
├── requirements.txt        # Python dependencies
├── PLAN.md                 # Step-by-step implementation roadmap
├── README.md               # This file
└── LICENSE                 # MIT
```

### Data files

**CWRU (bearings):**
- **Source**: [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter/welcome)
- **Content**: Vibration recordings. `Normal_0.mat` = healthy; `IR007_0.mat` = inner race fault; etc.

**NASA C-MAPSS (RUL):**
- **Source**: [NASA Prognostics](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/) / [GitHub mirror](https://github.com/edwardzjl/CMAPSSData)
- **Content**: Turbofan engine sensor data (unit, cycle, 3 op settings, 21 sensors). Train = run-to-failure; test = ends before failure with true RUL.

---

## 4. Setup — Step by step

### Prerequisites
- **Python 3.10+** (check: `python3 --version`)
- **pip** (usually comes with Python)

### 4.1 Create a virtual environment

A virtual environment keeps this project’s packages separate from others.

```bash
cd /path/to/ml
python3 -m venv venv
```

- **Linux/macOS**: `source venv/bin/activate`
- **Windows**: `venv\Scripts\activate`

You should see `(venv)` in your terminal prompt.

### 4.2 Install dependencies

```bash
pip install -r requirements.txt
```

This installs: numpy, pandas, scipy, matplotlib, TensorFlow, PyWavelets, scikit-learn, Streamlit, Jupyter.

### 4.3 Download data

```bash
python scripts/download_cwru.py
```

This fetches CWRU `.mat` files into `data/`. You need this before training.

### 4.4 (Optional) Verify data

```bash
python scripts/verify_data.py
```

Plots a short segment of a sample file. Confirms data loaded correctly.

---

## 5. How to run everything

### Full pipeline (in order)

```bash
# Activate venv first
source venv/bin/activate   # or venv\Scripts\activate on Windows

# 1. Download data (if not done)
python scripts/download_cwru.py

# 2. Train the feature-based model
python scripts/train.py

# 3. Demo: predict from a sample in the dataset
python scripts/demo.py

# 4. (Optional) Train raw model
python scripts/train_raw.py --arch 1dcnn

# 5. (Optional) Demo with raw model
python scripts/demo_raw.py data/IR007_0.mat

# 6. RUL prediction (NASA C-MAPSS)
python scripts/download_cmapss.py
python scripts/train_rul.py
python scripts/demo_rul.py

# 7. Web dashboard
streamlit run scripts/dashboard.py
```

### What each command does

| Command | What it does | Output |
|--------|--------------|--------|
| `python scripts/download_cwru.py` | Downloads CWRU `.mat` files | Files in `data/` |
| `python scripts/verify_data.py` | Loads one file, plots a segment | Plot (optional) |
| `python scripts/train.py` | Builds features, trains Dense model, saves it | `models/fault_classifier.keras` + `.npz`, confusion matrix in terminal |
| `python scripts/train_raw.py --arch 1dcnn` | Builds raw windows, trains 1D-CNN, saves it | `models/fault_classifier_raw.keras` + `.npz` |
| `python scripts/demo.py` | Uses feature model on first sample in dataset | Prints predicted class, health score, recommendation |
| `python scripts/run_predict.py` | Same, with optional `.mat` path | With path: `run_predict.py data/Normal_0.mat` |
| `python scripts/demo_raw.py data/IR007_0.mat` | Uses raw model on a specific `.mat` file | Same output format |
| `python scripts/download_cmapss.py` | Download NASA C-MAPSS FD001 | `data/cmapss/*.txt` |
| `python scripts/train_rul.py` | Train RUL predictor (LSTM) | `models/rul_predictor.keras` |
| `python scripts/demo_rul.py` | RUL demo on test engines | Predicted vs true RUL table |
| `streamlit run scripts/dashboard.py` | Starts web app | Open http://localhost:8501 in browser |

### Training options

```bash
# Feature model
python scripts/train.py --epochs 100 --batch-size 32 --no-class-weights

# Raw model
python scripts/train_raw.py --arch lstm --epochs 40 --batch-size 64
```

---

## 6. Understanding the two models

### Feature-based model (Dense)

1. **Input**: 9 numbers per window — RMS, peak, mean, std, kurtosis, spectral centroid, spectral bandwidth, wavelet_energy_d1, wavelet_energy_a1.
2. **Architecture**: A few Dense (fully connected) layers.
3. **Training**: ~50 epochs, early stopping, optional class weights.
4. **Output**: Probabilities for normal, inner_race, ball, outer_race.

**Pros**: Fast, interpretable (you see which features matter).  
**Cons**: We design features by hand; may miss patterns.

### Raw model (1D-CNN or LSTM)

1. **Input**: Raw 1024-sample windows (no hand-crafted features).
2. **Architecture**: Convolutions (1D-CNN) or LSTM layers.
3. **Training**: Same data, different representation.
4. **Output**: Same 4-class probabilities.

**Pros**: Can learn complex patterns from raw data.  
**Cons**: Less interpretable, more compute.

---

## 7. How to interpret results

### Example output

```
Predicted: inner_race
Confidence: 99.5%
Health score: 0%
Recommendation: Maintenance required — inspect immediately

All class probabilities: {'normal': 0.001, 'inner_race': 0.995, 'ball': 0.002, 'outer_race': 0.002}
```

- **Predicted**: The class with highest probability.
- **Confidence**: That probability.
- **Health score**: 0–100%. 100% = confidently normal; 0% = confidently faulty.
- **Recommendation**: Based on health score thresholds (see `scripts/demo.py` or `scripts/dashboard.py`).

### Performance metrics (during training)

- **Accuracy**: Fraction of correct predictions.
- **Confusion matrix**: Rows = true class, columns = predicted. Diagonal = correct.
- **Recall**: For each class, what fraction of true positives we caught.
- **Precision**: For each class, of what we predicted, how much was correct.

---

## 8. Troubleshooting

| Problem | What to do |
|--------|------------|
| `ModuleNotFoundError: No module named 'src'` | Run from project root: `cd /path/to/ml` before `python scripts/...` |
| `FileNotFoundError: Model not found` | Run training first: `python scripts/train.py` (or `train_raw.py` for raw model). |
| `No such file or directory: data/` | Run `python scripts/download_cwru.py` first. |
| CUDA / GPU messages | Safe to ignore if you don’t have a GPU; training runs on CPU. |
| Low accuracy | Try more epochs, check that data downloaded correctly, try both models. |

---

## 9. Learning resources

- **TensorFlow / Keras**: [keras.io](https://keras.io/) — Sequential API used here
- **CWRU dataset**: [Bearing Data Center](https://engineering.case.edu/bearingdatacenter/welcome)
- **Predictive maintenance**: [AWS – What is predictive maintenance?](https://aws.amazon.com/what-is/predictive-maintenance/)

---

## Results (typical performance)

| Model | Metric | Input |
|-------|--------|-------|
| Feature-based (Dense) | ~99–100% val accuracy | 9 features (RMS, peak, FFT, wavelet) |
| Raw-signal (1D-CNN) | ~99.9% val accuracy | 1024-sample windows |
| RUL predictor (LSTM) | Val RMSE ~15–25 cycles | NASA C-MAPSS FD001, 30-cycle windows |

**Bearing classes**: normal, inner_race, ball, outer_race  
**RUL**: Remaining Useful Life in cycles (regression)

---

## Project status & plan

- **Phases 1–6**: Done (data, features, feature model, raw model, polish, dashboard)
- **Phase 7.1**: Extra features (FFT, wavelet) — Done
- **Phase 7.3**: Class weights for imbalanced data — Done
- **Phase 7.4**: Web dashboard — Done
- **Phase 7.2**: NASA C-MAPSS RUL — Done

Full roadmap: **[PLAN.md](PLAN.md)**

---

## Quick reference (copy & paste)

```bash
cd /path/to/ml
source venv/bin/activate   # or venv\Scripts\activate on Windows

python scripts/download_cwru.py
python scripts/train.py
python scripts/demo.py

# Optional: raw model + RUL + web UI
python scripts/train_raw.py --arch 1dcnn
python scripts/demo_raw.py data/IR007_0.mat
python scripts/download_cmapss.py && python scripts/train_rul.py
python scripts/demo_rul.py
streamlit run scripts/dashboard.py
```

---

## License

MIT — see [LICENSE](LICENSE). Dataset usage follows CWRU terms.
