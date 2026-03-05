# Predictive Maintenance ML — Step-by-Step Plan

**Project:** `predictive-maintenance-tensorflow`  
**Dataset:** CWRU Bearing Data (real vibration, rotating machinery)  
**Goal:** Working ML pipeline, step-by-step implementation with understanding

---

## How We Work

- **One phase at a time** — we complete and understand each phase before moving on
- **Incremental code** — no big dumps; each step adds a small, testable piece
- **Verify as we go** — run and inspect results before the next step

---

## Phase 1: Setup & Data Access

**Goal:** Project structure ready, dataset downloaded, data loadable in Python.

| Step | What We Do | What You'll Learn |
|------|------------|-------------------|
| 1.1 | Create folder structure (`data/`, `src/`, `notebooks/`, `models/`) | Project layout for ML repos |
| 1.2 | Add `requirements.txt` (pandas, numpy, scipy, matplotlib) | Minimal dependencies |
| 1.3 | Download CWRU data (12k Drive End) — manual or script | Where to get real vibration data |
| 1.4 | Write `load_data.py` — load one `.mat` file, inspect shape | CWRU `.mat` structure |
| 1.5 | Verify: print sample data, plot a short segment | Sanity check before features |

**Deliverable:** You can load CWRU data and see raw vibration.

---

## Phase 2: Data Understanding & Exploration

**Goal:** Understand the data, visualize it, decide how to split for training.

| Step | What We Do | What You'll Learn |
|------|------------|-------------------|
| 2.1 | Explore in Jupyter: file contents, labels, sampling rate | CWRU fault types (inner/outer/ball) |
| 2.2 | Plot: healthy vs fault — time series | Visual difference normal vs fault |
| 2.3 | Optional: FFT / spectrum view | Why vibration = frequency domain |
| 2.4 | Decide: windows (e.g. 1024 samples) + labels per window | Sliding window for ML |

**Deliverable:** Notebook with clear plots and windowing strategy.

---

## Phase 3: Feature Engineering

**Goal:** Convert raw vibration → features that a model can use.

| Step | What We Do | What You'll Learn |
|------|------------|-------------------|
| 3.1 | Implement: RMS, peak, mean per window | Basic time-domain features |
| 3.2 | Add: kurtosis, std (optional: FFT-based features) | Statistical + frequency features |
| 3.3 | Build dataset: (X_features, y_labels) | ML-ready format |
| 3.4 | Train/val split (e.g. 80/20) | Avoid leakage |

**Deliverable:** `feature_engineering.py` + feature matrix + labels.

---

## Phase 4: Model v1 — Simple Classifier

**Goal:** First working model: fault vs healthy (binary).

| Step | What We Do | What You'll Learn |
|------|------------|-------------------|
| 4.1 | Define simple Dense model (2–3 layers) | Basic TF/Keras API |
| 4.2 | Train on features, binary labels | Training loop, metrics |
| 4.3 | Evaluate: accuracy, confusion matrix | Basic model evaluation |
| 4.4 | Save model, add `predict.py` stub | Model persistence |

**Deliverable:** Trained model, `predict.py` that loads and runs inference.

---

## Phase 5: Model v2 — Multi-class (Optional)

**Goal:** Classify fault type: healthy, inner race, outer race, ball.

| Step | What We Do | What You'll Learn |
|------|------------|-------------------|
| 5.1 | Extend labels to multi-class | Multi-class vs binary |
| 5.2 | Adjust model output (softmax) | One-hot encoding |
| 5.3 | Train, evaluate (per-class recall) | Imbalanced classes |
| 5.4 | Compare with v1 | When multi-class helps |

**Deliverable:** Multi-class model, optional upgrade to `predict.py`.

---

## Phase 6: Polish & GitHub-Ready

**Goal:** Repo looks professional, others can run it.

| Step | What We Do | What You'll Learn |
|------|------------|-------------------|
| 6.1 | Full README: goal, data, how to run | Good open-source docs |
| 6.2 | `train.py` CLI (optional args) | Reusable training script |
| 6.3 | Demo output: health score, recommendation | Real-world UX |
| 6.4 | Add visualizations (training curves, confusion matrix) | Presenting results |

**Deliverable:** Clean README, runnable scripts, nice plots.

---

## Current Status

- [x] Phase 1.3 — CWRU data downloaded
- [x] Phase 1.4 — load_data.py created
- [ ] Phase 1
- [ ] Phase 2
- [ ] Phase 3
- [ ] Phase 4
- [ ] Phase 5 (optional)
- [ ] Phase 6

---

## Next Step

**Start with Phase 1, Step 1.1:** Create folder structure.

After that, we'll do 1.2 (requirements), then 1.3 (data download), etc. — one step at a time.
