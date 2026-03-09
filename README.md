# Predictive Maintenance — TensorFlow

ML pipeline for **predictive maintenance** using real vibration data from the CWRU Bearing dataset. Detects bearing faults and supports maintenance decisions for rotating machinery (motors, pumps, fans).

---

## Project Status

We're building this **step by step**. See **[PLAN.md](PLAN.md)** for the full roadmap and current phase.

---

## Dataset

- **Source:** [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter/welcome)
- **Type:** Real vibration data (12k/48k Hz)
- **Labels:** Normal baseline + fault types (inner race, outer race, ball)

---

## Structure

```
predictive-maintenance-tensorflow/
├── data/           # CWRU .mat files
├── notebooks/      # Exploration & analysis
├── src/            # Preprocessing, features, training, prediction
├── models/         # Saved models
├── scripts/        # Download, setup utilities
├── PLAN.md         # Step-by-step implementation plan
├── CONTRIBUTING.md # Commit convention, code style
├── .cursor/rules/  # Cursor rules (e.g. Conventional Commits for ✨ Generate Message)
├── pyproject.toml  # Project config, Black, Ruff, Commitizen
├── .editorconfig   # Editor consistency
└── requirements.txt
```

---

## Getting Started

### 1. Create virtual environment and install

```bash
cd /path/to/ml
python3 -m venv venv
source venv/bin/activate   # Linux/macOS — or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

### 2. Full pipeline

```bash
source venv/bin/activate

# 1. Download data
python scripts/download_cwru.py

# 2. Verify data (optional)
python scripts/verify_data.py

# 3. Train model
python scripts/train.py

# 4. Demo: health score + recommendation
python scripts/demo.py

# 4b. Raw-signal model (1D-CNN or LSTM, optional)
python scripts/train_raw.py --arch 1dcnn
python scripts/demo_raw.py data/IR007_0.mat

# 5. Web dashboard (Phase 7.4)
streamlit run scripts/dashboard.py

# 6. Explore in Jupyter
jupyter notebook notebooks/exploration.ipynb
```

Or use venv Python directly: `./venv/bin/python scripts/train.py`

### 3. Quick demo (after training)

```bash
# Default: predict from a sample in the dataset
python scripts/demo.py

# Predict from a specific .mat file
python scripts/demo.py data/IR007_0.mat

# Or use run_predict for raw prediction output
python scripts/run_predict.py
python scripts/run_predict.py data/Normal_0.mat

# Raw-signal model demo (after training with train_raw.py)
python scripts/demo_raw.py
python scripts/demo_raw.py data/IR007_0.mat

# Web dashboard — upload .mat or pick from data/
streamlit run scripts/dashboard.py
```

Example output:
```
Machine ID: sample_001
Predicted: normal
Health score: 98%
Recommendation: No maintenance required
```

---

## Results

| Model | Val accuracy | Input |
|-------|--------------|-------|
| Feature-based (Dense) | ~99–100% | 9 features: RMS, peak, mean, std, kurtosis, spectral centroid/bandwidth, wavelet energies |
| Raw-signal (1D-CNN) | ~99.9% | 1024-sample windows |

**Classes:** normal, inner_race, ball, outer_race

---

## Follow the plan

1. Read [PLAN.md](PLAN.md) for the full roadmap.
2. Phases 1–6 complete; raw-signal models (1D-CNN, LSTM) available.

### Dev setup (optional)

```bash
source venv/bin/activate
pip install -e ".[dev]"
pre-commit install
pre-commit install --hook-type commit-msg   # Validate commit messages
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for commit convention and code style.

---

## Future Improvements

- **More features** — Phase 7.1 done: spectral centroid, spectral bandwidth, wavelet energies (db4) — FFT-based (spectral centroid, bandwidth), wavelet features
- **NASA C-MAPSS** — Add RUL (Remaining Useful Life) prediction
- **Class weights** — Handle imbalanced fault severities
- **Web dashboard** — Phase 7.4 done: `streamlit run scripts/dashboard.py`

---

## License

MIT — see [LICENSE](LICENSE). Dataset usage follows CWRU terms.
