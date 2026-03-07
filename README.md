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

# 5. Explore in Jupyter
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

| Metric | Value |
|--------|-------|
| Val accuracy | ~98% |
| Classes | normal, inner_race, ball, outer_race |
| Features | RMS, peak, mean, std, kurtosis |

---

## Follow the plan

1. Read [PLAN.md](PLAN.md) for the full roadmap.
2. Phases 1–4 are complete; Phase 6 polish in progress.

### Dev setup (optional)

```bash
source venv/bin/activate
pip install -e ".[dev]"
pre-commit install
pre-commit install --hook-type commit-msg   # Validate commit messages
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for commit convention and code style.

---

## License

MIT — dataset usage follows CWRU terms.
