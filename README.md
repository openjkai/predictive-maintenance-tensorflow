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

1. Read [PLAN.md](PLAN.md) to see the phases.
2. Start with **Phase 1** (setup & data access).
3. Run step by step as we add code.

### Dev setup (optional)

```bash
pip install -e ".[dev]"
pre-commit install
pre-commit install --hook-type commit-msg   # Validate commit messages
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for commit convention and code style.

---

## License

MIT — dataset usage follows CWRU terms.
