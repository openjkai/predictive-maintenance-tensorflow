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
├── PLAN.md         # Step-by-step implementation plan
├── README.md
└── requirements.txt
```

---

## Getting Started

1. Read [PLAN.md](PLAN.md) to see the phases.
2. Start with **Phase 1** (setup & data access).
3. Run step by step as we add code.

---

## License

MIT — dataset usage follows CWRU terms.
