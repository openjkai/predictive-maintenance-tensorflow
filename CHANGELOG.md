# Changelog

All notable changes to this project.

## [0.1.0] — 2025-03

### Added
- **Bearing fault detection** (CWRU): feature-based (9 features) and raw-signal (1D-CNN, LSTM)
- **RUL prediction** (NASA C-MAPSS FD001–004): LSTM for remaining useful life
- **Web dashboard**: Streamlit UI for bearing and RUL modes
- **Run-all script**: `python scripts/run_all.py` for full pipeline
- **Unit tests**: pytest for load_data, feature_engineering, load_cmapss, predict
- **Model comparison notebook**: side-by-side comparison of all models

### Datasets
- CWRU Bearing (12 files: normal, inner/ball/outer race at 0.007/0.014/0.021")
- NASA C-MAPSS FD001–004 (turbofan engine degradation)

### Models
- `fault_classifier.keras` — 9-feature Dense classifier
- `fault_classifier_raw.keras` — 1D-CNN/LSTM on raw windows
- `rul_predictor_fd00N.keras` — LSTM for RUL (N=1–4)
