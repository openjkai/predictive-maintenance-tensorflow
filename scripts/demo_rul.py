#!/usr/bin/env python3
"""
Demo: RUL (Remaining Useful Life) prediction — Phase 7.2
NASA C-MAPSS FD001 — predict cycles until failure.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.load_cmapss import load_fd001, CMAPSS_DIR
from src.predict import load_rul_model_and_meta, predict_rul

MODEL_PATH = ROOT / "models" / "rul_predictor.keras"


def main():
    if not MODEL_PATH.exists():
        print("RUL model not found. Run first:")
        print("  python scripts/download_cmapss.py")
        print("  python scripts/train_rul.py")
        sys.exit(1)

    print("NASA C-MAPSS — RUL Demo\n")

    # Load FD001 test data and true RUL
    train_df, test_df, true_rul = load_fd001(CMAPSS_DIR, fd=1)

    # Prepare test sequences (same logic as in load_cmapss.prepare_fd001)
    from src.load_cmapss import SENSOR_COLS, build_sequences, compute_train_rul
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler

    model, scaler_min, scaler_max, window_size, n_features, used_cols = load_rul_model_and_meta(
        MODEL_PATH
    )
    used_cols = list(used_cols)

    # Build test windows and scale
    test_engines = test_df["unit"].unique()
    predictions = []
    for unit in test_engines[:5]:  # First 5 engines
        unit_df = test_df[test_df["unit"] == unit].sort_values("cycle")
        arr = unit_df.iloc[:, used_cols].values.astype(np.float32)
        if len(arr) < window_size:
            continue
        window = arr[-window_size:]
        scale = scaler_max - scaler_min
        scale = np.where(scale < 1e-10, 1.0, scale)
        window_norm = (window - scaler_min) / scale * 2.0 - 1.0
        rul = float(model.predict(window_norm[np.newaxis, ...].astype(np.float32), verbose=0)[0, 0])
        rul = max(0, rul)
        predictions.append((unit, rul))

    # Show results (compare with true RUL for first 5)
    print("Engine  | Predicted RUL | True RUL | Error")
    print("-" * 45)
    for i, (unit, pred_rul) in enumerate(predictions):
        true = true_rul[i] if i < len(true_rul) else "?"
        err = abs(pred_rul - true) if i < len(true_rul) else ""
        print(f"  {unit:3d}  | {pred_rul:12.1f} | {str(true):>8} | {err}")

    print("\nRUL = Remaining Useful Life (cycles until failure)")
    print("Lower RUL = closer to failure. Model predicts from sensor trends.")


if __name__ == "__main__":
    main()
