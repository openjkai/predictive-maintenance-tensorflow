#!/usr/bin/env python3
"""
Run feature engineering — Phase 3 verification.
Builds X, y and prints shape + train/val split.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.feature_engineering import build_dataset
from src.load_data import DATA_DIR

if __name__ == "__main__":
    print("Phase 3 — Feature Engineering\n")

    X, y, label_names = build_dataset(DATA_DIR, binary=False)
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Feature names: {label_names}")
    print(f"Classes (unique labels): {len(np.unique(y))}")

    # Quick train/val split
    from src.feature_engineering import train_val_split

    X_train, X_val, y_train, y_val = train_val_split(X, y, val_frac=0.2, random_state=42)
    print(f"\nTrain: {X_train.shape[0]} | Val: {X_val.shape[0]}")
    print("Phase 3 complete — ready for Phase 4 (model).")
