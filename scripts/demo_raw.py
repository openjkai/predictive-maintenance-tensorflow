#!/usr/bin/env python3
"""
Demo: raw-signal model (1D-CNN/LSTM) — health score and recommendation.
Predicts from .mat file using raw vibration windows.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.predict import predict_from_file_raw

MODEL_PATH = ROOT / "models" / "fault_classifier_raw.keras"


def health_score(probability: float, is_normal: bool) -> float:
    """0–100 health score. Higher = healthier."""
    if is_normal:
        return probability * 100
    return (1 - probability) * 100


def recommendation(score: float) -> str:
    if score >= 90:
        return "No maintenance required"
    if score >= 70:
        return "Monitor — schedule inspection soon"
    if score >= 50:
        return "Maintenance recommended"
    return "Maintenance required — inspect immediately"


def main():
    if not MODEL_PATH.exists():
        print("Raw model not found. Run training first:")
        print("  python scripts/train_raw.py --arch 1dcnn")
        sys.exit(1)

    mat_path = sys.argv[1] if len(sys.argv) > 1 else None
    if mat_path and not Path(mat_path).exists():
        print(f"File not found: {mat_path}")
        sys.exit(1)

    print("Raw-signal model — Demo\n")

    result = predict_from_file_raw(
        mat_path or str(ROOT / "data" / "IR007_0.mat"),
        model_path=MODEL_PATH,
        n_windows=10,
    )

    pred = result["predicted_class"]
    prob = result["probability"]
    is_normal = pred == "normal"
    score = health_score(prob, is_normal)

    print(f"Input: {result.get('mat_path', 'N/A')}")
    print(f"Windows: {result.get('n_windows', 'N/A')}")
    print(f"Predicted: {pred}")
    print(f"Confidence: {prob:.1%}")
    print(f"Health score: {score:.0f}%")
    print(f"Recommendation: {recommendation(score)}")
    print(f"\nAll probs: {result['all_probs']}")


if __name__ == "__main__":
    main()
