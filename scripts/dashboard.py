#!/usr/bin/env python3
"""
Phase 7.4 — Web dashboard for health monitoring.
Run: streamlit run scripts/dashboard.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import streamlit as st

from src.predict import predict_from_file, predict_from_file_raw

MODEL_PATH = ROOT / "models" / "fault_classifier.keras"
RAW_MODEL_PATH = ROOT / "models" / "fault_classifier_raw.keras"
DATA_DIR = ROOT / "data"


def health_score(probability: float, is_normal: bool) -> float:
    """0–100 health score. Higher = healthier."""
    if is_normal:
        return probability * 100
    return (1 - probability) * 100


def recommendation(health_score_val: float) -> str:
    if health_score_val >= 90:
        return "No maintenance required"
    if health_score_val >= 70:
        return "Monitor — schedule inspection soon"
    if health_score_val >= 50:
        return "Maintenance recommended"
    return "Maintenance required — inspect immediately"


def main():
    st.set_page_config(page_title="Predictive Maintenance", page_icon="⚙️", layout="centered")
    st.title("⚙️ Predictive Maintenance")
    st.caption("Bearing fault detection — CWRU dataset • TensorFlow")

    # Model choice
    use_raw = st.radio(
        "Model",
        ["Feature-based (9 features)", "Raw-signal (1D-CNN)"],
        horizontal=True,
        help="Feature model uses hand-crafted + FFT + wavelet. Raw model uses 1024-sample windows.",
    )
    is_raw = "Raw" in use_raw

    model_path = RAW_MODEL_PATH if is_raw else MODEL_PATH
    if not model_path.exists():
        st.error(
            f"Model not found: {model_path.name}. Run training first:\n"
            f"- Feature: `python scripts/train.py`\n"
            f"- Raw: `python scripts/train_raw.py --arch 1dcnn`"
        )
        st.stop()

    # Input: file upload or select from data/
    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded = st.file_uploader("Upload .mat file", type=["mat"])
    with col2:
        data_files = list(DATA_DIR.glob("*.mat")) if DATA_DIR.exists() else []
        selected_file = st.selectbox(
            "Or pick from data/",
            [""] + [str(f.relative_to(ROOT)) for f in sorted(data_files)],
            format_func=lambda x: x or "(select)",
        )

    mat_path = None
    if uploaded:
        # Save to temp and use path
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as f:
            f.write(uploaded.read())
            mat_path = f.name
        label = uploaded.name
    elif selected_file:
        mat_path = ROOT / selected_file
        label = Path(selected_file).name
    else:
        st.info("Upload a CWRU .mat file or select one from the data folder.")
        st.stop()

    if not Path(mat_path).exists():
        st.error("Selected file not found.")
        st.stop()

    # Predict
    n_windows = st.slider("Windows for prediction", 5, 50, 10, help="More windows = more stable vote")
    if st.button("Run prediction", type="primary"):
        with st.spinner("Analyzing vibration data..."):
            try:
                if is_raw:
                    result = predict_from_file_raw(
                        mat_path, model_path=model_path, n_windows=n_windows
                    )
                else:
                    result = predict_from_file(
                        mat_path, model_path=model_path, n_windows=n_windows
                    )
        except Exception as e:
            st.exception(e)
            st.stop()

        pred = result["predicted_class"]
        prob = result["probability"]
        is_normal = pred == "normal"
        score = health_score(prob, is_normal)
        rec = recommendation(score)

        st.subheader("Result")
        st.metric("Predicted fault", pred.replace("_", " ").title())
        st.metric("Confidence", f"{prob:.1%}")
        st.metric("Health score", f"{score:.0f}%")
        st.info(f"**Recommendation:** {rec}")

        st.subheader("Class probabilities")
        probs = result["all_probs"]
        for cls, p in sorted(probs.items(), key=lambda x: -x[1]):
            st.progress(p, text=f"{cls}: {p:.1%}")

        st.caption(f"File: {label} • Windows: {result.get('n_windows', '—')}")


if __name__ == "__main__":
    main()
