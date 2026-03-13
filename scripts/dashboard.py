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


CMAPSS_DIR = ROOT / "data" / "cmapss"


def _rul_model_path(fd: int) -> Path:
    p = ROOT / "models" / f"rul_predictor_fd00{fd}.keras"
    if fd == 1 and not p.exists():
        legacy = ROOT / "models" / "rul_predictor.keras"
        if legacy.exists():
            return legacy
    return p


def main():
    st.set_page_config(page_title="Predictive Maintenance", page_icon="⚙️", layout="centered")
    st.title("⚙️ Predictive Maintenance")
    st.caption("Bearing fault detection • RUL prediction • TensorFlow")

    # Mode: Bearing (CWRU) or RUL (C-MAPSS)
    mode = st.radio("Mode", ["Bearing fault (CWRU)", "RUL (NASA C-MAPSS)"], horizontal=True)

    if mode == "RUL (NASA C-MAPSS)":
        _run_rul_mode()
        return

    # --- Bearing fault mode ---
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


def _run_rul_mode():
    """RUL prediction on C-MAPSS FD001 or FD002 test engines."""
    from src.load_cmapss import load_fd001
    from src.predict import load_rul_model_and_meta
    import numpy as np

    fd = st.radio("Dataset", [1, 2], format_func=lambda x: f"FD00{x} (1 op)" if x == 1 else "FD002 (6 op)")
    model_path = _rul_model_path(fd)

    if not model_path.exists():
        st.error(
            f"RUL model for FD00{fd} not found. Run:\n"
            "  python scripts/download_cmapss.py\n"
            f"  python scripts/train_rul.py --fd {fd}"
        )
        return

    test_file = CMAPSS_DIR / f"test_FD00{fd}.txt"
    if not test_file.exists():
        st.error("C-MAPSS data not found. Run: python scripts/download_cmapss.py")
        return

    n_engines = st.slider("Number of test engines to show", 5, 50, 10)
    if st.button("Predict RUL", type="primary"):
        with st.spinner("Loading model and data..."):
            train_df, test_df, true_rul = load_fd001(CMAPSS_DIR, fd=fd)
            model, scaler_min, scaler_max, window_size, n_features, used_cols = (
                load_rul_model_and_meta(model_path)
            )
            used_cols = list(used_cols)

        predictions = []
        for i, unit in enumerate(test_df["unit"].unique()[:n_engines]):
            unit_df = test_df[test_df["unit"] == unit].sort_values("cycle")
            arr = unit_df.iloc[:, used_cols].values.astype(np.float32)
            if len(arr) < window_size:
                continue
            window = arr[-window_size:]
            scale = scaler_max - scaler_min
            scale = np.where(scale < 1e-10, 1.0, scale)
            window_norm = (window - scaler_min) / scale * 2.0 - 1.0
            rul = float(
                model.predict(window_norm[np.newaxis, ...].astype(np.float32), verbose=0)[0, 0]
            )
            rul = max(0, rul)
            true = int(true_rul[i]) if i < len(true_rul) else "—"
            predictions.append({"Engine": unit, "Predicted RUL": f"{rul:.1f}", "True RUL": true})

        st.subheader("RUL predictions (cycles until failure)")
        st.dataframe(predictions)
        st.caption("RUL = Remaining Useful Life. Lower = closer to failure.")


if __name__ == "__main__":
    main()
