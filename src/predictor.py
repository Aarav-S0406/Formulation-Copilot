from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "tannin_reduction_model.joblib"
)


def assess_evidence_coverage(training_df, inputs):
    checks = {}

    categorical_features = [
        "crop",
        "processing_method",
        "pretreatment",
        "light_condition",
        "cooking_method",
        "sample_form",
    ]

    for feature in categorical_features:
        chosen = inputs.get(feature)

        if pd.isna(chosen):
            checks[feature] = "Not specified"
            continue

        observed = set(
            training_df[feature].dropna().astype(str).str.strip()
        )

        checks[feature] = (
            "Observed category"
            if str(chosen).strip() in observed
            else "Unseen category"
        )

    numeric_features = [
        "soaking_time_h",
        "germination_time_h",
        "processing_temperature_c",
        "cooking_time_min",
    ]

    for feature in numeric_features:
        chosen = inputs.get(feature)
        observed = pd.to_numeric(
            training_df[feature], errors="coerce"
        ).dropna()

        if pd.isna(chosen) or observed.empty:
            checks[feature] = "Insufficient evidence"
        elif observed.min() <= float(chosen) <= observed.max():
            checks[feature] = "Within observed range"
        else:
            checks[feature] = "Outside observed range"

    return checks


def load_model_artifact():
    """Load the saved model pipeline and its metadata."""
    artifact = joblib.load(MODEL_PATH)

    required_keys = {"pipeline", "features", "target"}
    if not required_keys.issubset(artifact):
        raise ValueError(
            f"Model artifact must contain {required_keys}"
        )

    return artifact


def predict_tannin_reduction(artifact, inputs):
    """Predict tannin reduction (%) for one processing condition."""
    features = artifact["features"]

    # Preserve the exact feature names and order used during training.
    row = pd.DataFrame([
        {feature: inputs.get(feature) for feature in features}
    ])

    prediction = artifact["pipeline"].predict(row)[0]
    return float(prediction)
