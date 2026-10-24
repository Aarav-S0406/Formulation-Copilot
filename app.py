import pandas as pd
import streamlit as st
import numpy as np
from src.predictor import load_model_artifact, predict_tannin_reduction
from src.recommender import CROPS, PRODUCTS, METHODS, profile, recommend, safety_check
from src.dataset import assess_dataset
from src.predictor import (
    load_model_artifact,
    predict_tannin_reduction,
    assess_evidence_coverage,
)

st.set_page_config(page_title="DesiProtein Formulation Copilot", page_icon="🌱", layout="wide")

@st.cache_resource
def get_model_artifact():
    return load_model_artifact()


try:
    model_artifact = get_model_artifact()
    model_error = None
except Exception as exc:
    model_artifact = None
    model_error = str(exc)

st.title("DesiProtein Formulation Copilot")
st.caption("Literature-backed processing recommender — first model scaffold")
st.caption(
    "Literature-backed recommendations + an experimental ML model "
    "for tannin reduction. Texture and flavour predictions are not available."
)


st.divider()
st.header("Experimental ML prediction")
st.write(
    "Estimate tannin reduction for a single crop-processing condition. "
    "This is a model estimate, not a guaranteed result or a food-safety assessment."
)

with st.expander("How reliable is this model?"):
    st.write("Evaluation: held-out source papers")
    col1, col2 = st.columns(2)

    col1.metric("Model MAE", "YOUR_MAE pp")
    col2.metric("Baseline MAE", "YOUR_BASELINE pp")

    st.caption(
        "MAE is the average absolute error in percentage points. "
        "Evaluation on held-out papers tests generalization to unseen "
        "sources, but the small dataset makes these results preliminary."
    )

if model_error:
    st.error(f"Could not load the trained model: {model_error}")

elif model_artifact:
    training_df = pd.read_csv(
        "data/processed/training_data.csv"
    )

    # Use categories observed in the training dataset.
    categorical_features = [
        "crop",
        "variety",
        "processing_method",
        "pretreatment",
        "light_condition",
        "cooking_method",
        "sample_form",
    ]

    numeric_features = [
        "soaking_time_h",
        "germination_time_h",
        "processing_temperature_c",
        "cooking_time_min",
    ]

    def observed_options(column):
        values = (
            training_df[column]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        return ["Not specified"] + sorted(values)

    with st.form("tannin_prediction_form"):
        st.caption(
            "Choose values from the categories represented in the "
            "training data. Numeric inputs are restricted to the "
            "observed training-data range."
        )

        col1, col2 = st.columns(2)

        with col1:
            crop_options = observed_options("crop")
            pred_crop = st.selectbox("Crop", crop_options[1:])

            variety = st.text_input(
                "Variety/accession",
                placeholder="e.g. Local; leave blank if unknown",
            )

            pred_method = st.selectbox(
                "Processing method",
                observed_options("processing_method")[1:],
            )

            pred_pretreatment = st.selectbox(
                "Pretreatment",
                observed_options("pretreatment"),
            )

            pred_light = st.selectbox(
                "Light condition",
                observed_options("light_condition"),
            )

        with col2:
            pred_cooking = st.selectbox(
                "Cooking method",
                observed_options("cooking_method"),
            )

            pred_sample = st.selectbox(
                "Sample form",
                observed_options("sample_form"),
            )

            numeric_values = {}

            for feature in numeric_features:
                values = pd.to_numeric(
                    training_df[feature], errors="coerce"
                ).dropna()

                low = float(values.min())
                high = float(values.max())
                default = float(values.median())

                numeric_values[feature] = st.number_input(
                    feature.replace("_", " "),
                    min_value=low,
                    max_value=high,
                    value=min(max(default, low), high),
                    step=1.0,
                )

        submitted = st.form_submit_button(
            "Predict tannin reduction",
            type="primary",
        )

    if submitted:
        inputs = {
            "crop": pred_crop,
            "variety": variety.strip() or np.nan,
            "processing_method": pred_method,
            "pretreatment": (
                np.nan if pred_pretreatment == "Not specified"
                else pred_pretreatment
            ),
            "light_condition": (
                np.nan if pred_light == "Not specified"
                else pred_light
            ),
            "cooking_method": (
                np.nan if pred_cooking == "Not specified"
                else pred_cooking
            ),
            "sample_form": (
                np.nan if pred_sample == "Not specified"
                else pred_sample
            ),
            **numeric_values,
        }

        try:
            prediction = predict_tannin_reduction(
                model_artifact, inputs
            )

            st.metric(
                "Estimated tannin reduction",
                f"{prediction:.1f}%",
            )

            if not 0 <= prediction <= 100:
                st.warning(
                    "The model output falls outside the physical "
                    "0–100% range. Treat this estimate as unreliable."
                )

            st.caption(
                "Model output for the selected conditions. "
                "Actual results depend on crop variety, assay method, "
                "experimental conditions and other factors. "
                "Validate against laboratory measurements."
            )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


        checks = assess_evidence_coverage(training_df, inputs)

        st.subheader("Evidence coverage")
        st.dataframe(
            pd.DataFrame(
                [{"Input": key, "Coverage": value}
                for key, value in checks.items()]
            ),
            hide_index=True,
            use_container_width=True,
        )

        if any(
            value in ("Unseen category", "Outside observed range",
                    "Insufficient evidence")
            for value in checks.values()
        ):
            st.warning(
                "One or more inputs have limited evidence coverage. "
                "Treat this prediction as exploratory and validate it experimentally."
            )


st.divider()
st.header("Compare processing experiments")

st.write(
    "Compare three candidate conditions for the same crop and "
    "processing method. These are model estimates, not measured results."
)

if model_artifact is not None:
    with st.form("experiment_comparison"):
        crop_options = sorted(
            training_df["crop"].dropna().astype(str).unique()
        )
        method_options = sorted(
            training_df["processing_method"]
            .dropna().astype(str).unique()
        )

        compare_crop = st.selectbox("Crop", crop_options)
        compare_method = st.selectbox(
            "Processing method", method_options
        )

        time_values = pd.to_numeric(
            training_df["soaking_time_h"], errors="coerce"
        ).dropna()

        low = float(time_values.min())
        high = float(time_values.max())

        defaults = [
            min(max(low, low), high),
            min(max((low + high) / 2, low), high),
            high,
        ]

        c1, c2, c3 = st.columns(3)
        times = []

        for col, label, default in zip(
            [c1, c2, c3],
            ["Experiment A", "Experiment B", "Experiment C"],
            defaults,
        ):
            with col:
                st.markdown(f"**{label}**")
                times.append(
                    st.number_input(
                        "Soaking time (hours)",
                        min_value=low,
                        max_value=high,
                        value=default,
                        key=f"comparison_{label}",
                    )
                )

        compare_submitted = st.form_submit_button(
            "Compare experiments"
        )

    if compare_submitted:
        results = []

        for label, time_h in zip(
            ["A", "B", "C"], times
        ):
            inputs = {
                "crop": compare_crop,
                "variety": np.nan,
                "processing_method": compare_method,
                "pretreatment": np.nan,
                "light_condition": np.nan,
                "cooking_method": np.nan,
                "sample_form": np.nan,
                "soaking_time_h": time_h,
                "germination_time_h": float(
                    pd.to_numeric(
                        training_df["germination_time_h"],
                        errors="coerce",
                    ).median()
                ),
                "processing_temperature_c": float(
                    pd.to_numeric(
                        training_df["processing_temperature_c"],
                        errors="coerce",
                    ).median()
                ),
                "cooking_time_min": float(
                    pd.to_numeric(
                        training_df["cooking_time_min"],
                        errors="coerce",
                    ).median()
                ),
            }

            prediction = predict_tannin_reduction(
                model_artifact, inputs
            )

            results.append({
                "Experiment": label,
                "Crop": compare_crop,
                "Method": compare_method,
                "Soaking time (h)": time_h,
                "Estimated tannin reduction (%)": round(
                    prediction, 2
                ),
            })

        result_df = pd.DataFrame(results)
        st.dataframe(result_df, hide_index=True, use_container_width=True)

        st.caption(
            "Other conditions are held at dataset medians or left "
            "unspecified. Differences are model outputs, not proof "
            "that changing soaking time causes the predicted change."
        )


with st.sidebar:
    crop = st.selectbox("Crop", list(CROPS))
    product = st.selectbox("Target product", list(PRODUCTS))
    method = st.selectbox("Processing method", list(METHODS))
    variety = st.text_input("Variety/accession (optional)")

p = profile(crop)
a, b = st.columns(2)
with a:
    st.subheader("Crop profile")
    st.metric("Reported protein range", p["protein"])
    st.write(p["limits"])
    st.caption(p["evidence"])
with b:
    st.subheader("Safety check")
    st.info(safety_check(crop))

st.subheader("Recommendation")
for r in recommend(crop, product, method):
    with st.container(border=True):
        st.markdown("**" + r["title"] + "**")
        st.write(r["text"])
        st.caption("Evidence: " + r["evidence"] + " | Basis: supplied DesiProtein reference")
        if r.get("caveat"):
            st.warning(r["caveat"])


st.subheader("Blend hypotheses from the reference")
st.caption("These are proposed experiments, not published optimal ratios.")
st.dataframe(pd.DataFrame([
 {"Product": "Paneer-like curd", "Starting blend": "Lablab + moth bean (70:30)", "Evidence": "Test"},
 {"Product": "Tempeh-style block", "Starting blend": "Lablab + rice bean (60:40)", "Evidence": "Test"},
 {"Product": "Crisps/bhujia-style snack", "Starting blend": "Moth bean + horse gram (80:20), roasted", "Evidence": "Test"},
 {"Product": "Nuggets/chunks", "Starting blend": "Moth/rice bean protein + pea or mung base (20:80)", "Evidence": "Test"},
]), hide_index=True, use_container_width=True)

st.divider()
st.header("Training-data readiness")
st.write("Upload a curated CSV extracted from measured research tables. Do not use crop averages or blend hypotheses as training rows.")
uploaded = st.file_uploader("Curated experimental CSV", type=["csv"])
if uploaded:
    try:
        df = pd.read_csv(uploaded)
        result = assess_dataset(df)
        st.write(f"Rows: **{result['rows']}** · Distinct papers: **{result['sources']}**")
        (st.success if result["ready"] else st.warning)(
            "Basic readiness checks passed; run train_model.py." if result["ready"] else "Not ready to train yet."
        )
        for issue in result["issues"]:
            st.write("- " + issue)
        st.dataframe(df.head(10), use_container_width=True)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")

with st.expander("What this prototype does and does not claim"):
    st.markdown("- Supports literature-backed processing recommendations and experiment planning.\n"
                "- Candidate ML target: tannin reduction; phytate reduction may be added if enough compatible data exist.\n"
                "- Does not predict flavour or nugget/paneer texture because the supplied reference says relevant data are missing.\n"
                "- Lablab safety rules are hard-coded reminders, not learned predictions.")
