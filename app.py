from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.predictor import (
    load_model_artifact,
    predict_tannin_reduction,
    assess_evidence_coverage,
)
from src.recommender import (
    CROPS,
    PRODUCTS,
    METHODS,
    profile,
    recommend,
    safety_check,
)
from src.dataset import assess_dataset


# --------------------------------------------------
# Configuration and paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed" / "training_data.csv"

NUMERIC_FEATURES = [
    "soaking_time_h",
    "germination_time_h",
    "processing_temperature_c",
    "cooking_time_min",
]

CATEGORICAL_FEATURES = [
    "crop",
    "variety",
    "processing_method",
    "pretreatment",
    "light_condition",
    "cooking_method",
    "sample_form",
]

st.set_page_config(
    page_title="DesiProtein Formulation Copilot",
    page_icon="🌱",
    layout="wide",
)


# --------------------------------------------------
# Load model and training data
# --------------------------------------------------

@st.cache_resource
def get_model_artifact():
    return load_model_artifact()


@st.cache_data
def get_training_data():
    return pd.read_csv(DATA_PATH)


try:
    model_artifact = get_model_artifact()
    model_error = None
except Exception as exc:
    model_artifact = None
    model_error = str(exc)

try:
    training_df = get_training_data()
    data_error = None
except Exception as exc:
    training_df = None
    data_error = str(exc)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("DesiProtein Formulation Copilot")

st.caption(
    "Literature-backed processing recommendations and "
    "experimental ML-assisted formulation research."
)

st.info(
    "The ML model estimates tannin reduction only. "
    "It does not predict flavour, texture, overall nutritional quality, "
    "or food safety."
)


# --------------------------------------------------
# Sidebar: formulation inputs
# --------------------------------------------------

with st.sidebar:
    st.header("Formulation setup")

    crop = st.selectbox("Crop", list(CROPS))
    product = st.selectbox("Target product", list(PRODUCTS))
    method = st.selectbox("Processing method", METHODS)

    variety = st.text_input(
        "Variety/accession (optional)",
        placeholder="Enter a known variety",
    )


# --------------------------------------------------
# Crop profile and safety
# --------------------------------------------------

st.header("Crop profile and safety")

p = profile(crop)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Crop profile")
    st.metric("Reported protein range", p["protein"])
    st.write(p["limits"])
    st.caption(p["evidence"])

with col2:
    st.subheader("Safety check")
    st.warning(safety_check(crop))


# --------------------------------------------------
# Rule-based recommendations
# --------------------------------------------------

st.header("Processing recommendations")

for recommendation in recommend(crop, product, method):
    with st.container(border=True):
        st.markdown("**" + recommendation["title"] + "**")
        st.write(recommendation["text"])

        st.caption(
            "Evidence: "
            + recommendation["evidence"]
            + " | Basis: supplied DesiProtein reference"
        )

        if recommendation.get("caveat"):
            st.warning(recommendation["caveat"])


# --------------------------------------------------
# Model evaluation information
# --------------------------------------------------

st.divider()
st.header("Experimental ML model")

with st.expander("How reliable is this model?"):
    st.write("Evaluation method: held-out source papers.")

    st.warning(
        "Evaluation metrics must be copied from the verified model "
        "evaluation run. They are intentionally not displayed until "
        "those results are confirmed."
    )

    st.caption(
        "MAE measures the average absolute prediction error in "
        "percentage points. A small dataset and a small number of "
        "independent papers mean that performance estimates remain "
        "preliminary. These metrics do not establish laboratory validity."
    )


# --------------------------------------------------
# Check data availability
# --------------------------------------------------

if model_error:
    st.error(f"Could not load the trained model: {model_error}")

if data_error:
    st.error(f"Could not load the training dataset: {data_error}")


# --------------------------------------------------
# Single-condition ML prediction
# --------------------------------------------------

if model_artifact is not None and training_df is not None:

    st.subheader("Predict tannin reduction")

    st.write(
        "Estimate tannin reduction for one set of processing conditions. "
        "The result is a model estimate, not a measured outcome."
    )

    required_columns = (
        CATEGORICAL_FEATURES
        + NUMERIC_FEATURES
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in training_df.columns
    ]

    if missing_columns:
        st.error(
            "The training dataset is missing required columns: "
            + ", ".join(missing_columns)
        )

    else:
        def observed_options(column):
            values = (
                training_df[column]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            return ["Not specified"] + sorted(values)

        numeric_ranges = {}
        numeric_error = None

        for feature in NUMERIC_FEATURES:
            values = pd.to_numeric(
                training_df[feature],
                errors="coerce",
            ).dropna()

            if values.empty:
                numeric_error = (
                    f"No numeric training values are available for "
                    f"{feature}. Prediction inputs cannot be configured "
                    "reliably until the dataset is corrected."
                )
                break

            numeric_ranges[feature] = {
                "min": float(values.min()),
                "max": float(values.max()),
                "median": float(values.median()),
            }

        if numeric_error:
            st.error(numeric_error)

        else:
            with st.form("tannin_prediction_form"):
                st.caption(
                    "Numeric values are restricted to the observed "
                    "training-data ranges. This does not guarantee that "
                    "the combination of selected conditions was studied."
                )

                col1, col2 = st.columns(2)

                with col1:
                    crop_options = observed_options("crop")

                    pred_crop = st.selectbox(
                        "Crop",
                        crop_options[1:],
                    )

                    pred_variety = st.text_input(
                        "Variety/accession",
                        placeholder="Leave blank if unknown",
                    )

                    method_options = observed_options(
                        "processing_method"
                    )

                    pred_method = st.selectbox(
                        "Processing method",
                        method_options[1:],
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

                    for feature in NUMERIC_FEATURES:
                        bounds = numeric_ranges[feature]

                        numeric_values[feature] = st.number_input(
                            feature.replace("_", " "),
                            min_value=bounds["min"],
                            max_value=bounds["max"],
                            value=bounds["median"],
                            step=1.0,
                            key=f"prediction_{feature}",
                        )

                submitted = st.form_submit_button(
                    "Predict tannin reduction",
                    type="primary",
                )

            if submitted:
                inputs = {
                    "crop": pred_crop,
                    "variety": pred_variety.strip() or np.nan,
                    "processing_method": pred_method,
                    "pretreatment": (
                        np.nan
                        if pred_pretreatment == "Not specified"
                        else pred_pretreatment
                    ),
                    "light_condition": (
                        np.nan
                        if pred_light == "Not specified"
                        else pred_light
                    ),
                    "cooking_method": (
                        np.nan
                        if pred_cooking == "Not specified"
                        else pred_cooking
                    ),
                    "sample_form": (
                        np.nan
                        if pred_sample == "Not specified"
                        else pred_sample
                    ),
                    **numeric_values,
                }

                try:
                    prediction = predict_tannin_reduction(
                        model_artifact,
                        inputs,
                    )

                    st.metric(
                        "Estimated tannin reduction",
                        f"{prediction:.1f}%",
                    )

                    if not 0 <= prediction <= 100:
                        st.warning(
                            "The prediction is outside the physical "
                            "0–100% range. Treat it as unreliable."
                        )

                    st.caption(
                        "Actual results depend on variety, measurement "
                        "method, experimental conditions and other factors. "
                        "Validate predictions using laboratory measurements."
                    )

                    # Evidence coverage
                    checks = assess_evidence_coverage(
                        training_df,
                        inputs,
                    )

                    st.subheader("Evidence coverage")

                    coverage_df = pd.DataFrame(
                        [
                            {
                                "Input": key,
                                "Coverage": value,
                            }
                            for key, value in checks.items()
                        ]
                    )

                    st.dataframe(
                        coverage_df,
                        hide_index=True,
                        use_container_width=True,
                    )

                    weak_coverage = any(
                        value in (
                            "Unseen category",
                            "Outside observed range",
                            "Insufficient evidence",
                        )
                        for value in checks.values()
                    )

                    if weak_coverage:
                        st.warning(
                            "Some inputs have limited evidence coverage. "
                            "This prediction should be treated as exploratory."
                        )

                except Exception as exc:
                    st.error(
                        f"Prediction or evidence assessment failed: {exc}"
                    )


# --------------------------------------------------
# Compare processing experiments
# --------------------------------------------------

st.divider()
st.header("Compare processing experiments")

st.write(
    "Compare three soaking-time settings for the same crop. "
    "Other conditions are held fixed at dataset medians or left "
    "unspecified. These are model estimates, not measured results."
)

if model_artifact is not None and training_df is not None:

    if "processing_method" not in training_df.columns:
        st.warning(
            "The training dataset does not contain processing_method."
        )

    elif "soaking_time_h" not in training_df.columns:
        st.warning(
            "The training dataset does not contain soaking_time_h."
        )

    else:
        method_values = (
            training_df["processing_method"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        # Use the exact category spelling stored in the CSV.
        soaking_category = next(
            (
                value for value in method_values
                if value.strip().lower() == "soaking"
            ),
            None,
        )

        time_values = pd.to_numeric(
            training_df["soaking_time_h"],
            errors="coerce",
        ).dropna()

        if soaking_category is None:
            st.info(
                "Experiment comparison is restricted to soaking, "
                "but the dataset has no 'Soaking' processing category. "
                "Check the category names in your CSV."
            )

        elif time_values.empty:
            st.warning(
                "No numeric soaking-time values are available "
                "for experiment comparison."
            )

        else:
            low = float(time_values.min())
            high = float(time_values.max())
            middle = (low + high) / 2

            crop_options = sorted(
                training_df["crop"]
                .dropna()
                .astype(str)
                .unique()
            )

            with st.form("experiment_comparison"):
                compare_crop = st.selectbox(
                    "Crop",
                    crop_options,
                    key="compare_crop",
                )

                st.write("Choose three soaking times.")

                col1, col2, col3 = st.columns(3)

                with col1:
                    time_a = st.number_input(
                        "Experiment A (hours)",
                        min_value=low,
                        max_value=high,
                        value=low,
                        key="comparison_a",
                    )

                with col2:
                    time_b = st.number_input(
                        "Experiment B (hours)",
                        min_value=low,
                        max_value=high,
                        value=middle,
                        key="comparison_b",
                    )

                with col3:
                    time_c = st.number_input(
                        "Experiment C (hours)",
                        min_value=low,
                        max_value=high,
                        value=high,
                        key="comparison_c",
                    )

                compare_submitted = st.form_submit_button(
                    "Compare experiments",
                    type="primary",
                )

            if compare_submitted:
                results = []

                median_values = {}

                for feature in NUMERIC_FEATURES:
                    values = pd.to_numeric(
                        training_df[feature],
                        errors="coerce",
                    ).dropna()

                    median_values[feature] = (
                        float(values.median())
                        if not values.empty
                        else np.nan
                    )

                experiment_times = [
                    ("A", time_a),
                    ("B", time_b),
                    ("C", time_c),
                ]

                try:
                    for label, time_h in experiment_times:
                        inputs = {
                            "crop": compare_crop,
                            "variety": np.nan,
                            "processing_method": soaking_category,
                            "pretreatment": np.nan,
                            "light_condition": np.nan,
                            "cooking_method": np.nan,
                            "sample_form": np.nan,
                            **median_values,
                        }

                        # Override only the variable being compared.
                        inputs["soaking_time_h"] = time_h

                        prediction = predict_tannin_reduction(
                            model_artifact,
                            inputs,
                        )

                        checks = assess_evidence_coverage(
                            training_df,
                            inputs,
                        )

                        limited_evidence = any(
                            value in (
                                "Unseen category",
                                "Outside observed range",
                                "Insufficient evidence",
                            )
                            for value in checks.values()
                        )

                        results.append(
                            {
                                "Experiment": label,
                                "Crop": compare_crop,
                                "Method": soaking_category,
                                "Soaking time (h)": time_h,
                                "Estimated tannin reduction (%)": round(
                                    prediction, 2
                                ),
                                "Limited evidence": (
                                    "Yes" if limited_evidence else "No"
                                ),
                            }
                        )

                    result_df = pd.DataFrame(results)

                    st.dataframe(
                        result_df,
                        hide_index=True,
                        use_container_width=True,
                    )

                    st.caption(
                        "These comparisons are exploratory. Individually "
                        "observed feature values do not prove that the "
                        "complete combination appeared in a source paper. "
                        "Differences are not proof of causation."
                    )

                    if any(
                        row["Limited evidence"] == "Yes"
                        for row in results
                    ):
                        st.warning(
                            "At least one experiment has limited evidence "
                            "coverage. Do not treat the ranking as a "
                            "validated recommendation."
                        )

                except Exception as exc:
                    st.error(
                        f"Experiment comparison failed: {exc}"
                    )


# --------------------------------------------------
# Proposed blend hypotheses
# --------------------------------------------------

st.divider()
st.header("Blend hypotheses")

st.caption(
    "These are proposed experiments, not published optimal ratios "
    "or ML-validated formulations."
)

blend_df = pd.DataFrame(
    [
        {
            "Product": "Paneer-like curd",
            "Starting blend": "Lablab + moth bean (70:30)",
            "Status": "Hypothesis to test",
        },
        {
            "Product": "Tempeh-style block",
            "Starting blend": "Lablab + rice bean (60:40)",
            "Status": "Hypothesis to test",
        },
        {
            "Product": "Crisps/bhujia-style snack",
            "Starting blend": "Moth bean + horse gram (80:20), roasted",
            "Status": "Hypothesis to test",
        },
        {
            "Product": "Nuggets/chunks",
            "Starting blend": "Moth/rice bean protein + pea or mung base (20:80)",
            "Status": "Hypothesis to test",
        },
    ]
)

st.dataframe(
    blend_df,
    hide_index=True,
    use_container_width=True,
)



# --------------------------------------------------
# Interactive recipe nutrition calculator
# --------------------------------------------------

st.divider()
st.header("Recipe nutrition & formulation")

st.write(
    "Adjust ingredient quantities to estimate dry-mix protein "
    "and inspect the amino-acid profile of the legume blend."
)

REFERENCE_DIR = BASE_DIR / "data" / "reference"

try:
    ingredients_df = pd.read_csv(
        REFERENCE_DIR / "ingredients.csv"
    )
    recipe_df = pd.read_csv(
        REFERENCE_DIR / "recipe_ingredients.csv"
    )
    amino_df = pd.read_csv(
        REFERENCE_DIR / "amino_acid_profiles.csv"
    )

    recipe_ids = sorted(
        recipe_df["recipe_id"].dropna().unique().tolist()
    )

    if not recipe_ids:
        st.warning("No recipes are available.")
    else:
        selected_recipe = st.selectbox(
            "Select recipe",
            recipe_ids,
            key="nutrition_recipe",
        )

        base_recipe = recipe_df[
            recipe_df["recipe_id"] == selected_recipe
        ][["ingredient", "quantity_g"]].copy()

        st.subheader("Adjust the dry-mix ingredients")
        st.caption(
            "Change quantities in grams. All results below recalculate "
            "from the edited quantities; they are estimates, not "
            "laboratory measurements."
        )

        edited_recipe = st.data_editor(
            base_recipe,
            hide_index=True,
            use_container_width=True,
            num_rows="fixed",
            disabled=["ingredient"],
            column_config={
                "quantity_g": st.column_config.NumberColumn(
                    "Quantity (g)",
                    min_value=0.0,
                    max_value=1000.0,
                    step=5.0,
                    format="%.1f",
                    required=True,
                ),
            },
            key=f"recipe_editor_{selected_recipe}",
        )

        if (
            edited_recipe["quantity_g"].isna().any()
            or (edited_recipe["quantity_g"] < 0).any()
        ):
            st.error("Ingredient quantities must be non-negative numbers.")

        else:
            calculated = edited_recipe.merge(
                ingredients_df,
                on="ingredient",
                how="left",
                validate="many_to_one",
            )

            if calculated["protein_g_per_100g"].isna().any():
                missing = calculated.loc[
                    calculated["protein_g_per_100g"].isna(),
                    "ingredient",
                ].tolist()
                st.error(
                    "Missing protein values for: "
                    + ", ".join(missing)
                )

            else:
                calculated["protein_contribution_g"] = (
                    calculated["quantity_g"]
                    * calculated["protein_g_per_100g"]
                    / 100
                )

                total_mass = float(
                    calculated["quantity_g"].sum()
                )
                total_protein = float(
                    calculated["protein_contribution_g"].sum()
                )

                if total_mass <= 0:
                    st.warning(
                        "Add at least one ingredient to calculate nutrition."
                    )
                else:
                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "Dry-mix mass",
                        f"{total_mass:.1f} g",
                    )
                    col2.metric(
                        "Estimated protein",
                        f"{total_protein:.1f} g",
                    )
                    col3.metric(
                        "Protein by mass",
                        f"{total_protein / total_mass * 100:.1f}%",
                    )

                    st.subheader("Protein contribution by ingredient")

                    breakdown = calculated[
                        [
                            "ingredient",
                            "quantity_g",
                            "protein_g_per_100g",
                            "protein_contribution_g",
                        ]
                    ].rename(
                        columns={
                            "ingredient": "Ingredient",
                            "quantity_g": "Quantity (g)",
                            "protein_g_per_100g": (
                                "Assumed protein (g/100 g)"
                            ),
                            "protein_contribution_g": (
                                "Protein contribution (g)"
                            ),
                        }
                    )

                    st.dataframe(
                        breakdown.round(2),
                        hide_index=True,
                        use_container_width=True,
                    )

                    st.caption(
                        "Protein values come from the reference CSV. "
                        "Ingredient variability, moisture, processing "
                        "losses and laboratory results are not accounted for."
                    )

                    # Amino-acid profile of the three-legume portion.
                    st.subheader("Legume amino-acid profile")

                    legume_map = {
                        "Rice Bean Flour": "Rice Bean",
                        "Moth Bean Flour": "Moth Bean",
                        "Red Lentil Flour": "Red Lentil",
                    }

                    calculated["crop"] = calculated[
                        "ingredient"
                    ].map(legume_map)

                    legumes = calculated[
                        calculated["crop"].notna()
                    ].copy()

                    if legumes.empty:
                        st.info(
                            "This recipe contains none of the three "
                            "supported legumes."
                        )
                    else:
                        legumes["legume_protein_g"] = (
                            legumes["quantity_g"]
                            * legumes["protein_g_per_100g"]
                            / 100
                        )

                        protein_by_crop = (
                            legumes.groupby("crop")[
                                "legume_protein_g"
                            ].sum()
                        )
                        total_legume_protein = float(
                            protein_by_crop.sum()
                        )

                        required_columns = {
                            "Amino Acid",
                            "Rice Bean",
                            "Moth Bean",
                            "Red Lentil",
                        }

                        missing_columns = (
                            required_columns - set(amino_df.columns)
                        )

                        if missing_columns:
                            st.error(
                                "Amino-acid CSV is missing columns: "
                                + ", ".join(sorted(missing_columns))
                            )
                        elif total_legume_protein <= 0:
                            st.info(
                                "Increase at least one legume quantity "
                                "to calculate its amino-acid profile."
                            )
                        else:
                            amino_results = []

                            for _, row in amino_df.iterrows():
                                weighted_value = 0.0

                                for crop_name in (
                                    "Rice Bean",
                                    "Moth Bean",
                                    "Red Lentil",
                                ):
                                    crop_protein = float(
                                        protein_by_crop.get(
                                            crop_name, 0.0
                                        )
                                    )

                                    if crop_protein == 0:
                                        continue

                                    concentration = pd.to_numeric(
                                        pd.Series([row[crop_name]]),
                                        errors="coerce",
                                    ).iloc[0]

                                    if pd.isna(concentration):
                                        weighted_value = np.nan
                                        break

                                    weighted_value += (
                                        crop_protein
                                        * float(concentration)
                                    )

                                if pd.notna(weighted_value):
                                    weighted_value /= (
                                        total_legume_protein
                                    )

                                amino_results.append({
                                    "Amino acid": row["Amino Acid"],
                                    "Estimated g per 100 g "
                                    "legume protein": (
                                        round(weighted_value, 2)
                                        if pd.notna(weighted_value)
                                        else np.nan
                                    ),
                                })

                            st.metric(
                                "Protein from the three legumes",
                                f"{total_legume_protein:.1f} g",
                            )

                            st.dataframe(
                                pd.DataFrame(amino_results),
                                hide_index=True,
                                use_container_width=True,
                            )

                            st.caption(
                                "This profile covers only rice bean, "
                                "moth bean and red lentil. It excludes "
                                "wheat gluten, textured soy protein and "
                                "chickpea flour. It is not a validated "
                                "protein-quality score or a PDCAAS/DIAAS "
                                "calculation."
                            )

except FileNotFoundError as exc:
    st.error(
        f"Nutrition reference file not found: {exc.filename}. "
        "Check the files in data/reference."
    )
except Exception as exc:
    st.error(f"Nutrition calculator error: {exc}")



# --------------------------------------------------
# Training-data readiness
# --------------------------------------------------

st.divider()
st.header("Training-data readiness")

st.write(
    "Upload a curated CSV extracted from measured research tables. "
    "Do not use crop averages or proposed blend hypotheses as training rows."
)

uploaded = st.file_uploader(
    "Curated experimental CSV",
    type=["csv"],
)

if uploaded is not None:
    try:
        uploaded_df = pd.read_csv(uploaded)
        result = assess_dataset(uploaded_df)

        st.write(
            f"Rows: **{result['rows']}** · "
            f"Distinct papers: **{result['sources']}**"
        )

        if result["ready"]:
            st.success(
                "Basic readiness checks passed. "
                "Review data quality and source separation before training."
            )
        else:
            st.warning("The dataset is not ready to train yet.")

        for issue in result["issues"]:
            st.write("- " + issue)

        st.dataframe(
            uploaded_df.head(10),
            hide_index=True,
            use_container_width=True,
        )

    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")


# --------------------------------------------------
# Scope and limitations
# --------------------------------------------------

with st.expander("What this prototype does and does not claim"):
    st.markdown(
        "- Provides literature-backed processing recommendations.\n"
        "- Estimates tannin reduction using an experimental ML model.\n"
        "- Does not establish the optimal formulation or processing conditions.\n"
        "- Does not predict texture, flavour, or protein quality.\n"
        "- Does not certify food safety.\n"
        "- Requires additional compatible, measured research data and "
        "laboratory validation before real-world deployment."
    )
