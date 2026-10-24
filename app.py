import pandas as pd
import streamlit as st
from src.recommender import CROPS, PRODUCTS, METHODS, profile, recommend, safety_check
from src.dataset import assess_dataset

st.set_page_config(page_title="DesiProtein Formulation Copilot", page_icon="🌱", layout="wide")
st.title("DesiProtein Formulation Copilot")
st.caption("Literature-backed processing recommender — first model scaffold")
st.warning("This version is NOT a trained predictive model. It uses source-aware rules and does not invent texture, flavour, or antinutrient predictions.")

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
