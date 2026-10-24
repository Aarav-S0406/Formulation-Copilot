TARGET = "tannin_reduction_pct"
REQUIRED = ["crop","processing_method","sample_form","source_id","measurement_unit","basis","measurement_method",TARGET]

def assess_dataset(df):
    rows = len(df)
    sources = df["source_id"].nunique(dropna=True) if "source_id" in df.columns else 0
    issues = []
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing: issues.append("Missing required columns: " + ", ".join(missing))
    if rows < 30: issues.append(f"{rows} rows; 30 is only a rough minimum screening threshold.")
    if sources < 5: issues.append(f"{sources} distinct papers; aim for at least 5 for study-held-out evaluation.")
    if TARGET not in df.columns or df[TARGET].notna().sum() < 30:
        issues.append("Need at least 30 non-missing tannin_reduction_pct values.")
    for col in ("source_id","measurement_unit","basis"):
        if col in df.columns and df[col].isna().any():
            issues.append(f"Some rows have missing {col}; resolve or explicitly document them.")
    ready = not missing and rows >= 30 and sources >= 5 and df[TARGET].notna().sum() >= 30
    return {"ready":ready,"rows":rows,"sources":sources,"issues":issues}
