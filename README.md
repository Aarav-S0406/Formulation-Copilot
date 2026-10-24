# Formulation Copilot — first model scaffold

Based on the supplied **DesiProtein: Crop, Processing & Formulation Reference**.

## Implemented
- Streamlit UI for horse gram, moth bean, rice bean and lablab.
- Source-aware rule-based recommendations and evidence caveats.
- Lablab safety reminder.
- CSV readiness checker.
- Baseline scikit-learn regression script for tannin reduction.
- Study-held-out evaluation by `source_id` and comparison against a mean baseline.
- Empty data template; no fake training observations.

## Run
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m streamlit run app.py
```

## Prepare data
1. Extract numeric tables from full research papers.
2. Use one row per measured treatment condition.
3. Preserve initial/final values, units, wet/dry basis, assay method and source paper.
4. Calculate `tannin_reduction_pct` only for comparable paired measurements:
   `((initial_tannin - final_tannin) / initial_tannin) * 100`
5. Save completed data as `data/processed/training_data.csv`.
6. Run `python train_model.py`.

The script requires at least 30 rows and 5 distinct papers as a basic screening gate. These thresholds do not guarantee scientific validity. It refuses to train on an empty template or tiny dataset.

## Scientific constraints
- Protein ranges are literature summaries, not individual training rows.
- Blend ratios are hypotheses, not validated optima.
- Do not claim flavour, paneer or nugget texture prediction.
- Do not merge incompatible units, assay methods, sample forms or wet/dry bases.
- Do not invent missing measurements.
- Lablab safety rules are hard-coded reminders, not model predictions.
- If data are sparse, retain the literature-backed recommender and be honest about limitations.
