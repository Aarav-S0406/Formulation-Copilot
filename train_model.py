"""Baseline model for tannin reduction. Refuses to train on tiny or untraceable data."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, r2_score
from src.dataset import assess_dataset, TARGET

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed" / "training_data.csv"
MODEL = ROOT / "models" / "tannin_reduction_model.joblib"
METRICS = ROOT / "models" / "evaluation_metrics.json"
NUM = ["soaking_time_h","germination_time_h","processing_temperature_c","cooking_time_min",
       "initial_tannin","initial_phytate","initial_trypsin_inhibitor","initial_protein"]
CAT = ["crop","variety","processing_method","pretreatment","light_condition","cooking_method","sample_form"]

def main():
    if not DATA.exists():
        raise SystemExit(f"Missing {DATA}. Copy the completed template there first.")
    df = pd.read_csv(DATA)
    check = assess_dataset(df)
    print(f"Rows: {check['rows']} | source papers: {check['sources']}")
    for issue in check["issues"]: print("- " + issue)
    if not check["ready"]:
        raise SystemExit("Refusing to train: data readiness checks failed.")
    df = df.dropna(subset=[TARGET,"source_id"]).copy()
    features = [c for c in CAT + NUM if c in df.columns]
    X, y, groups = df[features], pd.to_numeric(df[TARGET], errors="coerce"), df["source_id"].astype(str)
    valid = y.notna()
    X, y, groups = X.loc[valid], y.loc[valid], groups.loc[valid]
    splitter = GroupShuffleSplit(n_splits=50, test_size=.25, random_state=42)
    split = None
    for tr, te in splitter.split(X,y,groups):
        if len(tr) >= 20 and len(te) >= 5 and groups.iloc[te].nunique() >= 1:
            split = (tr,te); break
    if split is None: raise SystemExit("Could not form a usable held-out-paper split. Add more independent data.")
    tr, te = split
    nums = [c for c in features if c in NUM]
    cats = [c for c in features if c in CAT]
    transforms = []
    if nums: transforms.append(("num",Pipeline([("impute",SimpleImputer(strategy="median"))]),nums))
    if cats: transforms.append(("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),cats))
    pipe = Pipeline([("prep",ColumnTransformer(transforms)),
                     ("model",RandomForestRegressor(n_estimators=300,min_samples_leaf=2,random_state=42))])
    pipe.fit(X.iloc[tr],y.iloc[tr])
    pred = pipe.predict(X.iloc[te])
    baseline = [float(y.iloc[tr].mean())]*len(te)
    metrics = {
      "target":TARGET,"n_rows":len(df),"n_source_papers":int(groups.nunique()),
      "n_train":len(tr),"n_test":len(te),
      "held_out_source_papers":sorted(groups.iloc[te].unique().tolist()),
      "model_mae":float(mean_absolute_error(y.iloc[te],pred)),
      "mean_baseline_mae":float(mean_absolute_error(y.iloc[te],baseline)),
      "model_r2":float(r2_score(y.iloc[te],pred)) if len(te)>1 else None,
      "warning":"Exploratory metrics only; small literature datasets can be unstable."
    }
    MODEL.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump({"pipeline":pipe,"features":features,"target":TARGET},MODEL)
    METRICS.write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    print(json.dumps(metrics,indent=2))
    if metrics["model_mae"] >= metrics["mean_baseline_mae"]:
        print("WARNING: model did not beat the mean baseline on this split.")

if __name__ == "__main__": main()
