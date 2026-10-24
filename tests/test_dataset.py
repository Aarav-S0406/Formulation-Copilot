import pandas as pd
from src.dataset import assess_dataset

def test_empty_dataset_is_not_ready():
    assert assess_dataset(pd.DataFrame())["ready"] is False

def test_small_dataset_is_not_ready():
    df = pd.DataFrame({
      "crop":["Horse gram"]*3,"processing_method":["Soaking"]*3,"sample_form":["flour"]*3,
      "source_id":["paper-a"]*3,"measurement_unit":["mg/g"]*3,"basis":["dry"]*3,
      "measurement_method":["method-x"]*3,"tannin_reduction_pct":[10,12,14]
    })
    assert assess_dataset(df)["ready"] is False
