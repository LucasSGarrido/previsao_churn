from pathlib import Path

import pandas as pd

from src.data_loader import load_telco_data
from src.preprocessing import clean_data, run_preprocessing


def test_clean_data_replaces_blank_total_charges():
    df = pd.DataFrame(
        {
            "customerID": ["1"],
            "SeniorCitizen": [0],
            "TotalCharges": [" "],
        }
    )

    cleaned = clean_data(df)

    assert cleaned.loc[0, "TotalCharges"] == 0
    assert cleaned.loc[0, "SeniorCitizen"] == "No"


def test_preprocessing_outputs_no_missing_values():
    df = load_telco_data(Path.cwd())

    pipeline = run_preprocessing(df)

    assert pipeline["df_encoded"].isna().sum().sum() == 0
    assert pipeline["X_train"].isna().sum().sum() == 0
    assert pipeline["X_test"].isna().sum().sum() == 0
    assert len(pipeline["feature_names"]) > 20
