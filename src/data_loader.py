"""Carregamento e cache do dataset Telco Customer Churn."""

from pathlib import Path

import pandas as pd


def load_telco_data(base_dir: Path) -> pd.DataFrame:
    """Carrega o dataset Telco Customer Churn."""
    filepath = base_dir / "data" / "telco_churn.csv"
    df = pd.read_csv(filepath)
    return df
