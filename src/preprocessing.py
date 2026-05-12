"""Preprocessing pipeline: limpeza, encoding e feature engineering."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Limpeza e correção de tipos."""
    df = df.copy()

    # TotalCharges tem espaços em branco (11 registros com tenure=0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    # Converter SeniorCitizen para categórico legível
    df["SeniorCitizen"] = df["SeniorCitizen"].replace({0: "No", 1: "Yes", "0": "No", "1": "Yes"})

    # Remover customerID (não é feature)
    df.drop(columns=["customerID"], inplace=True)

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering."""
    df = df.copy()

    # Faixa de permanência
    bins = [0, 6, 12, 24, 48, 72]
    labels = ["0-6m", "6-12m", "1-2a", "2-4a", "4-6a"]
    df["tenure_group"] = pd.cut(df["tenure"], bins=bins, labels=labels, include_lowest=True)

    # Charge ratio
    df["avg_monthly_charge"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"],
        df["MonthlyCharges"],
    )

    # Número de serviços contratados
    service_cols = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["num_services"] = df[service_cols].apply(
        lambda row: sum(1 for v in row if v not in ["No", "No phone service", "No internet service", "No"]),
        axis=1,
    )

    return df


def encode_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Encoding de variáveis categóricas."""
    df = df.copy()
    encoders = {}

    # Target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0, 1: 1, 0: 0}).astype(int)

    # Colunas categóricas
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Label encoding para binárias
    binary_cols = [c for c in cat_cols if df[c].nunique() == 2]
    for col in binary_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # One-hot para multi-valor
    multi_cols = [c for c in cat_cols if c not in binary_cols]
    if multi_cols:
        df = pd.get_dummies(df, columns=multi_cols, drop_first=True, dtype=int)

    return df, encoders


def prepare_splits(df: pd.DataFrame, target: str = "Churn", test_size: float = 0.2):
    """Split estratificado e scaling."""
    X = df.drop(columns=[target])
    y = df[target]

    # Garantir matriz numerica e sem ausentes antes de scaler/SMOTE.
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0).astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    X_train = X_train.copy()
    X_test = X_test.copy()

    scaler = StandardScaler()
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "avg_monthly_charge", "num_services"]
    num_cols = [c for c in num_cols if c in X_train.columns]
    X_train.loc[:, num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test.loc[:, num_cols] = scaler.transform(X_test[num_cols])

    return X_train, X_test, y_train, y_test, scaler


def run_preprocessing(df: pd.DataFrame):
    """Pipeline completo de preprocessing."""
    df_clean = clean_data(df)
    df_feat = add_features(df_clean)
    df_encoded, encoders = encode_data(df_feat)
    X_train, X_test, y_train, y_test, scaler = prepare_splits(df_encoded)
    return {
        "df_clean": clean_data(df.copy()),
        "df_feat": add_features(clean_data(df.copy())),
        "df_encoded": df_encoded,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "encoders": encoders,
        "feature_names": X_train.columns.tolist(),
    }
