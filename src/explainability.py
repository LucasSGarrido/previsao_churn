"""Explicabilidade de modelos com SHAP e Feature Importance."""

import numpy as np
import pandas as pd
import shap
import plotly.express as px
import plotly.graph_objects as go

from src.eda import apply_layout, COLOR_PRIMARY, COLOR_DANGER, COLOR_ACCENT


def compute_shap_values(model, X_test: pd.DataFrame):
    """Calcula SHAP values para o modelo (usa TreeExplainer para modelos de árvore)."""
    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        explainer = shap.LinearExplainer(model, X_test)

    shap_values = explainer.shap_values(X_test)

    # Para modelos binários que retornam lista [class_0, class_1]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    return explainer, shap_values


def shap_importance_chart(shap_values, feature_names, top_n=15):
    """Gráfico de importância média SHAP (barras horizontais)."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": mean_abs,
    }).sort_values("importance", ascending=True).tail(top_n)

    fig = px.bar(
        importance, x="importance", y="feature", orientation="h",
        title=f"Top {top_n} Features — Importancia SHAP",
        labels={"importance": "Mean |SHAP|", "feature": "Feature"},
        color="importance",
        color_continuous_scale=[COLOR_PRIMARY, COLOR_ACCENT],
    )
    fig.update_layout(coloraxis_showscale=False)
    return apply_layout(fig, height=500)


def feature_importance_native(model, feature_names, top_n=15):
    """Feature importance nativa do modelo (para Random Forest e XGBoost)."""
    importances = model.feature_importances_
    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=True).tail(top_n)

    fig = px.bar(
        imp_df, x="importance", y="feature", orientation="h",
        title=f"Top {top_n} Features — Importancia Nativa do Modelo",
        labels={"importance": "Importancia", "feature": "Feature"},
        color="importance",
        color_continuous_scale=[COLOR_PRIMARY, COLOR_DANGER],
    )
    fig.update_layout(coloraxis_showscale=False)
    return apply_layout(fig, height=500)


def shap_waterfall_data(shap_values, X_test: pd.DataFrame, idx: int = 0):
    """Dados para waterfall de um cliente específico."""
    sv = shap_values[idx]
    features = X_test.columns.tolist()

    data = pd.DataFrame({
        "feature": features,
        "shap_value": sv,
        "feature_value": X_test.iloc[idx].values,
    }).sort_values("shap_value", key=abs, ascending=False).head(10)

    fig = go.Figure(go.Waterfall(
        orientation="h",
        y=data["feature"],
        x=data["shap_value"],
        connector={"line": {"color": "#6B7280"}},
        increasing={"marker": {"color": COLOR_DANGER}},
        decreasing={"marker": {"color": COLOR_PRIMARY}},
    ))
    fig.update_layout(title=f"SHAP Waterfall — Cliente #{idx}")
    return apply_layout(fig, height=450)
