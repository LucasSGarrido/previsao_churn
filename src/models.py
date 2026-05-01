"""Treinamento e avaliação de modelos de classificação."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import plotly.graph_objects as go
import plotly.express as px

from src.eda import apply_layout, COLOR_PRIMARY, COLOR_SECONDARY, COLOR_DANGER, COLOR_ACCENT


def train_models(X_train, y_train, X_test, y_test):
    """Treina 3 modelos e retorna resultados comparativos."""

    # SMOTE para balanceamento
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            random_state=42, eval_metric="logloss", use_label_encoder=False,
        ),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_res, y_res)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        cv_scores = cross_val_score(model, X_res, y_res, cv=5, scoring="roc_auc")

        results[name] = {
            "model": model,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "cv_auc_mean": cv_scores.mean(),
            "cv_auc_std": cv_scores.std(),
            "confusion": confusion_matrix(y_test, y_pred),
            "report": classification_report(y_test, y_pred, output_dict=True),
        }

    return results


def comparison_table(results: dict) -> pd.DataFrame:
    """Tabela comparativa de modelos."""
    rows = []
    for name, r in results.items():
        rows.append({
            "Modelo": name,
            "Accuracy": r["accuracy"],
            "Precision": r["precision"],
            "Recall": r["recall"],
            "F1-Score": r["f1"],
            "ROC-AUC": r["roc_auc"],
            "CV AUC (mean)": r["cv_auc_mean"],
        })
    return pd.DataFrame(rows)


def roc_curve_chart(results: dict, y_test):
    """Curva ROC comparativa."""
    fig = go.Figure()
    colors = [COLOR_PRIMARY, COLOR_SECONDARY, COLOR_ACCENT]

    for i, (name, r) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            name=f"{name} (AUC={r['roc_auc']:.3f})",
            line=dict(width=3, color=colors[i]),
        ))

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], name="Baseline",
        line=dict(dash="dash", color="#6B7280", width=1),
    ))
    fig.update_layout(
        title="Curva ROC Comparativa",
        xaxis_title="Taxa de Falso Positivo",
        yaxis_title="Taxa de Verdadeiro Positivo",
    )
    return apply_layout(fig, height=450)


def confusion_matrix_chart(cm, model_name: str):
    """Heatmap de Matriz de Confusão."""
    labels = ["Retido", "Cancelou"]
    fig = px.imshow(
        cm, text_auto=True,
        x=labels, y=labels,
        title=f"Matriz de Confusao — {model_name}",
        color_continuous_scale=[COLOR_PRIMARY, COLOR_DANGER],
        labels={"x": "Previsto", "y": "Real"},
    )
    fig.update_layout(coloraxis_showscale=False)
    return apply_layout(fig, height=400)
