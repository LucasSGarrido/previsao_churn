"""Simulador de impacto financeiro do churn."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.eda import apply_layout, COLOR_PRIMARY, COLOR_SECONDARY, COLOR_DANGER, COLOR_ACCENT


def classify_risk(probabilities: np.ndarray) -> pd.Series:
    """Classifica clientes em faixas de risco."""
    bins = [0, 0.3, 0.5, 0.7, 1.0]
    labels = ["Baixo", "Moderado", "Alto", "Critico"]
    return pd.cut(probabilities, bins=bins, labels=labels, include_lowest=True)


def financial_impact(
    y_proba: np.ndarray,
    avg_monthly_revenue: float = 64.76,
    avg_customer_lifetime_months: float = 12,
    cac: float = 300.0,
    retention_discount: float = 0.15,
):
    """Calcula o impacto financeiro de reter clientes de alto risco."""
    risk = classify_risk(y_proba)
    n_total = len(y_proba)

    risk_counts = risk.value_counts().to_dict()
    high_risk = risk_counts.get("Critico", 0) + risk_counts.get("Alto", 0)

    # Receita perdida sem intervenção
    revenue_at_risk = high_risk * avg_monthly_revenue * avg_customer_lifetime_months

    # Custo de reposição (adquirir novos clientes)
    replacement_cost = high_risk * cac

    # Custo de retenção (desconto)
    retention_cost = high_risk * avg_monthly_revenue * avg_customer_lifetime_months * retention_discount

    # Economia líquida ao reter
    net_savings = revenue_at_risk - retention_cost

    return {
        "total_customers": n_total,
        "high_risk_customers": high_risk,
        "risk_distribution": risk_counts,
        "revenue_at_risk": revenue_at_risk,
        "replacement_cost": replacement_cost,
        "retention_cost": retention_cost,
        "net_savings": net_savings,
        "roi_retention": (net_savings - retention_cost) / retention_cost if retention_cost > 0 else 0,
    }


def risk_distribution_chart(y_proba: np.ndarray):
    """Gráfico de distribuição de risco."""
    risk = classify_risk(y_proba)
    counts = risk.value_counts().reindex(["Baixo", "Moderado", "Alto", "Critico"]).fillna(0).reset_index()
    counts.columns = ["Risco", "Clientes"]

    color_map = {"Baixo": COLOR_PRIMARY, "Moderado": COLOR_SECONDARY, "Alto": "#FFA500", "Critico": COLOR_DANGER}

    fig = px.bar(
        counts, x="Risco", y="Clientes",
        title="Distribuicao de Risco de Churn",
        color="Risco",
        color_discrete_map=color_map,
        text="Clientes",
    )
    fig.update_traces(textposition="outside")
    return apply_layout(fig, height=400)


def financial_waterfall(impact: dict):
    """Waterfall do impacto financeiro."""
    fig = go.Figure(go.Waterfall(
        name="Impacto",
        orientation="v",
        measure=["relative", "relative", "relative", "total"],
        x=["Receita em Risco", "Custo Reposicao", "Custo Retencao", "Economia Liquida"],
        y=[
            impact["revenue_at_risk"],
            impact["replacement_cost"],
            -impact["retention_cost"],
            impact["net_savings"],
        ],
        connector={"line": {"color": "#6B7280"}},
        increasing={"marker": {"color": COLOR_PRIMARY}},
        decreasing={"marker": {"color": COLOR_DANGER}},
        totals={"marker": {"color": COLOR_SECONDARY}},
    ))
    fig.update_layout(title="Impacto Financeiro da Retencao")
    return apply_layout(fig, height=450)


def probability_histogram(y_proba: np.ndarray):
    """Histograma de probabilidades de churn."""
    fig = px.histogram(
        x=y_proba, nbins=50,
        title="Distribuicao de Probabilidades de Churn",
        labels={"x": "Probabilidade de Churn", "y": "Clientes"},
        color_discrete_sequence=[COLOR_ACCENT],
        opacity=0.8,
    )
    fig.add_vline(x=0.5, line_dash="dash", line_color=COLOR_DANGER, annotation_text="Threshold 50%")
    return apply_layout(fig, height=400)
