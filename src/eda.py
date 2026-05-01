"""Análise exploratória de dados — gráficos e estatísticas para o dashboard."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

COLOR_PRIMARY = "#00F5A0"
COLOR_SECONDARY = "#00D9F5"
COLOR_DANGER = "#E85D3F"
COLOR_ACCENT = "#7C5CFC"
COLOR_BG = "#0D1117"
COLOR_CARD = "#161B22"


def apply_layout(fig, height=400):
    """Layout padrão Cyber Emerald para gráficos Plotly."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#E6EDF3"),
        height=height,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
    return fig


def churn_rate_chart(df: pd.DataFrame):
    """Taxa geral de churn (donut)."""
    counts = df["Churn"].value_counts().reset_index()
    counts.columns = ["Churn", "count"]
    counts["Churn"] = counts["Churn"].map({"Yes": "Cancelou", "No": "Retido"})
    fig = px.pie(
        counts, names="Churn", values="count", hole=0.6,
        title="Taxa de Churn Geral",
        color="Churn",
        color_discrete_map={"Cancelou": COLOR_DANGER, "Retido": COLOR_PRIMARY},
    )
    return apply_layout(fig)


def churn_by_category(df: pd.DataFrame, col: str, title: str = None):
    """Taxa de churn por categoria."""
    ct = pd.crosstab(df[col], df["Churn"], normalize="index").reset_index()
    ct.columns = [col, "Retido", "Cancelou"]
    ct = ct.sort_values("Cancelou", ascending=False)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=ct[col], y=ct["Cancelou"], name="Cancelou", marker_color=COLOR_DANGER))
    fig.add_trace(go.Bar(x=ct[col], y=ct["Retido"], name="Retido", marker_color=COLOR_PRIMARY))
    fig.update_layout(barmode="stack", title=title or f"Churn por {col}")
    return apply_layout(fig)


def tenure_distribution(df: pd.DataFrame):
    """Distribuição de tenure por status de churn."""
    fig = px.histogram(
        df, x="tenure", color="Churn", nbins=36,
        title="Distribuicao de Permanencia (tenure)",
        labels={"tenure": "Meses de permanencia", "Churn": "Status"},
        color_discrete_map={"Yes": COLOR_DANGER, "No": COLOR_PRIMARY},
        barmode="overlay", opacity=0.7,
    )
    return apply_layout(fig)


def monthly_charges_box(df: pd.DataFrame):
    """Boxplot de MonthlyCharges por churn."""
    fig = px.box(
        df, x="Churn", y="MonthlyCharges",
        title="Mensalidade por Status de Churn",
        labels={"MonthlyCharges": "Mensalidade (USD)", "Churn": "Status"},
        color="Churn",
        color_discrete_map={"Yes": COLOR_DANGER, "No": COLOR_PRIMARY},
    )
    return apply_layout(fig)


def correlation_heatmap(df_encoded: pd.DataFrame):
    """Heatmap de correlação com a variável Churn."""
    corr = df_encoded.corr()["Churn"].drop("Churn").sort_values()
    top = pd.concat([corr.head(10), corr.tail(10)])
    fig = px.bar(
        x=top.values, y=top.index, orientation="h",
        title="Top Correlacoes com Churn",
        labels={"x": "Correlacao", "y": "Feature"},
        color=top.values,
        color_continuous_scale=[COLOR_PRIMARY, "#FFFFFF", COLOR_DANGER],
    )
    fig.update_layout(coloraxis_showscale=False)
    return apply_layout(fig, height=500)


def churn_by_contract(df: pd.DataFrame):
    """Churn por tipo de contrato."""
    return churn_by_category(df, "Contract", "Churn por Tipo de Contrato")


def churn_by_internet(df: pd.DataFrame):
    """Churn por tipo de internet."""
    return churn_by_category(df, "InternetService", "Churn por Tipo de Internet")


def churn_by_payment(df: pd.DataFrame):
    """Churn por método de pagamento."""
    return churn_by_category(df, "PaymentMethod", "Churn por Metodo de Pagamento")


def services_count_chart(df: pd.DataFrame):
    """Churn por número de serviços contratados."""
    ct = pd.crosstab(df["num_services"], df["Churn"], normalize="index").reset_index()
    ct.columns = ["num_services", "Retido", "Cancelou"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=ct["num_services"], y=ct["Cancelou"], name="Cancelou", marker_color=COLOR_DANGER))
    fig.add_trace(go.Bar(x=ct["num_services"], y=ct["Retido"], name="Retido", marker_color=COLOR_PRIMARY))
    fig.update_layout(barmode="stack", title="Churn por Numero de Servicos Contratados")
    return apply_layout(fig)


def tenure_group_chart(df: pd.DataFrame):
    """Churn por faixa de permanência."""
    return churn_by_category(df, "tenure_group", "Churn por Faixa de Permanencia")
