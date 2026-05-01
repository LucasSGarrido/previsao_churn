"""Dashboard de Previsao de Churn — Streamlit App."""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

# Diretório base do projeto
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.data_loader import load_telco_data
from src.preprocessing import run_preprocessing
from src.eda import (
    churn_rate_chart,
    tenure_distribution,
    monthly_charges_box,
    correlation_heatmap,
    churn_by_contract,
    churn_by_internet,
    churn_by_payment,
    services_count_chart,
    tenure_group_chart,
)
from src.models import train_models, comparison_table, roc_curve_chart, confusion_matrix_chart
from src.explainability import compute_shap_values, shap_importance_chart, feature_importance_native, shap_waterfall_data
from src.simulator import financial_impact, risk_distribution_chart, financial_waterfall, probability_histogram, classify_risk

# ============================================================
# CONFIGURACAO
# ============================================================
st.set_page_config(
    page_title="Previsao de Churn",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# CSS
with open(BASE_DIR / "src" / "styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ============================================================
# CACHE: Carregar e processar dados
# ============================================================
@st.cache_data(show_spinner=False)
def get_data():
    df = load_telco_data(BASE_DIR)
    return df

@st.cache_resource(show_spinner=False)
def get_pipeline(_df):
    return run_preprocessing(_df)

@st.cache_resource(show_spinner=False)
def get_models(_X_train, _y_train, _X_test, _y_test):
    return train_models(_X_train, _y_train, _X_test, _y_test)

@st.cache_resource(show_spinner=False)
def get_shap(_model, _X_test):
    return compute_shap_values(_model, _X_test)


# ============================================================
# DADOS
# ============================================================
with st.spinner("Carregando dados e treinando modelos..."):
    df_raw = get_data()
    pipeline = get_pipeline(df_raw)
    df_clean = pipeline["df_clean"]
    df_feat = pipeline["df_feat"]
    results = get_models(
        pipeline["X_train"], pipeline["y_train"],
        pipeline["X_test"], pipeline["y_test"],
    )

# Melhor modelo
best_name = max(results, key=lambda k: results[k]["roc_auc"])
best = results[best_name]


# ============================================================
# KPI HELPER
# ============================================================
def kpi_card(label: str, value: str, danger: bool = False):
    cls = "kpi-card kpi-danger" if danger else "kpi-card"
    return f"""
    <div class="{cls}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
    </div>
    """


# ============================================================
# HEADER
# ============================================================
st.markdown("## Previsao de Churn")
st.markdown("Analise preditiva de cancelamento de clientes — Telco Customer Dataset")
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ============================================================
# ABAS
# ============================================================
tabs = st.tabs([
    "Visao Geral",
    "Analise Exploratoria",
    "Modelos",
    "Explicabilidade",
    "Simulador de Impacto",
    "Predicao Individual",
    "Dados e Metodologia",
])

# ==============================================================
# ABA 1: VISAO GERAL
# ==============================================================
with tabs[0]:
    churn_count = df_clean["Churn"].value_counts()
    total = len(df_clean)
    churn_yes = churn_count.get("Yes", 0)
    churn_rate = churn_yes / total * 100
    avg_tenure = df_clean["tenure"].mean()
    avg_monthly = df_clean["MonthlyCharges"].mean()

    cols = st.columns(4)
    with cols[0]:
        st.markdown(kpi_card("Total de Clientes", f"{total:,}"), unsafe_allow_html=True)
    with cols[1]:
        st.markdown(kpi_card("Taxa de Churn", f"{churn_rate:.1f}%", danger=True), unsafe_allow_html=True)
    with cols[2]:
        st.markdown(kpi_card("Permanencia Media", f"{avg_tenure:.0f} meses"), unsafe_allow_html=True)
    with cols[3]:
        st.markdown(kpi_card("Mensalidade Media", f"${avg_monthly:.2f}"), unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(churn_rate_chart(df_clean), use_container_width=True)
    with col2:
        st.plotly_chart(churn_by_contract(df_clean), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(tenure_distribution(df_clean), use_container_width=True)
    with col4:
        st.plotly_chart(monthly_charges_box(df_clean), use_container_width=True)

    # Insights executivos
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### Insights Principais")

    mtm = df_clean[df_clean["Contract"] == "Month-to-month"]
    mtm_churn = mtm["Churn"].value_counts(normalize=True).get("Yes", 0) * 100
    fiber = df_clean[df_clean["InternetService"] == "Fiber optic"]
    fiber_churn = fiber["Churn"].value_counts(normalize=True).get("Yes", 0) * 100

    insights = [
        f"Clientes com contrato mensal tem taxa de churn de {mtm_churn:.1f}%, significativamente acima da media.",
        f"Clientes de Fibra Optica cancelam em {fiber_churn:.1f}% dos casos — sugere problema de satisfacao.",
        f"A permanencia media dos que cancelam e de {df_clean[df_clean['Churn']=='Yes']['tenure'].mean():.0f} meses vs {df_clean[df_clean['Churn']=='No']['tenure'].mean():.0f} dos retidos.",
        f"Melhor modelo preditivo: {best_name} com ROC-AUC de {best['roc_auc']:.3f}.",
    ]
    for ins in insights:
        st.markdown(f"- {ins}")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    with st.expander("Como interpretar estes graficos?"):
        st.markdown("""
**Taxa de Churn Geral (Donut):** Mostra a proporcao de clientes que cancelaram vs. os que permaneceram. A fatia vermelha representa a perda de base.

**Churn por Tipo de Contrato:** Compara a taxa de cancelamento entre contratos mensais, anuais e bianuais. Contratos curtos tendem a ter churn maior.

**Distribuicao de Permanencia:** Histograma que mostra quantos clientes cancelam em cada faixa de tempo. Picos no inicio indicam churn precoce.

**Mensalidade por Status:** Boxplot comparando o valor pago por quem cancelou vs. quem ficou. Valores mais altos entre churners sugerem insatisfacao com preco.
        """)

# ==============================================================
# ABA 2: ANALISE EXPLORATORIA
# ==============================================================
with tabs[1]:
    st.markdown("### Analise Exploratoria de Dados")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(churn_by_internet(df_clean), use_container_width=True)
    with col2:
        st.plotly_chart(churn_by_payment(df_clean), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(services_count_chart(df_feat), use_container_width=True)
    with col4:
        st.plotly_chart(tenure_group_chart(df_feat), use_container_width=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### Correlacoes com Churn")
    st.plotly_chart(correlation_heatmap(pipeline["df_encoded"]), use_container_width=True)

    # Perfil do churner
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### Perfil do Cliente que Cancela")
    churners = df_clean[df_clean["Churn"] == "Yes"]
    retained = df_clean[df_clean["Churn"] == "No"]
    profile = pd.DataFrame({
        "Metrica": ["Permanencia (meses)", "Mensalidade ($)", "Total Gasto ($)"],
        "Cancelou": [churners["tenure"].median(), churners["MonthlyCharges"].median(), churners["TotalCharges"].median()],
        "Retido": [retained["tenure"].median(), retained["MonthlyCharges"].median(), retained["TotalCharges"].median()],
    })
    st.dataframe(profile, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    with st.expander("Como interpretar esta analise?"):
        st.markdown("""
**Churn por Internet/Pagamento:** Identifica quais servicos e formas de pagamento estao mais associados ao cancelamento. Fibra optica e cheque eletronico costumam liderar.

**Churn por Numero de Servicos:** Clientes com poucos servicos contratados tendem a ter menos vinculo e cancelam mais facilmente.

**Correlacoes com Churn:** Barras positivas indicam features que **aumentam** a chance de churn; negativas indicam features que **protegem** contra o cancelamento.

**Perfil do Churner:** A tabela compara a mediana de cada metrica entre quem cancelou e quem ficou, revelando o perfil tipico de risco.
        """)

# ==============================================================
# ABA 3: MODELOS
# ==============================================================
with tabs[2]:
    st.markdown("### Comparativo de Modelos")
    comp = comparison_table(results)
    st.dataframe(
        comp.style.format({
            "Accuracy": "{:.3f}", "Precision": "{:.3f}", "Recall": "{:.3f}",
            "F1-Score": "{:.3f}", "ROC-AUC": "{:.3f}", "CV AUC (mean)": "{:.3f}",
        }).highlight_max(subset=["ROC-AUC", "F1-Score", "Recall"], color="rgba(0,245,160,0.15)"),
        use_container_width=True, hide_index=True,
    )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.plotly_chart(roc_curve_chart(results, pipeline["y_test"]), use_container_width=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### Matrizes de Confusao")
    cm_cols = st.columns(3)
    for i, (name, r) in enumerate(results.items()):
        with cm_cols[i]:
            st.plotly_chart(confusion_matrix_chart(r["confusion"], name), use_container_width=True)

    # Interpretação
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    with st.expander("Como interpretar estas metricas?"):
        st.markdown("""
| Metrica | Significado para Churn |
|---|---|
| **Precision** | De todos que o modelo disse que iam cancelar, quantos realmente cancelaram. |
| **Recall** | De todos que realmente cancelaram, quantos o modelo conseguiu identificar. |
| **F1-Score** | Harmonia entre Precision e Recall — equilibra falsos positivos e negativos. |
| **ROC-AUC** | Capacidade geral do modelo de separar churners de retidos. Quanto mais perto de 1, melhor. |

Para churn, **Recall** e tipicamente mais importante que Precision: e melhor abordar um cliente que nao ia cancelar do que deixar passar um que ia.

**Curva ROC:** Mostra o trade-off entre detectar churners (eixo Y) e gerar falsos alarmes (eixo X). Quanto mais a curva se aproxima do canto superior esquerdo, melhor.

**Matriz de Confusao:** Cada celula mostra quantos clientes foram classificados corretamente (diagonal) ou incorretamente (fora da diagonal).
        """)

# ==============================================================
# ABA 4: EXPLICABILIDADE
# ==============================================================
with tabs[3]:
    st.markdown("### Explicabilidade do Modelo")
    st.markdown(f"Usando: **{best_name}**")

    with st.spinner("Calculando SHAP values..."):
        explainer, shap_values = get_shap(best["model"], pipeline["X_test"])

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            shap_importance_chart(shap_values, pipeline["feature_names"]),
            use_container_width=True,
        )
    with col2:
        if hasattr(best["model"], "feature_importances_"):
            st.plotly_chart(
                feature_importance_native(best["model"], pipeline["feature_names"]),
                use_container_width=True,
            )
        else:
            st.info("Importancia nativa disponivel apenas para modelos de arvore.")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### Analise Individual (Waterfall)")
    
    # Criar lista de IDs reais para busca amigável
    test_indices = pipeline["X_test"].index
    customer_ids = df_raw.loc[test_indices, "customerID"].tolist()
    
    col_sel, _ = st.columns([2, 2])
    with col_sel:
        selected_id = st.selectbox(
            "Busque por ID do Cliente:",
            options=customer_ids,
            index=0,
            help="Selecione um cliente para ver os motivos especificos por tras do seu score de risco."
        )
    
    # Encontrar a posicao do ID selecionado para indexar os shap_values
    idx_pos = customer_ids.index(selected_id)
    
    st.plotly_chart(shap_waterfall_data(shap_values, pipeline["X_test"], idx_pos), use_container_width=True)

    # Explicação didática
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    with st.expander("Como interpretar SHAP?"):
        st.markdown("""
        **SHAP (SHapley Additive exPlanations)** atribui a cada feature uma contribuicao para a predicao individual.

        - **Barras vermelhas** empurram a predicao para "vai cancelar"
        - **Barras verdes** empurram a predicao para "vai ficar"
        - O tamanho da barra indica a **forca** da influencia

        Isso permite explicar para stakeholders nao-tecnicos **por que** o modelo tomou cada decisao.
        """)

# ==============================================================
# ABA 5: SIMULADOR DE IMPACTO
# ==============================================================
with tabs[4]:
    st.markdown("### Simulador de Impacto Financeiro")

    col1, col2, col3 = st.columns(3)
    with col1:
        avg_rev = st.number_input(
            "Receita Mensal Media (USD)",
            value=64.76, step=5.0,
            help="Valor medio que cada cliente paga por mes. O dataset tem media de ~USD 64.76.",
        )
    with col2:
        lifetime = st.number_input(
            "Lifetime Medio (meses)",
            value=12, step=1,
            help="Tempo medio que um cliente permanece ativo apos ser identificado como risco. Quanto maior, mais receita em jogo.",
        )
    with col3:
        cac = st.number_input(
            "CAC (USD)",
            value=300.0, step=25.0,
            help="Custo de Aquisicao de Cliente: quanto custa conquistar um novo cliente (marketing, vendas, onboarding).",
        )

    discount = st.slider(
        "Desconto de Retencao",
        min_value=5, max_value=50, value=15,
        format="%d%%",
        help="Percentual de desconto oferecido ao cliente de risco para incentiva-lo a permanecer.",
    ) / 100

    impact = financial_impact(best["y_proba"], avg_rev, lifetime, cac, discount)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    cols = st.columns(4)
    with cols[0]:
        st.markdown(kpi_card("Clientes Alto Risco", f"{impact['high_risk_customers']:,}", danger=True), unsafe_allow_html=True)
    with cols[1]:
        st.markdown(kpi_card("Receita em Risco", f"${impact['revenue_at_risk']:,.0f}", danger=True), unsafe_allow_html=True)
    with cols[2]:
        st.markdown(kpi_card("Custo de Retencao", f"${impact['retention_cost']:,.0f}"), unsafe_allow_html=True)
    with cols[3]:
        st.markdown(kpi_card("Economia Liquida", f"${impact['net_savings']:,.0f}"), unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(risk_distribution_chart(best["y_proba"]), use_container_width=True)
    with col2:
        st.plotly_chart(financial_waterfall(impact), use_container_width=True)

    st.plotly_chart(probability_histogram(best["y_proba"]), use_container_width=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### ROI da Retencao")
    roi_val = f"{impact['roi_retention']:.2f}"
    repl_val = f"{impact['replacement_cost']:,.0f}"
    ratio_val = f"{impact['replacement_cost'] / impact['retention_cost']:.1f}"
    st.markdown(f"- **Cada USD 1 investido** em retencao gera **USD {roi_val}** de retorno.")
    st.markdown(f"- **Custo de reposicao** (adquirir novos clientes): **USD {repl_val}**")
    st.markdown(f"- A retencao e **{ratio_val}x mais barata** que adquirir novos clientes.")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    with st.expander("Como interpretar o simulador?"):
        st.markdown("""
**Distribuicao de Risco:** Classifica cada cliente do conjunto de teste em faixas de probabilidade de churn (Baixo, Moderado, Alto, Critico).

**Impacto Financeiro (Waterfall):** Mostra o fluxo: receita em risco caso os clientes cancelem, quanto custaria repor esses clientes, quanto custa a campanha de retencao e a economia liquida final.

**Histograma de Probabilidades:** Mostra a distribuicao de scores do modelo. A linha vermelha (threshold 50%) separa quem o modelo classifica como churner.

**ROI da Retencao:** Compara o custo de reter um cliente existente vs. adquirir um novo. Em geral, reter e significativamente mais barato.
        """)

# ==============================================================
# ABA 6: PREDICAO INDIVIDUAL
# ==============================================================
with tabs[5]:
    st.markdown("### Predicao Individual de Churn")
    st.markdown("Preencha os dados de um cliente para prever o risco de cancelamento.")

    col1, col2, col3 = st.columns(3)
    with col1:
        tenure_input = st.number_input("Permanencia (meses)", 0, 72, 12)
        monthly_input = st.number_input("Mensalidade ($)", 18.0, 120.0, 70.0, step=5.0)
        contract_input = st.selectbox("Contrato", ["Month-to-month", "One year", "Two year"])

    with col2:
        internet_input = st.selectbox("Internet", ["DSL", "Fiber optic", "No"])
        phone_input = st.selectbox("Telefone", ["Yes", "No"])
        senior_input = st.selectbox("Senior Citizen", ["No", "Yes"])

    with col3:
        security_input = st.selectbox("Seguranca Online", ["Yes", "No", "No internet service"])
        support_input = st.selectbox("Suporte Tecnico", ["Yes", "No", "No internet service"])
        payment_input = st.selectbox("Pagamento", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)",
        ])

    if st.button("Prever Churn"):
        # Construir input no formato do modelo
        input_data = {
            "tenure": tenure_input,
            "MonthlyCharges": monthly_input,
            "TotalCharges": tenure_input * monthly_input,
        }

        # Preencher as colunas do modelo com valores padrão
        for col_name in pipeline["feature_names"]:
            if col_name not in input_data:
                input_data[col_name] = 0

        # Mapear inputs para features
        if senior_input == "Yes":
            input_data["SeniorCitizen"] = 1
        if phone_input == "Yes":
            if "PhoneService" in input_data:
                input_data["PhoneService"] = 1

        # Contract dummies
        for opt in ["One year", "Two year"]:
            key = f"Contract_{opt}"
            if key in input_data:
                input_data[key] = 1 if contract_input == opt else 0

        # Internet dummies
        for opt in ["Fiber optic", "No"]:
            key = f"InternetService_{opt}"
            if key in input_data:
                input_data[key] = 1 if internet_input == opt else 0

        # Payment dummies
        for opt in ["Credit card (automatic)", "Electronic check", "Mailed check"]:
            key = f"PaymentMethod_{opt}"
            if key in input_data:
                input_data[key] = 1 if payment_input == opt else 0

        # Feature engineering
        input_data["avg_monthly_charge"] = monthly_input
        input_data["num_services"] = sum([
            phone_input == "Yes",
            internet_input != "No",
            security_input == "Yes",
            support_input == "Yes",
        ])

        # Criar DataFrame com as mesmas colunas
        X_input = pd.DataFrame([input_data])[pipeline["feature_names"]]

        # Aplicar scaling nas mesmas colunas
        num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "avg_monthly_charge", "num_services"]
        num_cols = [c for c in num_cols if c in X_input.columns]
        X_input[num_cols] = pipeline["scaler"].transform(X_input[num_cols])

        # Prever
        prob = best["model"].predict_proba(X_input)[0][1]
        risk_label = classify_risk(np.array([prob]))[0]

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        result_cols = st.columns(3)
        with result_cols[0]:
            st.markdown(kpi_card("Probabilidade de Churn", f"{prob*100:.1f}%", danger=prob > 0.5), unsafe_allow_html=True)
        with result_cols[1]:
            st.markdown(kpi_card("Nivel de Risco", risk_label, danger=risk_label in ["Alto", "Critico"]), unsafe_allow_html=True)
        with result_cols[2]:
            verdict = "Reter urgente" if prob > 0.7 else "Monitorar" if prob > 0.3 else "Estavel"
            st.markdown(kpi_card("Acao Recomendada", verdict, danger=prob > 0.5), unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    with st.expander("Como funciona a predicao individual?"):
        st.markdown("""
**O formulario acima** recebe os dados de um unico cliente e os transforma nas mesmas features usadas no treinamento do modelo.

- **Probabilidade de Churn:** O score do modelo (0-100%). Acima de 50% o cliente e considerado provavel churner.
- **Nivel de Risco:** Classificacao em faixas — Baixo (0-30%), Moderado (30-50%), Alto (50-70%) e Critico (70-100%).
- **Acao Recomendada:** Sugestao automatica baseada no risco — "Estavel" para baixo risco, "Monitorar" para medio e "Reter urgente" para alto/critico.

Esta funcionalidade simula como o modelo seria usado em producao: recebendo dados em tempo real e gerando alertas de retencao.
        """)

# ==============================================================
# ABA 7: DADOS E METODOLOGIA
# ==============================================================
with tabs[6]:
    st.markdown("### Dados e Metodologia")

    with st.expander("Sobre o Dataset"):
        st.markdown("""
        **IBM Telco Customer Churn**
        - **Fonte:** IBM Sample Data Sets
        - **Registros:** 7.043 clientes
        - **Features:** 21 (demograficas, servicos e conta)
        - **Variavel-alvo:** Churn (Yes/No) — ~26.5% positivo (desbalanceado)
        """)

    with st.expander("Pipeline de Machine Learning"):
        st.markdown(f"""
        1. **Limpeza:** Conversao de TotalCharges, tratamento de 11 registros com tenure=0.
        2. **Feature Engineering:** tenure_group, avg_monthly_charge, num_services.
        3. **Encoding:** Label encoding (binarias) + One-Hot (multi-valor).
        4. **Balanceamento:** SMOTE (Synthetic Minority Over-sampling).
        5. **Modelos:** Logistic Regression, Random Forest (200 arvores), XGBoost (200 estimators).
        6. **Validacao:** 5-fold Cross Validation, split estratificado 80/20.
        7. **Explicabilidade:** SHAP (TreeExplainer).
        """)

    with st.expander("Limitacoes"):
        st.markdown("""
        - Os dados sao de uma empresa ficticia (IBM) — nao representam um cenario real especifico.
        - O dataset nao possui dimensao temporal (snapshot unico), impedindo analise de sobrevivencia.
        - SMOTE pode introduzir ruido em dados de alta dimensionalidade.
        - Explicabilidade SHAP e local — nao substitui validacao de negocio.
        """)

    with st.expander("Dicionario de Dados"):
        dict_data = pd.DataFrame({
            "Coluna": [
                "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
                "tenure", "PhoneService", "MultipleLines", "InternetService",
                "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
                "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
                "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
            ],
            "Descricao": [
                "ID unico do cliente", "Genero (Male/Female)", "Idoso (0/1)",
                "Tem parceiro", "Tem dependentes", "Meses como cliente",
                "Servico telefonico", "Multiplas linhas", "Tipo de internet",
                "Seguranca online", "Backup online", "Protecao de dispositivo",
                "Suporte tecnico", "Streaming TV", "Streaming Filmes",
                "Tipo de contrato", "Fatura digital", "Metodo de pagamento",
                "Mensalidade ($)", "Total gasto ($)", "Cancelou (Yes/No)",
            ],
        })
        st.dataframe(dict_data, use_container_width=True, hide_index=True)

    # Download
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### Exportar Dados")
    col1, col2 = st.columns(2)
    with col1:
        csv_clean = df_clean.to_csv(index=False).encode("utf-8")
        st.download_button("Download — Dataset Limpo (CSV)", csv_clean, "telco_churn_clean.csv", "text/csv")
    with col2:
        pred_df = pipeline["X_test"].copy()
        pred_df["Churn_Real"] = pipeline["y_test"].values
        pred_df["Churn_Previsto"] = best["y_pred"]
        pred_df["Probabilidade"] = best["y_proba"]
        csv_pred = pred_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download — Predicoes (CSV)", csv_pred, "telco_churn_predictions.csv", "text/csv")
