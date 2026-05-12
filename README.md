# Previsao de Churn — Dashboard de Machine Learning

Dashboard interativo de Machine Learning para previsao de cancelamento de clientes de telecomunicacoes.

https://previsaochurn-jak75owqgseod6htnfnrre.streamlit.app

## Sobre o Projeto

Este projeto analisa dados de 7.043 clientes de uma operadora de telecomunicacoes para identificar padroes de cancelamento (churn) e prever quais clientes tem maior probabilidade de sair.

## Stack

- Python 3.10+
- Streamlit (Dashboard interativo)
- Scikit-learn (Modelos de classificacao)
- XGBoost (Gradient Boosting)
- SHAP (Explicabilidade)
- Plotly (Visualizacoes)
- imbalanced-learn (SMOTE)

## Funcionalidades

- **Visao Geral:** KPIs e metricas de churn com insights executivos
- **Analise Exploratoria:** Graficos de distribuicao, correlacao e segmentacao
- **Modelos:** Comparativo entre Logistic Regression, Random Forest e XGBoost
- **Explicabilidade:** SHAP values com importancia global e analise individual
- **Simulador de Impacto:** Calculadora financeira de retencao com ROI
- **Predicao Individual:** Formulario para prever churn de um cliente especifico
- **Dados e Metodologia:** Documentacao, dicionario de dados e download

## Como Executar

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Dataset

IBM Telco Customer Churn — 7.043 registros, 21 features (demograficas, servicos e conta).

## Estrutura

```
previsao_churn/
  app.py
  requirements.txt
  data/telco_churn.csv
  src/
    data_loader.py
    preprocessing.py
    eda.py
    models.py
    explainability.py
    simulator.py
    styles.css
  docs/
    dicionario_dados.md
    metodologia.md
```

## Limitacoes

- Dados ficticios (IBM Sample)
- Snapshot unico sem dimensao temporal
- SMOTE pode introduzir ruido em alta dimensionalidade
