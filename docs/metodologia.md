# Metodologia

## Pipeline de Machine Learning

### 1. Coleta de Dados
Dataset IBM Telco Customer Churn — 7.043 registros de clientes de telecomunicacoes com 21 features.

### 2. Preprocessing
- Conversao de TotalCharges para numerico (11 registros com valores em branco)
- Feature engineering: tenure_group, avg_monthly_charge, num_services
- Label Encoding para variaveis binarias
- One-Hot Encoding para variaveis categoricas multi-valor
- StandardScaler nas features numericas
- Train/test split estratificado 80/20 (seed 42)

### 3. Balanceamento
SMOTE (Synthetic Minority Over-sampling Technique) para tratar o desbalanceamento da variavel-alvo (~26.5% positivo).

### 4. Modelos
| Modelo | Hiperparametros |
|---|---|
| Logistic Regression | max_iter=1000, class_weight=balanced |
| Random Forest | n_estimators=200, class_weight=balanced |
| XGBoost | n_estimators=200, max_depth=5, lr=0.1, scale_pos_weight |

### 5. Avaliacao
- Metricas: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Validacao cruzada 5-fold
- Curva ROC comparativa
- Matriz de confusao

### 6. Explicabilidade
SHAP (TreeExplainer) para importancia de features e explicacao de predicoes individuais.

### 7. Simulador Financeiro
Modelo de impacto financeiro parametrizavel com CAC, receita media e desconto de retencao.

## Limitacoes
- Dados ficticios (IBM Sample) — nao representam cenario real especifico
- Snapshot unico sem dimensao temporal
- SMOTE pode introduzir ruido em alta dimensionalidade
- Explicabilidade SHAP e local e nao substitui validacao de negocio
