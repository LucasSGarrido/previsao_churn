# Dicionario de Dados — Telco Customer Churn

| Coluna | Tipo | Descricao |
|---|---|---|
| customerID | string | Identificador unico do cliente |
| gender | categorico | Genero (Male/Female) |
| SeniorCitizen | binario | Cliente idoso (0 = Nao, 1 = Sim) |
| Partner | binario | Possui parceiro |
| Dependents | binario | Possui dependentes |
| tenure | inteiro | Meses como cliente (0 a 72) |
| PhoneService | binario | Possui servico telefonico |
| MultipleLines | categorico | Multiplas linhas (Yes/No/No phone service) |
| InternetService | categorico | Tipo de internet (DSL/Fiber optic/No) |
| OnlineSecurity | categorico | Servico de seguranca online |
| OnlineBackup | categorico | Servico de backup online |
| DeviceProtection | categorico | Protecao de dispositivo |
| TechSupport | categorico | Suporte tecnico |
| StreamingTV | categorico | Streaming de TV |
| StreamingMovies | categorico | Streaming de filmes |
| Contract | categorico | Tipo de contrato (Month-to-month/One year/Two year) |
| PaperlessBilling | binario | Fatura digital |
| PaymentMethod | categorico | Metodo de pagamento |
| MonthlyCharges | float | Mensalidade em USD |
| TotalCharges | float | Total gasto em USD |
| Churn | binario | Cancelou o servico (Yes/No) — variavel-alvo |

## Feature Engineering

| Feature | Descricao |
|---|---|
| tenure_group | Faixa de permanencia (0-6m, 6-12m, 1-2a, 2-4a, 4-6a) |
| avg_monthly_charge | Total gasto / tenure (charge medio real) |
| num_services | Quantidade de servicos contratados |
