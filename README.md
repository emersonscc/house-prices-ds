# House Prices — Projeto de Data Science

Previsão de preços de imóveis residenciais com base no dataset **House Prices: Advanced Regression Techniques** (Kaggle), referente ao município de Ames, Iowa. Este projeto cobre todo o ciclo de Data Science: análise exploratória, pré-processamento, modelagem comparativa e dashboard interativo.

**Disciplina:** Data Science
**Instituição:** Universidade de Passo Fundo (UPF) — Ciência da Computação
**Período:** 2026/1
**Autores:** Emerson Rosado Scalcon e Leandro Trevisan Martins

---

## 1. Sobre o projeto

O objetivo é prever o valor de venda (`SalePrice`) de imóveis residenciais a partir de 79 atributos descritivos (área, qualidade, localização, ano de construção, garagem, porão, entre outros). Foram treinados quatro modelos de regressão, comparados sob métricas de RMSE, MAE e R², e o melhor modelo foi integrado a um dashboard interativo construído com Streamlit.

### Principais resultados

| Modelo | RMSE Validação | R² Validação |
|---|---|---|
| Regressão Linear | $21.307 | 0.8996 |
| Ridge (L2) | $21.333 | 0.8994 |
| Random Forest | $24.613 | 0.8615 |
| **Gradient Boosting** | **$20.715** | **0.8961** |

O modelo vencedor, **Gradient Boosting**, prevê preços com erro mediano de aproximadamente $14.918 (MAE) em imóveis cujo preço médio é ~$180.000, o que representa um erro relativo de cerca de 8%.

---

## 2. Estrutura do repositório

```
house-prices-ds/
├── README.md                 ← este arquivo
├── requirements.txt          ← dependências Python
├── .gitignore
│
├── data/                     ← (não versionado) baixar do Kaggle
│   ├── train.csv
│   ├── test.csv
│   └── data_description.txt
│
├── notebooks/
│   ├── 01_eda.ipynb          ← análise exploratória
│   ├── 02_preprocessing.ipynb ← limpeza e feature engineering
│   └── 03_modeling.ipynb     ← treinamento e avaliação
│
├── dashboard/
│   └── app.py                ← dashboard Streamlit
│
└── report/
    ├── relatorio.pdf         ← relatório técnico final
    └── fig*.png              ← figuras geradas pelos notebooks
```

---

## 3. Como rodar o projeto do zero

### 3.1. Pré-requisitos

- **Python 3.10 ou superior** ([download](https://python.org/downloads))
  Durante a instalação, marque a opção **"Add Python to PATH"**.
- **Git** ([download](https://git-scm.com/downloads))
- **VS Code** ([download](https://code.visualstudio.com/)) com as extensões **Python** e **Jupyter** instaladas.
- Conta no **Kaggle** ([criar](https://www.kaggle.com)) para baixar o dataset.

### 3.2. Clonar o repositório

```bash
git clone https://github.com/SEU_USUARIO/house-prices-ds.git
cd house-prices-ds
```

### 3.3. Instalar as dependências

No terminal, dentro da pasta do projeto:

```bash
py -m pip install -r requirements.txt
```

Caso não tenha o `requirements.txt`, instale manualmente:

```bash
py -m pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib jupyter ipykernel streamlit
```

### 3.4. Baixar o dataset

1. Acesse [kaggle.com/competitions/house-prices-advanced-regression-techniques/data](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)
2. Faça login e clique em **Download All**
3. Extraia o arquivo `.zip`
4. Coloque `train.csv`, `test.csv` e `data_description.txt` dentro da pasta `data/` do projeto

> Os dados não são versionados no repositório por questões de tamanho e de licença do Kaggle. Cada usuário deve baixá-los individualmente.

### 3.5. Executar os notebooks (em ordem)

Abra o VS Code na pasta do projeto:

```bash
code .
```

Abra os notebooks na sequência abaixo e execute cada célula com `Shift+Enter`:

1. **`notebooks/01_eda.ipynb`** — Análise exploratória. Gera figuras 01-11 em `report/` e o arquivo `data/train_eda.csv`.
2. **`notebooks/02_preprocessing.ipynb`** — Limpeza, encoding e feature engineering. Gera `X_train.csv`, `X_val.csv`, `y_train.csv`, `y_val.csv`, `scaler.pkl` e `feature_names.csv` em `data/`.
3. **`notebooks/03_modeling.ipynb`** — Treinamento e avaliação dos 4 modelos. Gera `melhor_modelo.pkl`, `resultados_modelos.csv` e figuras 14-16 em `report/`.

> Na primeira execução, o VS Code pode pedir para selecionar um kernel. Escolha a versão do Python instalada localmente (ex.: `Python 3.12.x`).

### 3.6. Rodar o dashboard

Após executar os três notebooks:

```bash
py -m streamlit run dashboard/app.py
```

O dashboard abrirá automaticamente no navegador em `http://localhost:8501`. Para encerrar, pressione `Ctrl+C` no terminal.

---

## 4. Como usar o dashboard

O dashboard tem quatro abas, acessíveis pelo menu lateral:

- **📊 Visão Geral** — KPIs do dataset, distribuição de preços e visão das etapas do projeto.
- **🔍 Análise Exploratória** — exploração interativa com filtros por bairro e faixa de preço, scatter plots com linha de tendência, e tabela com o status das 5 hipóteses testadas.
- **🤖 Modelos** — comparação visual e tabular entre os 4 modelos treinados, com destaque do melhor.
- **💰 Simulador de Preço** — ajuste 8 características do imóvel (qualidade, área, garagem, banheiros, ano de construção, etc.) e visualize o preço estimado pelo modelo em tempo real, com comparação contra o histograma do mercado.

---

## 5. Metodologia resumida

### 5.1. Análise Exploratória (Notebook 01)

Foram formuladas 5 hipóteses e testadas com visualizações e estatísticas descritivas:

| # | Hipótese | Resultado |
|---|---|---|
| H1 | Área habitável é o fator mais correlacionado com o preço | Parcialmente confirmada (OverallQual supera GrLivArea) |
| H2 | OverallQual tem impacto não-linear no preço | Confirmada — relação exponencial entre qualidade 8 e 10 |
| H3 | Bairro explica variações além do tamanho | Confirmada — 2,3x de variação entre bairros (área similar) |
| H4 | Casas sem garagem têm preço sistematicamente menor | Confirmada — diferença de 67% na mediana |
| H5 | Imóveis mais novos tendem a ser mais caros | Confirmada — correlação positiva moderada (r = 0,52) |

### 5.2. Pré-processamento (Notebook 02)

- Remoção de 2 outliers (área > 4.000 pés² com preço < $200k) identificados na H1.
- Exclusão de colunas com mais de 50% de valores ausentes (`PoolQC`, `MiscFeature`, `Alley`, `Fence`).
- Imputação contextual: `None` para variáveis categóricas onde NaN indica ausência do atributo; 0 para variáveis numéricas correspondentes.
- Transformação logarítmica da variável-alvo (`log1p(SalePrice)`) para corrigir assimetria.
- Feature engineering: criação de 7 novas variáveis, incluindo `TotalSF` (área total = porão + acima do solo), `TotalBaths`, `HouseAge` e `YearsSinceRemod`. `TotalSF` atingiu correlação de 0,82 com SalePrice, superando todas as variáveis originais.
- Encoding ordinal manual para variáveis de qualidade (preservando ordem semântica) e One-Hot Encoding para variáveis nominais.
- Normalização (StandardScaler) com `fit` aplicado apenas no conjunto de treino para evitar data leakage.

### 5.3. Modelagem (Notebook 03)

Quatro modelos foram treinados e comparados:

- **Regressão Linear** — baseline interpretável.
- **Ridge (L2)** — regressão linear com regularização para verificar efeito da multicolinearidade.
- **Random Forest** — ensemble robusto, com `n_estimators=300` e `max_depth=15`.
- **Gradient Boosting** — modelo escolhido, com `n_estimators=500`, `learning_rate=0.05` e `subsample=0.8`.

A avaliação foi feita em conjunto de validação separado (20% do treino), com validação cruzada de 5 folds adicional para verificar estabilidade.

---

## 6. Tecnologias utilizadas

- **Python 3.14**
- **Pandas** e **NumPy** — manipulação de dados
- **Scikit-learn** — modelos de Machine Learning
- **Matplotlib** e **Seaborn** — visualizações
- **Streamlit** — dashboard interativo
- **Joblib** — serialização de modelos
- **Jupyter** — notebooks de desenvolvimento

---

## 7. Solução de problemas

**Erro `ModuleNotFoundError: No module named 'pandas'`**
O kernel do Jupyter está apontando para um Python diferente do que recebeu as bibliotecas. No notebook, clique no kernel atual (canto superior direito) e selecione o Python correto. Alternativamente, instale as bibliotecas no kernel ativo: `py -m pip install pandas`.

**Erro `streamlit: O termo não é reconhecido`**
O executável do Streamlit não está no PATH. Rode usando `py -m streamlit run dashboard/app.py`.

**Aviso `LF will be replaced by CRLF`**
É apenas informativo. Para silenciar: `git config --global core.autocrlf true`.

**Dashboard reclama de arquivo não encontrado**
Confirme que os três notebooks (01, 02 e 03) foram executados em ordem e que os arquivos `.pkl` e `.csv` correspondentes estão dentro de `data/`.

---

## 8. Referências

- Cock, D. D. (2011). Ames, Iowa: Alternative to the Boston Housing Data. *Journal of Statistics Education*, 19(3).
- Kaggle Competition: [House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- Documentação Scikit-learn: [scikit-learn.org](https://scikit-learn.org/stable/)
- Documentação Streamlit: [docs.streamlit.io](https://docs.streamlit.io)
