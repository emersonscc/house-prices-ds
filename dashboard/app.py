"""
House Prices — Dashboard Interativo
Projeto Final de Data Science | UPF 2026

Execução: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

# =========================================================
# CONFIGURAÇÃO DA PÁGINA
# =========================================================
st.set_page_config(
    page_title="House Prices — Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

sns.set_theme(style='whitegrid', palette='muted')

# =========================================================
# CARREGAMENTO DE DADOS (com cache para performance)
# =========================================================
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

@st.cache_data
def carregar_dados_brutos():
    return pd.read_csv(DATA_DIR / "train.csv")

@st.cache_data
def carregar_resultados():
    return pd.read_csv(DATA_DIR / "resultados_modelos.csv")

@st.cache_data
def carregar_features():
    return pd.read_csv(DATA_DIR / "feature_names.csv").iloc[:, 0].tolist()

@st.cache_resource
def carregar_modelo():
    return joblib.load(DATA_DIR / "melhor_modelo.pkl")

@st.cache_resource
def carregar_scaler():
    return joblib.load(DATA_DIR / "scaler.pkl")

# Carregamento principal
try:
    df = carregar_dados_brutos()
    resultados = carregar_resultados()
    features = carregar_features()
    modelo = carregar_modelo()
    scaler = carregar_scaler()
    dados_carregados = True
except FileNotFoundError as e:
    st.error(f"Arquivo não encontrado: {e}. Execute os notebooks 01, 02 e 03 antes.")
    dados_carregados = False
    st.stop()

# =========================================================
# SIDEBAR — NAVEGAÇÃO
# =========================================================
st.sidebar.title("🏠 House Prices")
st.sidebar.markdown("Projeto Final — Data Science")
st.sidebar.markdown("---")

aba = st.sidebar.radio(
    "Navegação",
    ["📊 Visão Geral", "🔍 Análise Exploratória", "🤖 Modelos", "💰 Simulador de Preço"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset:** Ames, Iowa")
st.sidebar.markdown(f"**Total de imóveis:** {len(df):,}")
st.sidebar.markdown(f"**Variáveis:** {df.shape[1]}")
st.sidebar.markdown("---")
st.sidebar.caption("UPF · Ciências Contábeis · 2026")

# =========================================================
# ABA 1 — VISÃO GERAL
# =========================================================
if aba == "📊 Visão Geral":
    st.title("Visão Geral do Projeto")
    st.markdown(
        "Este dashboard apresenta os resultados do projeto de Data Science sobre "
        "previsão de preços de imóveis com base no dataset **House Prices: Advanced "
        "Regression Techniques** (Kaggle), referente ao município de Ames, Iowa."
    )

    # KPIs principais
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Imóveis", f"{len(df):,}")
    col2.metric("Preço médio", f"${df['SalePrice'].mean():,.0f}")
    col3.metric("Preço mediano", f"${df['SalePrice'].median():,.0f}")
    col4.metric("Preço máximo", f"${df['SalePrice'].max():,.0f}")

    st.markdown("---")

    # Distribuição de preços
    col_esq, col_dir = st.columns([2, 1])

    with col_esq:
        st.subheader("Distribuição de preços")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(df['SalePrice'], bins=50, color='steelblue', edgecolor='white')
        ax.axvline(df['SalePrice'].median(), color='red', linestyle='--', linewidth=1.5,
                   label=f"Mediana: ${df['SalePrice'].median():,.0f}")
        ax.set_xlabel('Preço de venda (USD)')
        ax.set_ylabel('Frequência')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}k'))
        ax.legend()
        st.pyplot(fig)

    with col_dir:
        st.subheader("Estatísticas")
        stats = df['SalePrice'].describe()[['min', '25%', '50%', '75%', 'max', 'std']]
        stats_formatado = pd.DataFrame({
            'Métrica': ['Mínimo', '25%', 'Mediana', '75%', 'Máximo', 'Desvio padrão'],
            'Valor': [f"${v:,.0f}" for v in stats.values]
        })
        st.dataframe(stats_formatado, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader("Etapas do projeto")
    col1, col2, col3 = st.columns(3)
    col1.info("**1. EDA**\n\nAnálise de 5 hipóteses, correlações e padrões nos dados.")
    col2.info("**2. Pré-processamento**\n\nLimpeza, encoding e feature engineering (7 novas features).")
    col3.info("**3. Modelagem**\n\nComparação entre Regressão Linear, Ridge, Random Forest e Gradient Boosting.")

# =========================================================
# ABA 2 — ANÁLISE EXPLORATÓRIA
# =========================================================
elif aba == "🔍 Análise Exploratória":
    st.title("Análise Exploratória dos Dados")
    st.markdown("Explore as relações entre as variáveis e o preço de venda.")

    # Filtros
    st.sidebar.markdown("### Filtros da EDA")
    bairros_disp = sorted(df['Neighborhood'].unique())
    bairros_sel = st.sidebar.multiselect(
        "Bairros",
        options=bairros_disp,
        default=bairros_disp
    )
    faixa_preco = st.sidebar.slider(
        "Faixa de preço (USD)",
        int(df['SalePrice'].min()),
        int(df['SalePrice'].max()),
        (int(df['SalePrice'].min()), int(df['SalePrice'].max())),
        step=10000
    )

    df_filt = df[
        (df['Neighborhood'].isin(bairros_sel)) &
        (df['SalePrice'].between(*faixa_preco))
    ]

    st.markdown(f"**Imóveis no filtro:** {len(df_filt):,} de {len(df):,}")
    st.markdown("---")

    # Seletor de variável
    variaveis_numericas = ['GrLivArea', 'OverallQual', 'YearBuilt', 'GarageArea',
                           'TotalBsmtSF', 'LotArea', '1stFlrSF', 'FullBath']
    var_x = st.selectbox("Variável a analisar", variaveis_numericas)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"{var_x} × SalePrice")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(df_filt[var_x], df_filt['SalePrice'], alpha=0.4, color='steelblue',
                   edgecolors='white', linewidths=0.3)
        if len(df_filt) > 1:
            z = np.polyfit(df_filt[var_x], df_filt['SalePrice'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(df_filt[var_x].min(), df_filt[var_x].max(), 100)
            ax.plot(x_line, p(x_line), color='salmon', linewidth=2, label='Tendência')
            ax.legend()
        ax.set_xlabel(var_x)
        ax.set_ylabel('Preço (USD)')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}k'))
        st.pyplot(fig)
        if len(df_filt) > 1:
            corr = df_filt[var_x].corr(df_filt['SalePrice'])
            st.metric("Correlação com SalePrice", f"{corr:.3f}")

    with col2:
        st.subheader("Preço mediano por bairro")
        preco_bairro = df_filt.groupby('Neighborhood')['SalePrice'].median().sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(8, max(5, len(preco_bairro) * 0.25)))
        ax.barh(preco_bairro.index, preco_bairro.values, color='steelblue', edgecolor='white')
        ax.set_xlabel('Preço mediano (USD)')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}k'))
        st.pyplot(fig)

    st.markdown("---")

    # Hipóteses testadas
    st.subheader("Hipóteses testadas na EDA")
    hipoteses = pd.DataFrame({
        'Hipótese': [
            'H1 — GrLivArea é o fator de maior correlação com o preço',
            'H2 — OverallQual tem impacto não-linear no preço',
            'H3 — Bairro explica variações além do tamanho',
            'H4 — Casas sem garagem têm preço sistematicamente menor',
            'H5 — Imóveis mais novos tendem a ser mais caros',
        ],
        'Resultado': [
            'Parcialmente — OverallQual supera GrLivArea (r=0.79 vs 0.71)',
            'Confirmada — relação exponencial entre qualidade 8→10',
            'Confirmada — variação de 2,3x entre bairros (área similar)',
            'Confirmada — diferença de 67% na mediana',
            'Confirmada — correlação positiva moderada (r=0.52)',
        ],
        'Status': ['🟡', '🟢', '🟢', '🟢', '🟢']
    })
    st.dataframe(hipoteses, hide_index=True, use_container_width=True)

# =========================================================
# ABA 3 — COMPARAÇÃO DE MODELOS
# =========================================================
elif aba == "🤖 Modelos":
    st.title("Comparação entre Modelos")
    st.markdown("Quatro modelos foram treinados e comparados na previsão de preços.")

    # Tabela de resultados
    st.subheader("Tabela comparativa")
    tabela_exibir = resultados.copy()
    tabela_exibir['rmse_train'] = tabela_exibir['rmse_train'].apply(lambda x: f"${x:,.0f}")
    tabela_exibir['rmse_val'] = tabela_exibir['rmse_val'].apply(lambda x: f"${x:,.0f}")
    tabela_exibir['mae_val'] = tabela_exibir['mae_val'].apply(lambda x: f"${x:,.0f}")
    tabela_exibir['r2_train'] = tabela_exibir['r2_train'].apply(lambda x: f"{x:.4f}")
    tabela_exibir['r2_val'] = tabela_exibir['r2_val'].apply(lambda x: f"{x:.4f}")
    tabela_exibir['cv_rmse_log'] = tabela_exibir['cv_rmse_log'].apply(lambda x: f"{x:.4f}")
    tabela_exibir.columns = ['Modelo', 'RMSE Treino', 'RMSE Validação', 'MAE Validação',
                              'R² Treino', 'R² Validação', 'CV RMSE (log)']
    st.dataframe(tabela_exibir, hide_index=True, use_container_width=True)

    # Destacar melhor modelo
    melhor_idx = resultados['rmse_val'].idxmin()
    melhor_nome = resultados.loc[melhor_idx, 'modelo']
    melhor_rmse = resultados.loc[melhor_idx, 'rmse_val']
    melhor_r2 = resultados.loc[melhor_idx, 'r2_val']

    st.success(
        f"**Melhor modelo:** {melhor_nome} · "
        f"RMSE Validação: ${melhor_rmse:,.0f} · "
        f"R²: {melhor_r2:.4f}"
    )

    st.markdown("---")

    # Gráficos comparativos
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("RMSE na validação")
        fig, ax = plt.subplots(figsize=(8, 5))
        cores = ['#5b8db8', '#7ab3d0', '#2e7d5e', '#1a5c42']
        bars = ax.bar(resultados['modelo'], resultados['rmse_val'], color=cores, edgecolor='white')
        ax.set_ylabel('RMSE (USD)')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
        for bar, v in zip(bars, resultados['rmse_val']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                    f'${v:,.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        plt.setp(ax.get_xticklabels(), rotation=15, ha='right')
        st.pyplot(fig)

    with col2:
        st.subheader("R² na validação")
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(resultados['modelo'], resultados['r2_val'], color=cores, edgecolor='white')
        ax.set_ylabel('R²')
        ax.set_ylim(0.8, 1.0)
        for bar, v in zip(bars, resultados['r2_val']):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                    f'{v:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        plt.setp(ax.get_xticklabels(), rotation=15, ha='right')
        st.pyplot(fig)

    st.markdown("---")

    # Discussão
    st.subheader("Discussão dos resultados")
    st.markdown("""
    - **Gradient Boosting** vence em validação ($20.715), mas com RMSE de treino baixíssimo ($5.867)
      — sinal de leve overfitting controlado pela regularização interna (subsample=0.8).
    - **Random Forest** apresenta o caso clássico de overfitting: ótimo no treino ($14.246),
      pior na validação ($24.613). Memorizou mais do que generalizou.
    - **Regressão Linear** surpreende com performance próxima ao melhor modelo,
      mostrando que as relações no dataset são em grande parte lineares após feature engineering.
    - **Ridge (L2)** praticamente empata com a linear pura — multicolinearidade não é problema crítico aqui.
    """)

# =========================================================
# ABA 4 — SIMULADOR DE PREÇO
# =========================================================
elif aba == "💰 Simulador de Preço":
    st.title("Simulador de Preço de Imóvel")
    st.markdown(
        f"Use os controles abaixo para simular o preço de venda de um imóvel "
        f"usando o modelo **{resultados.loc[resultados['rmse_val'].idxmin(), 'modelo']}**."
    )

    st.sidebar.markdown("### Características principais")
    overall_qual = st.sidebar.slider("Qualidade geral (1-10)", 1, 10, 6)
    gr_liv_area = st.sidebar.slider("Área habitável (pés²)", 500, 4000, 1500, step=50)
    total_bsmt = st.sidebar.slider("Área do porão (pés²)", 0, 3000, 800, step=50)
    garage_cars = st.sidebar.slider("Vagas na garagem", 0, 4, 2)
    full_bath = st.sidebar.slider("Banheiros completos", 0, 4, 2)
    year_built = st.sidebar.slider("Ano de construção", 1900, 2010, 1980)
    year_remod = st.sidebar.slider("Ano da última reforma", 1950, 2010, 2000)
    lot_area = st.sidebar.slider("Área do terreno (pés²)", 1000, 30000, 9000, step=500)

    # Construir um exemplo baseado na mediana do treino
    X_train_csv = pd.read_csv(DATA_DIR / "X_train.csv")

    # Iniciar com valores medianos das features
    exemplo = X_train_csv.median().to_frame().T.copy()

    # Os dados já estão normalizados — precisamos converter os inputs do usuário
    # para o espaço normalizado. Para isso, recriamos um vetor não normalizado,
    # aplicamos o scaler, e depois substituímos no exemplo.
    # Como simplificação didática, ajustamos diretamente as features mais influentes
    # usando a relação proporcional ao desvio padrão.

    # Carregar valores originais (não normalizados) para referência
    # Aqui usamos uma abordagem prática: criamos um exemplo no espaço original
    # e aplicamos o scaler completo

    # Vamos reconstruir uma linha completa no espaço original
    df_orig = carregar_dados_brutos()

    # Inputs do usuário no espaço original
    inputs_usuario = {
        'OverallQual': overall_qual,
        'GrLivArea': gr_liv_area,
        'TotalBsmtSF': total_bsmt,
        'GarageCars': garage_cars,
        'FullBath': full_bath,
        'YearBuilt': year_built,
        'YearRemodAdd': year_remod,
        'LotArea': lot_area,
    }

    # Recalcular features engineered
    inputs_usuario['TotalSF'] = inputs_usuario['TotalBsmtSF'] + inputs_usuario['GrLivArea']
    inputs_usuario['TotalBaths'] = inputs_usuario['FullBath'] + 0.5  # média típica
    inputs_usuario['HouseAge'] = 2010 - inputs_usuario['YearBuilt']
    inputs_usuario['YearsSinceRemod'] = 2010 - inputs_usuario['YearRemodAdd']
    inputs_usuario['HasGarage'] = 1 if garage_cars > 0 else 0
    inputs_usuario['HasBasement'] = 1 if total_bsmt > 0 else 0

    # Predição usando heurística simplificada (modelo já treinado)
    # Como o dashboard é didático, mostramos a predição baseada nas features principais

    # Construir vetor de entrada completo a partir dos valores medianos do treino
    exemplo_dict = X_train_csv.median().to_dict()

    # Substituir as features ajustadas pelo usuário (após normalização)
    # Para cada feature do usuário, precisamos do valor padronizado.
    # Como o scaler foi salvo, podemos reverter para o espaço original via mean_ e scale_

    feature_names = X_train_csv.columns.tolist()
    medias = scaler.mean_
    escalas = scaler.scale_

    for feat, valor_usuario in inputs_usuario.items():
        if feat in feature_names:
            idx = feature_names.index(feat)
            valor_normalizado = (valor_usuario - medias[idx]) / escalas[idx]
            exemplo_dict[feat] = valor_normalizado

    exemplo_array = np.array([exemplo_dict[f] for f in feature_names]).reshape(1, -1)

    # Predição
    pred_log = modelo.predict(exemplo_array)[0]
    pred_usd = np.expm1(pred_log)

    # Exibir resultado
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.metric(
            "Preço estimado",
            f"${pred_usd:,.0f}",
            delta=f"{((pred_usd - df['SalePrice'].median()) / df['SalePrice'].median() * 100):+.1f}% vs. mediana"
        )

    st.markdown("---")

    # Resumo da configuração
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Configuração do imóvel")
        resumo = pd.DataFrame({
            'Característica': ['Qualidade geral', 'Área habitável', 'Área do porão',
                              'Vagas na garagem', 'Banheiros', 'Ano de construção',
                              'Última reforma', 'Área do terreno'],
            'Valor': [
                f"{overall_qual}/10",
                f"{gr_liv_area:,} pés²",
                f"{total_bsmt:,} pés²",
                f"{garage_cars}",
                f"{full_bath}",
                f"{year_built}",
                f"{year_remod}",
                f"{lot_area:,} pés²"
            ]
        })
        st.dataframe(resumo, hide_index=True, use_container_width=True)

    with col2:
        st.subheader("Comparação com o mercado")
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(df['SalePrice'], bins=50, color='steelblue', edgecolor='white', alpha=0.7)
        ax.axvline(pred_usd, color='red', linewidth=2.5, label=f'Sua simulação: ${pred_usd:,.0f}')
        ax.axvline(df['SalePrice'].median(), color='green', linestyle='--', linewidth=1.5,
                   label=f"Mediana do mercado: ${df['SalePrice'].median():,.0f}")
        ax.set_xlabel('Preço (USD)')
        ax.set_ylabel('Frequência')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}k'))
        ax.legend()
        st.pyplot(fig)

    st.caption(
        "⚠️ O simulador é uma demonstração didática. Valores de features não controladas "
        "pelos sliders usam a mediana do conjunto de treino."
    )
