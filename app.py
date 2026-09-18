"""
app.py

Laboratorio Estatistico Interativo
Dataset: Video Game Sales (vgsales.csv)

Todas as estatisticas exibidas usam a biblioteca propria core/minhastats.py.
NumPy/Pandas sao usados apenas para carregar dados e gerar graficos.
"""

import sys
import os
import random

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "core"))
import minhastats as ms

st.set_page_config(page_title="Laboratorio Estatistico - Video Game Sales", layout="wide")

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "vgsales.csv")
NUMERICAS = ["Year", "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales"]
CATEGORICAS = ["Platform", "Genre", "Publisher"]


@st.cache_data
def carregar_dados():
    return pd.read_csv(DATA_PATH)


df = carregar_dados()

st.title("Laboratorio Estatistico Interativo - Video Game Sales")
st.caption(f"Dataset com {df.shape[0]} registros. Estatisticas calculadas pela biblioteca propria core/minhastats.py.")

aba = st.sidebar.radio(
    "Modulos",
    [
        "0. Dados",
        "2. Estatistica descritiva",
        "3. Simulacao (Monte Carlo)",
        "4. Distribuicoes teoricas",
        "5. Correlacao e regressao",
        "6. Descobertas",
    ],
)

# Modulo 0
if aba.startswith("0"):
    st.header("Modulo 0 - Dados reais")
    st.write("Fonte: Kaggle - Video Game Sales (vgchartz.com).")
    st.dataframe(df.head(30), use_container_width=True)
    st.write("Variaveis numericas:", ", ".join(NUMERICAS))
    st.write("Variaveis categoricas:", ", ".join(CATEGORICAS))
    st.write("Total de registros:", df.shape[0])

# Modulo 2
elif aba.startswith("2"):
    st.header("Modulo 2 - Estatistica Descritiva")
    tipo = st.radio("Tipo de variavel", ["Numerica", "Categorica"], horizontal=True)

    if tipo == "Numerica":
        var = st.selectbox("Variavel", NUMERICAS)
        dados = df[var].dropna().tolist()

        c1, c2, c3 = st.columns(3)
        c1.metric("Media", f"{ms.media(dados):.2f}")
        c1.metric("Mediana", f"{ms.mediana(dados):.2f}")
        c2.metric("Desvio padrao", f"{ms.desvio_padrao(dados):.2f}")
        c2.metric("Variancia", f"{ms.variancia(dados):.2f}")
        c3.metric("Amplitude", f"{ms.amplitude(dados):.2f}")
        c3.metric("CV (%)", f"{ms.coeficiente_variacao(dados):.1f}")

        q1, q2, q3 = ms.quartis(dados)
        st.write(f"Quartis: Q1={q1:.2f}  Q2={q2:.2f}  Q3={q3:.2f}")

        fig, ax = plt.subplots()
        ax.hist(dados, bins=20)
        ax.set_xlabel(var)
        ax.set_ylabel("Frequencia")
        st.pyplot(fig)

        fig2, ax2 = plt.subplots()
        ax2.boxplot(dados, vert=False)
        st.pyplot(fig2)

        outliers = ms.detectar_outliers(dados)
        st.write(f"Outliers (regra IQR): {len(outliers)} de {len(dados)} ({100*len(outliers)/len(dados):.1f}%)")

        skew = ms.assimetria(dados)
        st.write(f"Assimetria: {skew:.2f} - {ms.interpretar_assimetria(skew)}")

    else:
        var = st.selectbox("Variavel", CATEGORICAS)
        dados = df[var].dropna().tolist()
        tabela = pd.DataFrame(ms.tabela_frequencias_categorica(dados)[:15])
        tabela["freq_rel"] = (tabela["freq_rel"] * 100).round(1)
        st.dataframe(tabela, use_container_width=True)

        fig, ax = plt.subplots()
        ax.bar(tabela["categoria"].astype(str), tabela["freq"])
        plt.xticks(rotation=60, ha="right")
        st.pyplot(fig)

# Modulo 3
elif aba.startswith("3"):
    st.header("Modulo 3 - Probabilidade e Simulacao")
    sub = st.radio("Experimento", ["Lei dos Grandes Numeros", "Teorema Central do Limite"])

    if sub == "Lei dos Grandes Numeros":
        st.write("Simulacao de lancamentos de moeda (P=0.5). A frequencia relativa converge para 0.5.")
        n = st.slider("Numero de lancamentos", 10, 20000, 2000, step=10)
        random.seed(0)
        resultados = [1 if random.random() < 0.5 else 0 for _ in range(n)]
        freqs = []
        soma = 0
        for i, r in enumerate(resultados, start=1):
            soma += r
            freqs.append(soma / i)

        fig, ax = plt.subplots()
        ax.plot(range(1, n + 1), freqs)
        ax.axhline(0.5, color="red", linestyle="--")
        ax.set_xlabel("Lancamentos")
        ax.set_ylabel("Frequencia relativa")
        st.pyplot(fig)
        st.write(f"Frequencia apos {n} lancamentos: {freqs[-1]:.4f}")

    else:
        st.write("Amostras repetidas de Global_Sales. A distribuicao das medias tende a Normal (TCL).")
        tam = st.slider("Tamanho da amostra", 2, 200, 30)
        rep = st.slider("Numero de repeticoes", 100, 5000, 1000, step=100)

        pop = df["Global_Sales"].dropna().tolist()
        random.seed(1)
        medias = [ms.media(random.choices(pop, k=tam)) for _ in range(rep)]

        c1, c2 = st.columns(2)
        with c1:
            st.write("Populacao original")
            fig1, ax1 = plt.subplots()
            ax1.hist(pop, bins=30)
            st.pyplot(fig1)
        with c2:
            st.write(f"Medias amostrais (n={tam})")
            fig2, ax2 = plt.subplots()
            ax2.hist(medias, bins=30)
            st.pyplot(fig2)

        st.write(f"Media das medias: {ms.media(medias):.3f} | Desvio padrao: {ms.desvio_padrao(medias):.3f}")

# Modulo 4
elif aba.startswith("4"):
    st.header("Modulo 4 - Distribuicoes Teoricas")
    var = st.selectbox("Variavel", NUMERICAS)
    dados = df[var].dropna().tolist()
    dist = st.selectbox("Distribuicao", ["Normal", "Exponencial", "Uniforme"])

    fig, ax = plt.subplots()
    ax.hist(dados, bins=30, density=True, alpha=0.6, label="Dados")
    xs = np.linspace(min(dados), max(dados), 300)

    if dist == "Normal":
        mu, sigma = ms.media(dados), ms.desvio_padrao(dados)
        ys = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xs - mu) / sigma) ** 2)
        ax.plot(xs, ys, color="red", label="Normal")
    elif dist == "Exponencial":
        pos = [d for d in dados if d >= 0]
        lam = 1 / ms.media(pos)
        xs2 = np.linspace(0, max(pos), 300)
        ax.plot(xs2, lam * np.exp(-lam * xs2), color="red", label="Exponencial")
    else:
        a, b = min(dados), max(dados)
        altura = 1 / (b - a)
        ax.plot([a, b], [altura, altura], color="red", label="Uniforme")

    ax.set_xlabel(var)
    ax.legend()
    st.pyplot(fig)
    st.write("Compare a curva teorica (vermelha) com o histograma dos dados reais para avaliar o ajuste.")

# Modulo 5
elif aba.startswith("5"):
    st.header("Modulo 5 - Correlacao e Regressao Linear")
    c1, c2 = st.columns(2)
    var_x = c1.selectbox("Variavel X", NUMERICAS, index=1)
    var_y = c2.selectbox("Variavel Y", NUMERICAS, index=5)

    if var_x == var_y:
        st.warning("Escolha variaveis diferentes.")
    else:
        sub = df[[var_x, var_y]].dropna()
        x, y = sub[var_x].tolist(), sub[var_y].tolist()
        r = ms.correlacao_pearson(x, y)
        reg = ms.regressao_linear(x, y)

        st.write(f"Correlacao de Pearson (r): {r:.3f}")
        st.write(f"Equacao: y = {reg['b0']:.3f} + {reg['b1']:.3f} * x")
        st.write(f"R2: {reg['r2']:.3f}")

        fig, ax = plt.subplots()
        ax.scatter(x, y, alpha=0.4)
        xs = np.linspace(min(x), max(x), 100)
        ax.plot(xs, [reg["prever"](v) for v in xs], color="red")
        ax.set_xlabel(var_x)
        ax.set_ylabel(var_y)
        st.pyplot(fig)

        valor_x = st.number_input(f"Valor de {var_x} para prever {var_y}", value=float(ms.media(x)))
        st.write(f"Predicao: {reg['prever'](valor_x):.3f}")

        st.info("Correlacao nao implica causalidade.")

# Modulo 6
else:
    st.header("Modulo 6 - Relatorio de Descobertas")
    st.write("As 3 descobertas estatisticas do trabalho estao documentadas no arquivo RELATORIO.md, "
             "com base nos numeros e graficos gerados pelos modulos acima.")
