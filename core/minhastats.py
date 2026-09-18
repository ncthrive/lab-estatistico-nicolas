"""
minhastats.py
==============
Biblioteca estatística implementada "na unha" (sem usar numpy.mean,
statistics.mean, scipy.stats, etc). Toda a matemática aqui é escrita
manualmente com loops e operações aritméticas básicas.

Cada função tem uma docstring com a fórmula matemática usada.
"""

import math


# ---------------------------------------------------------------------------
# 1. MEDIDAS DE TENDÊNCIA CENTRAL
# ---------------------------------------------------------------------------

def media(dados):
    """
    Média aritmética.
    Fórmula: x̄ = (Σ xi) / n
    """
    dados = list(dados)
    n = len(dados)
    if n == 0:
        raise ValueError("Lista vazia não tem média.")
    soma = 0.0
    for x in dados:
        soma += x
    return soma / n


def mediana(dados):
    """
    Mediana: valor central de uma lista ordenada.
    Se n for par, é a média dos dois valores centrais.
    """
    dados_ordenados = sorted(dados)
    n = len(dados_ordenados)
    if n == 0:
        raise ValueError("Lista vazia não tem mediana.")
    meio = n // 2
    if n % 2 == 1:
        return float(dados_ordenados[meio])
    else:
        return (dados_ordenados[meio - 1] + dados_ordenados[meio]) / 2.0


def moda(dados):
    """
    Moda: valor(es) que mais se repete(m).
    Retorna uma lista (pode haver mais de uma moda - multimodal).
    """
    contagem = {}
    for x in dados:
        contagem[x] = contagem.get(x, 0) + 1
    freq_max = max(contagem.values())
    modas = [valor for valor, freq in contagem.items() if freq == freq_max]
    return sorted(modas)


# ---------------------------------------------------------------------------
# 2. MEDIDAS DE DISPERSÃO
# ---------------------------------------------------------------------------

def amplitude(dados):
    """
    Amplitude: diferença entre o maior e o menor valor.
    Fórmula: A = max(x) - min(x)
    """
    return max(dados) - min(dados)


def variancia(dados, amostral=True):
    """
    Variância.
    Amostral (n-1, correção de Bessel): s² = Σ(xi - x̄)² / (n - 1)
    Populacional: σ² = Σ(xi - x̄)² / n
    """
    dados = list(dados)
    n = len(dados)
    if n < 2 and amostral:
        raise ValueError("Variância amostral requer ao menos 2 valores.")
    if n < 1:
        raise ValueError("Lista vazia.")
    m = media(dados)
    soma_quad = 0.0
    for x in dados:
        soma_quad += (x - m) ** 2
    divisor = (n - 1) if amostral else n
    return soma_quad / divisor


def desvio_padrao(dados, amostral=True):
    """
    Desvio padrão: raiz quadrada da variância.
    s = sqrt(s²)   ou   σ = sqrt(σ²)
    """
    return math.sqrt(variancia(dados, amostral=amostral))


def percentil(dados, p):
    """
    Percentil p (0-100) usando interpolação linear (método usado pelo
    NumPy/pandas por padrão - 'linear').
    Passo:
      1. ordenar os dados
      2. calcular a posição: pos = p/100 * (n - 1)
      3. interpolar entre os dois valores mais próximos dessa posição
    """
    dados_ordenados = sorted(dados)
    n = len(dados_ordenados)
    if n == 0:
        raise ValueError("Lista vazia.")
    if n == 1:
        return float(dados_ordenados[0])

    pos = (p / 100.0) * (n - 1)
    piso = math.floor(pos)
    teto = math.ceil(pos)

    if piso == teto:
        return float(dados_ordenados[int(pos)])

    fracao = pos - piso
    valor = dados_ordenados[piso] + fracao * (dados_ordenados[teto] - dados_ordenados[piso])
    return valor


def quartis(dados):
    """
    Retorna (Q1, Q2, Q3) usando a função percentil().
    """
    q1 = percentil(dados, 25)
    q2 = percentil(dados, 50)
    q3 = percentil(dados, 75)
    return q1, q2, q3


def coeficiente_variacao(dados, amostral=True):
    """
    Coeficiente de Variação (CV), em %.
    Fórmula: CV = (desvio_padrao / media) * 100
    Mede a dispersão relativa - útil para comparar variáveis
    com escalas diferentes.
    """
    m = media(dados)
    if m == 0:
        raise ValueError("Média zero: CV não é definido.")
    dp = desvio_padrao(dados, amostral=amostral)
    return (dp / abs(m)) * 100.0


def limites_iqr(dados, k=1.5):
    """
    Limites para detecção de outliers pela regra do IQR (Intervalo
    Interquartil).
    IQR = Q3 - Q1
    Limite inferior = Q1 - k*IQR
    Limite superior = Q3 + k*IQR
    """
    q1, _, q3 = quartis(dados)
    iqr = q3 - q1
    limite_inf = q1 - k * iqr
    limite_sup = q3 + k * iqr
    return limite_inf, limite_sup


def detectar_outliers(dados, k=1.5):
    """
    Retorna a lista de valores considerados outliers pela regra do IQR.
    """
    limite_inf, limite_sup = limites_iqr(dados, k=k)
    return [x for x in dados if x < limite_inf or x > limite_sup]


# ---------------------------------------------------------------------------
# 3. COVARIÂNCIA E CORRELAÇÃO
# ---------------------------------------------------------------------------

def covariancia(x, y, amostral=True):
    """
    Covariância entre duas variáveis.
    Amostral: cov(x,y) = Σ(xi - x̄)(yi - ȳ) / (n - 1)
    """
    x = list(x)
    y = list(y)
    n = len(x)
    if n != len(y):
        raise ValueError("x e y devem ter o mesmo tamanho.")
    if n < 2:
        raise ValueError("São necessários ao menos 2 pares de valores.")
    mx = media(x)
    my = media(y)
    soma = 0.0
    for i in range(n):
        soma += (x[i] - mx) * (y[i] - my)
    divisor = (n - 1) if amostral else n
    return soma / divisor


def correlacao_pearson(x, y):
    """
    Coeficiente de correlação de Pearson (r).
    Fórmula: r = cov(x,y) / (desvio_padrao(x) * desvio_padrao(y))
    Varia entre -1 (correlação negativa perfeita) e +1 (positiva perfeita).
    """
    cov = covariancia(x, y, amostral=True)
    dpx = desvio_padrao(x, amostral=True)
    dpy = desvio_padrao(y, amostral=True)
    if dpx == 0 or dpy == 0:
        raise ValueError("Desvio padrão zero: correlação não definida.")
    return cov / (dpx * dpy)


# ---------------------------------------------------------------------------
# 4. REGRESSÃO LINEAR SIMPLES (MÍNIMOS QUADRADOS)
# ---------------------------------------------------------------------------

def regressao_linear(x, y):
    """
    Regressão linear simples pelo método dos mínimos quadrados.
    Modelo: y = b0 + b1*x

    b1 (inclinação) = Σ(xi - x̄)(yi - ȳ) / Σ(xi - x̄)²
    b0 (intercepto)  = ȳ - b1 * x̄

    Retorna um dicionário com b0, b1, r2 (coeficiente de determinação)
    e uma função de predição.
    """
    x = list(x)
    y = list(y)
    n = len(x)
    if n != len(y) or n < 2:
        raise ValueError("x e y precisam do mesmo tamanho (>=2).")

    mx = media(x)
    my = media(y)

    num = 0.0
    den = 0.0
    for i in range(n):
        num += (x[i] - mx) * (y[i] - my)
        den += (x[i] - mx) ** 2

    if den == 0:
        raise ValueError("Variância de x é zero: regressão indefinida.")

    b1 = num / den
    b0 = my - b1 * mx

    # R² = 1 - (SS_res / SS_tot)
    ss_res = 0.0
    ss_tot = 0.0
    for i in range(n):
        y_pred = b0 + b1 * x[i]
        ss_res += (y[i] - y_pred) ** 2
        ss_tot += (y[i] - my) ** 2

    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    def prever(novo_x):
        return b0 + b1 * novo_x

    return {"b0": b0, "b1": b1, "r2": r2, "prever": prever}


# ---------------------------------------------------------------------------
# 5. TABELA DE FREQUÊNCIAS
# ---------------------------------------------------------------------------

def tabela_frequencias_categorica(dados):
    """
    Tabela de frequências para variável categórica.
    Retorna lista de dicts: [{'categoria':..., 'freq':..., 'freq_rel':...}]
    """
    contagem = {}
    for x in dados:
        contagem[x] = contagem.get(x, 0) + 1
    n = len(dados)
    tabela = []
    for categoria, freq in sorted(contagem.items(), key=lambda kv: -kv[1]):
        tabela.append({
            "categoria": categoria,
            "freq": freq,
            "freq_rel": freq / n,
        })
    return tabela


def tabela_frequencias_continua(dados, n_classes=None):
    """
    Tabela de frequências para variável contínua, dividida em classes
    (intervalos) de igual amplitude - regra de Sturges usada para
    escolher o número de classes quando não informado.
    Fórmula de Sturges: k = 1 + 3.322 * log10(n)
    """
    dados = list(dados)
    n = len(dados)
    if n_classes is None:
        n_classes = max(1, round(1 + 3.322 * math.log10(n)))

    minimo = min(dados)
    maximo = max(dados)
    largura = (maximo - minimo) / n_classes if maximo > minimo else 1.0

    classes = []
    for i in range(n_classes):
        limite_inf = minimo + i * largura
        limite_sup = minimo + (i + 1) * largura
        classes.append([limite_inf, limite_sup, 0])

    for x in dados:
        idx = int((x - minimo) / largura) if largura > 0 else 0
        if idx >= n_classes:
            idx = n_classes - 1
        classes[idx][2] += 1

    tabela = []
    for limite_inf, limite_sup, freq in classes:
        tabela.append({
            "limite_inf": limite_inf,
            "limite_sup": limite_sup,
            "freq": freq,
            "freq_rel": freq / n,
        })
    return tabela


# ---------------------------------------------------------------------------
# 6. ASSIMETRIA (usada para interpretação textual automática)
# ---------------------------------------------------------------------------

def assimetria(dados):
    """
    Coeficiente de assimetria de Pearson (baseado em média e mediana):
    Skew ≈ 3 * (média - mediana) / desvio_padrao
    Valor positivo → cauda à direita; negativo → cauda à esquerda;
    próximo de 0 → aproximadamente simétrica.
    """
    m = media(dados)
    med = mediana(dados)
    dp = desvio_padrao(dados, amostral=True)
    if dp == 0:
        return 0.0
    return 3 * (m - med) / dp


def interpretar_assimetria(valor_assimetria):
    """Retorna uma frase interpretando o coeficiente de assimetria."""
    if valor_assimetria > 0.5:
        return "A distribuição é assimétrica à direita (cauda longa para valores altos)."
    elif valor_assimetria < -0.5:
        return "A distribuição é assimétrica à esquerda (cauda longa para valores baixos)."
    else:
        return "A distribuição é aproximadamente simétrica."
