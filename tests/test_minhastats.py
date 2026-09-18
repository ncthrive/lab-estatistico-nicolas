"""
test_minhastats.py
====================
Testes automatizados que comparam cada função de minhastats.py com a
implementação de referência do NumPy/SciPy. Tolerância numérica: 1e-6
(diferenças de arredondamento de ponto flutuante).

Rodar com:  pytest tests/test_minhastats.py -v
"""

import sys
import os
import math
import random

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
import minhastats as ms

TOLERANCIA = 1e-6

random.seed(42)
DADOS_A = [random.uniform(0, 100) for _ in range(200)]
DADOS_B = [random.uniform(0, 50) for _ in range(200)]
DADOS_CATEGORICOS = ["A", "B", "A", "C", "B", "A", "D", "C", "A", "B"]


def test_media():
    assert math.isclose(ms.media(DADOS_A), np.mean(DADOS_A), rel_tol=TOLERANCIA)


def test_mediana():
    assert math.isclose(ms.mediana(DADOS_A), np.median(DADOS_A), rel_tol=TOLERANCIA)


def test_moda():
    dados = [1, 2, 2, 3, 3, 3, 4]
    assert ms.moda(dados) == [3]


def test_amplitude():
    esperado = max(DADOS_A) - min(DADOS_A)
    assert math.isclose(ms.amplitude(DADOS_A), esperado, rel_tol=TOLERANCIA)


def test_variancia_amostral():
    esperado = np.var(DADOS_A, ddof=1)
    assert math.isclose(ms.variancia(DADOS_A, amostral=True), esperado, rel_tol=TOLERANCIA)


def test_variancia_populacional():
    esperado = np.var(DADOS_A, ddof=0)
    assert math.isclose(ms.variancia(DADOS_A, amostral=False), esperado, rel_tol=TOLERANCIA)


def test_desvio_padrao_amostral():
    esperado = np.std(DADOS_A, ddof=1)
    assert math.isclose(ms.desvio_padrao(DADOS_A, amostral=True), esperado, rel_tol=TOLERANCIA)


def test_desvio_padrao_populacional():
    esperado = np.std(DADOS_A, ddof=0)
    assert math.isclose(ms.desvio_padrao(DADOS_A, amostral=False), esperado, rel_tol=TOLERANCIA)


def test_percentil_25():
    esperado = np.percentile(DADOS_A, 25)
    assert math.isclose(ms.percentil(DADOS_A, 25), esperado, rel_tol=TOLERANCIA)


def test_percentil_75():
    esperado = np.percentile(DADOS_A, 75)
    assert math.isclose(ms.percentil(DADOS_A, 75), esperado, rel_tol=TOLERANCIA)


def test_quartis():
    q1, q2, q3 = ms.quartis(DADOS_A)
    assert math.isclose(q1, np.percentile(DADOS_A, 25), rel_tol=TOLERANCIA)
    assert math.isclose(q2, np.percentile(DADOS_A, 50), rel_tol=TOLERANCIA)
    assert math.isclose(q3, np.percentile(DADOS_A, 75), rel_tol=TOLERANCIA)


def test_coeficiente_variacao():
    esperado = (np.std(DADOS_A, ddof=1) / np.mean(DADOS_A)) * 100
    assert math.isclose(ms.coeficiente_variacao(DADOS_A), esperado, rel_tol=TOLERANCIA)


def test_covariancia():
    esperado = np.cov(DADOS_A, DADOS_B, ddof=1)[0][1]
    assert math.isclose(ms.covariancia(DADOS_A, DADOS_B), esperado, rel_tol=TOLERANCIA)


def test_correlacao_pearson():
    esperado = np.corrcoef(DADOS_A, DADOS_B)[0][1]
    assert math.isclose(ms.correlacao_pearson(DADOS_A, DADOS_B), esperado, rel_tol=TOLERANCIA)


def test_regressao_linear():
    # y = 2x + 5 com ruído leve
    x = list(range(1, 101))
    y = [2 * xi + 5 + random.uniform(-2, 2) for xi in x]
    resultado = ms.regressao_linear(x, y)

    b1_esperado, b0_esperado = np.polyfit(x, y, 1)
    assert math.isclose(resultado["b1"], b1_esperado, rel_tol=1e-3)
    assert math.isclose(resultado["b0"], b0_esperado, rel_tol=1e-2)
    assert 0.9 < resultado["r2"] <= 1.0  # ajuste deve ser forte


def test_detectar_outliers():
    dados = [10, 12, 11, 13, 12, 11, 100]  # 100 é outlier
    outliers = ms.detectar_outliers(dados)
    assert 100 in outliers


def test_tabela_frequencias_categorica():
    tabela = ms.tabela_frequencias_categorica(DADOS_CATEGORICOS)
    total_freq = sum(item["freq"] for item in tabela)
    assert total_freq == len(DADOS_CATEGORICOS)
    # 'A' aparece 4 vezes - deve ser a categoria mais frequente
    assert tabela[0]["categoria"] == "A"
    assert tabela[0]["freq"] == 4


def test_tabela_frequencias_continua_soma():
    tabela = ms.tabela_frequencias_continua(DADOS_A, n_classes=8)
    total_freq = sum(item["freq"] for item in tabela)
    assert total_freq == len(DADOS_A)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
