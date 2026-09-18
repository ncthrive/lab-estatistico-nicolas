# Laboratorio Estatistico Interativo - Video Game Sales

Aplicacao em Streamlit que carrega um dataset real de vendas de video
games e permite explora-lo com estatistica descritiva, probabilidade e
simulacao, distribuicoes teoricas e regressao linear. O nucleo
estatistico foi implementado do zero (sem usar funcoes prontas de
estatistica) e validado contra NumPy/SciPy.

## Integrantes

| Nome completo | Matricula |
|---|---|
| Nicolas Bomfim | 72650395 |

Nome do grupo: Nicolas

## Dataset

- Nome: Video Game Sales
- Fonte original: Kaggle - gregorut/videogamesales (https://www.kaggle.com/datasets/gregorut/videogamesales)
- Registros usados: 1050 (minimo exigido: 1000)
- Variaveis numericas (6): Year, NA_Sales, EU_Sales, JP_Sales, Other_Sales, Global_Sales
- Variaveis categoricas (3): Platform, Genre, Publisher
- Arquivo local: data/vgsales.csv

## Estrutura do repositorio

```
lab_estatistico/
├── app.py                    aplicacao Streamlit (interface e modulos)
├── core/
│   └── minhastats.py         biblioteca estatistica propria
├── data/
│   └── vgsales.csv           dataset real
├── tests/
│   └── test_minhastats.py    testes automatizados (pytest)
├── requirements.txt
├── README.md
└── RELATORIO.md
```

## Como instalar e rodar

Pre-requisito: Python 3.10 ou superior.

```bash
git clone <URL_DO_REPOSITORIO>
cd lab_estatistico

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

python -m pip install -r requirements.txt

python -m pytest tests/ -v

python -m streamlit run app.py
```

O Streamlit abre automaticamente no navegador em http://localhost:8501.

## Modulos implementados

- Modulo 0: carregamento e visao geral do dataset.
- Modulo 1: core/minhastats.py com media, mediana, moda, amplitude,
  variancia e desvio padrao (amostral e populacional), quartis,
  coeficiente de variacao, covariancia, correlacao de Pearson e
  regressao linear por minimos quadrados, testados contra NumPy/SciPy.
- Modulo 2: estatistica descritiva interativa, tabela de frequencias,
  histograma, boxplot, grafico de barras, deteccao de outliers (IQR) e
  interpretacao automatica da assimetria.
- Modulo 3: simulacao de Monte Carlo (Lei dos Grandes Numeros e
  Teorema Central do Limite), com parametros controlaveis.
- Modulo 4: ajuste visual de distribuicoes teoricas (Normal,
  Exponencial, Uniforme) sobre os dados reais.
- Modulo 5: correlacao, regressao linear, equacao da reta, R2 e
  predicao interativa.
- Modulo 6: resumo das descobertas (detalhadas em RELATORIO.md).
