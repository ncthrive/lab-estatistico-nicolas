# Relatório — Laboratório Estatístico Interativo (Video Game Sales)

## 1. Dataset escolhido e justificativa

Escolhi o dataset **Video Game Sales** (fonte: Kaggle —
`gregorut/videogamesales`, coletado originalmente do site vgchartz.com).
Ele reúne vendas globais (em milhões de cópias) de jogos por região,
plataforma, gênero e publicadora.

Motivos da escolha:
- Atende aos requisitos mínimos (usamos 1.050 registros; dataset original
  tem 16.598 registros no total).
- Possui **6 variáveis numéricas** (`Year`, `NA_Sales`, `EU_Sales`,
  `JP_Sales`, `Other_Sales`, `Global_Sales`) e **3 variáveis categóricas**
  (`Platform`, `Genre`, `Publisher`), superando o mínimo de 4 numéricas + 2
  categóricas exigido.
- Tema acessível e permite discussões estatísticas ricas (assimetria forte
  em vendas, correlação entre mercados regionais, comparação entre gêneros
  e plataformas).
- Sem valores nulos nas colunas usadas, simplificando o pipeline.

## 2. Decisões de implementação do núcleo estatístico

Todas as funções estão em `core/minhastats.py`, implementadas apenas com
`math` e laços/`sum` manuais (nenhuma função pronta de estatística de
terceiros é usada nos cálculos exibidos ao usuário). NumPy/Pandas são
usados somente para **carregar/manipular** os dados e para os testes de
**validação**.

### Fórmulas utilizadas

- **Média:** x̄ = (Σxᵢ) / n
- **Mediana:** valor central da lista ordenada (ou média dos dois centrais,
  se n for par)
- **Moda:** valor(es) de maior frequência
- **Amplitude:** A = max(x) − min(x)
- **Variância amostral:** s² = Σ(xᵢ − x̄)² / (n − 1)
- **Variância populacional:** σ² = Σ(xᵢ − x̄)² / n
- **Desvio padrão:** raiz quadrada da variância correspondente
- **Percentil (interpolação linear):** posição = (p/100)·(n−1), com
  interpolação entre os dois valores adjacentes — mesmo método usado por
  padrão pelo NumPy (`numpy.percentile`, `interpolation='linear'`)
- **Coeficiente de variação:** CV = (s / x̄) · 100%
- **Covariância amostral:** cov(x,y) = Σ(xᵢ − x̄)(yᵢ − ȳ) / (n − 1)
- **Correlação de Pearson:** r = cov(x,y) / (sₓ · s_y)
- **Regressão linear (mínimos quadrados):**
  - b₁ = Σ(xᵢ − x̄)(yᵢ − ȳ) / Σ(xᵢ − x̄)²
  - b₀ = ȳ − b₁·x̄
  - R² = 1 − (SS_res / SS_tot)
- **Regra do IQR para outliers:** um valor é outlier se estiver fora de
  [Q1 − 1.5·IQR, Q3 + 1.5·IQR], onde IQR = Q3 − Q1
- **Assimetria (coeficiente de Pearson):** 3·(média − mediana) / desvio padrão
- **Regra de Sturges (nº de classes):** k = 1 + 3.322·log₁₀(n)

## 3. Validação contra NumPy/SciPy

O arquivo `tests/test_minhastats.py` compara cada função com a referência
NumPy/SciPy usando `math.isclose` com tolerância relativa de `1e-6` (para
regressão linear, `1e-2` a `1e-3`, devido a ruído aleatório introduzido no
teste). Rodamos manualmente a comparação durante o desenvolvimento e todos
os resultados bateram, por exemplo:

| Métrica | minhastats | NumPy/SciPy |
|---|---|---|
| Média (amostra de 200 valores) | 48.422393 | 48.422393 |
| Mediana | 46.984785 | 46.984785 |
| Variância amostral | 860.654366 | 860.654366 |
| Desvio padrão amostral | 29.336911 | 29.336911 |
| Percentil 25 | 22.868091 | 22.868091 |
| Percentil 75 | 72.978161 | 72.978161 |
| Covariância | 34.070462 | 34.070462 |
| Correlação de Pearson | 0.077696 | 0.077696 |
| Regressão — coeficiente angular (b1) | 1.999087 | 1.999087 |
| Regressão — intercepto (b0) | 5.003152 | 5.003152 |

Todas as diferenças ficaram abaixo da tolerância definida (arredondamento
de ponto flutuante apenas).

## 4. Descobertas estatísticas (Módulo 6)

### Descoberta 1 — Vendas globais são fortemente assimétricas à direita

- Média de `Global_Sales`: **4,17 milhões de cópias**
- Mediana: **2,80 milhões**
- Desvio padrão: **4,73**
- Coeficiente de assimetria: **≈ 0,87** (assimetria à direita)
- Coeficiente de variação: **≈ 113,5%**
- Outliers pela regra do IQR: **96 de 1.050 jogos (≈ 9,1%)**

A média fica bem acima da mediana, confirmando uma distribuição com cauda
longa à direita: a maioria dos jogos vende relativamente pouco, enquanto
um pequeno grupo de "blockbusters" (Wii Sports, GTA V, etc.) puxa a média
para cima. O CV alto (>100%) mostra dispersão relativa muito grande —
típico de mercados dominados por poucos sucessos.

### Descoberta 2 — Vendas na América do Norte explicam boa parte das vendas globais

- Correlação de Pearson entre `NA_Sales` e `Global_Sales`: **r ≈ 0,92**
  (correlação forte e positiva)
- Regressão linear: Global_Sales ≈ 0,73 + 1,70 · NA_Sales
- R² ≈ **0,848** — o mercado norte-americano sozinho explica cerca de 85%
  da variação nas vendas globais.

Isso sugere que a América do Norte é historicamente o mercado mais
representativo do desempenho global de um jogo neste dataset. **Importante:**
essa forte correlação não implica causalidade — o tamanho do mercado
americano, o padrão de lançamento simultâneo e o próprio peso da região no
cálculo de `Global_Sales` podem explicar parte dessa relação.

### Descoberta 3 — Ação e plataformas de sétima geração dominam o catálogo, mas "Platform" e "Role-Playing" vendem mais por jogo

- Gênero mais frequente: **Action** (211 jogos, 20,1% do dataset), seguido
  por Sports (13,6%) e Shooter (13,1%).
- Plataforma mais frequente: **PS2** (175 jogos, 16,7%), seguida por X360
  (12,7%) e PS3 (11,8%).
- Porém, ao comparar a **média de vendas globais por gênero** entre os 5
  mais frequentes, o gênero **Platform** tem a maior média por jogo
  (5,00 milhões), seguido por Role-Playing (4,51) e Shooter (4,51) —
  todos acima da média geral do dataset (4,17). Action, apesar de ser o
  gênero mais comum, tem média abaixo da geral (3,76).

Ou seja: ter mais jogos lançados (Action) não significa vender mais por
título — gêneros mais raros como Platform (puxado por franquias icônicas
como Mario) têm desempenho médio por jogo superior.

## 5. Prints dos módulos

_(adicionar aqui capturas de tela de cada módulo da aplicação rodando)_
