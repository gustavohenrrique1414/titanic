# Previsão de Sobrevivência no Titanic

Este projeto aplica técnicas de **Machine Learning** para prever a sobrevivência de passageiros no Titanic com base em características demográficas e socioeconômicas.

O projeto segue um fluxo completo de ciência de dados:

EDA → Pré-processamento → Treinamento de Modelos → Avaliação de Desempenho

O objetivo é identificar **quais fatores influenciaram a sobrevivência** e construir modelos capazes de realizar previsões confiáveis.

---

# Dataset

O dataset contém informações sobre passageiros do Titanic, incluindo características pessoais e detalhes da viagem.

Principais variáveis:

- **Pclass** — Classe do passageiro (1ª, 2ª ou 3ª)
- **Sex** — Sexo do passageiro
- **Age** — Idade
- **Fare** — Valor da passagem
- **Embarked** — Porto de embarque
- **SibSp** — Número de irmãos/cônjuges a bordo
- **Parch** — Número de pais/filhos a bordo

Variável alvo:

- **Survived**
  - `0` = Não sobreviveu
  - `1` = Sobreviveu

---

# Análise Exploratória de Dados (EDA)

A análise exploratória revelou padrões importantes sobre os fatores associados à sobrevivência.

### Distribuição das classes

O dataset apresenta um **leve desbalanceamento**, com maior número de passageiros que não sobreviveram.

Por isso, a avaliação dos modelos considerou métricas além da acurácia.

---

### Impacto do sexo

O **sexo do passageiro foi o fator mais determinante** para a sobrevivência.

Principais observações:

- Mulheres tiveram **taxa de sobrevivência muito maior**
- Homens tiveram **maior probabilidade de não sobreviver**

Esse padrão reflete a política adotada durante o desastre conhecida como **"mulheres e crianças primeiro"**.

---

### Classe do passageiro

A classe socioeconômica teve grande impacto na sobrevivência.

Observações importantes:

- Passageiros da **1ª classe tiveram maior taxa de sobrevivência**
- Passageiros da **3ª classe tiveram as menores taxas**
- Isso sugere influência de fatores como **localização da cabine e acesso aos botes salva-vidas**

---

### Distribuição de idade

A maior parte dos passageiros estava entre **20 e 40 anos**.

Observações:

- Crianças tiveram **probabilidade ligeiramente maior de sobreviver**
- Idade teve impacto menor que sexo e classe social

---

### Valor da passagem

A variável **Fare** apresentou distribuição assimétrica com presença de outliers.

Observações:

- Passagens mais caras estão associadas a **maiores taxas de sobrevivência**
- Essa variável está fortemente relacionada à **classe do passageiro**

---

### Estrutura familiar

As variáveis **SibSp** e **Parch** indicam tamanho da família a bordo.

Padrões identificados:

- Passageiros com **famílias pequenas tiveram maiores taxas de sobrevivência**
- Passageiros viajando sozinhos tiveram **menor probabilidade de sobreviver**
- Famílias muito grandes também tiveram menor sobrevivência

---

# Pré-processamento dos dados

O pré-processamento foi implementado **manualmente utilizando estratégia Hold-Out**, evitando o uso de pipelines automáticos.

Um princípio fundamental adotado foi:

> Todos os transformadores são ajustados apenas nos dados de treino (fit) e aplicados no conjunto de validação (transform), evitando **data leakage**.

Principais etapas:

### Age
- Imputação de valores faltantes com **mediana**
- Padronização usando **StandardScaler**

### Fare
- Imputação com **mediana**
- Escalonamento usando **MinMaxScaler**

### Sex
- Imputação com valor mais frequente
- Codificação usando **OrdinalEncoder**

