# 🚢 **TITANIC SURVIVAL PREDICTION**

> *"Por trás de cada linha deste dataset há um nome, um rosto, uma história interrompida. Este projeto é uma homenagem às 1.500+ vidas perdidas e um estudo sobre como dados podem revelar padrões de desigualdade, privilégio e sobrevivência."*

---

## 📋 **Índice**

1. [📖 Introdução Histórica](#-introdução-histórica)
2. [🔍 Exploração de Dados (EDA)](#-exploração-de-dados-eda)
3. [🔧 Pré-Processamento](#-pré-processamento)
4. [🤖 Modelagem & Resultados](#-modelagem--resultados)
5. [ Conclusões & Lições](#-conclusões--lições)
6. [💻 Como Usar](#-como-usar)

---

## 📖 **Introdução Histórica**

### **Uma Noite que Mudou a História**

> *"15 de abril de 1912, 2h20 da manhã. O navio considerado 'inafundável' encontra seu destino nas águas geladas do Atlântico Norte."*

| Estatística | Valor |
|-------------|-------|
| **Passageiros a bordo** | ~2.224 |
| **Sobreviventes** | 705 |
| **Vítimas** | ~1.519 |
| **Taxa de sobrevivência** | 31.7% |

**Objetivo do Projeto:**
- ✅ Analisar padrões de sobrevivência usando Exploratory Data Analysis (EDA)
- ✅ Pré-processar dados evitando data leakage
- ✅ Comparar múltiplos algoritmos de Machine Learning
- ✅ Entender quais fatores determinaram quem viveu e quem morreu

---

## 🔍 **Exploração de Dados (EDA)**

### **📊 Visão Geral do Dataset**

```python
data.shape  # (891, 12)
data.columns  # PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, 
              # Parch, Ticket, Fare, Cabin, Embarked
```

### **⚠️ Dados Missing**

| Variável | Missing | % | Impacto |
|----------|---------|---|---------|
| **Cabin** | 687 | 77% | ❌ Excluída |
| **Age** | 177 | 20% | ⚠️ Imputação necessária |
| **Embarked** | 2 | <1% | ✅ Quase completo |

> *"77% dos registros de cabine se perderam. Não nos dados — na realidade. Quando o navio afundou, as listas de passageiros foram destruídas. O que temos são fragmentos de uma tragédia."*

### **📈 Estatísticas Descritivas**

| Feature | Mean | Std | Min | 50% | Max |
|---------|------|-----|-----|-----|-----|
| **Age** | 29.7 | 14.5 | 0.42 | 28 | 80 |
| **Fare** | 32.20 | 49.69 | 0 | 14.45 | 512.33 |
| **Pclass** | 2.31 | 0.84 | 1 | 3 | 3 |
| **Survived** | 0.38 | 0.49 | 0 | 0 | 1 |

### **🎯 Principais Insights da EDA**

| Insight | Evidência | Impacto |
|---------|-----------|---------|
| **Mulheres sobreviveram mais** | 74% vs 19% (homens) | ⭐⭐⭐ |
| **1ª classe teve prioridade** | 63% vs 24% (3ª classe) | ⭐⭐⭐ |
| **Tarifa mais alta = mais vida** | Mediana £30 vs £13 | ⭐⭐ |
| **Crianças foram priorizadas** | Mediana idade menor | ⭐⭐ |
| **Cherbourg = mais sobreviventes** | 55% vs 34% (Southampton) | ⭐ |

### **🔗 Correlações com Sobrevivência**

```
Feature      | Correlação | Interpretação
-------------|------------|----------------
Pclass       | -0.34      | Classe menor = mais sobrevivência
Fare         | +0.26      | Mais dinheiro = mais chances
Age          | -0.08      | Idade maior = menos chances
SibSp        | -0.04      | Pouco impacto
Parch        | +0.08      | Pouco impacto
```

> *"Pclass e Fare são duas faces da mesma moeda: classe social. Juntas, elas explicam grande parte da variância na sobrevivência."*

---

## 🔧 **Pré-Processamento**

### **🛡️ Regra de Ouro: FIT no Treino, TRANSFORM na Validação**

```python
# ✅ CORRETO
age_imputer.fit_transform(X_train[['Age']])   # Aprende do TREINO
age_imputer.transform(X_val[['Age']])         # Aplica na VALIDAÇÃO

# ❌ ERRADO (Data Leakage!)
age_imputer.fit_transform(X[['Age']])         # Vaza informação do futuro!
```

> *"Nunca use informações do conjunto de validação no treino. Isso seria como estudar com a prova antes de fazer o exame."*

### **📋 Estratégias de Imputação**

| Feature | Estratégia | Justificativa |
|---------|------------|---------------|
| **Age** | Mediana | Distribuição com outliers |
| **Fare** | Mediana | Altamente assimétrico |
| **Sex** | Moda | Variável categórica binária |
| **Embarked** | Moda | Apenas 2 missing |

### **⚖️ Escalonamento**

| Feature | Scaler | Razão |
|---------|--------|-------|
| **Age** | StandardScaler | Distribuição ~normal |
| **Fare** | MinMaxScaler | Outliers extremos (£0-£512) |

### **🔢 Encoding**

| Feature | Método | Categorias |
|---------|--------|------------|
| **Sex** | Ordinal | female=0, male=1 |
| **Embarked** | OneHot | C, Q, S |
| **Pclass** | OneHot | 1, 2, 3 |

### **📊 Features Finais**

```python
['PassengerId', 'Survived', 'Sex', 'SibSp', 'Parch', 'Fare', 
 'Embarked_C', 'Embarked_Q', 'Embarked_S', 
 'Pclass_1', 'Pclass_2', 'Pclass_3']
```

---

## 🤖 **Modelagem & Resultados**

### **🥊 Os Contendores**

| Modelo | Tipo | Característica |
|--------|------|----------------|
| **Logistic Regression** | Linear | Interpretável, simples |
| **Random Forest** | Ensemble | Poderoso, prone a overfitting |
| **Gradient Boosting** | Ensemble | Otimizador sequencial |
| **SVC** | Kernel-based | Mestre de fronteiras |

### **📈 Performance Comparison**

| Modelo | Train Accuracy | Val Accuracy | Gap | AUC | Veredito |
|--------|---------------|--------------|-----|-----|----------|
| **Gradient Boosting** | 91% | 84% | +7% | 0.836 | ⭐ Equilibrado |
| **Random Forest** | 98% | 81% | +17% | 0.801 | ⚠️ Overfitting |
| **Logistic Regression** | 80% | 80% | 0% | 0.822 | ✅ Generalizou |
| **SVC** | 84% | 80% | +4% | **0.846** | 🏆 Melhor AUC |

### **🏆 Vencedores por Categoria**

| Categoria | Modelo | Métrica |
|-----------|--------|---------|
| **Generalização** | Logistic Regression | Gap ~0% |
| **Discriminação (AUC)** | **SVC** | **0.846** |
| **Precisão em Mortes** | Gradient Boosting | 98 True Negatives |
| **Interpretabilidade** | Logistic Regression | Coeficientes claros |

### **⚠️ Lição Chave: Overfitting**

> *"O Random Forest teve 98% de acurácia no treino, mas caiu para 81% na validação. Ele não aprendeu os padrões de sobrevivência; ele **decorou** os passageiros do treino. Isso é **Overfitting**: inteligência artificial que falha na vida real."*

### **🎯 Matriz de Confusão (Melhor Modelo)**

```
                    Predicted
                  ┌─────────┬─────────┐
                  │  Não    │   Sim   │
        ┌─────────┼─────────┼─────────┤
Actual  │  Não    │   94    │    7    │
        ├─────────┼─────────┼─────────┤
        │  Sim    │   24    │   28    │
        └─────────┴─────────┴─────────┘
```

---

## 📊 **Conclusões & Lições**

### **🔍 O Que os Dados Revelam**

1. **✅ Classe Social > Tudo**
   - Passageiros da 1ª classe tinham 3x mais chances de sobreviver
   - Acesso privilegiado aos botes salva-vidas

2. **✅ Gênero Foi Crucial**
   - 74% das mulheres sobreviveram vs 19% dos homens
   - Política "mulheres e crianças primeiro" foi real

3. **✅ Dinheiro Importava**
   - Fare correlacionado positivamente com sobrevivência
   - Privilégio econômico salvou vidas

4. **✅ Crianças Foram Protegidas**
   - Mediana de idade menor entre sobreviventes

### **💡 Lições de Machine Learning**

| Lição | Explicação |
|-------|------------|
| **Valide Sempre** | Sem validação, escolheríamos o Random Forest (overfitting) |
| **Simplicidade Vence** | Logistic Regression generalizou melhor que modelos complexos |
| **AUC > Accuracy** | SVC venceu em discriminação, não em acurácia bruta |
| **Entenda os Dados** | EDA revelou features importantes antes de modelar |

### **🔮 Reflexão Final**

> *"O Titanic não foi apenas um acidente naval. Foi um espelho da sociedade de 1912.*

*Os dados mostram claramente:*
- *✅ Riqueza salvou vidas*
- *✅ Gênero determinou destino*
- *✅ Classe social foi sentença*
- *✅ Crianças foram protegidas*

*Quando treinamos modelos com esses dados, não estamos apenas prevendo 0 ou 1. Estamos capturando padrões de **desigualdade estrutural** que custaram mais de 1.500 vidas.*

*O Titanic afundou há 114 anos. Mas as lições sobre privilégio, desigualdade e valor da vida humana continuam relevantes hoje.*

*Que este projeto seja mais que código. Seja memória."*
