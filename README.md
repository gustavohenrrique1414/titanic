## 📌 Projeto de Machine Learning – Previsão de Sobrevivência no Titanic

Este projeto apresenta uma análise completa de dados e construção de modelos de Machine Learning utilizando o clássico dataset do Titanic. O trabalho abrange desde a Análise Exploratória de Dados (EDA), passando pelo pré-processamento manual dos dados, até a implementação e avaliação de diferentes algoritmos de classificação.

Durante a etapa de EDA, foram identificados padrões importantes relacionados à sobrevivência dos passageiros, destacando a influência de variáveis socioeconômicas como classe (*Pclass*) e tarifa (*Fare*), além de fatores demográficos como sexo (*Sex*). Também foram tratados aspectos como distribuição dos dados, presença de outliers e correlações entre variáveis.

O pré-processamento foi realizado de forma manual, seguindo boas práticas para evitar *data leakage*, com aplicação de imputação de valores ausentes, normalização/padronização e codificação de variáveis categóricas. Essa abordagem garantiu um pipeline consistente e reprodutível.

Foram treinados e avaliados diferentes modelos de classificação, incluindo Regressão Logística, Random Forest, Gradient Boosting e SVM. A avaliação foi conduzida com base em métricas como acurácia, precision, recall, F1-score e AUC-ROC, permitindo uma análise detalhada do desempenho e da capacidade de generalização dos modelos.

Os resultados indicaram que modelos baseados em ensemble, como Gradient Boosting, apresentaram melhor desempenho geral, enquanto modelos mais simples demonstraram maior estabilidade. Também foi possível identificar desafios relacionados ao desbalanceamento das classes, especialmente na identificação de sobreviventes.

---

🎓 **Contexto Acadêmico**

Este projeto foi desenvolvido como atividade final da disciplina **Introdução ao Machine Learning**, na qual participei como **aluno especial de mestrado**, cursando a disciplina de forma isolada.

---

O projeto demonstra a aplicação prática de técnicas fundamentais de ciência de dados e aprendizado de máquina, incluindo análise exploratória, engenharia de atributos, pré-processamento e avaliação de modelos, seguindo boas práticas utilizadas em cenários reais.
