# Titanic Survival Prediction

This project applies machine learning techniques to predict passenger survival on the Titanic.  
The workflow includes **exploratory data analysis (EDA)**, **data preprocessing**, **feature engineering**, and **model comparison**.

The objective is to identify the key factors that influenced survival and build predictive models capable of estimating survival probability.

---

# Dataset

The dataset contains demographic and travel information about Titanic passengers.

Main features:

- `Pclass` – Passenger class (1st, 2nd, 3rd)
- `Sex` – Passenger gender
- `Age` – Passenger age
- `Fare` – Ticket fare
- `Embarked` – Port of embarkation
- `SibSp` – Number of siblings/spouses aboard
- `Parch` – Number of parents/children aboard

Target variable:

- `Survived`
  - `0` = Did not survive
  - `1` = Survived

---

# Exploratory Data Analysis (EDA)

The exploratory analysis provided important insights about the structure of the data and the factors influencing survival.

## Dataset Balance

The dataset shows a **moderate class imbalance**, with a larger proportion of passengers who did not survive.  
This imbalance requires careful evaluation using metrics beyond simple accuracy.

---

## Gender Impact on Survival

Gender is the **strongest predictor** of survival.

Key observations:

- Women had a **significantly higher survival rate**
- Most male passengers did **not survive**
- The survival policy during the disaster followed the well-known **"women and children first" principle**

This makes `Sex` one of the most important features for classification.

---

## Passenger Class and Socioeconomic Status

Passenger class (`Pclass`) shows a strong relationship with survival probability.

Insights:

- **1st class passengers had the highest survival rate**
- **3rd class passengers had the lowest survival rate**
- Higher class passengers likely had **better cabin locations and easier access to lifeboats**

This suggests that **socioeconomic status played a major role** in survival outcomes.

---

## Age Distribution

The age distribution reveals that:

- Most passengers were between **20 and 40 years old**
- Younger passengers had **slightly higher survival rates**
- Children had relatively better survival chances compared to adults

However, age alone is **not as strong a predictor** as gender or passenger class.

---

## Fare Distribution

The `Fare` variable is highly skewed and reflects passenger class.

Observations:

- Higher fares correlate with **higher survival probability**
- This variable indirectly captures **wealth and class hierarchy**
- Extreme fare values suggest the presence of **outliers**

Because of its skewness, transformations or scaling may improve model performance.

---

## Family Structure

Family-related features (`SibSp` and `Parch`) provide additional insights:

- Passengers traveling with **small families had higher survival rates**
- Passengers traveling **alone often had lower survival probability**
- Very large families tended to have **lower survival rates**

This suggests that **family size influences evacuation dynamics**.

---

## Missing Data Analysis

Missing values were primarily found in:

- `Age`
- `Embarked`
- `Cabin` (very large amount of missing data)

Handling strategy:

- `Age` → median imputation
- `Embarked` → most frequent category
- `Cabin` → removed or transformed into derived features

These preprocessing steps help maintain dataset consistency without introducing significant bias.

---

# Data Preprocessing

A preprocessing pipeline was created to ensure consistent transformations across models.

Steps included:

- Missing value imputation
- Feature scaling for numerical variables
- Categorical encoding using one-hot encoding
- Feature engineering

Pipeline components used:

- `SimpleImputer`
- `StandardScaler`
- `OneHotEncoder`
- `ColumnTransformer`

This approach improves reproducibility and prevents data leakage.

---

# Models Evaluated

Multiple classification algorithms were tested and compared:

- Logistic Regression
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVC)

Each model was trained and evaluated using validation datasets.

---

# Model Performance

Models were evaluated using several metrics:

- Accuracy
- ROC-AUC
- Confusion Matrix
- ROC Curves

Approximate ROC-AUC results:

| Model | ROC-AUC |
|------|------|
| Logistic Regression | ~0.82 |
| Random Forest | ~0.80 |
| Gradient Boosting | ~0.83 |
| SVC | ~0.85 |

---

# Key Modeling Insights

### Best Overall Model
Support Vector Machine achieved the **highest ROC-AUC**, indicating strong classification capability.

### Overfitting Detection
Random Forest achieved extremely high training accuracy but lower validation accuracy, indicating **overfitting**.

### Gradient Boosting Stability
Gradient Boosting showed **consistent performance across training and validation**, suggesting good generalization.

### Logistic Regression Baseline
Despite its simplicity, Logistic Regression performed competitively and provides **high interpretability**.

---

# Key Insights from the Analysis

The most influential factors affecting survival appear to be:

1. **Gender**
2. **Passenger class**
3. **Ticket fare**
4. **Age**
5. **Family size**

These variables capture both **demographic characteristics and socioeconomic status**, which strongly influenced survival outcomes during the disaster.

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

# Project Structure

