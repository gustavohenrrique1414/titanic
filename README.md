# Titanic Survival Prediction

This project aims to predict passenger survival on the Titanic using machine learning models.  
The workflow includes **exploratory data analysis (EDA)**, **data preprocessing**, **model training**, and **performance evaluation**.

---

# Dataset

The dataset contains demographic and travel information about Titanic passengers.

Main features include:

- `Pclass` – Passenger class (1st, 2nd, 3rd)
- `Sex` – Gender
- `Age` – Passenger age
- `Fare` – Ticket price
- `Embarked` – Port of embarkation
- `SibSp` – Number of siblings/spouses aboard
- `Parch` – Number of parents/children aboard

Target variable:

- `Survived`
  - `0` = Did not survive
  - `1` = Survived

---

# Exploratory Data Analysis (EDA)

The exploratory analysis revealed several important patterns:

### Class Distribution
The dataset is **moderately imbalanced**, with more passengers who did not survive.

### Survival by Gender
Gender is one of the strongest predictors:
- **Women had a much higher survival rate**
- **Men had a significantly lower survival rate**

### Survival by Passenger Class
Passengers in **1st class had the highest survival probability**, while **3rd class passengers had the lowest**.

### Age Distribution
Most passengers were between **20 and 40 years old**. Younger passengers had slightly higher survival rates.

### Fare Distribution
Higher fares generally corresponded to **higher survival probability**, likely reflecting passenger class.

### Missing Values
Missing values were primarily found in:
- `Age`
- `Embarked`

These were handled using **imputation strategies** during preprocessing.

---

# Data Preprocessing

The preprocessing pipeline included:

- Missing value imputation (`SimpleImputer`)
- Numerical feature scaling (`StandardScaler`)
- Categorical encoding (`OneHotEncoder` / `OrdinalEncoder`)
- Feature engineering (family size, categorical transformations)

A structured **pipeline approach** was used to ensure reproducibility.

---

# Models Evaluated

Several machine learning models were trained and compared:

- Logistic Regression
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVC)

Models were evaluated using validation datasets and multiple performance metrics.

---

# Model Performance

The models were compared using:

- Accuracy
- ROC-AUC
- Confusion Matrix
- ROC Curves

### Key Observations

- **SVC achieved the highest ROC-AUC score**
- **Gradient Boosting performed consistently well**
- **Random Forest showed signs of overfitting**, with very high training accuracy but lower validation performance
- Logistic Regression provided a strong and interpretable baseline

Approximate ROC-AUC scores:

| Model | ROC-AUC |
|------|------|
| Logistic Regression | ~0.82 |
| Random Forest | ~0.80 |
| Gradient Boosting | ~0.83 |
| SVC | ~0.85 |

---

# Evaluation Metrics

Performance was assessed using:

- **Accuracy**
- **ROC-AUC**
- **Precision**
- **Recall**
- **Confusion Matrix**

These metrics allow a better understanding of classification performance beyond accuracy alone.

---

# Key Insights

The analysis suggests that the most important factors influencing survival include:

- Passenger **gender**
- Passenger **class**
- **Fare paid**
- **Age**

Women and passengers in higher classes had a significantly greater probability of survival.

---

# Future Improvements

Potential improvements to the project include:

- Hyperparameter optimization using **cross-validation**
- Model calibration for better probability estimates
- Advanced ensemble methods such as **stacking**
- Feature importance analysis using **SHAP values**

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

