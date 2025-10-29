# 💳 Credit Risk Prediction using Logistic Regression

## 📊 Overview  
This project focuses on predicting **credit risk** — identifying whether a loan applicant is likely to default or not — using **Logistic Regression**.  
It demonstrates a complete **machine learning pipeline**, from data preprocessing and exploratory analysis to model optimization and evaluation.

---

## 🧠 Objectives  
- Analyze and preprocess customer financial and demographic data  
- Build a predictive model to classify loan default risk  
- Optimize the model using **Stratified K-Fold Cross Validation** and **GridSearchCV**  
- Evaluate model performance using multiple metrics and visualize results  

---

## 🗂️ Dataset Information  

| Feature | Description |
|----------|-------------|
| person_age | Age of the individual |
| person_income | Annual income |
| person_home_ownership | Home ownership type |
| person_emp_length | Employment length (in years) |
| loan_intent | Purpose of the loan |
| loan_grade | Loan grade assigned |
| loan_amnt | Loan amount |
| loan_int_rate | Interest rate |
| loan_status | **Target variable — 0: Non-default, 1: Default** |
| loan_percent_income | Loan amount as a percentage of income |
| cb_person_default_on_file | Historical default record |
| cb_person_cred_hist_length | Credit history length |

---

## ⚙️ Tech Stack  

- **Python 🐍**  
- **Pandas, NumPy, Matplotlib, Seaborn** → Data analysis & visualization  
- **Scikit-learn** → Modeling, preprocessing & evaluation  
- **Joblib** → Model persistence  

---

## 🧩 Model Workflow  

### 1️⃣ Data Cleaning & Preprocessing  
- Handled missing values and outliers  
- Encoded categorical features using `OneHotEncoder`  
- Scaled numerical variables using `StandardScaler`  

### 2️⃣ Exploratory Data Analysis (EDA)  
- Visualized class imbalance  
- Explored feature distributions and correlations  
- Interpreted patterns for credit risk prediction  

### 3️⃣ Model Development  
- Implemented **Logistic Regression**  
- Tuned hyperparameters (`C`, `penalty`, `solver`) using **GridSearchCV**  
- Used **Stratified K-Fold** for balanced validation  

### 4️⃣ Evaluation Metrics  

| Metric | Value |
|--------|-------|
| Accuracy | 0.88 |
| Precision (Defaulters) | 0.79 |
| Recall (Defaulters) | 0.58 |
| ROC-AUC | 0.88 |

### 5️⃣ Model Saving  
Exported the final model using **Joblib** for deployment readiness:

```python
import joblib

# Save the trained model
joblib.dump(model, 'credit_risk_model.pkl')
```
## 📈 Results & Insights  

- The **Logistic Regression model** shows strong discrimination between risky and non-risky borrowers.  
- **Loan grade**, **interest rate**, and **loan percent income** were key predictors.  
- Balanced performance achieved across metrics with **high interpretability**.  

---

## 🚀 Future Improvements  

- Experiment with **XGBoost**, **Random Forest**, or **SMOTE** for imbalance handling.  
- Deploy the model via **Streamlit** or **FastAPI** for real-time credit scoring.  
- Add explainability features using **SHAP** or **LIME**.  

---

## 📚 Learning Outcome  

This project highlights:  
✔️ End-to-end **Machine Learning workflow** design  
✔️ **Data-driven** credit risk evaluation  
✔️ Application of **Logistic Regression** in finance  

---

## 👨‍💻 Author  

**Vasu Agrawal**  
📍 *B.Tech Computer Engineering | NMIMS Shirpur*  
🔗 [LinkedIn](https://www.linkedin.com/in/vasu-agrawal20/) • [GitHub](https://github.com/vasuag09)

# To load the saved model later
model = joblib.load('credit_risk_model.pkl')
