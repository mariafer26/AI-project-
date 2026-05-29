# 🧠 Machine Learning / AI Project

## Team: Pablo Posada, Samuel Salazar, Maria Fernanda Alvarez

### 📅 Timeline

### 🟡 Phase 1: Proposal (Anteproyecto)
**Due:** May 5,    

---

## 1. Problem & Idea

### What problem are you trying to solve?
This project aims to predict whether a customer is likely to leave a service or company, using structured customer information such as demographics, billing, and subscription details.

### Why is it interesting or useful?
Customer churn is a major business problem because losing customers directly affects revenue and long-term growth. If a company can identify customers with high churn risk early, it can take preventive actions such as personalized offers, support, or retention strategies.

### Task type
**Classification**

---

## 2. Proposed Approach

### What methods or models do you plan to use?
We plan to start with traditional machine learning models for tabular data, such as:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting / XGBoost (if feasible)

In addition, we plan to incorporate an **AI agent** that takes the model prediction and customer profile as input, then generates a natural language explanation of the churn risk and possible retention actions.

### Brief explanation of why it makes sense
This is a coherent approach because churn prediction is a binary classification problem. Models such as Logistic Regression and Random Forest are well suited for structured tabular datasets and provide a strong baseline.  
The AI agent adds an explainability layer, making the prediction easier to interpret and more useful for business or customer support scenarios.

> ❗ It does not need to be correct, but it must be coherent.

---

## 3. Data

### Dataset source
We plan to use the **Telco Customer Churn dataset**, available on platforms such as Kaggle and IBM sample datasets.

### What does it contain?
The dataset contains customer information related to demographics, account configuration, services, and billing.

#### Features (X)
Possible features include:
- gender
- SeniorCitizen
- Partner
- Dependents
- tenure
- PhoneService
- InternetService
- OnlineSecurity
- Contract
- PaymentMethod
- MonthlyCharges
- TotalCharges

#### Target variable (y)
- **Churn** (Yes / No)

### Approximate size
The dataset contains approximately **7,000 rows** and around **20 features**, depending on the selected version.

---

## 4. Initial Exploration (EDA)

### What did you observe?
In the initial exploration, we expect to analyze:

- The distribution of the target variable (**Churn**)
- Numerical variables such as **tenure**, **MonthlyCharges**, and **TotalCharges**
- Possible class imbalance between churn and non-churn customers
- Potential missing values, formatting issues, or anomalies
- Relationships between churn and relevant categorical variables such as **Contract** or **PaymentMethod**

### Possible observations
Some expected patterns are:

- The dataset may show **class imbalance**, with fewer churn cases than non-churn cases
- Customers with shorter tenure may be more likely to churn
- Customers with month-to-month contracts may present higher churn risk
- Some numeric features may contain outliers or values requiring cleaning

### Plots to include
We plan to include at least **1–2 plots**, such as:

1. **Count plot of the churn variable**
2. **Histogram or boxplot of tenure grouped by churn**
3. *(Optional)* Boxplot of MonthlyCharges by churn

---

## 5. Question & Objective

### Question
Can we accurately predict customer churn from structured customer data, and can an AI agent explain the prediction in a useful and understandable way?

### Objective
To build a machine learning classification model that predicts customer churn and complement it with an AI agent capable of explaining the prediction and suggesting possible retention actions.

---

## 6. Evaluation

### How will you measure performance?
We plan to evaluate the model using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

### Brief justification
Accuracy is useful as a general metric, but it may be misleading if the dataset is imbalanced.  
For that reason, **Precision**, **Recall**, and especially **F1-score** are important to better evaluate performance on churn cases.  
**ROC-AUC** will also help measure how well the model separates churn and non-churn customers across different thresholds.

---

## 7. References

1. IBM Sample Data Sets – Telco Customer Churn
2. Scikit-learn Documentation – Classification Models
3. Aurélien Géron, *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*
4. Kaggle – Telco Customer Churn Dataset