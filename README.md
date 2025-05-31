# 💳 Credit Default Prediction Web App

This project is a machine learning-powered web application for detecting fraudulent credit card transactions. Built using **Streamlit**, it allows users to upload a credit card transaction dataset and:

- Visualize class distribution
- Train machine learning models (Random Forest & XGBoost)
- Evaluate model performance
- Explore feature importance using SHAP (SHapley Additive exPlanations)

---

## 🚀 Features
### 📤 Dataset Upload
Users can upload a CSV file (`creditcard.csv`) containing credit card transaction data.

### 📋 Data Summary
- Displays the shape of the dataset  
- Shows the first 5 rows of the dataset  
- Provides information on the number of missing values  

### 🛠️ Feature Engineering
- Adds new features such as `Hour` (extracted from `Time`) and `Amount_log` (log-transformed transaction amount)

### 📊 Visualization
- Class distribution (Legit vs Fraud)  
- Feature correlation heatmap  
- Distribution of transaction amounts and hourly transaction frequency  

### ⚙️ Preprocessing & Resampling
- Standardization of `Amount` and `Time` columns  
- Stratified train-test split  
- Data imbalance handling using SMOTE  

### 🧠 Model Training & Evaluation
- Models used: **Random Forest** and **XGBoost**  
- Evaluation metrics: **ROC-AUC Score** and **Confusion Matrix**  
- Visualization of model performance using charts  

### 🔍 Model Interpretation with SHAP

- Displays SHAP summary plots to identify the most important features in **XGBoost** predictions

## 📁 Project Structure

├── 📓 Credit_Default_Prediction.ipynb # Jupyter Notebook (exploration, training, evaluation)
├── 📄 README.md # Project documentation
├── 📄 requirements.txt # List of required Python packages
└── 🖥️ streamlit_app.py # Main Streamlit app for deployment

## 🧪 Dataset

For the Credit Default Prediction dataset, i download the dataset from kaggle dataset by an author Laura Fink, here's the original link:
[E-Commerce Sales Forecast]([https://www.kaggle.com/code/allunia/e-commerce-sales-forecast/input?select=data.csv](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud))


## SHAP Feature Importance

This app includes global interpretability using SHAP summary plots for the XGBoost model. It helps to visualize which features most influence the fraud predictions.

## Requirements

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- imbalanced-learn
- shap
- streamlit

## 📬 Contact
[Linkedin](https://www.linkedin.com/in/carmenita-lamba-6a7555220/)

carmenitalamba17@gmail.com
