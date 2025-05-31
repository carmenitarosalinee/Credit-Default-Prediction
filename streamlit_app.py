# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import shap

st.set_page_config(page_title="Credit Default Prediction", layout="wide")

st.title("📊 Credit Default Prediction")
st.markdown("Visualization and prediction using machine learning on credit transaction data.")

# Upload CSV
uploaded_file = st.file_uploader("Upload dataset (CSV)", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    st.subheader("1. Data Summary")
    st.write("Shape:", df.shape)
    st.write(df.head())

    st.write("Missing Values:")
    st.write(df.isnull().sum())

    # Feature engineering
    df['Hour'] = df['Time'] // 3600 % 24
    df['Amount_log'] = np.log1p(df['Amount'])

    # Visualisasi Distribusi Kelas
    st.subheader("2. Class Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='Class', data=df, ax=ax)
    st.pyplot(fig)

    # Preprocessing
    X = df.drop('Class', axis=1)
    y = df['Class']
    scaler = StandardScaler()
    X[['Amount', 'Time']] = scaler.fit_transform(X[['Amount', 'Time']])
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Model Training
    st.subheader("3. Model Training & Evaluation")

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_res, y_train_res)
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:,1]

    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb.fit(X_train_res, y_train_res)
    y_pred_xgb = xgb.predict(X_test)
    y_prob_xgb = xgb.predict_proba(X_test)[:,1]

    st.write("**ROC-AUC Random Forest:**", round(roc_auc_score(y_test, y_prob_rf), 4))
    st.write("**ROC-AUC XGBoost:**", round(roc_auc_score(y_test, y_prob_xgb), 4))

    # SHAP Explanation
    st.subheader("4. SHAP Feature Importance (XGBoost)")
    explainer = shap.Explainer(xgb, X_train)
    shap_values = explainer(X_train)

    shap.summary_plot(shap_values, X_train, plot_type="bar")
    st.pyplot(bbox_inches='tight')
else:
    st.info("Please upload the `creditcard.csv` file to get started.")
