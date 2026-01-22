# app.py
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
import numpy as np

# --- Load & Prepare Dataset ---
@st.cache_data
def load_and_train_models():
    df = pd.read_csv(r"C:\Users\gurra\SVM\archive\train_u6lujuX_CVtuZ9i.csv")
    df.drop(columns=["Loan_ID"], inplace=True)

    # Fill missing numerical columns
    num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
    for col in num_cols:
        df[col].fillna(df[col].mean(), inplace=True)

    # Fill missing categorical columns
    cat_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
    for col in cat_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Encode categorical columns
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Split features and target
    X = df.drop("Loan_Status", axis=1)
    y = df["Loan_Status"]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train three SVM models
    svm_linear = SVC(kernel='linear', C=1, probability=True).fit(X_scaled, y)
    svm_poly = SVC(kernel='poly', degree=3, C=1, probability=True).fit(X_scaled, y)
    svm_rbf = SVC(kernel='rbf', C=1, gamma='scale', probability=True).fit(X_scaled, y)

    models = {
        'Linear SVM': svm_linear,
        'Polynomial SVM': svm_poly,
        'RBF SVM': svm_rbf
    }

    return scaler, label_encoders, models, X.columns.tolist()

scaler, label_encoders, models, feature_order = load_and_train_models()

# --- Streamlit App ---
st.title("Smart Loan Approval System")
st.write("This system uses Support Vector Machines to predict loan approval.")

st.sidebar.header("Applicant Details")
applicant_income = st.sidebar.number_input("Applicant Income", min_value=0)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0)
credit_history_input = st.sidebar.selectbox("Credit History", ["Yes", "No"])
self_employed_input = st.sidebar.selectbox("Employment Status", ["Yes", "No"])
property_area_input = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Model selection
kernel_choice = st.sidebar.radio(
    "Select SVM Kernel",
    ('Linear SVM', 'Polynomial SVM', 'RBF SVM')
)

# Prepare input for model
def prepare_input():
    input_dict = {
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': 0,  # optional, user can extend later
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': 360,  # default full term
        'Credit_History': 1 if credit_history_input=="Yes" else 0,
        'Gender': 0,  # default dummy value
        'Married': 0,
        'Dependents': 0,
        'Education': 1,  # assume Graduate
        'Self_Employed': 1 if self_employed_input=="Yes" else 0,
        'Property_Area': label_encoders['Property_Area'].transform([property_area_input])[0]
    }
    # Convert to dataframe and order columns as in training
    df_input = pd.DataFrame([input_dict])
    df_input = df_input[feature_order]
    # Scale
    return scaler.transform(df_input)

# Prediction
if st.sidebar.button("Check Loan Eligibility"):
    X_user = prepare_input()
    model = models[kernel_choice]
    pred = model.predict(X_user)[0]
    conf = model.predict_proba(X_user).max()

    if pred == 1:
        st.success(f" Loan Approved! (Confidence: {conf*100:.2f}%)")
    else:
        st.error(f" Loan Rejected! (Confidence: {conf*100:.2f}%)")

    st.info(f"Kernel Used: {kernel_choice}")
    explanation = ("Based on credit history and income pattern, the applicant is "
                   "likely to repay the loan." if pred==1 else
                   "unlikely to repay the loan.")
    st.write(explanation)
