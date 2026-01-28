# -------------------------
# 1️⃣ Imports
# -------------------------
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# -------------------------
# 2️⃣ App Title & Description
# -------------------------
st.set_page_config(page_title="Smart Loan Approval System", layout="wide")
st.title(" Smart Loan Approval System – Stacking Model")
st.markdown("""
**This system uses a Stacking Ensemble Machine Learning model to predict whether a loan
will be approved by combining multiple ML models for better decision making.**
""")

# -------------------------
# 3️⃣ Load Dataset
# -------------------------
# Replace path with your local CSV file
df = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")

# -------------------------
# 4️⃣ Data Preprocessing
# -------------------------
# Drop irrelevant column
df = df.drop(columns=["Loan_ID"])

# Encode target
df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

# Fill missing values correctly
numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

categorical_cols = df.select_dtypes(exclude=np.number).columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Encode categorical features
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
df["Married"] = df["Married"].map({"Yes": 1, "No": 0})
df["Education"] = df["Education"].map({"Graduate": 1, "Not Graduate": 0})
df["Self_Employed"] = df["Self_Employed"].map({"Yes": 1, "No": 0})
property_map = {"Urban": 2, "Semiurban": 1, "Rural": 0}
df["Property_Area"] = df["Property_Area"].map(property_map)

# Features & target
X = df.drop(columns=["Loan_Status"])
y = df["Loan_Status"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------
# 5️⃣ Sidebar Inputs
# -------------------------
st.sidebar.header(" Applicant Details")

applicant_income = st.sidebar.number_input("Applicant Income", min_value=0.0, step=1000.0)
coapplicant_income = st.sidebar.number_input("Co-Applicant Income", min_value=0.0, step=1000.0)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0.0, step=1000.0)
loan_term = st.sidebar.number_input("Loan Amount Term (Months)", min_value=1, step=1)
credit_history = st.sidebar.radio("Credit History", ["Yes", "No"])
employment = st.sidebar.selectbox("Employment Status", ["Salaried", "Self-Employed"])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Encode user inputs
credit_val = 1 if credit_history == "Yes" else 0
employment_val = 1 if employment == "Salaried" else 0
property_val = property_map[property_area]

user_data = np.array([[ 
    applicant_income,
    coapplicant_income,
    loan_amount,
    loan_term,
    credit_val,
    employment_val,
    property_val
]])

# -------------------------
# 6️⃣ Display Stacking Architecture
# -------------------------
st.subheader(" Stacking Model Architecture")
st.info("""
**Base Models Used**
• Logistic Regression  
• Decision Tree  
• Random Forest  

**Meta Model Used**
• Logistic Regression  

*Predictions from base models are combined and passed to the meta-model
to make the final decision.*
""")

# -------------------------
# 7️⃣ Train Base Models
# -------------------------
lr = LogisticRegression()
dt = DecisionTreeClassifier(max_depth=5)
rf = RandomForestClassifier(n_estimators=100, random_state=42)

lr.fit(X_train_scaled, y_train)
dt.fit(X_train_scaled, y_train)
rf.fit(X_train_scaled, y_train)

# -------------------------
# 8️⃣ Prediction Button
# -------------------------
st.markdown("###  Prediction")

if st.button("Check Loan Eligibility (Stacking Model)"):
    
    user_scaled = scaler.transform(user_data)
    
    # Base model predictions
    lr_pred = lr.predict(user_scaled)[0]
    dt_pred = dt.predict(user_scaled)[0]
    rf_pred = rf.predict(user_scaled)[0]

    # Meta-model dataset (Stacking)
    stacked_input = np.array([[lr_pred, dt_pred, rf_pred]])
    meta_model = LogisticRegression()
    meta_model.fit(
        np.column_stack([lr.predict(X_train_scaled),
                         dt.predict(X_train_scaled),
                         rf.predict(X_train_scaled)]),
        y_train
    )
    final_pred = meta_model.predict(stacked_input)[0]
    confidence = max(meta_model.predict_proba(stacked_input)[0]) * 100

    # -------------------------
    # 9️⃣ Output Section
    # -------------------------
    st.subheader(" Base Model Predictions")
    st.write(f"**Logistic Regression:** {'Approved' if lr_pred else 'Rejected'}")
    st.write(f"**Decision Tree:** {'Approved' if dt_pred else 'Rejected'}")
    st.write(f"**Random Forest:** {'Approved' if rf_pred else 'Rejected'}")

    st.subheader(" Final Stacking Decision")
    if final_pred == 1:
        st.success(" Loan Approved")
    else:
        st.error(" Loan Rejected")
    
    st.write(f" **Confidence Score:** {confidence:.2f}%")

    st.subheader(" Business Explanation")
    explanation = (
        f"Based on the applicant’s income, credit history, employment status, "
        f"and combined predictions from multiple machine learning models, "
        f"the applicant is **{'likely' if final_pred else 'unlikely'} to repay the loan**. "
        f"Therefore, the stacking model predicts **loan {'approval' if final_pred else 'rejection'}**."
    )
    st.info(explanation)
