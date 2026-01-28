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
st.title("🎯 Smart Loan Approval System – Stacking Model")
st.markdown("""
**This system uses a Stacking Ensemble Machine Learning model to predict whether a loan
will be approved by combining multiple ML models for better decision making.**
""")

# -------------------------
# 3️⃣ Load Dataset
# -------------------------
df = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")

# -------------------------
# 4️⃣ Data Preprocessing
# -------------------------
df = df.drop(columns=["Loan_ID"])
df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

# Fix 'Dependents' column
df['Dependents'] = df['Dependents'].replace('3+', 3).astype(float)

# Fill missing numeric values
numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Fill missing categorical values
categorical_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Encode categorical features
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0}).astype(float)
df["Married"] = df["Married"].map({"Yes": 1, "No": 0}).astype(float)
df["Education"] = df["Education"].map({"Graduate": 1, "Not Graduate": 0}).astype(float)
df["Self_Employed"] = df["Self_Employed"].map({"Yes": 1, "No": 0}).astype(float)
property_map = {"Urban": 2, "Semiurban": 1, "Rural": 0}
df["Property_Area"] = df["Property_Area"].map(property_map).astype(float)

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
st.sidebar.header("📋 Applicant Details")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
married = st.sidebar.selectbox("Married", ["Yes", "No"])
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
applicant_income = st.sidebar.number_input("Applicant Income", min_value=0.0, step=1000.0)
coapplicant_income = st.sidebar.number_input("Co-Applicant Income", min_value=0.0, step=1000.0)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0.0, step=1000.0)
loan_term = st.sidebar.number_input("Loan Amount Term (Months)", min_value=1, step=1)
dependents = st.sidebar.selectbox("Dependents", ["0", "1", "2", "3+"])
credit_history = st.sidebar.radio("Credit History", ["Yes", "No"])
employment = st.sidebar.selectbox("Employment Status", ["Salaried", "Self-Employed"])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Encode user inputs
gender_val = 1 if gender == "Male" else 0
married_val = 1 if married == "Yes" else 0
education_val = 1 if education == "Graduate" else 0
dependents_val = 3.0 if dependents == "3+" else float(dependents)
credit_val = 1 if credit_history == "Yes" else 0
employment_val = 1 if employment == "Salaried" else 0
property_val = property_map[property_area]

# Create input array in the same order as X_train columns
user_data = np.array([[
    gender_val,
    married_val,
    dependents_val,
    education_val,
    employment_val,       # Self_Employed
    applicant_income,
    coapplicant_income,
    loan_amount,
    loan_term,
    credit_val,
    property_val
]]).astype(float)

# Safety check: feature number match
if user_data.shape[1] != X_train_scaled.shape[1]:
    st.error(f"Feature mismatch! Expected {X_train_scaled.shape[1]} features, got {user_data.shape[1]}")
    st.stop()

# -------------------------
# 6️⃣ Display Stacking Architecture
# -------------------------
st.subheader("🧩 Stacking Model Architecture")
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
st.markdown("### 🔘 Prediction")

if st.button("Check Loan Eligibility (Stacking Model)"):
    
    user_scaled = scaler.transform(user_data)
    
    # Base model predictions
    lr_pred = lr.predict(user_scaled)[0]
    dt_pred = dt.predict(user_scaled)[0]
    rf_pred = rf.predict(user_scaled)[0]

    # Meta-model dataset (Stacking)
    stacked_input_train = np.column_stack([
        lr.predict(X_train_scaled),
        dt.predict(X_train_scaled),
        rf.predict(X_train_scaled)
    ])
    meta_model = LogisticRegression()
    meta_model.fit(stacked_input_train, y_train)
    
    stacked_input_user = np.array([[lr_pred, dt_pred, rf_pred]])
    final_pred = meta_model.predict(stacked_input_user)[0]
    confidence = max(meta_model.predict_proba(stacked_input_user)[0]) * 100

    # -------------------------
    # 9️⃣ Output Section
    # -------------------------
    st.subheader("📊 Base Model Predictions")
    st.write(f"**Logistic Regression:** {'Approved' if lr_pred else 'Rejected'}")
    st.write(f"**Decision Tree:** {'Approved' if dt_pred else 'Rejected'}")
    st.write(f"**Random Forest:** {'Approved' if rf_pred else 'Rejected'}")

    st.subheader("🧠 Final Stacking Decision")
    if final_pred == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")
    
    st.write(f"📈 **Confidence Score:** {confidence:.2f}%")

    st.subheader("💼 Business Explanation")
    explanation = (
        f"Based on the applicant’s income, credit history, employment status, "
        f"dependents, and combined predictions from multiple machine learning models, "
        f"the applicant is **{'likely' if final_pred else 'unlikely'} to repay the loan**. "
        f"Therefore, the stacking model predicts **loan {'approval' if final_pred else 'rejection'}**."
    )
    st.info(explanation)
