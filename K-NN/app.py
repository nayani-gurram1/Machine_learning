# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# -----------------------------
# Load dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\gurra\K-NN\credit_risk_dataset.csv")
    
    # Impute missing numerical values
    num_imputer = SimpleImputer(strategy="median")
    df[['person_emp_length', 'loan_int_rate']] = num_imputer.fit_transform(
        df[['person_emp_length', 'loan_int_rate']]
    )
    
    # Drop other categorical columns to simplify (use only numeric features + credit history)
    df = df[['person_age','person_income','loan_amnt','cb_person_cred_hist_length','loan_status']]
    return df

df = load_data()

# Features and target
X = df.drop('loan_status', axis=1)
y = df['loan_status']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# Streamlit Header
# -----------------------------
st.set_page_config(page_title="Customer Risk Prediction System (KNN)", layout="centered")
st.title("Customer Risk Prediction System (KNN)")
st.write("This system predicts customer risk by comparing them with similar customers.")

# -----------------------------
# Sidebar - User Inputs
# -----------------------------
st.sidebar.header("Input Customer Data")

age = st.sidebar.slider("Age", int(df['person_age'].min()), int(df['person_age'].max()), 30)
income = st.sidebar.number_input("Annual Income ($)", min_value=0, value=50000, step=1000)
loan_amount = st.sidebar.number_input("Loan Amount ($)", min_value=0, value=10000, step=500)
credit_history = st.sidebar.selectbox("Credit History", ["No", "Yes"])
k_value = st.sidebar.slider("Number of Neighbors (K)", 1, 15, 5)

# Map credit history to numeric
credit_num = 1 if credit_history == "Yes" else 0

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Customer Risk"):

    # Prepare input as dataframe
    input_df = pd.DataFrame([[age, income, loan_amount, credit_num]], columns=X.columns)
    
    # Scale input
    input_scaled = scaler.transform(input_df)
    
    # Train KNN
    knn = KNeighborsClassifier(n_neighbors=k_value)
    knn.fit(X_scaled, y)
    
    # Predict
    prediction = knn.predict(input_scaled)[0]
    
    # Neighbors explanation
    distances, indices = knn.kneighbors(input_scaled)
    neighbor_classes = y.iloc[indices[0]]
    majority_class = neighbor_classes.mode()[0]
    
    # -----------------------------
    # Display Prediction
    # -----------------------------
    if prediction == 1:
        st.markdown("<h2 style='text-align:center; color:red;'>🔴 High Risk Customer</h2>", unsafe_allow_html=True)
    else:
        st.markdown("<h2 style='text-align:center; color:green;'>🟢 Low Risk Customer</h2>", unsafe_allow_html=True)
    
    # Nearest neighbors explanation
    st.subheader("Nearest Neighbors Explanation")
    st.write(f"Number of neighbors considered: {k_value}")
    st.write(f"Majority class among neighbors: {'High Risk' if majority_class==1 else 'Low Risk'}")
    
    # Optional nearest neighbors table
    st.write("Nearest Customers in Dataset:")
    st.dataframe(df.iloc[indices[0]])
    
    # Business insight
    st.subheader("Business Insight")
    st.write("This decision is based on similarity with nearby customers in feature space. "
             "Customers with similar age, income, loan amount, and credit history influence the risk prediction.")
