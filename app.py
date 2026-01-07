import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# -------------------------------------------------
# Custom CSS for Better Look
# -------------------------------------------------
st.markdown("""
<style>
.metric-box {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
}
.big-text {
    font-size: 32px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Title
# -------------------------------------------------
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>📊 Customer Churn Prediction Dashboard</h1>", unsafe_allow_html=True)

# -------------------------------------------------
# Load Data
# -------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

df = load_data()

with st.expander(" View Dataset"):
    st.dataframe(df.head())

# -------------------------------------------------
# Preprocessing
# -------------------------------------------------
X = df.drop(["Churn", "customerID"], axis=1, errors="ignore")
y = df["Churn"].map({"Yes": 1, "No": 0})

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------------------------
# Train Model
# -------------------------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# -------------------------------------------------
# PERFORMANCE SECTION
# -------------------------------------------------
st.markdown("##  Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="metric-box">
        <p class="big-text" style="color: #2196F3;">{:.2f}%</p>
        <p>Model Accuracy</p>
    </div>
    """.format(accuracy * 100), unsafe_allow_html=True)

with col2:
    if accuracy >= 0.8:
        st.success(" Excellent Model Performance")
    elif accuracy >= 0.6:
        st.warning(" Average Model Performance")
    else:
        st.error(" Poor Model Performance")

# -------------------------------------------------
# CONFUSION MATRIX
# -------------------------------------------------
st.markdown("##  Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(
    cm,
    index=["Actual No Churn", "Actual Churn"],
    columns=["Predicted No Churn", "Predicted Churn"]
)

st.dataframe(
    cm_df.style
        .background_gradient(cmap="Greens")
        .set_properties(**{
            "font-size": "18px",
            "font-weight": "bold",
            "text-align": "center"
        })
)

tn, fp, fn, tp = cm.ravel()

col3, col4, col5, col6 = st.columns(4)

col3.metric("True Negatives", tn)
col4.metric("False Positives", fp)
col5.metric("False Negatives", fn)
col6.metric("True Positives", tp)

# -------------------------------------------------
# PREDICTION SECTION
# -------------------------------------------------
st.markdown("##  Predict Churn for New Customer")

st.info("Enter customer details below and click **Predict Churn**")

input_data = {}
for col in X.columns:
    input_data[col] = st.number_input(col, value=0.0)

input_df = pd.DataFrame([input_data])
input_scaled = scaler.transform(input_df)

if st.button(" Predict Churn"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f" **Customer is Likely to Churn**  \nProbability: **{probability:.2f}**")
    else:
        st.success(f" **Customer is Likely to Stay**  \nProbability: **{probability:.2f}**")

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center;'>Developed using Streamlit & Logistic Regression</p>",
    unsafe_allow_html=True
)
