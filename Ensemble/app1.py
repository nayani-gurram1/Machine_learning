# app_customer_segmentation.py

import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1️⃣ App Title & Description
# -----------------------------
st.title("🟢 Customer Segmentation Dashboard")
st.write("""
This system uses **K-Means Clustering** to group customers based on their purchasing behavior and similarities.
👉 Discover hidden customer groups without predefined labels.
""")

# -----------------------------
# Load dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Wholesale customers data.csv")
    # Clean column names
    df.columns = df.columns.str.strip().replace({'Delicassen':'Delicatessen'})
    return df

df = load_data()

# -----------------------------
# 2️⃣ Input Section (Sidebar)
# -----------------------------
st.sidebar.header("Clustering Controls")

# Get numeric features
numeric_features = df.select_dtypes(include='number').columns.tolist()
# Remove Channel & Region
numeric_features = [f for f in numeric_features if f not in ['Channel','Region']]

# Feature selection
selected_features = st.sidebar.multiselect(
    "Select at least 2 features for clustering",
    options=numeric_features,
    default=numeric_features[:2]
)

# Ensure at least 2 features
if len(selected_features) < 2:
    st.sidebar.warning("Please select at least 2 features.")

# Number of clusters
k = st.sidebar.slider("Select number of clusters (K)", min_value=2, max_value=10, value=3)

# Optional: Random state
random_state = st.sidebar.number_input("Random State (optional)", min_value=0, max_value=1000, value=42)

# -----------------------------
# 3️⃣ Clustering Control
# -----------------------------
if st.sidebar.button("🟦 Run Clustering") and len(selected_features) >= 2:
    
    # -----------------------------
    # 3a. Data Preparation
    # -----------------------------
    X = df[selected_features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # -----------------------------
    # 4️⃣ K-Means Clustering
    # -----------------------------
    kmeans = KMeans(n_clusters=k, random_state=random_state)
    clusters = kmeans.fit_predict(X_scaled)
    df['Cluster'] = clusters
    
    # -----------------------------
    # 4a. Visualization
    # -----------------------------
    st.subheader("Cluster Visualization")
    
    plt.figure(figsize=(8,6))
    sns.scatterplot(
        x=df[selected_features[0]],
        y=df[selected_features[1]],
        hue=df['Cluster'],
        palette='Set1',
        s=100,
        alpha=0.7
    )
    
    # Plot cluster centers (inverse transform to original scale)
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=250, marker='X', label='Centers')
    
    plt.xlabel(selected_features[0])
    plt.ylabel(selected_features[1])
    plt.title(f"K-Means Clustering (K={k})")
    plt.legend()
    st.pyplot(plt)
    
    # -----------------------------
    # 5️⃣ Cluster Summary Table
    # -----------------------------
    st.subheader("Cluster Summary")
    summary = df.groupby('Cluster')[selected_features].agg(['mean', 'count'])
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    st.dataframe(summary)
    
    # -----------------------------
    # 6️⃣ Business Interpretation
    # -----------------------------
    st.subheader("Business Insights")
    for cluster_id in range(k):
        cluster_data = df[df['Cluster']==cluster_id][selected_features]
        avg_spend = cluster_data.mean()
        high_features = avg_spend[avg_spend > avg_spend.mean()].index.tolist()
        low_features = avg_spend[avg_spend <= avg_spend.mean()].index.tolist()
        
        insight = f"🟢 Cluster {cluster_id}: "
        if high_features:
            insight += f"High spend in {', '.join(high_features)}. "
        if low_features:
            insight += f"Lower spend in {', '.join(low_features)}. "
        insight += "Customers in this cluster exhibit similar purchasing patterns."
        st.write(insight)
    
    # -----------------------------
    # 7️⃣ User Guidance / Insight Box
    # -----------------------------
    st.info("💡 Customers in the same cluster exhibit similar purchasing behaviour and can be targeted with similar business strategies.")
