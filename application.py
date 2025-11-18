import streamlit as st
import pandas as pd
import joblib as joblib


scaler = joblib.load("scaler.pkl")
kmeans = joblib.load("kmeans_model.pkl")

cluster_labels = {
    0: "Budget Customers",
    1: "Standard Shoppers",
    2: "Target Customers (High Income & Spending)",
    3: "Potential Customers (High Income, Low Spending)",
    4: "Low Income, High Spending"
}


st.title("Customer Segmentation using K-Means")
st.markdown("Enter new customer details.")


income = st.number_input("Annual Income (k$)", min_value=10, max_value=150, value=50)
spending = st.number_input("Spending Score", min_value=1, max_value=100, value=50)



if st.button("Predict Cluster"):
    new_data = pd.DataFrame([[income, spending]], columns=['Annual Income (k$)', 'Spending Score'])
    new_scaled = scaler.transform(new_data)
    cluster = kmeans.predict(new_scaled)[0]

    st.success(f"Predicted Cluster: {cluster} - {cluster_labels.get(cluster, 'Unknown')}")
