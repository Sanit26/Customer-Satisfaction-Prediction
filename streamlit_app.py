# streamlit_app.py

import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.title("Customer Satisfaction Prediction App")

# Add background color to the entire page
st.markdown("""
    <style>
    .main {
        background-color: #ffe6f2;  /* Light blue */
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_data
def load_data():
    df = pd.read_csv("customer_support_tickets.csv")
    return df

# Load data
df = load_data()

# Drop rows with missing target
df = df.dropna(subset=["Customer Satisfaction Rating"])

# Encode categorical columns
cat_cols = ["Customer Gender", "Product Purchased", "Ticket Type", "Ticket Priority", "Ticket Channel"]
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Select features
features = ["Customer Age", "Customer Gender", "Ticket Type", "Ticket Priority", "Ticket Channel"]
X = df[features]
y = df["Customer Satisfaction Rating"]

# Train model
model = RandomForestRegressor()
model.fit(X, y)

# UI for input
st.sidebar.header("Enter Ticket Details")

age = st.sidebar.slider("Customer Age", 18, 80, 30)
gender = st.sidebar.selectbox("Customer Gender", encoders["Customer Gender"].classes_)
ticket_type = st.sidebar.selectbox("Ticket Type", encoders["Ticket Type"].classes_)
priority = st.sidebar.selectbox("Ticket Priority", encoders["Ticket Priority"].classes_)
channel = st.sidebar.selectbox("Ticket Channel", encoders["Ticket Channel"].classes_)

# Encode inputs
input_data = pd.DataFrame([[
    age,
    encoders["Customer Gender"].transform([gender])[0],
    encoders["Ticket Type"].transform([ticket_type])[0],
    encoders["Ticket Priority"].transform([priority])[0],
    encoders["Ticket Channel"].transform([channel])[0]
]], columns=features)

# Predict
if st.sidebar.button("Predict Satisfaction Rating"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Satisfaction Rating: {round(prediction, 2)} / 5")

# streamlit run streamlit_app.py