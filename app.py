import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("🏠 Real Estate Investment Advisor")

# Load models
clf = joblib.load("classification_model.pkl")
reg = joblib.load("regression_model.pkl")

# Inputs
bhk = st.number_input("BHK", 1, 10)
size = st.number_input("Size (SqFt)")

if st.button("Predict"):
    data = np.array([[bhk, size]])

    pred_class = clf.predict(data)
    pred_price = reg.predict(data)

    st.success(f"Good Investment: {pred_class[0]}")
    st.success(f"Future Price: {pred_price[0]}")
