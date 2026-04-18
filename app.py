import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression

st.title("🏠 Real Estate Investment Advisor")

@st.cache_data
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/your-username/your-repo/main/small_data.csv")
    return df

@st.cache_resource
def train_models(df):
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop(['Good_Investment', 'Future_Price'], axis=1)
    y_c = df['Good_Investment']
    y_r = df['Future_Price']

    clf = LogisticRegression(max_iter=1000)
    reg = LinearRegression()

    clf.fit(X, y_c)
    reg.fit(X, y_r)

    return clf, reg, X.columns

# Load data
df = load_data()

# Train model
clf, reg, columns = train_models(df)

# USER INPUT (simple demo)
bhk = st.number_input("BHK", 1, 10)
size = st.number_input("Size (SqFt)")

if st.button("Predict"):
    input_df = pd.DataFrame([[bhk, size]], columns=['BHK', 'Size_in_SqFt'])

    # Add missing columns
    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[columns]

    pred_class = clf.predict(input_df)
    pred_price = reg.predict(input_df)

    st.success(f"Good Investment: {pred_class[0]}")
    st.success(f"Future Price: ₹ {round(pred_price[0], 2)} Lakhs")
