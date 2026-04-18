import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression

st.title("🏠 Real Estate Investment Advisor")

# -----------------------------
# LOAD SMALL DATA (INLINE SAMPLE)
# -----------------------------
@st.cache_data
def load_data():
    # Small synthetic dataset (no GitHub, no CSV issues)
    data = {
        "BHK": [1,2,3,2,3,4,2,3,1,4],
        "Size_in_SqFt": [500,800,1200,900,1500,2000,850,1300,600,1800],
        "Price_in_Lakhs": [20,35,60,40,75,120,38,70,25,110]
    }
    df = pd.DataFrame(data)

    # Create targets
    median_price = df['Price_in_Lakhs'].median()

    df['Good_Investment'] = np.where(
        (df['Price_in_Lakhs'] <= median_price) &
        (df['BHK'] >= 2),
        1, 0
    )

    r = 0.08
    t = 5
    df['Future_Price'] = df['Price_in_Lakhs'] * ((1 + r) ** t)

    return df


# -----------------------------
# TRAIN MODEL
# -----------------------------
@st.cache_resource
def train_model(df):
    X = df[['BHK', 'Size_in_SqFt']]
    y_c = df['Good_Investment']
    y_r = df['Future_Price']

    clf = LogisticRegression()
    reg = LinearRegression()

    clf.fit(X, y_c)
    reg.fit(X, y_r)

    return clf, reg


df = load_data()
clf, reg = train_model(df)

# -----------------------------
# USER INPUT
# -----------------------------
bhk = st.slider("BHK", 1, 5, 2)
size = st.slider("Size (SqFt)", 400, 2500, 800)

if st.button("Predict"):
    input_data = np.array([[bhk, size]])

    pred_class = clf.predict(input_data)[0]
    pred_price = reg.predict(input_data)[0]

    st.success(f"Good Investment: {'Yes' if pred_class==1 else 'No'}")
    st.success(f"Future Price (5 yrs): ₹ {round(pred_price,2)} Lakhs")
