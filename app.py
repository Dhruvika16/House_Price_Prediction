import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# -------------------------------
# Train Model (No .pkl needed)
# -------------------------------
@st.cache_resource
def train_model():
    url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    data = pd.read_csv(url)

    X = data.drop('medv', axis=1)
    y = data['medv']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)

    return model, score


model, r2 = train_model()

# -------------------------------
# UI Design
# -------------------------------
st.set_page_config(page_title="House Price Predictor", layout="centered")

st.title("🏡 House Price Prediction App")
st.markdown("### 📊 ML Model Dashboard")
st.write("Enter house details below to predict price")

# -------------------------------
# Input Fields
# -------------------------------
crim = st.number_input("Crime Rate", 0.0, 100.0, 0.1)
zn = st.number_input("Residential Land Zone (%)", 0.0, 100.0, 25.0)
indus = st.number_input("Industrial Area (%)", 0.0, 30.0, 5.0)
chas = st.selectbox("Near River? (1 = Yes, 0 = No)", [0, 1])
nox = st.number_input("Nitric Oxide Level", 0.0, 1.0, 0.5)
rm = st.number_input("Average Rooms", 1.0, 10.0, 5.0)
age = st.number_input("Old Houses (%)", 0.0, 100.0, 50.0)
dis = st.number_input("Distance to Employment", 0.0, 15.0, 5.0)
rad = st.number_input("Highway Access Index", 1, 24, 5)
tax = st.number_input("Property Tax", 100, 800, 300)
ptratio = st.number_input("Pupil-Teacher Ratio", 10.0, 30.0, 18.0)
b = st.number_input("Population Proportion", 0.0, 400.0, 300.0)
lstat = st.number_input("Lower Status Population (%)", 0.0, 40.0, 10.0)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Price"):
    features = np.array([[crim, zn, indus, chas, nox, rm, age, dis,
                          rad, tax, ptratio, b, lstat]])

    prediction = model.predict(features)

    st.success(f"💰 Estimated Price: ${prediction[0]*1000:,.2f}")

# -------------------------------
# Model Performance
# -------------------------------
st.markdown("---")
st.subheader("📈 Model Performance")

st.write(f"R² Score: {r2:.2f}")

if r2 > 0.8:
    st.success("Model is performing well ✅")
else:
    st.warning("Model performance can be improved ⚠️")