import pickle
import numpy as np
import streamlit as st

st.set_page_config(page_title="Premium Wine Quality Prediction", page_icon="🍷", layout="centered")

# Load model and scaler
try:
    model = pickle.load(open("wine_model_streamlit.pkl", "rb"))
    scaler = pickle.load(open("scaler_streamlit.pkl", "rb"))
except:
    st.error("❌ Model or scaler file not found. Ensure the files are in the same folder as app.py")
    st.stop()

st.title("🍷 Premium Wine Quality Prediction")
st.write("Analyze wine chemistry & predict its quality")

# Input fields
fixed_acidity = st.number_input("Fixed Acidity", 0.0, 20.0, 7.4)
volatile_acidity = st.number_input("Volatile Acidity", 0.0, 2.0, 0.7)
citric_acid = st.number_input("Citric Acid", 0.0, 2.0, 0.0)
residual_sugar = st.number_input("Residual Sugar", 0.0, 20.0, 1.9)
chlorides = st.number_input("Chlorides", 0.0, 1.0, 0.076)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", 0.0, 100.0, 11.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", 0.0, 200.0, 34.0)
density = st.number_input("Density", 0.990, 1.5, 0.9978)
pH = st.number_input("pH", 2.0, 5.0, 3.51)
sulphates = st.number_input("Sulphates", 0.0, 2.0, 0.56)
alcohol = st.number_input("Alcohol (%)", 0.0, 20.0, 9.4)

# Predict button
if st.button("🔮 Predict Quality"):
    features = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                          chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                          density, pH, sulphates, alcohol]])

    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)[0]

    st.subheader("🔮 Wine Quality Prediction")
    quality_text = "Excellent Quality Wine" if prediction >= 7 else (
                   "Good Quality Wine" if prediction >= 6 else "Average Quality Wine")

    st.write(f"⚠️ {quality_text}")
    st.write(f"Predicted Score: **{prediction} / 10**")
