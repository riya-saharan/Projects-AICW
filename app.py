import streamlit as st
import pickle
import numpy as np

# -----------------------------
# Load model + scaler safely
# -----------------------------
try:
    with open("wine_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    st.error("❌ Model or scaler file not found. Ensure 'wine_model.pkl' and 'scaler.pkl' are in the same folder as app.py.")
    st.stop()

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="Premium Wine Quality Prediction",
    page_icon="🍷",
    layout="centered"
)

st.title("🍇 Premium Wine Quality Prediction")
st.write("Analyze wine chemistry & predict its quality accurately.")

st.markdown("---")

st.subheader("🔬 Enter Wine Chemical Properties")

# -----------------------------
# Input sliders
# -----------------------------
fixed_acidity = st.number_input("Fixed Acidity", 4.0, 16.0, 8.0)
volatile_acidity = st.number_input("Volatile Acidity", 0.1, 1.5, 0.5)
citric_acid = st.number_input("Citric Acid", 0.0, 1.0, 0.3)
residual_sugar = st.number_input("Residual Sugar", 0.5, 20.0, 2.5)
chlorides = st.number_input("Chlorides", 0.01, 0.2, 0.08)
free_sulfur = st.number_input("Free Sulfur Dioxide", 1, 100, 20)
total_sulfur = st.number_input("Total Sulfur Dioxide", 1, 250, 100)
density = st.number_input("Density", 0.9900, 1.0050, 0.9960)
ph = st.number_input("pH Level", 2.5, 4.5, 3.3)
sulphates = st.number_input("Sulphates", 0.2, 2.0, 0.7)
alcohol = st.number_input("Alcohol %", 8.0, 15.0, 10.0)

# Collect input
input_data = np.array([[
    fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
    free_sulfur, total_sulfur, density, ph, sulphates, alcohol
]])

# -----------------------------
# BUTTON → Predict
# -----------------------------
if st.button("Predict Wine Quality 🍷"):
    
    # Scale input
    scaled_data = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(scaled_data)[0]
    confidence = np.max(model.predict_proba(scaled_data)) * 100 if hasattr(model, "predict_proba") else 0
    
    # -----------------------------
    # Output Section
    # -----------------------------
    st.markdown("---")
    st.subheader("🔮 Prediction Result")

    # Label logic
    if prediction <= 4:
        status = "❌ Low Quality Wine"
    elif prediction == 5 or prediction == 6:
        status = "⚠️ Average Quality Wine"
    else:
        status = "✅ Premium Quality Wine"

    # Display result
    st.markdown(f"### {status}")
    st.markdown(f"**Predicted Score:** `{prediction} / 10`")
    st.markdown(f"**Confidence:** `{confidence:.2f}%`")

    # Interpretation
    st.markdown("### 🔎 Interpretation")
    st.write(
        "- Higher alcohol % increases quality.\n"
        "- High volatile acidity decreases quality.\n"
        "- Balanced sulphates & acidity improve taste.\n"
        "- Clean density & pH indicate better fermentation."
    )

st.markdown("---")
st.caption("Built with ❤️ by Riya Saharan")
