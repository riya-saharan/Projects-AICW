import streamlit as st
import pickle
import numpy as np

# --------------------------------
# PAGE CONFIGURATION
# --------------------------------
st.set_page_config(
    page_title="Premium Wine Quality Predictor",
    page_icon="🍷",
    layout="centered"
)

# --------------------------------
# AESTHETIC CSS
# --------------------------------
st.markdown("""
<style>

body {
    background-color: #f5f1f8 !important;
}

/* Main Card */
.main-card {
    background: linear-gradient(145deg, #ffffff, #f3e9ff);
    padding: 30px;
    border-radius: 18px;
    box-shadow: 0px 4px 20px rgba(120, 0, 150, 0.15);
}

/* Title Style */
.title-text {
    font-size: 42px;
    font-weight: 800;
    text-align: center;
    color: #4b004f;
}

/* Subheading */
.sub-text {
    font-size: 20px;
    text-align: center;
    color: #6b006e;
    margin-top: -10px;
}

/* Result Box */
.result-card {
    padding: 22px;
    border-radius: 15px;
    margin-top: 20px;
    text-align: center;
    font-size: 20px;
    font-weight: 600;
}

.low {
    background-color: #ffe5e5;
    color: #a30000;
}

.avg {
    background-color: #fff4cc;
    color: #936c00;
}

.high {
    background-color: #e6ffe6;
    color: #005f00;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------
# LOAD MODEL + SCALER
# --------------------------------
try:
    with open("wine_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    st.error("❌ Model or scaler file not found. Upload 'wine_model.pkl' and 'scaler.pkl'.")
    st.stop()

# --------------------------------
# HEADER
# --------------------------------
st.markdown("<h1 class='title-text'>🍇 Premium Wine Quality Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-text'>AI-powered analysis of your wine's chemical composition</p>", unsafe_allow_html=True)

st.markdown("<div class='main-card'>", unsafe_allow_html=True)

# --------------------------------
# INPUT FIELDS
# --------------------------------
st.subheader("🔬 Enter Wine Chemical Properties")

fixed_acidity = st.slider("Fixed Acidity", 4.0, 16.0, 8.0)
volatile_acidity = st.slider("Volatile Acidity", 0.1, 1.5, 0.5)
citric_acid = st.slider("Citric Acid", 0.0, 1.0, 0.3)
residual_sugar = st.slider("Residual Sugar", 0.5, 20.0, 2.5)
chlorides = st.slider("Chlorides", 0.01, 0.2, 0.08)
free_sulfur = st.slider("Free Sulfur Dioxide", 1, 100, 20)
total_sulfur = st.slider("Total Sulfur Dioxide", 1, 250, 100)
density = st.slider("Density", 0.9900, 1.0050, 0.9960)
ph = st.slider("pH Level", 2.5, 4.5, 3.3)
sulphates = st.slider("Sulphates", 0.2, 2.0, 0.7)
alcohol = st.slider("Alcohol %", 8.0, 15.0, 10.0)

# Collect input
input_data = np.array([[
    fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
    free_sulfur, total_sulfur, density, ph, sulphates, alcohol
]])

# --------------------------------
# PREDICT BUTTON
# --------------------------------
if st.button("Predict Quality 🍷"):
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]
    confidence = np.max(model.predict_proba(scaled_data)) * 100 if hasattr(model, "predict_proba") else 0

    # Determine quality label
    if prediction <= 4:
        label = "❌ Low Quality Wine"
        css_class = "low"
    elif prediction in [5, 6]:
        label = "⚠️ Average Quality Wine"
        css_class = "avg"
    else:
        label = "✅ Premium Quality Wine"
        css_class = "high"

    # Show result
    st.markdown(
        f"<div class='result-card {css_class}'>{label}<br><br>"
        f"Score: <b>{prediction} / 10</b><br>"
        f"Confidence: <b>{confidence:.2f}%</b></div>",
        unsafe_allow_html=True
    )

    st.subheader("🔎 Interpretation")
    st.write(
        "- Higher alcohol % improves quality.\n"
        "- High volatile acidity lowers quality.\n"
        "- Balanced sulphates & pH improve taste.\n"
        "- Density indicates fermentation quality."
    )

st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.caption("Crafted with ❤️ by Riya Saharan")
