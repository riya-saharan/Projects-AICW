import streamlit as st
import numpy as np
import pickle

# ---------------------- Load Model & Scaler ------------------------
model = pickle.load(open("wine_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ---------------------- App UI Design ------------------------------
st.set_page_config(page_title="Premium Wine Quality Predictor", page_icon="🍷", layout="wide")

st.markdown(
    """
    <h1 style='text-align:center; color:#8B0000;'>🍷 Premium Wine Quality Prediction</h1>
    <h4 style='text-align:center; color:#444;'>Analyze your wine features and predict its quality instantly</h4>
    <br>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.title("ℹ️ About This App")
    st.write("""
    This app predicts the **Wine Quality Score** using machine learning.

    **Model:** RandomForestClassifier  
    **Dataset:** Wine Quality (UCI Repository)  
    **Developer:** Riya Saharan  
    """)
    st.write("Made with ❤️ using Streamlit")

# ---------------------- Input Section ------------------------------
st.subheader("🔍 Enter Wine Chemical Properties")

col1, col2, col3 = st.columns(3)

with col1:
    fixed_acidity = st.number_input("Fixed Acidity", 0.0, 20.0, 7.4)
    volatile_acidity = st.number_input("Volatile Acidity", 0.0, 2.0, 0.70)
    citric_acid = st.number_input("Citric Acid", 0.0, 2.0, 0.00)

with col2:
    residual_sugar = st.number_input("Residual Sugar", 0.0, 20.0, 1.9)
    chlorides = st.number_input("Chlorides", 0.0, 1.0, 0.076)
    free_sulfur = st.number_input("Free Sulfur Dioxide", 0.0, 100.0, 11.0)

with col3:
    total_sulfur = st.number_input("Total Sulfur Dioxide", 0.0, 300.0, 34.0)
    density = st.number_input("Density", 0.0, 2.0, 0.9978)
    pH = st.number_input("pH Level", 0.0, 14.0, 3.51)
    sulphates = st.number_input("Sulphates", 0.0, 2.0, 0.56)
    alcohol = st.number_input("Alcohol %", 0.0, 20.0, 9.4)

# Prepare input
input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                        chlorides, free_sulfur, total_sulfur, density, pH, sulphates, alcohol]])

scaled = scaler.transform(input_data)

# ---------------------- Predict Button ------------------------------
if st.button("✨ Predict Quality", use_container_width=True):
    with st.spinner("Analyzing your wine... 🍷"):
        pred = model.predict(scaled)[0]
        proba = model.predict_proba(scaled)[0][pred]

    # ---------------------- Result Card ------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📌 Prediction Results")

    # Define Quality Labels
    if pred <= 4:
        quality_text = "❌ Poor Quality Wine"
        color = "#B22222"
        tips = "High acidity or low alcohol might be affecting the taste."
    elif pred == 5:
        quality_text = "⚠️ Average Quality Wine"
        color = "#DAA520"
        tips = "Balanced but lacks richness. Maybe adjust fermentation."
    else:
        quality_text = "✅ Premium Quality Wine"
        color = "#228B22"
        tips = "Smooth profile with strong aroma and balance!"

    st.markdown(
        f"""
        <div style="padding:20px; border-radius:12px; background-color:{color}; color:white;">
            <h2 style='text-align:center;'>{quality_text}</h2>
            <h3 style='text-align:center;'>Predicted Score: {pred}/10</h3>
            <h4 style='text-align:center;'>Confidence: {proba*100:.2f}%</h4>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div style="margin-top:20px; padding:15px; background:#f9f5f0; border-radius:10px;">
            <h4>🔎 Interpretation:</h4>
            <p>{tips}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
