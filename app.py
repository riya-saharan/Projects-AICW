import streamlit as st
import numpy as np
import pickle

# ---------------------- Load Model & Scaler -------------------------
model = pickle.load(open("wine_model_streamlit.pkl", "rb"))
scaler = pickle.load(open("scaler_streamlit.pkl", "rb"))

# ---------------------- Page Setup ---------------------------------
st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷", layout="wide")

st.markdown(
    """
    <h1 style='text-align:center; color:#8B0000;'>🍷 Premium Wine Quality Prediction</h1>
    <h4 style='text-align:center; color:#444;'>Analyze wine chemistry & predict its quality</h4>
    <hr>
    """,
    unsafe_allow_html=True
)

# ---------------------- Sidebar Inputs ------------------------------
with st.sidebar:
    st.title("🔧 Wine Properties")

    fixed_acidity = st.slider("Fixed Acidity", 0.0, 20.0, 7.4)
    volatile_acidity = st.slider("Volatile Acidity", 0.0, 2.0, 0.70)
    citric_acid = st.slider("Citric Acid", 0.0, 2.0, 0.30)
    residual_sugar = st.slider("Residual Sugar", 0.0, 20.0, 1.9)
    chlorides = st.slider("Chlorides", 0.0, 1.0, 0.076)
    free_sulfur = st.slider("Free Sulfur Dioxide", 0.0, 100.0, 11.0)
    total_sulfur = st.slider("Total Sulfur Dioxide", 0.0, 300.0, 34.0)
    density = st.slider("Density", 0.9900, 1.0100, 0.9978)
    pH = st.slider("pH Level", 0.0, 14.0, 3.51)
    sulphates = st.slider("Sulphates", 0.0, 2.0, 0.56)
    alcohol = st.slider("Alcohol %", 0.0, 20.0, 9.4)

    st.markdown("---")
    st.info("📌 Adjust the sliders to input wine chemical values.")

# --------------- Prepare the input for prediction -------------------
input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                        chlorides, free_sulfur, total_sulfur, density, pH, sulphates, alcohol]])

scaled = scaler.transform(input_data)

# ---------------------- Prediction Section --------------------------
st.subheader("🔮 Wine Quality Prediction")

if st.button("✨ Predict Quality", use_container_width=True):
    
    pred = model.predict(scaled)[0]
    proba = model.predict_proba(scaled)[0][pred]

    # Quality Labels
    if pred <= 4:
        quality = "❌ Poor Quality Wine"
        color = "#B22222"
        desc = "High acidity or low alcohol may be affecting the taste."
    elif pred <= 6:
        quality = "⚠️ Average Quality Wine"
        color = "#DAA520"
        desc = "Moderate flavor profile. Could be improved."
    else:
        quality = "✅ Premium Quality Wine"
        color = "#228B22"
        desc = "Balanced, smooth & aromatic — high-quality wine!"

    # Result Card
    st.markdown(
        f"""
        <div style="padding:25px; border-radius:12px; background-color:{color}; color:white;">
            <h2 style='text-align:center;'>{quality}</h2>
            <h3 style='text-align:center;'>Predicted Score: {pred} / 10</h3>
            <h4 style='text-align:center;'>Confidence: {proba*100:.2f}%</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Explanation Box
    st.markdown(
        f"""
        <div style="margin-top:20px; padding:20px; background:#f4f4f4; border-radius:10px;">
            <h4>🔎 Interpretation:</h4>
            <p style="font-size:16px;">{desc}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

else:
    st.info("💡 Click the Predict button to see wine quality results!")

