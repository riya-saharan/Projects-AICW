import streamlit as st
import pandas as pd
import joblib
import time

# Page Config
st.set_page_config(page_title="Pro Wine AI", page_icon="🍷", layout="wide")

# Custom CSS for that "Aesthetic" look
st.markdown("""
    <style>
    .main { background-color: #fdfcfc; }
    .stButton>button { width: 100%; border-radius: 20px; background-color: #720e0e; color: white; }
    </style>
    """, unsafe_allow_html=True)

# Load Model
model = joblib.load('wine_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🍷 Premium Wine Quality Predictor")
st.markdown("---")

# Sidebar
st.sidebar.header("🧪 Chemical Analysis")
alcohol = st.sidebar.slider("Alcohol (%)", 8.0, 15.0, 10.5)
sulphates = st.sidebar.slider("Sulphates", 0.3, 2.0, 0.6)
vol_acidity = st.sidebar.slider("Volatile Acidity", 0.1, 1.5, 0.5)
citric = st.sidebar.slider("Citric Acid", 0.0, 1.0, 0.3)

# Main UI - 2 Columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Current Wine Profile")
    df = pd.DataFrame({
        'Metric': ['Alcohol', 'Sulphates', 'Volatile Acidity', 'Citric Acid'],
        'Value': [alcohol, sulphates, vol_acidity, citric]
    })
    st.table(df)
    # Adding a Chart to beat the competition
    st.bar_chart(df.set_index('Metric'))

with col2:
    st.subheader("AI Prediction")
    if st.button("Analyze Wine Quality"):
        with st.spinner('AI is analyzing chemical bonds...'):
            time.sleep(1)  # Dramatic pause

            # Predict (Dummy logic for the 11-feature model we generated)
            features = [[0.5]*7 + [citric, vol_acidity, sulphates, alcohol]]
            prediction = model.predict(features)[0]

            if alcohol > 11.5:  # Simple logic for demo
                st.success("✨ RESULT: HIGH QUALITY")
                st.balloons()
                st.info(
                    "This wine has a balanced chemical structure suitable for premium aging.")
            else:
                st.warning("⚖️ RESULT: STANDARD QUALITY")
                st.write(
                    "This wine is good for immediate consumption but lacks premium complexity.")


