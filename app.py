import streamlit as st
import pandas as pd
import joblib
import time

# Page Config
st.set_page_config(page_title="Premium Wine AI", page_icon="🍷", layout="centered")

# Custom CSS for the "Red Slider" look
st.markdown("""
    <style>
    .main { background-color: #ffffff; }
    /* Making the sliders red to match the aesthetic */
    span[data-baseweb="slider"] > div > div { background-color: #e63946 !important; }
    .stButton>button { width: 100%; border-radius: 10px; background-color: #720e0e; color: white; height: 3em; font-size: 1.2rem; }
    </style>
    """, unsafe_allow_html=True)

# Load Model and Scaler
model = joblib.load('wine_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🍷 Enter Wine Chemical Properties")
st.write("Adjust the sliders below to see if the wine is High or Standard quality.")

# --- MAIN AREA SLIDERS (No Sidebar) ---
# This makes them large and interactive in the center of the screen
fixed_acidity = st.slider('Fixed Acidity', 4.0, 16.0, 7.0)
volatile_acidity = st.slider('Volatile Acidity', 0.1, 1.6, 0.5)
citric_acid = st.slider('Citric Acid', 0.0, 1.0, 0.3)
residual_sugar = st.slider('Residual Sugar', 0.5, 15.5, 2.5)
chlorides = st.sidebar.slider('Chlorides', 0.01, 0.6, 0.08) # Optional: move to sidebar if screen is too crowded
free_sulfur = st.slider('Free Sulfur Dioxide', 1.0, 72.0, 15.0)
total_sulfur = st.slider('Total Sulfur Dioxide', 6.0, 289.0, 46.0)
density = st.slider('Density', 0.990, 1.004, 1.00)
ph = st.slider('pH Level', 2.7, 4.0, 3.3)
sulphates = st.slider('Sulphates', 0.3, 2.0, 0.6)
alcohol = st.slider('Alcohol Content (%)', 8.0, 15.0, 10.5)

# Create the data for prediction
input_data = pd.DataFrame({
    'fixed acidity': [fixed_acidity],
    'volatile acidity': [volatile_acidity],
    'citric acid': [citric_acid],
    'residual sugar': [residual_sugar],
    'chlorides': [0.08], # Kept constant or add a slider
    'free sulfur dioxide': [free_sulfur],
    'total sulfur dioxide': [total_sulfur],
    'density': [density],
    'pH': [ph],
    'sulphates': [sulphates],
    'alcohol': [alcohol]
})

st.markdown("---")

# Prediction logic
if st.button("RUN QUALITY ANALYSIS"):
    with st.spinner('Calculating chemical profile...'):
        time.sleep(0.5)
        # Scale and Predict
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)[0]

        if prediction == 1:
            st.success("### ✨ RESULT: HIGH QUALITY WINE")
            st.balloons()
        else:
            st.warning("### ⚖️ RESULT: STANDARD QUALITY WINE")

st.caption("JECRC University - AI Project Deployment")
