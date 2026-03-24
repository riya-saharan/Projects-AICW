import streamlit as st
import pandas as pd
import joblib
import time

# Page Config (Keeping your "Pro Wine AI" title)
st.set_page_config(page_title="Pro Wine AI", page_icon="🍷", layout="wide")

# Custom CSS (Keeping your exact Aesthetic)
st.markdown("""
    <style>
    .main { background-color: #fdfcfc; }
    .stButton>button { width: 100%; border-radius: 20px; background-color: #720e0e; color: white; height: 3em; font-weight: bold;}
    </style>
    """, unsafe_allow_html=True)

# Load Model and Scaler
model = joblib.load('wine_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🍷 Premium Wine Quality Predictor")
st.markdown("---")

# --- SIDEBAR: ALL 11 SLIDERS (Now they all work!) ---
st.sidebar.header("🧪 Chemical Analysis")

fixed_acidity = st.sidebar.slider('Fixed Acidity', 4.0, 16.0, 8.0)
vol_acidity = st.sidebar.slider("Volatile Acidity", 0.1, 1.5, 0.5)
citric = st.sidebar.slider("Citric Acid", 0.0, 1.0, 0.3)
residual_sugar = st.sidebar.slider('Residual Sugar', 0.9, 15.5, 2.5)
chlorides = st.sidebar.slider('Chlorides', 0.01, 0.6, 0.08)
free_sulfur = st.sidebar.slider('Free Sulfur Dioxide', 1.0, 72.0, 15.0)
total_sulfur = st.sidebar.slider('Total Sulfur Dioxide', 6.0, 289.0, 46.0)
density = st.sidebar.slider('Density', 0.990, 1.004, 0.996)
ph = st.sidebar.slider('pH Level', 2.7, 4.0, 3.3)
sulphates = st.sidebar.slider("Sulphates", 0.3, 2.0, 0.6)
alcohol = st.sidebar.slider("Alcohol (%)", 8.0, 15.0, 10.5)

# Main UI - 2 Columns (Keeping your layout)
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Current Wine Profile")
    # This dataframe now includes ALL ingredients and updates LIVE
    df_display = pd.DataFrame({
        'Metric': ['Fixed Acidity', 'Volatile Acidity', 'Citric Acid', 'Residual Sugar', 'Chlorides', 'Free SO2', 'Total SO2', 'Density', 'pH', 'Sulphates', 'Alcohol'],
        'Value': [fixed_acidity, vol_acidity, citric, residual_sugar, chlorides, free_sulfur, total_sulfur, density, ph, sulphates, alcohol]
    })
    st.table(df_display)
    
    # The Chart now shows all 11 bars!
    st.bar_chart(df_display.set_index('Metric'))

with col2:
    st.subheader("AI Prediction")
    if st.button("Analyze Wine Quality"):
        with st.spinner('AI is analyzing chemical bonds...'):
            time.sleep(0.8) 

            # Create the input for the model with all 11 values
            features = pd.DataFrame([[fixed_acidity, vol_acidity, citric, residual_sugar, chlorides, free_sulfur, total_sulfur, density, ph, sulphates, alcohol]], 
                                    columns=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'])
            
            # Scale and Predict
            scaled_input = scaler.transform(features)
            prediction = model.predict(scaled_input)[0]

            if prediction == 1:
                st.success("✨ RESULT: HIGH QUALITY")
                st.balloons()
            else:
                st.warning("⚖️ RESULT: STANDARD QUALITY")

st.markdown("---")
st.caption("JECRC University - AI Model Deployment")
