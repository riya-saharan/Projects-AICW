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
    .stButton>button { width: 100%; border-radius: 20px; background-color: #720e0e; color: white; height: 3em; font-weight: bold;}
    </style>
    """, unsafe_allow_html=True)

# Load Model and Scaler (Make sure these match your filenames in GitHub)
model = joblib.load('wine_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🍷 Premium Wine Quality Predictor")
st.markdown("---")

# --- SIDEBAR: ALL 11 INTERACTIVE SLIDERS ---
st.sidebar.header("🧪 Chemical Analysis")

def get_user_input():
    # Now all 11 ingredients are interactive!
    fixed_acidity = st.sidebar.slider('Fixed Acidity', 4.0, 16.0, 8.3)
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
    
    # Put them in a DataFrame for the model
    data = {
        'fixed acidity': fixed_acidity,
        'volatile acidity': vol_acidity,
        'citric acid': citric,
        'residual sugar': residual_sugar,
        'chlorides': chlorides,
        'free sulfur dioxide': free_sulfur,
        'total sulfur dioxide': total_sulfur,
        'density': density,
        'pH': ph,
        'sulphates': sulphates,
        'alcohol': alcohol
    }
    return pd.DataFrame(data, index=[0])

input_df = get_user_input()

# Main UI - 2 Columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Wine Chemical Profile")
    # This table updates LIVE when you move the sliders!
    st.dataframe(input_df.T.rename(columns={0: 'Value'}), use_container_width=True)
    
    # Bar chart updates LIVE too
    st.bar_chart(input_df.T)

with col2:
    st.subheader("🤖 AI Prediction Engine")
    
    if st.button("Analyze Wine Quality"):
        with st.spinner('AI is analyzing chemical bonds...'):
            time.sleep(0.8) 
            
            # Scale the input using your scaler
            scaled_input = scaler.transform(input_df)
            
            # Predict using your model
            prediction = model.predict(scaled_input)[0]

            if prediction == 1:
                st.success("✨ RESULT: HIGH QUALITY")
                st.balloons()
            else:
                st.warning("⚖️ RESULT: STANDARD QUALITY")

st.markdown("---")
st.caption("JECRC CSE Department - AI Model Deployment Project 2026")
