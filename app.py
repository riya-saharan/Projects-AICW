import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# 1. Page Config & Professional Styling
st.set_page_config(page_title="Vinho-AI Analytics", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #fcfcfc; }
    div[data-testid="stMetric"] { background-color: #ffffff; border: 1px solid #e0e0e0; padding: 15px; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

# 2. Load your model assets
# model = joblib.load('wine_model.pk1')
# scaler = joblib.load('scaler.pk1')

st.title("🍷 Vinho-AI: Advanced Wine Fingerprinting")
st.write("Adjust the parameters below to see the chemical analysis and AI prediction update live.")

# 3. Interactive Input Grid (The "Different Way")
st.subheader("🧪 Chemical Analysis Parameters")
col1, col2, col3, col4 = st.columns(4)

with col1:
    alc = st.number_input("Alcohol (%)", 8.0, 15.0, 10.5, step=0.1)
    ph = st.number_input("pH Level", 2.7, 4.0, 3.3, step=0.01)

with col2:
    vol = st.number_input("Volatile Acidity", 0.1, 1.5, 0.7, step=0.01)
    sul = st.number_input("Sulphates", 0.3, 2.0, 0.6, step=0.1)

with col3:
    sug = st.number_input("Residual Sugar", 0.9, 15.0, 2.5, step=0.1)
    fso = st.number_input("Free SO2", 1.0, 72.0, 15.0, step=1.0)

with col4:
    tso = st.number_input("Total SO2", 6.0, 289.0, 46.0, step=1.0)
    den = st.number_input("Density", 0.990, 1.004, 0.996, step=0.001)

# 4. Main Dashboard Area
st.divider()
left_chart, right_gauge = st.columns([2, 1])

# Data for visuals
data_dict = {"Alcohol": alc, "pH": ph, "Volatile": vol, "Sulphates": sul, "Sugar": sug, "Density": den}

with left_chart:
    st.subheader("📊 Chemical Fingerprint")
    # Horizontal bar chart that updates instantly as you type numbers
    fig_bar = go.Figure(go.Bar(
        x=list(data_dict.values()),
        y=list(data_dict.keys()),
        orientation='h',
        marker_color='#1f77b4', # The professional blue you like
        bordercolor="white"
    ))
    fig_bar.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=10), template="plotly_white")
    st.plotly_chart(fig_bar, use_container_width=True)

with right_gauge:
    st.subheader("🎯 AI Quality Prediction")
    # A professional Gauge for the final score
    prediction = 7.4 # Replace with: model.predict(scaler.transform(...))
    
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prediction,
        gauge = {'axis': {'range': [0, 10]},
                 'bar': {'color': "#1f77b4"},
                 'steps': [
                     {'range': [0, 5], 'color': "#eeeeee"},
                     {'range': [5, 7], 'color': "#dddddd"},
                     {'range': [7, 10], 'color': "#d4af37"}]}, # Gold for high quality
        domain = {'x': [0, 1], 'y': [0, 1]}
    ))
    fig_gauge.update_layout(height=350, margin=dict(t=0, b=0))
    st.plotly_chart(fig_gauge, use_container_width=True)

st.info("💡 **Pro-Tip:** High alcohol and balanced acidity typically correlate with premium ratings.")
