import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# UI Setup
st.set_page_config(page_title="Vinho Analytics", layout="wide")

# Custom CSS for the "Pro" look
st.markdown("""
    <style>
    .stNumberInput { border: 1px solid #1f77b4; border-radius: 5px; }
    div[data-testid="stMetric"] { background-color: #f0f2f6; padding: 10px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🧪 Advanced Wine Chemical Analysis")
st.markdown("---")

# --- STEP 1: INPUT GRID (The "Different Way") ---
st.subheader("1. Precise Chemical Composition")
col1, col2, col3, col4 = st.columns(4)

with col1:
    alc = st.number_input("Alcohol %", 8.0, 15.0, 10.5)
    ph = st.number_input("pH Level", 2.7, 4.0, 3.3)
with col2:
    sul = st.number_input("Sulphates", 0.3, 2.0, 0.6)
    vol = st.number_input("Volatile Acidity", 0.1, 1.5, 0.7)
with col3:
    sug = st.number_input("Residual Sugar", 0.9, 15.0, 2.5)
    fso = st.number_input("Free SO2", 1.0, 72.0, 15.0)
with col4:
    tso = st.number_input("Total SO2", 6.0, 289.0, 46.0)
    den = st.number_input("Density", 0.990, 1.004, 0.996)

# --- STEP 2: PREDICTION LOGIC ---
# Construct the dataframe for your model
input_data = pd.DataFrame([[alc, ph, sul, vol, sug, fso, tso, den]], 
                         columns=['alcohol', 'pH', 'sulphates', 'volatile acidity', 
                                  'residual sugar', 'free sulfur dioxide', 
                                  'total sulfur dioxide', 'density'])

# Prediction (Placeholder until your model is loaded)
# prediction = model.predict(scaler.transform(input_data))[0]
prediction = 6.8  # Example value

# --- STEP 3: VISUAL ANALYSIS ---
st.markdown("---")
viz_left, viz_right = st.columns([2, 1])

with viz_left:
    st.subheader("📊 Composition Analysis")
    # Horizontal Bar Chart that changes instantly with the number inputs
    fig_bar = go.Figure(go.Bar(
        x=list(input_data.iloc[0]),
        y=list(input_data.columns),
        orientation='h',
        marker_color='#1f77b4'
    ))
    fig_bar.update_layout(height=400, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig_bar, use_container_width=True)

with viz_right:
    st.subheader("🎯 AI Quality Score")
    # Gauge Chart for a premium feel
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prediction,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Wine Grade"},
        gauge = {'axis': {'range': [None, 10]},
                 'bar': {'color': "#1f77b4"},
                 'steps' : [
                     {'range': [0, 5], 'color': "lightgray"},
                     {'range': [5, 7], 'color': "gray"},
                     {'range': [7, 10], 'color': "gold"}]}
    ))
    st.plotly_chart(fig_gauge, use_container_width=True)
