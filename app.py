import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events

# 1. Page Setup
st.set_page_config(page_title="Wine Quality Tuner", layout="wide")

# 2. Memory: This keeps the orange lines where you put them
if 'wine_params' not in st.session_state:
    st.session_state.wine_params = {
        "Alcohol": 10.5, "pH": 3.3, "Sulphates": 0.6, 
        "Residual Sugar": 2.5, "Volatile Acidity": 0.7,
        "Total SO2": 46.0, "Free SO2": 15.0
    }

st.title("🍷 Live Wine Quality Tuner")
st.info("Drag/Click the chart to move the orange handles and update the prediction.")

# 3. Build the "Handle" Chart
fig = go.Figure()

# Add the Blue Bars (The "Track")
fig.add_trace(go.Bar(
    x=list(st.session_state.wine_params.values()),
    y=list(st.session_state.wine_params.keys()),
    orientation='h',
    marker_color='rgba(31, 119, 180, 0.4)', # Semi-transparent blue
    hoverinfo='skip'
))

# Add the Orange Lines (The "Handles" from the link)
for i, (key, val) in enumerate(st.session_state.wine_params.items()):
    fig.add_shape(
        type="line",
        x0=val, x1=val, 
        y0=i-0.4, y1=i+0.4,
        line=dict(color="Orange", width=6), # High visibility orange handle
    )

fig.update_layout(
    xaxis=dict(range=[0, 60], title="Value"), # Adjust range based on your data
    yaxis=dict(title="Chemical Properties"),
    height=500,
    template="plotly_white",
    clickmode='event+select'
)

# 4. CAPTURE THE INTERACTION
# This makes the "orange lines" move to where you click
selected = plotly_events(fig, click_event=True, override_height=500)

if selected:
    point_index = selected[0]['pointNumber']
    clicked_key = list(st.session_state.wine_params.keys())[point_index]
    new_value = selected[0]['x']
    
    # Update the handle position
    st.session_state.wine_params[clicked_key] = round(new_value, 2)
    st.rerun()

# 5. Prediction Logic
st.divider()
# model = joblib.load('wine_model.pk1')
# scaler = joblib.load('scaler.pk1')
# prediction = model.predict(...)

st.subheader("Results")
col1, col2 = st.columns(2)
with col1:
    st.write("**Current Values:**", st.session_state.wine_params)
with col2:
    st.metric("Predicted Quality", "7.2 / 10")
