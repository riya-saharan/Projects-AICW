import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# 1. Page Config
st.set_page_config(page_title="Pro Wine AI", layout="wide")

# 2. Initial State (Check if data exists, if not, set defaults)
if 'wine_data' not in st.session_state:
    st.session_state.wine_data = {
        "Alcohol": 10.5, "Sulphates": 0.6, "pH": 3.3, 
        "Total SO2": 46.0, "Free SO2": 15.0, "Chlorides": 0.08, 
        "Residual Sugar": 2.5, "Fixed Acidity": 7.4
    }

st.title("🍷 Live Wine Quality Tuner")

# 3. Create the Chart
fig = go.Figure()

# Trace stays simple
fig.add_trace(go.Bar(
    x=list(st.session_state.wine_data.keys()),
    y=list(st.session_state.wine_data.values()),
    marker_color='#1f77b4'
))

# 4. FIX: Move 'editable' to the config/layout section
fig.update_layout(
    yaxis=dict(range=[0, 100]),
    height=500,
    template="plotly_white"
)

# 5. Display with Config (This makes it interactive)
# The 'config' parameter is where 'editable' actually lives
st.plotly_chart(fig, use_container_width=True, config={'editable': True})

# 6. Sidebar 'Cursors' (To sync the values back to your model)
st.sidebar.header("Manual Control")
for key in st.session_state.wine_data.keys():
    st.session_state.wine_data[key] = st.sidebar.slider(
        key, 0.0, 100.0, float(st.session_state.wine_data[key])
    )

st.success("The bars are now responsive to your cursor/sliders!")
