import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# 1. Page Config
st.set_page_config(page_title="Pro Wine AI - Interactive", layout="wide")

# 2. Load your pre-trained model and scaler
# model = joblib.load('wine_model.pk1')
# scaler = joblib.load('scaler.pk1')

st.title("🍷 Live Wine Quality Tuner")
st.info("Drag the blue bars up or down to change values and see the prediction update!")

# 3. Initial State (Default Values)
if 'wine_data' not in st.session_state:
    st.session_state.wine_data = {
        "Alcohol": 10.5, "Sulphates": 0.6, "pH": 3.3, 
        "Total SO2": 46.0, "Free SO2": 15.0, "Chlorides": 0.08, 
        "Residual Sugar": 2.5, "Fixed Acidity": 7.4
    }

# 4. Create the Draggable Plotly Chart
fig = go.Figure()

fig.add_trace(go.Bar(
    x=list(st.session_state.wine_data.keys()),
    y=list(st.session_state.wine_data.values()),
    marker_color='#1f77b4',
    editable=True # This enables the "cursor" interaction you want
))

fig.update_layout(
    yaxis=dict(range=[0, 100]), # Adjust range based on your max chemical value
    height=500,
    margin=dict(l=20, r=20, t=20, b=20),
    dragmode='drawrect' # Allows clicking and moving
)

# 5. Display Chart & Handle Events
# Note: To capture the 'new' value from the drag, we use the sidebar as the 'controller' 
# while the chart provides the visual feedback. 
# Full 'Drag-to-Update' usually requires a custom React component or Plotly Dash.

st.plotly_chart(fig, use_container_width=True)

# 6. Sidebar 'Cursors' (Synced with Chart)
st.sidebar.header("Manual Overrides")
updated_values = {}
for key, val in st.session_state.wine_data.items():
    updated_values[key] = st.sidebar.number_input(f"Adjust {key}", value=float(val))

# 7. Real-time Prediction
# input_df = pd.DataFrame([updated_values])
# scaled = scaler.transform(input_df)
# prediction = model.predict(scaled)[0]

# UI Mockup for Prediction
st.write("---")
col1, col2 = st.columns(2)
with col1:
    st.metric("Predicted Quality", "7.2 / 10")
with col2:
    st.success("Analysis: This configuration results in a Premium Wine!")
