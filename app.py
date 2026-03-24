import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events

# 1. Setup
st.set_page_config(page_title="Pro Wine Tuner", layout="wide")

# 2. Memory: Store the chemical values
if 'wine_values' not in st.session_state:
    st.session_state.wine_values = {
        "Alcohol": 10.5, "pH": 3.3, "Sulphates": 0.6, "Sugar": 2.5
    }

st.title("🍷 Interactive Chemical Tuner")
st.write("Click on the chart to move the **Orange Line** and change the chemical value.")

# 3. Build the "Draggable" Chart
fig = go.Figure()

# Add the Blue background bars (The "track")
fig.add_trace(go.Bar(
    x=list(st.session_state.wine_values.values()),
    y=list(st.session_state.wine_values.keys()),
    orientation='h',
    marker_color='rgba(31, 119, 180, 0.3)', # Faded blue
    hoverinfo='skip'
))

# Add the Orange "Handles" (The draggable lines)
for i, (key, val) in enumerate(st.session_state.wine_values.items()):
    fig.add_shape(
        type="line",
        x0=val, x1=val, 
        y0=i-0.4, y1=i+0.4,
        line=dict(color="Orange", width=5),
    )

fig.update_layout(
    xaxis=dict(range=[0, 20]),
    height=400,
    template="plotly_white",
    clickmode='event+select'
)

# 4. CAPTURE THE MOVEMENT
# When you click a new spot, the orange line "jumps" there
selected_point = plotly_events(fig, click_event=True, override_height=400)

if selected_point:
    p = selected_point[0]
    clicked_feature = list(st.session_state.wine_values.keys())[p['pointNumber']]
    new_x = p['x'] # Where your cursor clicked
    
    # Update the value and refresh
    st.session_state.wine_values[clicked_feature] = round(new_x, 2)
    st.rerun()

# 5. Prediction
st.divider()
# prediction = model.predict(...)
st.subheader(f"Target Quality: 7.4 / 10")
st.write(f"Current Settings: {st.session_state.wine_values}")
