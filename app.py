import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Wine Quality AI",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS  – deep wine-red luxury theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;1,400&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Dark wine background */
.stApp {
    background: linear-gradient(135deg, #0d0608 0%, #1a0a10 40%, #0f0510 100%);
}

h1, h2, h3 { font-family: 'Cormorant Garamond', serif; }

/* Metric cards */
[data-testid="metric-container"] {
    background: rgba(139,0,40,0.15);
    border: 1px solid rgba(200,50,80,0.3);
    border-radius: 12px;
    padding: 16px 20px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(20, 5, 10, 0.95);
    border-right: 1px solid rgba(180,30,60,0.25);
}

section[data-testid="stSidebar"] .stSlider > div > div {
    background: rgba(139,0,40,0.3);
}

/* Divider */
hr { border-color: rgba(180,30,60,0.25) !important; }

/* Info / success boxes */
.stAlert { border-radius: 10px; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #8B0028, #C0284C);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #A0003A, #D03055);
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(180,30,60,0.4);
}

/* Quality score big display */
.quality-score {
    font-family: 'Cormorant Garamond', serif;
    font-size: 72px;
    font-weight: 600;
    text-align: center;
    background: linear-gradient(135deg, #FF6B8A, #FFD700);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
}
.quality-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 13px;
    text-align: center;
    color: rgba(255,255,255,0.5);
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 4px;
}
.wine-card {
    background: rgba(139,0,40,0.12);
    border: 1px solid rgba(200,50,80,0.2);
    border-radius: 16px;
    padding: 24px;
    margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MODEL LOADING  (with graceful fallback)
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    """Load model and scaler. Returns (model, scaler, loaded_ok)."""
    model_paths = ["wine_model.pkl", "wine_model.pk1"]  # handles typo .pk1 too
    scaler_paths = ["scaler.pkl", "scaler.pk1"]

    model, scaler = None, None

    for p in model_paths:
        if os.path.exists(p):
            try:
                model = joblib.load(p)
                break
            except Exception:
                pass

    for p in scaler_paths:
        if os.path.exists(p):
            try:
                scaler = joblib.load(p)
                break
            except Exception:
                pass

    return model, scaler


model, scaler = load_models()
model_loaded = model is not None and scaler is not None


# ─────────────────────────────────────────────
# FEATURE CONFIG  – ranges, defaults, tooltips
# ─────────────────────────────────────────────
FEATURES = {
    "fixed acidity":        {"min": 4.0,   "max": 16.0,  "default": 8.0,   "step": 0.1,  "unit": "g/L",  "tip": "Tartaric acid concentration"},
    "volatile acidity":     {"min": 0.1,   "max": 1.6,   "default": 0.5,   "step": 0.01, "unit": "g/L",  "tip": "Acetic acid (high = vinegar taste)"},
    "citric acid":          {"min": 0.0,   "max": 1.0,   "default": 0.3,   "step": 0.01, "unit": "g/L",  "tip": "Adds freshness & flavor"},
    "residual sugar":       {"min": 1.0,   "max": 16.0,  "default": 2.5,   "step": 0.1,  "unit": "g/L",  "tip": "Sugar remaining after fermentation"},
    "chlorides":            {"min": 0.01,  "max": 0.62,  "default": 0.08,  "step": 0.001,"unit": "g/L",  "tip": "Salt content"},
    "free sulfur dioxide":  {"min": 1.0,   "max": 72.0,  "default": 15.0,  "step": 1.0,  "unit": "mg/L", "tip": "Free SO₂ prevents microbial growth"},
    "total sulfur dioxide": {"min": 6.0,   "max": 289.0, "default": 46.0,  "step": 1.0,  "unit": "mg/L", "tip": "Total SO₂ in free & bound forms"},
    "density":              {"min": 0.990, "max": 1.004, "default": 0.997, "step": 0.0001,"unit": "g/cm³","tip": "Depends on alcohol & sugar"},
    "pH":                   {"min": 2.7,   "max": 4.1,   "default": 3.3,   "step": 0.01, "unit": "",     "tip": "Acidity scale (lower = more acidic)"},
    "sulphates":            {"min": 0.3,   "max": 2.0,   "default": 0.6,   "step": 0.01, "unit": "g/L",  "tip": "Wine additive, antimicrobial"},
    "alcohol":              {"min": 8.0,   "max": 15.0,  "default": 10.5,  "step": 0.1,  "unit": "% vol","tip": "Percentage of alcohol by volume"},
}

QUALITY_LABELS = {
    3: ("Poor",      "#E74C3C"),
    4: ("Below Avg", "#E67E22"),
    5: ("Average",   "#F39C12"),
    6: ("Good",      "#27AE60"),
    7: ("Very Good", "#2ECC71"),
    8: ("Excellent", "#1ABC9C"),
}


# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
if "wine_params" not in st.session_state:
    st.session_state.wine_params = {k: v["default"] for k, v in FEATURES.items()}

if "history" not in st.session_state:
    st.session_state.history = []  # list of (params_dict, predicted_quality)


# ─────────────────────────────────────────────
# PREDICTION HELPER
# ─────────────────────────────────────────────
def predict_quality(params: dict):
    """Return (quality_int, proba_array) or None on error."""
    if not model_loaded:
        return None, None
    try:
        df = pd.DataFrame([params])
        df = df[list(FEATURES.keys())]  # ensure correct column order
        X_scaled = scaler.transform(df)
        quality = int(model.predict(X_scaled)[0])
        proba = model.predict_proba(X_scaled)[0]
        return quality, proba
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None


# ─────────────────────────────────────────────
# SIDEBAR  – sliders
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🍷 Wine Parameters")
    st.markdown("Adjust the chemical properties to tune your wine profile.")
    st.divider()

    # Group sliders into logical sections
    groups = {
        "🧪 Acidity": ["fixed acidity", "volatile acidity", "citric acid", "pH"],
        "🍬 Sugar & Salt": ["residual sugar", "chlorides"],
        "🛡️ Sulfur Dioxide": ["free sulfur dioxide", "total sulfur dioxide"],
        "🧬 Physical": ["density", "sulphates", "alcohol"],
    }

    for group_name, keys in groups.items():
        st.markdown(f"**{group_name}**")
        for key in keys:
            cfg = FEATURES[key]
            label = f"{key.title()} ({cfg['unit']})" if cfg["unit"] else key.title()
            st.session_state.wine_params[key] = st.slider(
                label,
                min_value=cfg["min"],
                max_value=cfg["max"],
                value=float(st.session_state.wine_params[key]),
                step=cfg["step"],
                help=cfg["tip"],
                key=f"slider_{key}",
            )
        st.markdown("")

    st.divider()
    if st.button("🔄 Reset to Defaults", use_container_width=True):
        for k, v in FEATURES.items():
            st.session_state.wine_params[k] = v["default"]
        st.rerun()

    if st.button("📌 Save to History", use_container_width=True):
        quality, _ = predict_quality(st.session_state.wine_params)
        if quality:
            st.session_state.history.append(
                (dict(st.session_state.wine_params), quality)
            )
            st.success(f"Saved! Quality = {quality}/10")


# ─────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────
st.markdown("# 🍷 Wine Quality AI")
st.markdown(
    "<p style='color:rgba(255,255,255,0.5);font-size:15px;margin-top:-12px;'>"
    "Machine-learning powered prediction using Gradient Boosting · Red Wine · UCI Dataset"
    "</p>",
    unsafe_allow_html=True,
)

if not model_loaded:
    st.warning(
        "⚠️ **Model files not found.** Place `wine_model.pkl` and `scaler.pkl` "
        "in the same folder as `app.py`. Using demo mode."
    )

quality, proba = predict_quality(st.session_state.wine_params)

# ── Row 1: Quality Score + Gauge + Radar ─────
col_score, col_gauge, col_radar = st.columns([1, 1.6, 1.6])

with col_score:
    if quality:
        label, color = QUALITY_LABELS.get(quality, ("Unknown", "#AAA"))
        st.markdown(f"""
        <div class="wine-card" style="text-align:center;padding:32px 16px;">
            <div style='font-family:Cormorant Garamond,serif;font-size:14px;
                        color:rgba(255,255,255,0.4);letter-spacing:3px;
                        text-transform:uppercase;margin-bottom:8px;'>
                Predicted Quality
            </div>
            <div style='font-family:Cormorant Garamond,serif;font-size:80px;
                        font-weight:600;color:{color};line-height:1;'>
                {quality}
            </div>
            <div style='font-size:13px;color:rgba(255,255,255,0.5);
                        letter-spacing:3px;text-transform:uppercase;'>
                out of 10
            </div>
            <div style='margin-top:12px;font-size:18px;font-weight:500;color:{color};'>
                {label}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Key metrics
        st.metric("🍶 Alcohol", f"{st.session_state.wine_params['alcohol']:.1f}%")
        st.metric("🧪 pH", f"{st.session_state.wine_params['pH']:.2f}")
        st.metric("💧 Volatile Acidity", f"{st.session_state.wine_params['volatile acidity']:.2f} g/L")
    else:
        st.info("Awaiting prediction…")

with col_gauge:
    # Gauge chart
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=quality if quality else 5,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Quality Score", "font": {"size": 16, "color": "rgba(255,255,255,0.7)"}},
        delta={"reference": 5.5, "increasing": {"color": "#2ECC71"}, "decreasing": {"color": "#E74C3C"}},
        gauge={
            "axis": {"range": [3, 8], "tickwidth": 1, "tickcolor": "rgba(255,255,255,0.3)"},
            "bar": {"color": color if quality else "#8B0028", "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [3, 4.5], "color": "rgba(231,76,60,0.2)"},
                {"range": [4.5, 5.5], "color": "rgba(243,156,18,0.2)"},
                {"range": [5.5, 6.5], "color": "rgba(39,174,96,0.2)"},
                {"range": [6.5, 8],   "color": "rgba(26,188,156,0.25)"},
            ],
            "threshold": {
                "line": {"color": "gold", "width": 3},
                "thickness": 0.75,
                "value": quality if quality else 5,
            },
        },
        number={"font": {"color": color if quality else "#AAA", "size": 48}},
    ))
    fig_gauge.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="rgba(255,255,255,0.7)",
        height=280,
        margin=dict(l=20, r=20, t=30, b=10),
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Probability bar for each class
    if proba is not None and model_loaded:
        classes = model.classes_
        fig_prob = go.Figure()
        bar_colors = [QUALITY_LABELS.get(int(c), ("?", "#AAA"))[1] for c in classes]
        fig_prob.add_trace(go.Bar(
            x=[f"Q{c}" for c in classes],
            y=proba * 100,
            marker_color=bar_colors,
            text=[f"{p*100:.1f}%" for p in proba],
            textposition="outside",
            textfont=dict(color="rgba(255,255,255,0.7)", size=11),
        ))
        fig_prob.update_layout(
            title=dict(text="Probability per Quality Class", font=dict(size=13, color="rgba(255,255,255,0.6)")),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="rgba(255,255,255,0.6)",
            height=200,
            margin=dict(l=10, r=10, t=40, b=10),
            yaxis=dict(showgrid=False, showticklabels=False, range=[0, 110]),
            xaxis=dict(showgrid=False),
            showlegend=False,
        )
        st.plotly_chart(fig_prob, use_container_width=True)

with col_radar:
    # Radar / Spider chart – normalized feature values
    params = st.session_state.wine_params
    keys_radar = ["alcohol", "pH", "sulphates", "citric acid",
                  "volatile acidity", "residual sugar", "fixed acidity"]

    normalized = []
    for k in keys_radar:
        cfg = FEATURES[k]
        norm = (params[k] - cfg["min"]) / (cfg["max"] - cfg["min"])
        normalized.append(round(norm, 3))

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=normalized + [normalized[0]],
        theta=keys_radar + [keys_radar[0]],
        fill="toself",
        fillcolor="rgba(139,0,40,0.3)",
        line=dict(color="#FF6B8A", width=2),
        name="Current",
    ))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1],
                            showticklabels=False, gridcolor="rgba(255,255,255,0.1)"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.1)",
                             tickfont=dict(color="rgba(255,255,255,0.7)", size=11)),
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="rgba(255,255,255,0.7)",
        title=dict(text="Feature Profile (Normalized)", font=dict(size=13)),
        height=450,
        margin=dict(l=40, r=40, t=50, b=10),
        showlegend=False,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

st.divider()

# ── Row 2: Horizontal Bar Chart (like your original) + Feature Importance ──
col_bar, col_imp = st.columns(2)

with col_bar:
    st.markdown("### Current Parameter Values")
    params = st.session_state.wine_params

    # Normalize for display as % of range
    bar_vals, bar_labels, bar_pct = [], [], []
    for k, cfg in FEATURES.items():
        v = params[k]
        pct = (v - cfg["min"]) / (cfg["max"] - cfg["min"]) * 100
        bar_vals.append(v)
        bar_pct.append(pct)
        bar_labels.append(f"{k.title()} ({cfg['unit']})" if cfg["unit"] else k.title())

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=bar_pct,
        y=bar_labels,
        orientation="h",
        marker=dict(
            color=bar_pct,
            colorscale=[[0, "rgba(139,0,40,0.4)"], [0.5, "rgba(180,30,60,0.7)"], [1, "#FF6B8A"]],
            showscale=False,
        ),
        text=[f"{v:.3g}" for v in bar_vals],
        textposition="outside",
        textfont=dict(color="rgba(255,255,255,0.6)", size=11),
        hovertemplate="<b>%{y}</b><br>Value: %{text}<br>Range: %{x:.1f}%<extra></extra>",
    ))
    # Orange handles (your original orange line concept)
    for i, pct in enumerate(bar_pct):
        fig_bar.add_shape(
            type="line",
            x0=pct, x1=pct,
            y0=i - 0.45, y1=i + 0.45,
            line=dict(color="orange", width=4),
        )
    fig_bar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="rgba(255,255,255,0.6)",
        height=420,
        margin=dict(l=10, r=60, t=10, b=10),
        xaxis=dict(range=[0, 120], title="% of Range", gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with col_imp:
    st.markdown("### Feature Importance")
    if model_loaded and hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feat_names = list(FEATURES.keys())
        sorted_idx = np.argsort(importances)
        fig_imp = go.Figure(go.Bar(
            x=importances[sorted_idx],
            y=[feat_names[i].title() for i in sorted_idx],
            orientation="h",
            marker=dict(
                color=importances[sorted_idx],
                colorscale=[[0, "rgba(139,0,40,0.3)"], [1, "#FFD700"]],
                showscale=False,
            ),
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
        ))
        fig_imp.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="rgba(255,255,255,0.6)",
            height=420,
            margin=dict(l=10, r=20, t=10, b=10),
            xaxis=dict(title="Importance", gridcolor="rgba(255,255,255,0.08)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        )
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("Feature importance not available for this model type.")

st.divider()

# ── Row 3: History comparison ────────────────
st.markdown("### 📊 Comparison History")
if st.session_state.history:
    history_data = []
    for i, (p, q) in enumerate(st.session_state.history):
        row = {"#": i + 1, "Quality": q,
               "Label": QUALITY_LABELS.get(q, ("?",""))[0]}
        for k in ["alcohol", "pH", "volatile acidity", "sulphates"]:
            row[k.title()] = p[k]
        history_data.append(row)

    df_hist = pd.DataFrame(history_data)
    st.dataframe(
        df_hist,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Quality": st.column_config.ProgressColumn(
                "Quality", min_value=3, max_value=8, format="%d"
            )
        },
    )
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()
else:
    st.markdown(
        "<p style='color:rgba(255,255,255,0.3);font-style:italic;'>"
        "No history yet. Adjust parameters and click 'Save to History' in the sidebar.</p>",
        unsafe_allow_html=True,
    )

st.divider()

# ── Footer ───────────────────────────────────
st.markdown(
    "<p style='text-align:center;color:rgba(255,255,255,0.2);font-size:12px;'>"
    "Wine Quality AI · Red Wine UCI Dataset · Gradient Boosting Classifier"
    "</p>",
    unsafe_allow_html=True,
)
