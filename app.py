"""
app.py – Streamlit dashboard for PJM Electricity Demand Forecasting
===================================================================
Run:  streamlit run app.py

Features
--------
- Sidebar: model selector, date-range picker, forecast-horizon slider
- Main panel:
    * Actual vs Predicted line chart (Plotly)
    * Forecast curve for next N hours (recursive)
    * Metrics cards (MAE / RMSE / MAPE / R2) from metrics.json
    * Best-model badge
- Models loaded once at startup (no retraining per interaction)
- Graceful error if a model file is missing (clear on-screen message, no crash)
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PJM Energy Demand Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── App background ── */
.stApp { background: linear-gradient(135deg,#0E1117 0%,#131720 100%); color:#F5F7FA; }

/* ── Sidebar background ── */
section[data-testid="stSidebar"] {
    background: #1C2333 !important;
    border-right: 1px solid #2D3748 !important;
}

/* ── Sidebar: ALL text bright white ── */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] *,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] .stMarkdown {
    color: #F5F7FA !important;
    opacity: 1 !important;
}

/* ── Sidebar: section headers ── */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #FFFFFF !important;
    font-weight: 700 !important;
}

/* ── Sidebar: selectbox / dropdown ── */
section[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: #263147 !important;
    border-color: #3B82F6 !important;
    color: #F5F7FA !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] span {
    color: #F5F7FA !important;
}

/* ── Sidebar: date input boxes ── */
section[data-testid="stSidebar"] input {
    background-color: #263147 !important;
    color: #F5F7FA !important;
    border: 1px solid #3B82F6 !important;
    border-radius: 6px !important;
}

/* ── Sidebar: slider label + value ── */
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"],
section[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"],
section[data-testid="stSidebar"] .stSlider p {
    color: #F5F7FA !important;
    font-weight: 500 !important;
}

/* ── Sidebar: caption text ── */
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] small {
    color: #94A3B8 !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: rgba(28,35,51,0.95) !important;
    border: 1px solid #2D3748 !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
    transition: transform .2s, box-shadow .2s;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 24px rgba(59,130,246,0.2);
}
[data-testid="stMetricLabel"] p {
    color: #CBD5E1 !important;
    font-size: .75rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: .06em !important;
    opacity: 1 !important;
}
[data-testid="stMetricValue"] {
    color: #60A5FA !important;
    font-weight: 700 !important;
    font-size: 1.6rem !important;
}

/* ── Subheaders + main text ── */
h1, h2, h3, .stSubheader { color: #F5F7FA !important; }
p, .stText { color: #E2E8F0 !important; }

/* ── Dataframe text ── */
[data-testid="stDataFrame"] * { color: #F5F7FA !important; }

/* ── Hero title ── */
.hero {
    font-size: 2rem; font-weight: 700;
    background: linear-gradient(90deg, #60A5FA, #93C5FD, #BFDBFE);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}

/* ── Best model badge ── */
.badge {
    display: inline-block;
    background: linear-gradient(135deg, #16A34A, #22C55E);
    color: #fff; padding: 4px 14px; border-radius: 20px;
    font-size: .85rem; font-weight: 700; margin-left: 8px;
}

hr { border-color: #2D3748 !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
MODELS_DIR   = "models"
DATA_PATH    = "data/PJME_hourly.csv"
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")

MODEL_FILES = {
    "Linear Regression": "linear_model.pkl",
    "Random Forest":     "rf_model.pkl",
    "XGBoost":           "xgboost_model.pkl",
    "LSTM":              "lstm_model.h5",
}

COLORS = {
    "actual":           "#58a6ff",
    "Linear Regression":"#ffa657",
    "Random Forest":    "#3fb950",
    "XGBoost":          "#d2a8ff",
    "LSTM":             "#ff7b72",
    "forecast":         "#f0883e",
}

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0E1117",
    plot_bgcolor="#1C2333",
    font=dict(family="Inter", color="#F5F7FA", size=13),
    xaxis=dict(
        gridcolor="#2D3748", zerolinecolor="#2D3748",
        tickfont=dict(color="#CBD5E1", size=12),
        title_font=dict(color="#F5F7FA", size=13),
    ),
    yaxis=dict(
        gridcolor="#2D3748", zerolinecolor="#2D3748",
        tickfont=dict(color="#CBD5E1", size=12),
        title_font=dict(color="#F5F7FA", size=13),
    ),
    margin=dict(l=10, r=10, t=44, b=10),
    legend=dict(
        bgcolor="rgba(28,35,51,0.92)",
        bordercolor="#3B82F6",
        borderwidth=1,
        font=dict(color="#F5F7FA", size=12),
    ),
    hoverlabel=dict(
        bgcolor="#1C2333",
        bordercolor="#3B82F6",
        font=dict(color="#F5F7FA", size=12),
    ),
    title_font=dict(color="#F5F7FA", size=15, family="Inter"),
)


# ── Data + model loaders (cached) ────────────────────────────────────────────

@st.cache_data(show_spinner="Loading dataset...")
def _load_data():
    from src.utils import load_and_clean
    from src.features import build_features
    df   = load_and_clean(DATA_PATH)
    feat = build_features(df)
    return df, feat

@st.cache_resource(show_spinner=False)
def _load_model(model_name: str):
    fname = MODEL_FILES[model_name]
    path  = os.path.join(MODELS_DIR, fname)
    if not os.path.exists(path):
        return None
    if model_name == "LSTM":
        import tensorflow as tf
        return tf.keras.models.load_model(path, compile=False)
    return joblib.load(path)

@st.cache_resource(show_spinner=False)
def _load_lstm_scalers():
    p = os.path.join(MODELS_DIR, "lstm_scaler.pkl")
    return joblib.load(p) if os.path.exists(p) else None

@st.cache_data(show_spinner=False)
def _load_metrics():
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH) as f:
        return json.load(f)


# ── Prediction helpers ───────────────────────────────────────────────────────

def _predict_tabular(model, feat_df):
    from src.features import FEATURE_COLS
    return model.predict(feat_df[FEATURE_COLS])

def _predict_lstm(model, scalers, feat_df):
    from src.features import FEATURE_COLS, make_sequences
    X = scalers["scaler_X"].transform(feat_df[FEATURE_COLS].values)
    y_dummy = np.zeros(len(X))
    Xs, _   = make_sequences(X, y_dummy, 24)
    preds_s = model.predict(Xs, verbose=0).ravel()
    return scalers["scaler_y"].inverse_transform(preds_s.reshape(-1, 1)).ravel()

def _run_forecast(model_name: str, n_hours: int):
    from src.predict import predict as do_predict
    key = {"Linear Regression":"linear","Random Forest":"rf",
           "XGBoost":"xgboost","LSTM":"lstm"}[model_name]
    return do_predict(key, n_hours)


# ── Load everything ──────────────────────────────────────────────────────────

if not os.path.exists(DATA_PATH):
    st.error("data/PJME_hourly.csv not found. Run download_data.py or place the file manually.")
    st.stop()

df, feat_df = _load_data()
metrics_all = _load_metrics()
from src.features import TRAIN_CUTOFF, TARGET_COL, FEATURE_COLS
test_feat = feat_df[feat_df.index >= TRAIN_CUTOFF]
test_actual = test_feat[TARGET_COL]


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚡ PJM Forecasting")
    st.markdown("---")

    available = [m for m in MODEL_FILES if
                 os.path.exists(os.path.join(MODELS_DIR, MODEL_FILES[m]))]
    if not available:
        st.warning("No trained models found.\nRun: `python src/train.py`")
        st.stop()

    model_name = st.selectbox("Select Model", available)

    st.markdown("### Date Range (Test Set)")
    date_min = test_actual.index.min().date()
    date_max = test_actual.index.max().date()
    d_start  = st.date_input("From", value=date_min, min_value=date_min, max_value=date_max)
    d_end    = st.date_input("To",   value=min(date_min + pd.Timedelta(days=90), date_max),
                              min_value=date_min, max_value=date_max)

    st.markdown("### Forecast Horizon")
    fc_hours = st.slider("Hours ahead", 1, 168, 24)

    st.markdown("---")
    if metrics_all:
        best_model = min(metrics_all, key=lambda m: metrics_all[m]["MAE"])
        st.markdown(f"**Best Model (MAE):** {best_model}")
    st.caption("PJM Interconnection · 2002–2018 · Hourly")


# ── Hero ─────────────────────────────────────────────────────────────────────

col_title, col_badge = st.columns([4, 1])
with col_title:
    st.markdown("<div class='hero'>⚡ PJM Electricity Demand Forecasting</div>", unsafe_allow_html=True)
    st.caption("Real PJM Interconnection data · Linear Regression · Random Forest · XGBoost · LSTM")
with col_badge:
    if metrics_all:
        st.markdown(f"<br><span class='badge'>Best: {best_model}</span>", unsafe_allow_html=True)

st.markdown("---")

# ── KPI cards ────────────────────────────────────────────────────────────────

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Hours", f"{len(df):,}")
k2.metric("Avg Load",    f"{df[TARGET_COL].mean():,.0f} MW")
k3.metric("Peak Load",   f"{df[TARGET_COL].max():,.0f} MW")
k4.metric("Min Load",    f"{df[TARGET_COL].min():,.0f} MW")

st.markdown("---")

# ── Load selected model + get predictions ────────────────────────────────────

model = _load_model(model_name)
if model is None:
    st.warning(f"Model file for **{model_name}** not found in `{MODELS_DIR}/`. "
               f"Run `python src/train.py` first.")
    st.stop()

with st.spinner(f"Running {model_name} predictions on test set..."):
    if model_name == "LSTM":
        scalers  = _load_lstm_scalers()
        if scalers is None:
            st.error("lstm_scaler.pkl not found. Re-run `python src/train.py`.")
            st.stop()
        raw_preds = _predict_lstm(model, scalers, test_feat)
        # LSTM preds offset by lookback
        pred_index = test_actual.index[24:]
        pred_series = pd.Series(raw_preds, index=pred_index)
        actual_aligned = test_actual[24:]
    else:
        raw_preds = _predict_tabular(model, test_feat)
        pred_series = pd.Series(raw_preds, index=test_actual.index)
        actual_aligned = test_actual

# ── Actual vs Predicted chart ─────────────────────────────────────────────────

st.subheader(f"Actual vs Predicted — {model_name}")

mask = (actual_aligned.index >= pd.Timestamp(d_start)) & \
       (actual_aligned.index <= pd.Timestamp(d_end) + pd.Timedelta(hours=23))

act_slice  = actual_aligned[mask]
pred_slice = pred_series[pred_series.index.isin(act_slice.index)]

fig = go.Figure()
fig.add_trace(go.Scatter(x=act_slice.index,  y=act_slice.values,
                          name="Actual",    line=dict(color=COLORS["actual"], width=1.5)))
fig.add_trace(go.Scatter(x=pred_slice.index, y=pred_slice.values,
                          name=model_name, line=dict(color=COLORS.get(model_name, "#ff7b72"),
                                                      width=1.5, dash="dot")))
fig.update_layout(**PLOTLY_LAYOUT, title="Actual vs Predicted (MW)",
                   xaxis_title="Date", yaxis_title="Load (MW)", height=420, hovermode="x unified")
st.plotly_chart(fig, width='stretch')

# ── Metrics cards ─────────────────────────────────────────────────────────────

st.subheader(f"Model Metrics — {model_name}")

if metrics_all and model_name in metrics_all:
    m = metrics_all[model_name]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE (MW)",  f"{m['MAE']:,.1f}")
    c2.metric("RMSE (MW)", f"{m['RMSE']:,.1f}")
    c3.metric("MAPE (%)",  f"{m['MAPE']:.2f}")
    c4.metric("R²",        f"{m['R2']:.4f}")
else:
    st.info("Metrics not found. Run `python src/train.py` to generate `models/metrics.json`.")

st.markdown("---")

# ── Forecast ──────────────────────────────────────────────────────────────────

st.subheader(f"Recursive {fc_hours}h Forecast — {model_name}")
st.caption("Step-by-step lag reconstruction — no future ground truth used.")

with st.spinner(f"Generating {fc_hours}-hour recursive forecast..."):
    try:
        fc_series = _run_forecast(model_name, fc_hours)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=actual_aligned.index[-168:], y=actual_aligned.values[-168:],
            name="Recent Actual", line=dict(color=COLORS["actual"], width=1.5)))
        fig2.add_trace(go.Scatter(
            x=fc_series.index, y=fc_series.values,
            name="Forecast", line=dict(color=COLORS["forecast"], width=2.5, dash="dot"),
            fill="tozeroy", fillcolor="rgba(240,136,62,0.07)"))
        fig2.update_layout(**PLOTLY_LAYOUT,
                           title=f"Next {fc_hours}h Forecast (MW)",
                           xaxis_title="Datetime", yaxis_title="Load (MW)",
                           height=360, hovermode="x unified")
        st.plotly_chart(fig2, width='stretch')

        fc_df = fc_series.reset_index()
        fc_df.columns = ["Datetime", "Forecasted MW"]
        fc_df["Datetime"] = fc_df["Datetime"].dt.strftime("%Y-%m-%d %H:%M")
        fc_df["Forecasted MW"] = fc_df["Forecasted MW"].round(1)
        st.dataframe(fc_df, width='stretch', hide_index=True)
    except Exception as e:
        st.error(f"Forecast failed: {e}")

st.markdown("---")

# ── Model Comparison table ────────────────────────────────────────────────────

if metrics_all:
    st.subheader("All Models Comparison")
    comp_df = pd.DataFrame(metrics_all).T.reset_index()
    comp_df.columns = ["Model", "MAE (MW)", "RMSE (MW)", "MAPE (%)", "R2"]
    best_idx = comp_df["MAE (MW)"].idxmin()

    def highlight_best(row):
        return ["background-color: rgba(35,134,54,0.25)" if row.name == best_idx
                else "" for _ in row]

    st.dataframe(
        comp_df.style.apply(highlight_best, axis=1).format(
            {"MAE (MW)": "{:.1f}", "RMSE (MW)": "{:.1f}", "MAPE (%)": "{:.2f}", "R2": "{:.4f}"}
        ),
        width='stretch', hide_index=True
    )

    # Saved comparison chart
    img_path = os.path.join("images", "model_comparison.png")
    if os.path.exists(img_path):
        st.image(img_path, caption="Model Comparison Chart", width='stretch')
