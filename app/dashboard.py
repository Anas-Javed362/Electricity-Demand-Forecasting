"""
Streamlit dashboard for PJM Energy Consumption Forecasting.

Tabs
----
  🔮 Forecast      – Actual vs predicted time series (all models)
  🔬 Explainability – SHAP feature importance (XGBoost)
  📊 Model Metrics  – MAE / RMSE / MAPE comparison table + bar chart
  🧪 MLflow         – Link to experiment tracking UI
  ⚡ Live Predict   – Interactive single-point prediction widget

Run
---
    streamlit run app/dashboard.py
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Page config + global CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PJM Energy Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* ── Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Dark gradient background ── */
    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
        color: #e6edf3;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
        border-right: 1px solid #30363d;
    }
    [data-testid="stSidebar"] .stMarkdown h2 {
        color: #58a6ff;
    }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background: rgba(22, 27, 34, 0.8);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 16px 20px;
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(88, 166, 255, 0.15);
    }
    [data-testid="stMetricLabel"] {
        color: #8b949e !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    [data-testid="stMetricValue"] {
        color: #58a6ff !important;
        font-weight: 700 !important;
        font-size: 1.6rem !important;
    }

    /* ── Tab styling ── */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(22, 27, 34, 0.9);
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
        border: 1px solid #30363d;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #8b949e;
        font-weight: 500;
        padding: 8px 16px;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    /* ── Section headers ── */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #58a6ff;
        border-bottom: 2px solid #21262d;
        padding-bottom: 8px;
        margin-bottom: 20px;
    }

    /* ── Status badge ── */
    .badge-ok   { background:#238636; color:#fff; padding:3px 10px; border-radius:20px; font-size:.75rem; font-weight:600; }
    .badge-warn { background:#9e6a03; color:#fff; padding:3px 10px; border-radius:20px; font-size:.75rem; font-weight:600; }

    /* ── Hero title ── */
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #58a6ff, #79c0ff, #a5d6ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.2;
    }
    .hero-subtitle {
        color: #8b949e;
        font-size: 1rem;
        margin-top: 4px;
    }

    /* ── Cards ── */
    .glass-card {
        background: rgba(22, 27, 34, 0.7);
        border: 1px solid #30363d;
        border-radius: 16px;
        padding: 20px 24px;
        backdrop-filter: blur(12px);
    }

    /* ── Divider ── */
    hr { border-color: #21262d; }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #1f6feb, #388bfd);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 16px rgba(88, 166, 255, 0.4);
    }

    /* ── Inputs ── */
    .stNumberInput input, .stTextInput input, .stSelectbox select {
        background: #161b22 !important;
        border: 1px solid #30363d !important;
        color: #e6edf3 !important;
        border-radius: 8px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Constants / Colours
# ---------------------------------------------------------------------------
COLORS = {
    "actual": "#58a6ff",
    "xgboost": "#3fb950",
    "lstm": "#ff7b72",
    "prophet": "#d2a8ff",
    "bg": "#0d1117",
    "card": "#161b22",
    "border": "#30363d",
    "text": "#e6edf3",
    "muted": "#8b949e",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color=COLORS["text"]),
    xaxis=dict(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"]),
    yaxis=dict(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"]),
    margin=dict(l=0, r=0, t=40, b=0),
    legend=dict(bgcolor="rgba(22,27,34,0.8)", bordercolor=COLORS["border"], borderwidth=1),
)

MODEL_DIR = "models"


# ---------------------------------------------------------------------------
# Data loaders (cached)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def load_energy_data():
    """Load raw PJME_hourly.csv."""
    path = os.path.join("data", "PJME_hourly.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["Datetime"])
    df = df.rename(columns={"PJME_MW": "load_mw"})
    df = df.set_index("Datetime").sort_index()
    return df


@st.cache_data(ttl=300, show_spinner=False)
def load_features():
    """Build full feature set."""
    from src.feature_engineering import build_feature_set, get_train_test

    df = build_feature_set()
    X_train, y_train, X_test, y_test = get_train_test(df)
    return df, X_train, y_train, X_test, y_test


@st.cache_data(ttl=300, show_spinner=False)
def load_xgb_predictions():
    path = os.path.join(MODEL_DIR, "xgboost_model.joblib")
    if not os.path.exists(path):
        return None
    model = joblib.load(path)
    _, _, _, X_test, y_test = _get_feature_sets()
    preds = model.predict(X_test)
    return pd.Series(preds, index=y_test.index, name="XGBoost")


@st.cache_data(ttl=300, show_spinner=False)
def load_prophet_predictions():
    path = os.path.join(MODEL_DIR, "prophet_forecast.joblib")
    if not os.path.exists(path):
        return None
    data = joblib.load(path)
    fc = data["forecast_test"]
    test_df = data["test_df"]
    fc.index = pd.to_datetime(test_df["ds"].values)
    return fc[["yhat", "yhat_lower", "yhat_upper"]]


@st.cache_data(ttl=300, show_spinner=False)
def load_prophet_72h():
    path = os.path.join(MODEL_DIR, "prophet_forecast.joblib")
    if not os.path.exists(path):
        return None
    data = joblib.load(path)
    return data.get("forecast_72h")


@st.cache_data(ttl=300, show_spinner=False)
def load_shap_data():
    path = os.path.join(MODEL_DIR, "shap_values.joblib")
    if not os.path.exists(path):
        return None, None
    d = joblib.load(path)
    return d["shap_values"], d["X_sample"]


@st.cache_data(ttl=300, show_spinner=False)
def load_metrics():
    path = os.path.join(MODEL_DIR, "model_metrics.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, index_col=0)


# Helper so other cached funcs can call feature loading safely
def _get_feature_sets():
    return load_features()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## ⚡ PJM Forecasting")
    st.markdown("---")

    st.markdown("### 📅 Date Range")
    date_start = st.date_input("From", value=pd.to_datetime("2017-01-01"))
    date_end = st.date_input("To", value=pd.to_datetime("2018-06-30"))

    st.markdown("### 🔭 Display Options")
    show_actual = st.checkbox("Actual Load", value=True)
    show_xgb = st.checkbox("XGBoost", value=True)
    show_prophet = st.checkbox("Prophet", value=True)
    show_ci = st.checkbox("Prophet CI Band", value=True)

    st.markdown("---")
    st.markdown("### 🔗 Services")
    st.markdown("- [FastAPI Docs](http://localhost:8000/docs)")
    st.markdown("- [MLflow UI](http://localhost:5000)")

    st.markdown("---")
    st.caption("PJM Interconnection · 2002–2018 · Hourly")


# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class='glass-card' style='margin-bottom:24px'>
        <div class='hero-title'>⚡ PJM Energy Consumption Forecasting</div>
        <div class='hero-subtitle'>
            XGBoost (Optuna) · LSTM · Prophet · MLflow · FastAPI · Streamlit
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Quick KPI row
raw_df = load_energy_data()
if raw_df is not None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Hours", f"{len(raw_df):,}")
    col2.metric("Avg Load", f"{raw_df['load_mw'].mean():,.0f} MW")
    col3.metric("Peak Load", f"{raw_df['load_mw'].max():,.0f} MW")
    col4.metric("Min Load", f"{raw_df['load_mw'].min():,.0f} MW")
else:
    st.warning("⚠️ data/PJME_hourly.csv not found. Run `python download_data.py` first.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🔮 Forecast", "🔬 Explainability", "📊 Model Metrics", "🧪 MLflow", "⚡ Live Predict"]
)


# ============================================================
# TAB 1 – FORECAST
# ============================================================
with tab1:
    st.markdown("<div class='section-header'>Actual vs Predicted – Test Period</div>", unsafe_allow_html=True)

    data_available = raw_df is not None
    if not data_available:
        st.info("Load PJME_hourly.csv to see forecast charts.")
    else:
        try:
            _, _, _, X_test, y_test = _get_feature_sets()
            date_start_ts = pd.Timestamp(date_start)
            date_end_ts = pd.Timestamp(date_end)

            actual = y_test[(y_test.index >= date_start_ts) & (y_test.index <= date_end_ts)]

            fig = go.Figure()

            # Actual
            if show_actual and len(actual) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=actual.index,
                        y=actual.values,
                        name="Actual",
                        line=dict(color=COLORS["actual"], width=1.5),
                        opacity=0.85,
                    )
                )

            # XGBoost
            if show_xgb:
                xgb_preds = load_xgb_predictions()
                if xgb_preds is not None:
                    xgb_slice = xgb_preds[
                        (xgb_preds.index >= date_start_ts) & (xgb_preds.index <= date_end_ts)
                    ]
                    fig.add_trace(
                        go.Scatter(
                            x=xgb_slice.index,
                            y=xgb_slice.values,
                            name="XGBoost",
                            line=dict(color=COLORS["xgboost"], width=1.5, dash="dot"),
                        )
                    )

            # Prophet
            if show_prophet:
                prophet_fc = load_prophet_predictions()
                if prophet_fc is not None:
                    p_slice = prophet_fc[
                        (prophet_fc.index >= date_start_ts) & (prophet_fc.index <= date_end_ts)
                    ]
                    if len(p_slice) > 0:
                        if show_ci:
                            fig.add_trace(
                                go.Scatter(
                                    x=list(p_slice.index) + list(p_slice.index[::-1]),
                                    y=list(p_slice["yhat_upper"]) + list(p_slice["yhat_lower"][::-1]),
                                    fill="toself",
                                    fillcolor="rgba(210,168,255,0.08)",
                                    line=dict(color="rgba(0,0,0,0)"),
                                    name="Prophet CI",
                                    showlegend=True,
                                )
                            )
                        fig.add_trace(
                            go.Scatter(
                                x=p_slice.index,
                                y=p_slice["yhat"],
                                name="Prophet",
                                line=dict(color=COLORS["prophet"], width=1.5, dash="dashdot"),
                            )
                        )

            fig.update_layout(
                **PLOTLY_LAYOUT,
                title="Energy Consumption – Actual vs Forecast (MW)",
                xaxis_title="Datetime",
                yaxis_title="Load (MW)",
                height=480,
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error building forecast chart: {e}")

    # 72h forward forecast
    st.markdown("<div class='section-header' style='margin-top:32px'>72-Hour Forward Forecast (Prophet)</div>", unsafe_allow_html=True)
    fc_72 = load_prophet_72h()
    if fc_72 is not None:
        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=list(fc_72["ds"]) + list(fc_72["ds"][::-1]),
                y=list(fc_72["yhat_upper"]) + list(fc_72["yhat_lower"][::-1]),
                fill="toself",
                fillcolor="rgba(210,168,255,0.10)",
                line=dict(color="rgba(0,0,0,0)"),
                name="95% CI",
            )
        )
        fig2.add_trace(
            go.Scatter(
                x=fc_72["ds"],
                y=fc_72["yhat"],
                name="Forecast",
                line=dict(color=COLORS["prophet"], width=2),
            )
        )
        fig2.update_layout(
            **PLOTLY_LAYOUT,
            title="72-Hour Forward Energy Forecast (MW)",
            xaxis_title="Datetime",
            yaxis_title="Load (MW)",
            height=340,
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Train the Prophet model (`python src/train_prophet.py`) to see the 72h forward forecast.")


# ============================================================
# TAB 2 – EXPLAINABILITY
# ============================================================
with tab2:
    st.markdown("<div class='section-header'>SHAP Feature Importance (XGBoost)</div>", unsafe_allow_html=True)

    shap_values, X_sample = load_shap_data()

    if shap_values is None:
        st.info("Train XGBoost (`python src/train_xgboost.py`) to generate SHAP values.")
    else:
        from src.feature_engineering import FEATURE_COLS

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        importance_df = (
            pd.DataFrame({"Feature": FEATURE_COLS, "Mean |SHAP|": mean_abs_shap})
            .sort_values("Mean |SHAP|", ascending=True)
            .tail(20)
        )

        fig3 = go.Figure(
            go.Bar(
                x=importance_df["Mean |SHAP|"],
                y=importance_df["Feature"],
                orientation="h",
                marker=dict(
                    color=importance_df["Mean |SHAP|"],
                    colorscale=[[0, "#1f6feb"], [0.5, "#388bfd"], [1, "#79c0ff"]],
                    showscale=False,
                ),
                text=importance_df["Mean |SHAP|"].round(1),
                textposition="outside",
            )
        )
        fig3.update_layout(
            **PLOTLY_LAYOUT,
            title="Top 20 Features by Mean |SHAP Value|",
            xaxis_title="Mean |SHAP Value| (MW)",
            yaxis_title="",
            height=600,
        )
        st.plotly_chart(fig3, use_container_width=True)

        # Static SHAP summary image
        shap_img = os.path.join(MODEL_DIR, "shap_summary.png")
        if os.path.exists(shap_img):
            st.markdown("<div class='section-header' style='margin-top:32px'>SHAP Summary Plot</div>", unsafe_allow_html=True)
            st.image(shap_img, caption="SHAP Summary – direction and magnitude of feature contributions", use_container_width=True)


# ============================================================
# TAB 3 – MODEL METRICS
# ============================================================
with tab3:
    st.markdown("<div class='section-header'>Cross-Model Performance Comparison</div>", unsafe_allow_html=True)

    metrics_df = load_metrics()

    if metrics_df is None:
        st.info("Run `python src/evaluate.py` to generate model metrics.")
    else:
        col_a, col_b = st.columns([1, 1])

        with col_a:
            st.dataframe(
                metrics_df.style
                .format("{:.2f}")
                .highlight_min(color="#0e4429", axis=0)
                .highlight_max(color="#6e1408", axis=0),
                use_container_width=True,
                height=200,
            )

        with col_b:
            # Grouped bar chart
            fig4 = go.Figure()
            bar_colors = [COLORS["xgboost"], COLORS["prophet"], COLORS["lstm"]]
            metrics_to_plot = [c for c in metrics_df.columns if c in ["MAE (MW)", "RMSE (MW)", "MAPE (%)"]]
            for i, model_name in enumerate(metrics_df.index):
                fig4.add_trace(
                    go.Bar(
                        name=model_name,
                        x=metrics_to_plot,
                        y=[metrics_df.loc[model_name, m] for m in metrics_to_plot],
                        marker_color=bar_colors[i % len(bar_colors)],
                    )
                )
            fig4.update_layout(
                **PLOTLY_LAYOUT,
                barmode="group",
                title="Metrics Comparison",
                height=300,
                legend=dict(orientation="h", y=1.1),
            )
            st.plotly_chart(fig4, use_container_width=True)

        # Per-metric podium
        st.markdown("<div class='section-header' style='margin-top:24px'>Best Model per Metric</div>", unsafe_allow_html=True)
        cols = st.columns(len(metrics_to_plot))
        for col, metric in zip(cols, metrics_to_plot):
            best_model = metrics_df[metric].idxmin()
            best_val = metrics_df[metric].min()
            col.metric(f"Best {metric}", f"{best_val:.2f}", best_model)


# ============================================================
# TAB 4 – MLFLOW
# ============================================================
with tab4:
    st.markdown("<div class='section-header'>MLflow Experiment Tracking</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class='glass-card'>
            <p>All training runs are tracked in <code>mlruns/</code> using MLflow.</p>
            <p>Start the MLflow UI with:</p>
            <pre style='background:#0d1117;padding:12px;border-radius:8px;color:#79c0ff'>mlflow ui --host 0.0.0.0 --port 5000</pre>
            <p>Or via Docker Compose:</p>
            <pre style='background:#0d1117;padding:12px;border-radius:8px;color:#79c0ff'>docker-compose up mlflow</pre>
            <p>Then open: <a href='http://localhost:5000' target='_blank'>http://localhost:5000</a></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Show mlruns contents if available
    mlruns_path = "mlruns"
    if os.path.exists(mlruns_path):
        experiments = [d for d in os.listdir(mlruns_path) if os.path.isdir(os.path.join(mlruns_path, d))]
        st.markdown(f"**Experiment directories detected:** {len(experiments)}")

        run_rows = []
        for exp_id in experiments:
            exp_path = os.path.join(mlruns_path, exp_id)
            for run_id in os.listdir(exp_path):
                meta_path = os.path.join(exp_path, run_id, "meta.yaml")
                tags_path = os.path.join(exp_path, run_id, "tags")
                if os.path.exists(meta_path):
                    import yaml as _yaml
                    try:
                        with open(meta_path) as f:
                            meta = _yaml.safe_load(f)
                        run_name = meta.get("run_name", run_id[:8])
                        status = meta.get("status", "")
                        run_rows.append({"Run": run_name, "Status": status, "ID": run_id[:8]})
                    except Exception:
                        pass

        if run_rows:
            st.dataframe(pd.DataFrame(run_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No MLflow runs found yet. Train a model first.")


# ============================================================
# TAB 5 – LIVE PREDICT
# ============================================================
with tab5:
    st.markdown("<div class='section-header'>Interactive Single-Point Forecast</div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='glass-card'>"
        "Enter weather conditions and datetime to get an instant XGBoost energy forecast."
        "</div><br>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        pred_date = st.date_input("📅 Date", value=pd.to_datetime("2025-06-01"), key="pred_date")
        pred_hour = st.slider("🕐 Hour of Day", 0, 23, 14)
        pred_temp = st.number_input("🌡️ Temperature (°C)", value=22.0, min_value=-30.0, max_value=50.0, step=0.5)
    with c2:
        pred_rhum = st.number_input("💧 Relative Humidity (%)", value=65.0, min_value=0.0, max_value=100.0, step=1.0)
        pred_wspd = st.number_input("💨 Wind Speed (km/h)", value=12.0, min_value=0.0, max_value=150.0, step=1.0)

    if st.button("⚡ Forecast Now", use_container_width=False):
        model_path = os.path.join(MODEL_DIR, "xgboost_model.joblib")
        if not os.path.exists(model_path):
            st.error("XGBoost model not found. Run `python src/train_xgboost.py` first.")
        else:
            dt_str = f"{pred_date} {pred_hour:02d}:00"
            try:
                from src.predict import predict as do_predict

                result = do_predict(dt_str, pred_temp, pred_rhum, pred_wspd)

                st.markdown("---")
                res_col1, res_col2, res_col3 = st.columns(3)
                res_col1.metric("⚡ Forecast", f"{result['forecast_mw']:,.1f} MW")
                res_col2.metric("📅 Datetime", dt_str)
                res_col3.metric("🤖 Model", result["model"].title())

                # Quick load range gauge
                load_pct = min(result["forecast_mw"] / 60_000, 1.0) * 100
                fig5 = go.Figure(
                    go.Indicator(
                        mode="gauge+number+delta",
                        value=result["forecast_mw"],
                        number={"suffix": " MW", "font": {"size": 28, "color": COLORS["actual"]}},
                        gauge={
                            "axis": {"range": [0, 60_000], "tickcolor": COLORS["muted"]},
                            "bar": {"color": COLORS["xgboost"]},
                            "bgcolor": COLORS["card"],
                            "bordercolor": COLORS["border"],
                            "steps": [
                                {"range": [0, 20_000], "color": "#0e4429"},
                                {"range": [20_000, 40_000], "color": "#1f6feb"},
                                {"range": [40_000, 60_000], "color": "#6e1408"},
                            ],
                            "threshold": {
                                "line": {"color": COLORS["actual"], "width": 3},
                                "thickness": 0.75,
                                "value": result["forecast_mw"],
                            },
                        },
                        title={"text": "Forecasted Energy Load", "font": {"color": COLORS["text"]}},
                    )
                )
                fig5.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter", color=COLORS["text"]),
                    height=320,
                )
                st.plotly_chart(fig5, use_container_width=True)

            except Exception as e:
                st.error(f"Prediction failed: {e}")
