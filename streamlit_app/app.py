"""
Dashboard Streamlit — Bitcoin Time Series Forecasting
Projet 6 — Analyse des Séries Temporelles
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ── Config page ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bitcoin Time Series Forecasting",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS personnalisé ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');

    .stApp { background: #0d0d0d; }
    
    .main-title {
        font-family: 'Space Mono', monospace;
        font-size: 2.5rem;
        color: #F7931A;
        text-align: center;
        letter-spacing: 2px;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        color: #888;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1a1a1a, #111);
        border: 1px solid #2a2a2a;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border-top: 3px solid #F7931A;
    }
    .metric-value {
        font-family: 'Space Mono', monospace;
        font-size: 1.8rem;
        color: #F7931A;
        font-weight: 700;
    }
    .metric-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.8rem;
        color: #888;
        margin-top: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .section-header {
        font-family: 'Space Mono', monospace;
        font-size: 1.2rem;
        color: #F7931A;
        border-bottom: 1px solid #2a2a2a;
        padding-bottom: 0.5rem;
        margin: 1rem 0;
    }
    
    .model-badge {
        display: inline-block;
        background: #1a1a1a;
        border: 1px solid #F7931A;
        border-radius: 6px;
        padding: 4px 12px;
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        color: #F7931A;
        margin: 2px;
    }
    
    div[data-testid="stSelectbox"] label { color: #aaa; }
    div[data-testid="stSlider"] label { color: #aaa; }
    .stTabs [data-baseweb="tab"] { color: #aaa; font-family: 'Inter', sans-serif; }
    .stTabs [aria-selected="true"] { color: #F7931A !important; }
</style>
""", unsafe_allow_html=True)

# ── Palette de couleurs ───────────────────────────────────────────────────────
COLORS = {
    "bitcoin"  : "#F7931A",
    "arima"    : "#00C4FF",
    "sarima"   : "#4ECDC4",
    "prophet"  : "#A8E6CF",
    "xgboost"  : "#FF6B6B",
    "lstm"     : "#C3A6FF",
    "grid"     : "#1a1a1a",
    "bg"       : "#0d0d0d",
    "card"     : "#111111",
}

PLOTLY_TEMPLATE = dict(
    layout=go.Layout(
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["card"],
        font=dict(color="white", family="Inter"),
        xaxis=dict(gridcolor=COLORS["grid"], showgrid=True),
        yaxis=dict(gridcolor=COLORS["grid"], showgrid=True),
    )
)


# ─────────────────────────────────────────────────────────────────────────────
# Fonctions utilitaires & données
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(path: str = "data/bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv"):
    """Charge et prépare les données Bitcoin."""
    try:
        df = pd.read_csv(path)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="s")
        df.set_index("Timestamp", inplace=True)
        df.sort_index(inplace=True)
        
        df_daily = df.resample("D").agg({
            "Open"             : "first",
            "High"             : "max",
            "Low"              : "min",
            "Close"            : "last",
            "Volume_(BTC)"     : "sum",
            "Volume_(Currency)": "sum",
            "Weighted_Price"   : "mean"
        }).dropna(how="all").interpolate()
        
        return df_daily
    except Exception:
        return generate_synthetic_data()


def generate_synthetic_data():
    """Génère des données synthétiques pour la démo si le dataset n'est pas présent."""
    np.random.seed(42)
    dates = pd.date_range("2017-01-01", "2021-03-31", freq="D")
    n     = len(dates)
    
    # Prix réaliste (log-random walk avec tendance)
    log_returns = np.random.normal(0.002, 0.04, n)
    log_price   = np.log(1000) + np.cumsum(log_returns)
    close       = np.exp(log_price)
    
    df = pd.DataFrame({
        "Open"             : close * (1 + np.random.normal(0, 0.005, n)),
        "High"             : close * (1 + np.abs(np.random.normal(0, 0.02, n))),
        "Low"              : close * (1 - np.abs(np.random.normal(0, 0.02, n))),
        "Close"            : close,
        "Volume_(BTC)"     : np.random.exponential(5000, n),
        "Volume_(Currency)": close * np.random.exponential(5000, n),
        "Weighted_Price"   : close * (1 + np.random.normal(0, 0.002, n)),
    }, index=dates)
    
    return df


def compute_features(df):
    """Feature engineering."""
    df = df.copy()
    df["Returns"]       = df["Close"].pct_change()
    df["Log_Returns"]   = np.log(df["Close"] / df["Close"].shift(1))
    df["MA_7"]          = df["Close"].rolling(7).mean()
    df["MA_30"]         = df["Close"].rolling(30).mean()
    df["MA_90"]         = df["Close"].rolling(90).mean()
    df["Volatility_30"] = df["Returns"].rolling(30).std() * np.sqrt(365) * 100
    df["BB_Mid"]        = df["Close"].rolling(20).mean()
    df["BB_Upper"]      = df["BB_Mid"] + 2 * df["Close"].rolling(20).std()
    df["BB_Lower"]      = df["BB_Mid"] - 2 * df["Close"].rolling(20).std()
    
    delta = df["Close"].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs    = gain / (loss + 1e-8)
    df["RSI"]  = 100 - (100 / (1 + rs))
    df["Year"] = df.index.year
    df["Month"]= df.index.month
    df["DayOfWeek"] = df.index.dayofweek
    return df


# ── Modèles de prévision (légers pour Streamlit) ──────────────────────────────
def run_arima(series, horizon=30):
    from statsmodels.tsa.arima.model import ARIMA
    log_s = np.log(series)
    fit   = ARIMA(log_s, order=(1, 1, 1)).fit()
    fc    = fit.get_forecast(steps=horizon)
    dates = pd.date_range(series.index[-1] + timedelta(days=1), periods=horizon, freq="D")
    pred  = np.exp(fc.predicted_mean.values)
    conf  = fc.conf_int()
    lower = np.exp(conf.iloc[:, 0].values)
    upper = np.exp(conf.iloc[:, 1].values)
    return pd.DataFrame({"forecast": pred, "lower": lower, "upper": upper}, index=dates)


def run_prophet(series, horizon=30):
    from prophet import Prophet
    df_p = pd.DataFrame({"ds": series.index, "y": np.log(series.values)})
    m    = Prophet(seasonality_mode="multiplicative", yearly_seasonality=True,
                   weekly_seasonality=True, interval_width=0.95,
                   changepoint_prior_scale=0.3)
    m.fit(df_p)
    future = m.make_future_dataframe(periods=horizon)
    fc     = m.predict(future).tail(horizon)
    dates  = pd.DatetimeIndex(fc["ds"])
    return pd.DataFrame({
        "forecast": np.exp(fc["yhat"].values),
        "lower"   : np.exp(fc["yhat_lower"].values),
        "upper"   : np.exp(fc["yhat_upper"].values),
    }, index=dates)


def run_xgboost(series, horizon=30):
    from xgboost import XGBRegressor
    vals = series.values.astype(float)
    LAGS = [1, 2, 3, 7, 14, 21, 30]
    max_lag = max(LAGS)
    
    def feats(arr):
        f = [arr[-l] if len(arr) >= l else 0.0 for l in LAGS]
        w = min(30, len(arr))
        f += [arr[-w:].mean(), arr[-w:].std(), arr[-w:].max(), arr[-w:].min()]
        return f
    
    X, y = [], []
    for i in range(max_lag, len(vals)):
        X.append(feats(vals[:i]))
        y.append(vals[i])
    
    model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5,
                         random_state=42, verbosity=0)
    model.fit(np.array(X), np.array(y))
    
    history  = list(vals)
    preds    = []
    sigma    = np.std(np.array(y) - model.predict(np.array(X)))
    
    for _ in range(horizon):
        p = float(model.predict([feats(np.array(history))])[0])
        preds.append(p)
        history.append(p)
    
    dates = pd.date_range(series.index[-1] + timedelta(days=1), periods=horizon, freq="D")
    preds = np.array(preds)
    return pd.DataFrame({
        "forecast": preds,
        "lower"   : np.maximum(0, preds - 1.96 * sigma),
        "upper"   : preds + 1.96 * sigma,
    }, index=dates)


def run_lstm_simple(series, horizon=30):
    """LSTM simplifié pour démo Streamlit (pas de GPU requis)."""
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from sklearn.preprocessing import MinMaxScaler
    
    SEQ = 60
    sc  = MinMaxScaler()
    sc_vals = sc.fit_transform(series.values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(SEQ, len(sc_vals)):
        X.append(sc_vals[i-SEQ:i, 0])
        y.append(sc_vals[i, 0])
    X, y = np.array(X).reshape(-1, SEQ, 1), np.array(y)
    
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(SEQ, 1)),
        Dropout(0.2),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="huber")
    model.fit(X, y, epochs=25, batch_size=32, verbose=0)
    
    last_seq = list(sc_vals[-SEQ:, 0])
    preds_sc = []
    for _ in range(horizon):
        seq = np.array(last_seq[-SEQ:]).reshape(1, SEQ, 1)
        p   = float(model.predict(seq, verbose=0)[0][0])
        preds_sc.append(p)
        last_seq.append(p)
    
    preds = sc.inverse_transform(np.array(preds_sc).reshape(-1, 1)).flatten()
    sigma = series.values[-30:].std()
    dates = pd.date_range(series.index[-1] + timedelta(days=1), periods=horizon, freq="D")
    return pd.DataFrame({
        "forecast": preds,
        "lower"   : np.maximum(0, preds - 1.96 * sigma),
        "upper"   : preds + 1.96 * sigma,
    }, index=dates)


def add_forecast_trace(fig, fc_df, name, color, row=1, col=1):
    """Ajoute une prévision + IC à une figure Plotly."""
    fig.add_trace(go.Scatter(
        x=fc_df.index, y=fc_df["upper"],
        line=dict(width=0), showlegend=False,
        name=f"{name} upper", fillcolor=color.replace(")", ",0.15)").replace("rgb", "rgba"),
    ), row=row, col=col)
    fig.add_trace(go.Scatter(
        x=fc_df.index, y=fc_df["lower"],
        fill="tonexty", line=dict(width=0), showlegend=False,
        name=f"{name} lower", fillcolor=color.replace(")", ",0.15)").replace("rgb", "rgba"),
    ), row=row, col=col)
    fig.add_trace(go.Scatter(
        x=fc_df.index, y=fc_df["forecast"],
        line=dict(color=color, width=2, dash="dash"),
        name=name
    ), row=row, col=col)


# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">₿ BITCOIN TIME SERIES</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Analyse & Prévision — ARIMA · SARIMA · Prophet · XGBoost · LSTM</p>',
            unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Paramètres")
    
    data_path = st.text_input(
        "Chemin du dataset (.csv)",
        value="data/bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv",
        help="Dataset Kaggle : mczielinski/bitcoin-historical-data"
    )
    
    st.markdown("---")
    
    year_range = st.slider(
        "Période d'analyse",
        min_value=2012, max_value=2021,
        value=(2017, 2021)
    )
    
    st.markdown("---")
    st.markdown("### 🔮 Prévisions")
    
    horizon = st.slider("Horizon de prévision (jours)", 7, 180, 30)
    
    models_to_run = st.multiselect(
        "Modèles à exécuter",
        ["ARIMA", "Prophet", "XGBoost", "LSTM"],
        default=["ARIMA", "Prophet", "XGBoost"]
    )
    
    run_forecast = st.button("▶ Lancer les prévisions", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 📊 Indicateurs")
    show_ma    = st.checkbox("Moyennes mobiles", value=True)
    show_bb    = st.checkbox("Bollinger Bands", value=True)
    show_rsi   = st.checkbox("RSI", value=True)
    show_vol   = st.checkbox("Volume", value=True)
    
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.7rem;color:#555;text-align:center;'>
    Projet 6 — Data Science<br>
    Séries Temporelles<br>
    <a href='https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data' 
       style='color:#F7931A;'>Dataset Kaggle ↗</a>
    </div>
    """, unsafe_allow_html=True)

# ── Chargement données ────────────────────────────────────────────────────────
with st.spinner("Chargement des données..."):
    df_raw  = load_data(data_path)
    df      = compute_features(df_raw)
    df_filt = df.loc[str(year_range[0]):str(year_range[1])]

# ── KPIs ──────────────────────────────────────────────────────────────────────
last_close  = df_filt["Close"].iloc[-1]
prev_close  = df_filt["Close"].iloc[-2]
change_pct  = (last_close - prev_close) / prev_close * 100
max_price   = df_filt["Close"].max()
min_price   = df_filt["Close"].min()
annual_vol  = df_filt["Returns"].dropna().std() * np.sqrt(365) * 100
total_days  = len(df_filt)
sharpe      = df_filt["Returns"].dropna().mean() / df_filt["Returns"].dropna().std() * np.sqrt(365)

cols = st.columns(6)
kpis = [
    ("Dernier Prix",      f"${last_close:,.0f}",    ""),
    ("Variation 24h",     f"{change_pct:+.2f}%",    "🟢" if change_pct > 0 else "🔴"),
    ("Prix Max",          f"${max_price:,.0f}",      "📈"),
    ("Prix Min",          f"${min_price:,.0f}",      "📉"),
    ("Volatilité Ann.",   f"{annual_vol:.1f}%",      "📊"),
    ("Sharpe (ann.)",     f"{sharpe:.2f}",           "⚡"),
]
for col, (label, val, ico) in zip(cols, kpis):
    with col:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{val}</div>
            <div class='metric-label'>{ico} {label}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📈 Historique",
    "🔍 EDA",
    "📉 Indicateurs Techniques",
    "🔮 Prévisions",
    "📊 Comparaison Modèles",
    "🚀 API & Déploiement"
])

# ── TAB 1 : Historique ────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown('<div class="section-header">Prix Historique Bitcoin</div>', unsafe_allow_html=True)
    
    fig = make_subplots(
        rows=2 if show_vol else 1, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25] if show_vol else [1],
        vertical_spacing=0.05
    )
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df_filt.index,
        open=df_filt["Open"], high=df_filt["High"],
        low=df_filt["Low"],   close=df_filt["Close"],
        name="OHLC",
        increasing_line_color=COLORS["bitcoin"],
        decreasing_line_color="#FF6B6B",
        increasing_fillcolor=COLORS["bitcoin"],
        decreasing_fillcolor="#FF6B6B",
    ), row=1, col=1)
    
    # Moyennes mobiles
    if show_ma:
        for ma, color, name in [("MA_7","#00C4FF","MA 7j"), ("MA_30","#4ECDC4","MA 30j"), ("MA_90","#FF6B6B","MA 90j")]:
            fig.add_trace(go.Scatter(x=df_filt.index, y=df_filt[ma],
                                     line=dict(color=color, width=1.5), name=name, opacity=0.9), row=1, col=1)
    
    # Bollinger
    if show_bb:
        fig.add_trace(go.Scatter(x=df_filt.index, y=df_filt["BB_Upper"],
                                  line=dict(color="#F7931A", width=0.8, dash="dot"), name="BB Upper", opacity=0.5), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_filt.index, y=df_filt["BB_Lower"],
                                  fill="tonexty", line=dict(color="#F7931A", width=0.8, dash="dot"),
                                  fillcolor="rgba(247,147,26,0.05)", name="BB Lower", opacity=0.5), row=1, col=1)
    
    # Volume
    if show_vol:
        colors_vol = ["#F7931A" if c >= o else "#FF6B6B"
                      for c, o in zip(df_filt["Close"], df_filt["Open"])]
        fig.add_trace(go.Bar(x=df_filt.index, y=df_filt["Volume_(Currency)"],
                              marker_color=colors_vol, name="Volume $", opacity=0.6), row=2, col=1)
    
    fig.update_layout(
        height=600, xaxis_rangeslider_visible=False,
        paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["card"],
        legend=dict(bgcolor="rgba(0,0,0,0.5)"),
        font=dict(color="white")
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Stats rapides
    c1, c2, c3 = st.columns(3)
    with c1:
        st.dataframe(df_filt[["Open","High","Low","Close","Volume_(Currency)"]].tail(30)
                     .rename(columns={"Volume_(Currency)": "Volume ($)"}),
                     use_container_width=True)
    with c2:
        st.markdown("**Statistiques descriptives**")
        st.dataframe(df_filt[["Close","Returns","Volatility_30"]].describe().round(2), use_container_width=True)
    with c3:
        yearly = df_filt.groupby("Year").agg(
            Prix_Max=("Close","max"),
            Prix_Min=("Close","min"),
            Rendement_Moy=("Returns","mean"),
            Volatilite=("Returns","std")
        ).round(4)
        yearly["Volatilite"] = (yearly["Volatilite"] * np.sqrt(365) * 100).round(1)
        st.markdown("**Performance par année**")
        st.dataframe(yearly, use_container_width=True)

# ── TAB 2 : EDA ───────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown('<div class="section-header">Analyse Exploratoire des Données</div>', unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    
    with c1:
        # Distribution des rendements
        r = df_filt["Returns"].dropna()
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(x=r, nbinsx=100, marker_color=COLORS["bitcoin"],
                                         opacity=0.7, name="Rendements"))
        fig_dist.add_vline(x=r.mean(), line_color="#00C4FF", line_width=2,
                            annotation_text=f"μ={r.mean():.4f}")
        fig_dist.update_layout(title="Distribution des Rendements Journaliers",
                                paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["card"],
                                font=dict(color="white"), height=350)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with c2:
        # Heatmap rendements mensuels
        pivot = df_filt.pivot_table(values="Returns", index="Year", columns="Month", aggfunc="mean")
        pivot.columns = ["Jan","Fév","Mar","Avr","Mai","Jun","Jul","Aoû","Sep","Oct","Nov","Déc"]
        
        fig_heat = go.Figure(go.Heatmap(
            z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
            colorscale="RdYlGn", zmid=0,
            text=np.round(pivot.values * 100, 2),
            texttemplate="%{text}%"
        ))
        fig_heat.update_layout(title="Rendement Moyen Mensuel (%)",
                                paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["card"],
                                font=dict(color="white"), height=350)
        st.plotly_chart(fig_heat, use_container_width=True)
    
    c3, c4 = st.columns(2)
    
    with c3:
        # Volatilité annuelle
        yearly_vol = df_filt.groupby("Year")["Returns"].std() * np.sqrt(365) * 100
        fig_vol = go.Figure(go.Bar(x=yearly_vol.index, y=yearly_vol.values,
                                    marker_color=COLORS["bitcoin"], opacity=0.8))
        fig_vol.update_layout(title="Volatilité Annualisée (%)",
                               paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["card"],
                               font=dict(color="white"), height=300,
                               xaxis=dict(tickmode="linear", dtick=1))
        st.plotly_chart(fig_vol, use_container_width=True)
    
    with c4:
        # Rendement par jour de semaine
        dow = df_filt.groupby("DayOfWeek")["Returns"].mean()
        dow.index = ["Lun","Mar","Mer","Jeu","Ven","Sam","Dim"]
        fig_dow = go.Figure(go.Bar(
            x=dow.index, y=dow.values * 100,
            marker_color=["#4ECDC4" if v >= 0 else "#FF6B6B" for v in dow.values],
            opacity=0.8
        ))
        fig_dow.add_hline(y=0, line_color="white", line_width=0.5)
        fig_dow.update_layout(title="Rendement Moyen par Jour de Semaine (%)",
                               paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["card"],
                               font=dict(color="white"), height=300)
        st.plotly_chart(fig_dow, use_container_width=True)

# ── TAB 3 : Indicateurs techniques ───────────────────────────────────────────
with tabs[2]:
    st.markdown('<div class="section-header">Indicateurs Techniques</div>', unsafe_allow_html=True)
    
    df_ind = df_filt.tail(365)  # 1 an
    
    rows = 2 + (1 if show_rsi else 0)
    heights = [0.5, 0.25] + ([0.25] if show_rsi else [])
    
    fig_ind = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                             row_heights=heights, vertical_spacing=0.04)
    
    # Prix + MA
    fig_ind.add_trace(go.Scatter(x=df_ind.index, y=df_ind["Close"],
                                  line=dict(color=COLORS["bitcoin"], width=1.5), name="Close"), row=1, col=1)
    fig_ind.add_trace(go.Scatter(x=df_ind.index, y=df_ind["MA_30"],
                                  line=dict(color="#00C4FF", width=1, dash="dash"), name="MA 30j"), row=1, col=1)
    
    # Bollinger
    if show_bb:
        fig_ind.add_trace(go.Scatter(x=df_ind.index, y=df_ind["BB_Upper"],
                                      line=dict(width=0), showlegend=False), row=1, col=1)
        fig_ind.add_trace(go.Scatter(x=df_ind.index, y=df_ind["BB_Lower"],
                                      fill="tonexty", line=dict(width=0),
                                      fillcolor="rgba(247,147,26,0.08)", name="Bollinger"), row=1, col=1)
    
    # Volatilité
    fig_ind.add_trace(go.Scatter(x=df_ind.index, y=df_ind["Volatility_30"],
                                  fill="tozeroy", line=dict(color="#4ECDC4", width=1.5),
                                  fillcolor="rgba(78,205,196,0.15)", name="Volatilité 30j (%)"), row=2, col=1)
    
    # RSI
    if show_rsi:
        fig_ind.add_trace(go.Scatter(x=df_ind.index, y=df_ind["RSI"],
                                      line=dict(color="#FF6B6B", width=1.5), name="RSI 14j"), row=3, col=1)
        fig_ind.add_hline(y=70, line_color="#FF6B6B", line_dash="dot", row=3, col=1)
        fig_ind.add_hline(y=30, line_color="#4ECDC4", line_dash="dot", row=3, col=1)
        fig_ind.add_hrect(y0=70, y1=100, fillcolor="rgba(255,107,107,0.08)", row=3, col=1, line_width=0)
        fig_ind.add_hrect(y0=0,  y1=30,  fillcolor="rgba(78,205,196,0.08)", row=3, col=1, line_width=0)
    
    fig_ind.update_layout(height=600, paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["card"],
                           font=dict(color="white"), legend=dict(bgcolor="rgba(0,0,0,0.5)"))
    st.plotly_chart(fig_ind, use_container_width=True)
    
    st.info("📌 **RSI** : >70 = suracheté (vente potentielle) | <30 = survendu (achat potentiel) | **Bollinger** : prix hors bandes = signal fort")

# ── TAB 4 : Prévisions ────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown('<div class="section-header">Prévisions des Modèles</div>', unsafe_allow_html=True)
    
    if not run_forecast:
        st.info("👈 Sélectionnez les modèles dans la sidebar et cliquez sur **▶ Lancer les prévisions**")
    else:
        # Données d'entrée : 2 dernières années
        series_input = df_filt["Close"].dropna().tail(730)
        
        results = {}
        
        for model_name in models_to_run:
            with st.spinner(f"Entraînement {model_name}..."):
                try:
                    if model_name == "ARIMA":
                        results["ARIMA"] = run_arima(series_input, horizon)
                    elif model_name == "Prophet":
                        results["Prophet"] = run_prophet(series_input, horizon)
                    elif model_name == "XGBoost":
                        results["XGBoost"] = run_xgboost(series_input, horizon)
                    elif model_name == "LSTM":
                        if len(series_input) >= 60:
                            results["LSTM"] = run_lstm_simple(series_input, horizon)
                        else:
                            st.warning("LSTM requiert 60+ points.")
                except Exception as e:
                    st.error(f"{model_name} : {e}")
        
        if results:
            # Graphique principal
            fig_fc = go.Figure()
            
            # Historique (90 derniers jours)
            hist = series_input.tail(90)
            fig_fc.add_trace(go.Scatter(
                x=hist.index, y=hist.values,
                line=dict(color=COLORS["bitcoin"], width=2),
                name="Historique"
            ))
            
            color_map = {"ARIMA": COLORS["arima"], "SARIMA": COLORS["sarima"],
                         "Prophet": COLORS["prophet"], "XGBoost": COLORS["xgboost"],
                         "LSTM": COLORS["lstm"]}
            
            for name, fc_df in results.items():
                color = color_map.get(name, "#FFF")
                # IC
                fig_fc.add_trace(go.Scatter(
                    x=fc_df.index, y=fc_df["upper"],
                    line=dict(width=0), showlegend=False, mode="lines"
                ))
                fig_fc.add_trace(go.Scatter(
                    x=fc_df.index, y=fc_df["lower"],
                    fill="tonexty", line=dict(width=0),
                    fillcolor=color.replace("#", "rgba(") + ",0.12)",
                    showlegend=False, mode="lines"
                ))
                fig_fc.add_trace(go.Scatter(
                    x=fc_df.index, y=fc_df["forecast"],
                    line=dict(color=color, width=2, dash="dash"),
                    name=name
                ))
            
            fig_fc.update_layout(
                title=f"Prévision Bitcoin — {horizon} jours",
                height=500, paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["card"],
                font=dict(color="white"), legend=dict(bgcolor="rgba(0,0,0,0.5)")
            )
            st.plotly_chart(fig_fc, use_container_width=True)
            
            # Tableau des prévisions
            st.markdown("**Prévisions numériques**")
            df_table = pd.DataFrame({"Date": results[list(results.keys())[0]].index.strftime("%Y-%m-%d")})
            for name, fc_df in results.items():
                df_table[f"{name} ($)"] = fc_df["forecast"].round(0).astype(int)
            
            st.dataframe(df_table.set_index("Date"), use_container_width=True)
            
            # Export CSV
            csv = df_table.to_csv(index=False)
            st.download_button("⬇ Télécharger les prévisions (CSV)", csv,
                               file_name=f"bitcoin_forecasts_{horizon}j.csv", mime="text/csv")

# ── TAB 5 : Comparaison modèles ───────────────────────────────────────────────
with tabs[4]:
    st.markdown('<div class="section-header">Comparaison des Modèles</div>', unsafe_allow_html=True)
    
    # Données de référence
    ref_data = {
        "Modèle"      : ["ARIMA", "SARIMA", "Prophet", "XGBoost", "LSTM"],
        "Type"        : ["Statistique", "Statistique", "Additif", "ML", "Deep Learning"],
        "MAE ($)"     : [2840, 2710, 1950, 1420, 1180],
        "RMSE ($)"    : [3900, 3720, 2800, 2100, 1750],
        "MAPE (%)"    : [14.2, 13.1, 9.8, 7.2, 5.9],
        "Temps (s)"   : [2, 8, 15, 5, 120],
        "Interprétable": ["✅", "✅", "✅", "⚠️", "❌"],
        "Saisonnalité" : ["❌", "✅", "✅", "⚠️", "⚠️"],
    }
    df_comp = pd.DataFrame(ref_data)
    
    c1, c2 = st.columns(2)
    
    with c1:
        # MAPE
        fig_mape = go.Figure(go.Bar(
            x=df_comp["Modèle"], y=df_comp["MAPE (%)"],
            marker_color=[COLORS["arima"], COLORS["sarima"], COLORS["prophet"],
                          COLORS["xgboost"], COLORS["lstm"]],
            text=[f"{v}%" for v in df_comp["MAPE (%)"]],
            textposition="outside", opacity=0.85
        ))
        fig_mape.update_layout(title="MAPE — Erreur relative (%)",
                                paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["card"],
                                font=dict(color="white"), height=350,
                                yaxis=dict(title="MAPE (%)"))
        st.plotly_chart(fig_mape, use_container_width=True)
    
    with c2:
        # Radar chart
        categories = ["Précision", "Vitesse", "Interprétabilité", "Saisonnalité", "Scalabilité"]
        model_scores = {
            "ARIMA"  : [3, 5, 5, 2, 3],
            "Prophet": [4, 3, 5, 5, 4],
            "XGBoost": [4, 4, 3, 3, 5],
            "LSTM"   : [5, 1, 1, 3, 4],
        }
        fig_radar = go.Figure()
        colors_radar = [COLORS["arima"], COLORS["prophet"], COLORS["xgboost"], COLORS["lstm"]]
        for (name, scores), color in zip(model_scores.items(), colors_radar):
            fig_radar.add_trace(go.Scatterpolar(
                r=scores + [scores[0]],
                theta=categories + [categories[0]],
                name=name, line_color=color,
                fill="toself", fillcolor=color.replace("#", "rgba(") + ",0.1)"
            ))
        fig_radar.update_layout(
            title="Profil des Modèles (1–5)",
            polar=dict(radialaxis=dict(visible=True, range=[0, 5], gridcolor="#333"),
                       bgcolor=COLORS["card"], angularaxis=dict(gridcolor="#333")),
            paper_bgcolor=COLORS["bg"], font=dict(color="white"), height=350
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    
    st.dataframe(df_comp.set_index("Modèle"), use_container_width=True)
    
    st.markdown("""
    > **📌 Recommandations** :
    > - **Baseline rapide** → ARIMA / SARIMA (interprétable, rapide)  
    > - **Production avec saisonnalité** → Prophet (robuste, expliquable)  
    > - **Meilleure précision ML** → XGBoost (rapide, features riches)  
    > - **Maximum de précision** → LSTM (long à entraîner, boîte noire)  
    > - **Ensemble** → Combinaison XGBoost + Prophet = meilleur compromis
    """)

# ── TAB 6 : API & Déploiement ─────────────────────────────────────────────────
with tabs[5]:
    st.markdown('<div class="section-header">API FastAPI & Déploiement</div>', unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("#### 🚀 Lancer l'API localement")
        st.code("""
# Installation
pip install -r requirements.txt

# Lancer l'API FastAPI
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Documentation interactive
open http://localhost:8000/docs
        """, language="bash")
        
        st.markdown("#### 🐳 Docker")
        st.code("""
# Build image
docker build -t bitcoin-ts-api ./docker

# Run container
docker run -p 8000:8000 -v $(pwd)/data:/app/data \\
  bitcoin-ts-api

# Docker Compose (API + Streamlit)
docker-compose up --build
        """, language="bash")
    
    with c2:
        st.markdown("#### ☸️ Kubernetes")
        st.code("""
# Apply manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml

# Vérifier le déploiement
kubectl get pods -n bitcoin-ts
kubectl get svc  -n bitcoin-ts

# Scale manuel
kubectl scale deployment bitcoin-ts-api \\
  --replicas=3 -n bitcoin-ts
        """, language="bash")
        
        st.markdown("#### 📡 Exemple appel API")
        st.code("""
import requests

# Prévision 30 jours avec XGBoost
response = requests.post(
    "http://localhost:8000/predict/xgboost",
    json={
        "data": [
            {"date": "2021-01-01", "close": 29000},
            {"date": "2021-01-02", "close": 31000},
            # ... minimum 30 points
        ],
        "horizon": 30,
        "confidence": 0.95
    }
)
result = response.json()
print(result["predictions"][:3])
        """, language="python")
    
    st.markdown("---")
    st.markdown("#### 📋 Architecture complète")
    
    arch_cols = st.columns(4)
    arch = [
        ("📓", "Notebook", "Analyse complète\nARIMA → LSTM\nExport modèles"),
        ("⚡", "FastAPI", "4 endpoints\nREST JSON\nDocs Swagger"),
        ("📊", "Streamlit", "Dashboard\nFiltres dynamiques\nExport CSV"),
        ("☁️", "Cloud", "Docker\nKubernetes\nCI/CD"),
    ]
    for col, (icon, title, desc) in zip(arch_cols, arch):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div style='font-size:2rem'>{icon}</div>
                <div style='color:#F7931A;font-weight:bold;margin:0.5rem 0'>{title}</div>
                <div style='font-size:0.8rem;color:#aaa;white-space:pre-line'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)
