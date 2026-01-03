import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.ensemble import IsolationForest
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# ==================================================
# PAGE CONFIG + GLOBAL STYLE
# ==================================================
st.set_page_config(
    page_title="🚀 AI Quant Financial Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main {padding-top:10px;}
.block-container {padding-left:2rem;padding-right:2rem;}
.metric-label {font-size:14px;}
.big-title {font-size:48px;font-weight:800;}
.sub-title {font-size:20px;color:#888;}
.card {
    background-color:#0e1117;
    padding:25px;
    border-radius:18px;
    box-shadow:0 4px 20px rgba(0,0,0,0.3);
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# SESSION STATE
# ==================================================
if "page" not in st.session_state:
    st.session_state.page = "welcome"

# ==================================================
# WELCOME PAGE
# ==================================================
if st.session_state.page == "welcome":
    st.markdown("<div class='big-title'>🚀 AI Quant Financial Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Institutional-grade analytics powered by AI</div>", unsafe_allow_html=True)
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### 📈 Technical Intelligence")
        st.markdown("- SMA / EMA / RSI / MACD\n- Support & Resistance\n- VWAP & Volatility")
    with c2:
        st.markdown("### 🔮 AI Forecasting")
        st.markdown("- Prophet-based forecasting\n- Confidence intervals\n- Trend detection")
    with c3:
        st.markdown("### 🧠 Risk & Anomaly")
        st.markdown("- Isolation Forest anomalies\n- Sharpe Ratio\n- Max Drawdown")

    st.markdown("---")
    if st.button("🚀 Launch Dashboard", use_container_width=True):
        st.session_state.page = "dashboard"
    st.stop()

# ==================================================
# DATA CACHE
# ==================================================
@st.cache_data(ttl=3600)
def load_stock_data(ticker, period):
    df = yf.download(ticker, period=period, auto_adjust=False, progress=False)
    if df is None or df.empty:
        return None
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df.dropna()

@st.cache_data(ttl=3600)
def load_fundamentals(ticker):
    info = yf.Ticker(ticker).info
    return {
        "Market Cap": info.get("marketCap"),
        "P/E Ratio": info.get("trailingPE"),
        "Forward P/E": info.get("forwardPE"),
        "EPS": info.get("trailingEps"),
        "Dividend Yield": info.get("dividendYield"),
        "Beta": info.get("beta"),
        "52W High": info.get("fiftyTwoWeekHigh"),
        "52W Low": info.get("fiftyTwoWeekLow"),
        "Sector": info.get("sector"),
        "Industry": info.get("industry"),
    }

# ==================================================
# FEATURE ENGINEERING
# ==================================================
def technical_indicators(df):
    df = df.copy()

    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["EMA20"] = df["Close"].ewm(span=20).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9).mean()

    df["VWAP"] = (df["Volume"] * (df["High"] + df["Low"] + df["Close"]) / 3).cumsum() / df["Volume"].cumsum()

    df["Support"] = df["Low"].rolling(20).min()
    df["Resistance"] = df["High"].rolling(20).max()

    return df

def detect_anomalies(df):
    returns = df["Close"].pct_change().dropna()
    model = IsolationForest(contamination=0.03, random_state=42)
    preds = model.fit_predict(returns.values.reshape(-1, 1))
    df["Anomaly"] = 1
    df.loc[returns.index, "Anomaly"] = preds
    return df

# ==================================================
# FORECASTING
# ==================================================
def prepare_prophet(df):
    p = df["Close"].reset_index()
    p.columns = ["ds", "y"]
    return p

@st.cache_resource
def train_prophet(df):
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05
    )
    model.fit(df)
    return model

def forecast_price(df, days):
    p = prepare_prophet(df)
    model = train_prophet(p)
    future = model.make_future_dataframe(periods=days)
    return model.predict(future)

# ==================================================
# RISK METRICS
# ==================================================
def risk_metrics(df):
    returns = df["Close"].pct_change().dropna()
    sharpe = np.sqrt(252) * returns.mean() / returns.std()
    max_dd = ((df["Close"] / df["Close"].cummax()) - 1).min()
    volatility = returns.std() * np.sqrt(252) * 100
    return sharpe, max_dd * 100, volatility

# ==================================================
# SIDEBAR
# ==================================================
st.sidebar.title("⚙️ Controls")

ticker = st.sidebar.text_input("Ticker", "AAPL").upper()
period = st.sidebar.selectbox("Period", ["6mo", "1y", "2y", "5y", "10y"], index=2)
forecast_days = st.sidebar.slider("Forecast Days", 7, 365, 90)
indicators = st.sidebar.multiselect(
    "Indicators",
    ["SMA20", "SMA50", "EMA20", "VWAP", "Support", "Resistance"],
    default=["SMA20", "SMA50"]
)

# ==================================================
# LOAD DATA
# ==================================================
df = load_stock_data(ticker, period)

if df is None:
    st.error("❌ Invalid ticker or no data available.")
    st.stop()

df = technical_indicators(df)
df = detect_anomalies(df)
forecast = forecast_price(df, forecast_days)
fundamentals = load_fundamentals(ticker)
sharpe, max_dd, volatility = risk_metrics(df)

# ==================================================
# DASHBOARD
# ==================================================
st.markdown(f"## 📊 {ticker} Quant Dashboard")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Last Price", f"${df['Close'].iloc[-1]:.2f}")
m2.metric("Sharpe Ratio", f"{sharpe:.2f}")
m3.metric("Volatility", f"{volatility:.2f}%")
m4.metric("Max Drawdown", f"{max_dd:.2f}%")

tabs = st.tabs(["📈 Price", "📊 Indicators", "🔮 Forecast", "🏦 Fundamentals", "📉 Risk", "📥 Export"])

# --------------------------------------------------
# PRICE
# --------------------------------------------------
with tabs[0]:
    fig = go.Figure()
    fig.add_candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Price"
    )
    for ind in indicators:
        fig.add_trace(go.Scatter(x=df.index, y=df[ind], name=ind))
    anomalies = df[df["Anomaly"] == -1]
    fig.add_trace(go.Scatter(
        x=anomalies.index,
        y=anomalies["Close"],
        mode="markers",
        marker=dict(color="red", size=7),
        name="Anomaly"
    ))
    fig.update_layout(height=520, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# INDICATORS
# --------------------------------------------------
with tabs[1]:
    rsi_fig = go.Figure()
    rsi_fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI"))
    rsi_fig.add_hline(y=70, line_dash="dash")
    rsi_fig.add_hline(y=30, line_dash="dash")
    rsi_fig.update_layout(height=300)
    st.plotly_chart(rsi_fig, use_container_width=True)

    macd_fig = go.Figure()
    macd_fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD"))
    macd_fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal"))
    macd_fig.update_layout(height=300)
    st.plotly_chart(macd_fig, use_container_width=True)

# --------------------------------------------------
# FORECAST
# --------------------------------------------------
with tabs[2]:
    fig_f = go.Figure()
    fig_f.add_trace(go.Scatter(x=df.index, y=df["Close"], name="History"))
    fig_f.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], name="Forecast"))
    fig_f.add_trace(go.Scatter(
        x=forecast["ds"],
        y=forecast["yhat_upper"],
        fill=None,
        name="Upper"
    ))
    fig_f.add_trace(go.Scatter(
        x=forecast["ds"],
        y=forecast["yhat_lower"],
        fill="tonexty",
        name="Lower"
    ))
    fig_f.update_layout(height=520)
    st.plotly_chart(fig_f, use_container_width=True)

# --------------------------------------------------
# FUNDAMENTALS
# --------------------------------------------------
with tabs[3]:
    st.dataframe(pd.DataFrame(fundamentals, index=["Value"]).T, use_container_width=True)

# --------------------------------------------------
# RISK
# --------------------------------------------------
with tabs[4]:
    returns = df["Close"].pct_change()
    hist = go.Figure()
    hist.add_histogram(x=returns, nbinsx=60)
    hist.update_layout(height=420, title="Return Distribution")
    st.plotly_chart(hist, use_container_width=True)

# --------------------------------------------------
# EXPORT
# --------------------------------------------------
with tabs[5]:
    st.download_button(
        "⬇️ Download CSV",
        df.to_csv().encode("utf-8"),
        file_name=f"{ticker}_{datetime.now().date()}.csv",
        mime="text/csv"
    )
