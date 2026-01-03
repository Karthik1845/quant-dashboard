import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.ensemble import IsolationForest
import numpy as np
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# --------------------------------------------------
# Page Config + Theme
# --------------------------------------------------
st.set_page_config(
    page_title="🚀 AI Quant Financial Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main {padding-top:15px;}
.block-container {padding-left:2rem;padding-right:2rem;}
.metric-label {font-size:14px;}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Cached Data
# --------------------------------------------------
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


# --------------------------------------------------
# Feature Engineering
# --------------------------------------------------
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

    tr = pd.concat([
        df["High"] - df["Low"],
        abs(df["High"] - df["Close"].shift()),
        abs(df["Low"] - df["Close"].shift())
    ], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    df["VWAP"] = (df["Volume"] * (df["High"] + df["Low"] + df["Close"]) / 3).cumsum() / df["Volume"].cumsum()

    return df


def support_resistance(df, window=20):
    df["Support"] = df["Low"].rolling(window).min()
    df["Resistance"] = df["High"].rolling(window).max()
    return df


def detect_anomalies(df):
    returns = df["Close"].pct_change().dropna()
    model = IsolationForest(contamination=0.03, random_state=42)
    preds = model.fit_predict(returns.values.reshape(-1, 1))
    df["Anomaly"] = 1
    df.loc[returns.index, "Anomaly"] = preds
    return df


# --------------------------------------------------
# Forecasting
# --------------------------------------------------
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


# --------------------------------------------------
# Risk & Performance
# --------------------------------------------------
def risk_metrics(df):
    returns = df["Close"].pct_change().dropna()
    sharpe = np.sqrt(252) * returns.mean() / returns.std()
    max_dd = ((df["Close"] / df["Close"].cummax()) - 1).min()
    return sharpe, max_dd * 100


def summary_metrics(df):
    cp = df["Close"].iloc[-1]
    chg = df["Close"].pct_change().iloc[-1] * 100
    vol = df["Close"].pct_change().rolling(30).std().iloc[-1] * 100
    avg_vol = df["Volume"].tail(30).mean()
    trend = "📈 Bullish" if chg > 0 else "📉 Bearish"
    return cp, chg, vol, avg_vol, trend


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("⚙️ Controls")

ticker = st.sidebar.text_input("Ticker", "AAPL").upper()
period = st.sidebar.selectbox("Period", ["6mo", "1y", "2y", "5y", "10y"], index=2)
forecast_days = st.sidebar.slider("Forecast Days", 7, 180, 60)
show_ind = st.sidebar.multiselect(
    "Indicators",
    ["SMA20", "SMA50", "EMA20", "VWAP", "Support", "Resistance"],
    default=["SMA20", "SMA50"]
)
refresh = st.sidebar.button("🔄 Refresh")

# --------------------------------------------------
# Load Pipeline
# --------------------------------------------------
if refresh or "data" not in st.session_state:
    data = load_stock_data(ticker, period)
    if data is not None:
        data = technical_indicators(data)
        data = support_resistance(data)
        data = detect_anomalies(data)
        forecast = forecast_price(data, forecast_days)
        fundamentals = load_fundamentals(ticker)
        metrics = summary_metrics(data)
        sharpe, max_dd = risk_metrics(data)

        st.session_state.update({
            "data": data,
            "forecast": forecast,
            "fundamentals": fundamentals,
            "metrics": metrics,
            "sharpe": sharpe,
            "max_dd": max_dd
        })
    else:
        st.session_state.data = None

df = st.session_state.get("data")

# --------------------------------------------------
# Dashboard
# --------------------------------------------------
if df is not None:
    cp, chg, vol, avg_vol, trend = st.session_state.metrics

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Price", f"${cp:.2f}", f"{chg:.2f}%")
    c2.metric("Trend", trend)
    c3.metric("Volatility", f"{vol:.2f}%")
    c4.metric("Avg Volume", f"{avg_vol:,.0f}")
    c5.metric("Sharpe", f"{st.session_state.sharpe:.2f}")
    c6.metric("Max Drawdown", f"{st.session_state.max_dd:.2f}%")

    tabs = st.tabs(["📈 Price", "📊 Indicators", "🔮 Forecast", "🏦 Fundamentals", "📉 Risk", "📥 Export"])

    # -------- Price
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

        for ind in show_ind:
            fig.add_trace(go.Scatter(x=df.index, y=df[ind], name=ind))

        anomalies = df[df["Anomaly"] == -1]
        fig.add_trace(go.Scatter(
            x=anomalies.index,
            y=anomalies["Close"],
            mode="markers",
            marker=dict(color="red", size=7),
            name="Anomaly"
        ))

        fig.update_layout(height=500, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width="stretch")

    # -------- Indicators
    with tabs[1]:
        rsi_fig = go.Figure()
        rsi_fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI"))
        rsi_fig.add_hline(y=70, line_dash="dash")
        rsi_fig.add_hline(y=30, line_dash="dash")
        rsi_fig.update_layout(height=300)
        st.plotly_chart(rsi_fig, use_container_width="stretch")

        macd_fig = go.Figure()
        macd_fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD"))
        macd_fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal"))
        macd_fig.update_layout(height=300)
        st.plotly_chart(macd_fig, use_container_width="stretch")

    # -------- Forecast
    with tabs[2]:
        fc = st.session_state.forecast
        fig_f = go.Figure()
        fig_f.add_trace(go.Scatter(x=df.index, y=df["Close"], name="History"))
        fig_f.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat"], name="Forecast"))
        fig_f.add_trace(go.Scatter(
            x=fc["ds"], y=fc["yhat_lower"],
            fill="tonexty", name="Confidence"
        ))
        fig_f.update_layout(height=500)
        st.plotly_chart(fig_f, use_container_width="stretch")

    # -------- Fundamentals
    with tabs[3]:
        print(df.dtypes)
        st.dataframe(pd.DataFrame(st.session_state.fundamentals, index=["Value"]).T)
        


    # -------- Risk
    with tabs[4]:
        returns = df["Close"].pct_change()
        hist = go.Figure()
        hist.add_histogram(x=returns, nbinsx=50)
        hist.update_layout(height=400, title="Return Distribution")
        st.plotly_chart(hist, use_container_width="stretch")

    # -------- Export
    with tabs[5]:
        st.download_button(
            "⬇️ Download CSV",
            df.to_csv().encode("utf-8"),
            file_name=f"{ticker}_{datetime.now().date()}.csv",
            mime="text/csv"
        )

else:
    st.warning("Invalid ticker or no data available.")

