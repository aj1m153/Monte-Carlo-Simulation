import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from prophet import Prophet
from scipy import stats
import requests
import time
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="QuantView Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS  –  dark terminal-finance aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;600;800&display=swap');

:root {
    --bg:        #0d0f14;
    --surface:   #13161d;
    --border:    #1e2330;
    --accent:    #00e5a0;
    --accent2:   #5c6bff;
    --warn:      #ff6b6b;
    --text:      #e2e8f0;
    --muted:     #64748b;
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Syne', sans-serif;
}

/* Header */
.app-header {
    padding: 2rem 0 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.app-title {
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -1px;
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.app-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: var(--muted);
    margin-top: 0.3rem;
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* Metric cards */
.metric-grid { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.5rem; }
.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.4rem;
    flex: 1;
    min-width: 140px;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
}
.metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-size: 1.5rem;
    font-weight: 800;
    color: var(--text);
    line-height: 1;
}
.metric-value.pos { color: var(--accent); }
.metric-value.neg { color: var(--warn); }
.metric-delta {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: var(--muted);
    margin-top: 0.3rem;
}

/* Section headers */
.section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--accent);
    font-family: 'Space Mono', monospace;
    letter-spacing: 1px;
    text-transform: uppercase;
    border-left: 3px solid var(--accent);
    padding-left: 0.8rem;
    margin: 1.8rem 0 1rem;
}

/* Info boxes */
.info-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    line-height: 1.7;
    color: var(--muted);
}
.info-box strong { color: var(--text); }

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stTextInput label {
    color: var(--muted) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}
[data-testid="stSidebar"] h2 {
    color: var(--accent) !important;
    font-size: 0.85rem !important;
    font-family: 'Space Mono', monospace !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background-color: var(--surface) !important;
    border-radius: 8px;
    gap: 4px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    color: var(--muted) !important;
    background: transparent !important;
    border-radius: 6px !important;
}
.stTabs [aria-selected="true"] {
    background: var(--bg) !important;
    color: var(--accent) !important;
}

/* Dataframe */
[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 8px; }

/* Expander */
.streamlit-expanderHeader {
    background-color: var(--surface) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
    color: var(--muted) !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
POPULAR_STOCKS = {
    "Apple (AAPL)": "AAPL", "Microsoft (MSFT)": "MSFT",
    "Amazon (AMZN)": "AMZN", "Alphabet (GOOGL)": "GOOGL",
    "NVIDIA (NVDA)": "NVDA", "Tesla (TSLA)": "TSLA",
    "Meta (META)": "META",  "JPMorgan (JPM)": "JPM",
    "Berkshire (BRK-B)": "BRK-B", "S&P 500 ETF (SPY)": "SPY",
    "Custom…": "CUSTOM",
}
PLOTLY_THEME = dict(
    paper_bgcolor="#0d0f14", plot_bgcolor="#0d0f14",
    font=dict(family="Space Mono, monospace", color="#e2e8f0", size=11),
    xaxis=dict(gridcolor="#1e2330", linecolor="#1e2330", showgrid=True),
    yaxis=dict(gridcolor="#1e2330", linecolor="#1e2330", showgrid=True),
    margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(bgcolor="#13161d", bordercolor="#1e2330", borderwidth=1),
)

# ─────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────
def get_yf_session() -> requests.Session:
    """Return a browser-spoofed session to avoid Yahoo Finance rate limits."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    })
    return session

@st.cache_data(ttl=3600)
def fetch_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    time.sleep(1)  # reduce request frequency on shared Cloud IPs
    session = get_yf_session()
    df = yf.download(ticker, period=period, auto_adjust=True,
                     progress=False, session=session)
    df.index = pd.to_datetime(df.index)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df

@st.cache_data(ttl=3600)
def fetch_ff3_factors() -> pd.DataFrame:
    """
    Download Fama-French 3 factors from Kenneth French's library via yfinance proxy.
    Fallback: compute approximate factors from ETF proxies.
    """
    try:
        import pandas_datareader.data as web
        ff = web.DataReader("F-F_Research_Data_Factors", "famafrench",
                            start="2015-01-01")[0]
        ff = ff / 100
        ff.index = pd.to_datetime(ff.index.to_timestamp())
        ff.columns = ["Mkt_RF", "SMB", "HML", "RF"]
        return ff
    except Exception:
        pass

    # ETF-based approximation
    spy  = fetch_data("SPY",  "5y")["Close"].pct_change()
    iwm  = fetch_data("IWM",  "5y")["Close"].pct_change()   # small-cap proxy
    iwd  = fetch_data("IWD",  "5y")["Close"].pct_change()   # value proxy
    iwf  = fetch_data("IWF",  "5y")["Close"].pct_change()   # growth proxy
    rf_rate = 0.05 / 252

    merged = pd.DataFrame({"SPY": spy, "IWM": iwm, "IWD": iwd, "IWF": iwf}).dropna()
    merged["Mkt_RF"] = merged["SPY"] - rf_rate
    merged["SMB"]    = merged["IWM"] - merged["SPY"]
    merged["HML"]    = merged["IWD"] - merged["IWF"]
    merged["RF"]     = rf_rate
    return merged[["Mkt_RF", "SMB", "HML", "RF"]]


def compute_capm(stock_returns: pd.Series, market_returns: pd.Series,
                 rf: float = 0.05) -> dict:
    excess_stock  = stock_returns  - rf / 252
    excess_market = market_returns - rf / 252
    df = pd.DataFrame({"stock": excess_stock, "market": excess_market}).dropna()

    slope, intercept, r, p, se = stats.linregress(df["market"], df["stock"])
    beta   = slope
    alpha  = intercept * 252                          # annualise
    r2     = r ** 2
    capm_expected = rf + beta * (market_returns.mean() * 252 - rf)
    sharpe = (stock_returns.mean() * 252 - rf) / (stock_returns.std() * np.sqrt(252))
    treynor = (stock_returns.mean() * 252 - rf) / beta if beta != 0 else np.nan

    return dict(beta=beta, alpha=alpha, r_squared=r2, p_value=p,
                expected_return=capm_expected, sharpe=sharpe, treynor=treynor,
                excess_stock=df["stock"], excess_market=df["market"])


def compute_ff3(stock_returns: pd.Series, ff_factors: pd.DataFrame) -> dict:
    """Fama-French 3-Factor OLS regression."""
    rf    = ff_factors["RF"]
    mkt   = ff_factors["Mkt_RF"]
    smb   = ff_factors["SMB"]
    hml   = ff_factors["HML"]

    # align on common dates (monthly or daily)
    common = stock_returns.index.intersection(ff_factors.index)
    if len(common) < 30:
        # resample stock to monthly if FF data is monthly
        stock_monthly = stock_returns.resample("ME").apply(
            lambda x: (1 + x).prod() - 1)
        common = stock_monthly.index.intersection(ff_factors.index)
        s = stock_monthly.loc[common]
    else:
        s = stock_returns.loc[common]

    excess = s - ff_factors.loc[common, "RF"]
    X = pd.DataFrame({
        "Mkt_RF": ff_factors.loc[common, "Mkt_RF"],
        "SMB":    ff_factors.loc[common, "SMB"],
        "HML":    ff_factors.loc[common, "HML"],
    })
    X = sm_ols_fit(X, excess)
    return X


def sm_ols_fit(X: pd.DataFrame, y: pd.Series) -> dict:
    """Manual OLS with t-stats (no statsmodels dependency)."""
    Xm = np.column_stack([np.ones(len(X)), X.values])
    coefs, res, rank, sv = np.linalg.lstsq(Xm, y.values, rcond=None)
    n, k = Xm.shape
    sse  = np.sum((y.values - Xm @ coefs) ** 2)
    mse  = sse / (n - k)
    cov  = mse * np.linalg.pinv(Xm.T @ Xm)
    se   = np.sqrt(np.diag(cov))
    t    = coefs / se
    p    = 2 * (1 - stats.t.cdf(np.abs(t), df=n - k))
    ss_tot = np.sum((y.values - y.values.mean()) ** 2)
    r2   = 1 - sse / ss_tot if ss_tot > 0 else 0

    labels = ["Alpha", "Beta_Mkt", "Beta_SMB", "Beta_HML"]
    return dict(
        alpha=coefs[0] * 252, beta_mkt=coefs[1], beta_smb=coefs[2], beta_hml=coefs[3],
        se=dict(zip(labels, se)), t=dict(zip(labels, t)), p=dict(zip(labels, p)),
        r_squared=r2, n=n
    )


def run_prophet(df: pd.DataFrame, periods: int = 365,
                ci: float = 0.90) -> tuple[pd.DataFrame, Prophet]:
    prophet_df = df[["Close"]].reset_index()
    prophet_df.columns = ["ds", "y"]
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
    prophet_df["ds"] = prophet_df["ds"].dt.tz_localize(None)

    m = Prophet(
        interval_width=ci,
        changepoint_prior_scale=0.05,
        seasonality_mode="multiplicative",
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
    )
    m.fit(prophet_df)
    future   = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)
    return forecast, m


def label_color(val: float, inverse: bool = False) -> str:
    if inverse:
        return "pos" if val < 0 else "neg"
    return "pos" if val >= 0 else "neg"


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ QuantView Pro")
    st.markdown("---")

    stock_choice = st.selectbox("Stock", list(POPULAR_STOCKS.keys()), index=0)
    if stock_choice == "Custom…":
        custom_ticker = st.text_input("Enter Ticker Symbol", value="AAPL",
                                      max_chars=10).upper().strip()
        ticker = custom_ticker
    else:
        ticker = POPULAR_STOCKS[stock_choice]

    st.markdown("---")
    st.markdown("## 📆 Data Range")
    hist_period = st.select_slider("Historical Period",
                                   options=["1y", "2y", "3y", "5y", "10y"],
                                   value="5y")

    st.markdown("---")
    st.markdown("## 🔮 Prophet Settings")
    forecast_days = st.slider("Forecast Horizon (days)", 30, 730, 365, 30)
    ci_level = st.select_slider("Confidence Interval", [0.80, 0.90, 0.95], value=0.90)

    st.markdown("---")
    st.markdown("## 💹 Monte Carlo")
    n_mc_paths = st.slider("Simulation Paths", 100, 5000, 1000, 100)

    st.markdown("---")
    run_btn = st.button("▶  RUN ANALYSIS", use_container_width=True, type="primary")

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <p class="app-title">QuantView Pro</p>
  <p class="app-sub">Monte Carlo · Prophet Forecast · CAPM · Fama-French 3-Factor</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MAIN LOGIC
# ─────────────────────────────────────────────
if not run_btn:
    st.markdown("""
    <div class="info-box" style="text-align:center; padding:3rem;">
      <strong style="font-size:1.1rem;">Select a stock in the sidebar and click ▶ RUN ANALYSIS</strong><br><br>
      This tool provides:<br>
      📈 &nbsp;Price history & Monte Carlo simulation<br>
      🔮 &nbsp;Prophet time-series forecast with 90% CI<br>
      📐 &nbsp;CAPM regression (alpha, beta, Sharpe, Treynor)<br>
      🧮 &nbsp;Fama-French 3-Factor model (Mkt-RF, SMB, HML)
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─── LOAD DATA ────────────────────────────────
with st.spinner(f"Fetching {ticker} data…"):
    try:
        df = fetch_data(ticker, hist_period)
        if df.empty:
            st.error(f"No data found for **{ticker}**. Please check the symbol.")
            st.stop()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        st.stop()

try:
    time.sleep(0.5)
    info = yf.Ticker(ticker, session=get_yf_session()).info
except Exception:
    info = {}
comp_name  = info.get("longName", ticker)
sector     = info.get("sector", "—")
industry   = info.get("industry", "—")
market_cap = info.get("marketCap", 0)
currency   = info.get("currency", "USD")

returns      = df["Close"].pct_change().dropna()
last_price   = float(df["Close"].iloc[-1])
prev_price   = float(df["Close"].iloc[-2])
price_chg    = last_price - prev_price
pct_chg      = price_chg / prev_price * 100
ann_vol      = float(returns.std() * np.sqrt(252) * 100)
ann_ret      = float(returns.mean() * 252 * 100)
ytd_start    = df[df.index.year == datetime.now().year]["Close"].iloc[0] if any(
    df.index.year == datetime.now().year) else df["Close"].iloc[0]
ytd_ret      = (last_price / float(ytd_start) - 1) * 100

cap_str = f"${market_cap/1e9:.1f}B" if market_cap > 1e9 else (
          f"${market_cap/1e6:.0f}M" if market_cap else "—")

# ─── KPI CARDS ────────────────────────────────
st.markdown(f"### {comp_name} &nbsp;<code style='font-size:0.8rem;background:#1e2330;padding:3px 8px;border-radius:4px'>{ticker}</code>", unsafe_allow_html=True)
st.markdown(f"<span style='font-family:Space Mono;font-size:0.75rem;color:#64748b'>{sector} · {industry} · {cap_str}</span>", unsafe_allow_html=True)

st.markdown(f"""
<div class="metric-grid">
  <div class="metric-card">
    <div class="metric-label">Last Price</div>
    <div class="metric-value">${last_price:.2f}</div>
    <div class="metric-delta">{currency}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Day Change</div>
    <div class="metric-value {label_color(pct_chg)}">{pct_chg:+.2f}%</div>
    <div class="metric-delta">${price_chg:+.2f}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Ann. Return</div>
    <div class="metric-value {label_color(ann_ret)}">{ann_ret:.1f}%</div>
    <div class="metric-delta">Annualised</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Ann. Volatility</div>
    <div class="metric-value">{ann_vol:.1f}%</div>
    <div class="metric-delta">1-yr σ</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">YTD Return</div>
    <div class="metric-value {label_color(ytd_ret)}">{ytd_ret:+.1f}%</div>
    <div class="metric-delta">{datetime.now().year}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tabs = st.tabs([
    "📈 Price & Monte Carlo",
    "🔮 Prophet Forecast",
    "📐 CAPM Analysis",
    "🧮 Fama-French 3F",
    "📋 Summary",
])

# ══════════════════════════════════════════════
#  TAB 1 – PRICE + MONTE CARLO
# ══════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-title">Historical Price</div>', unsafe_allow_html=True)

    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(
        x=df.index, y=df["Close"],
        line=dict(color="#00e5a0", width=1.5),
        fill="tozeroy", fillcolor="rgba(0,229,160,0.06)",
        name="Close",
    ))
    # 50-day / 200-day MA
    df["MA50"]  = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()
    fig_price.add_trace(go.Scatter(x=df.index, y=df["MA50"],
        line=dict(color="#5c6bff", width=1, dash="dot"), name="MA 50"))
    fig_price.add_trace(go.Scatter(x=df.index, y=df["MA200"],
        line=dict(color="#ff6b6b", width=1, dash="dot"), name="MA 200"))
    fig_price.update_layout(**PLOTLY_THEME, title=f"{ticker} — Adjusted Close Price",
                            height=380)
    st.plotly_chart(fig_price, use_container_width=True)

    # Volume bar
    fig_vol = go.Figure(go.Bar(
        x=df.index, y=df["Volume"],
        marker_color=np.where(df["Close"].diff() >= 0, "#00e5a0", "#ff6b6b"),
        name="Volume",
    ))
    fig_vol.update_layout(**PLOTLY_THEME, title="Volume", height=200,
                          yaxis=dict(gridcolor="#1e2330"))
    st.plotly_chart(fig_vol, use_container_width=True)

    # ── Monte Carlo ──
    st.markdown('<div class="section-title">Monte Carlo Simulation (GBM)</div>',
                unsafe_allow_html=True)

    S0    = last_price
    mu    = float(returns.mean())
    sigma = float(returns.std())
    T_mc  = 252
    dt    = 1 / 252

    np.random.seed(42)
    paths = np.zeros((n_mc_paths, T_mc + 1))
    paths[:, 0] = S0
    for step in range(1, T_mc + 1):
        z = np.random.standard_normal(n_mc_paths)
        paths[:, step] = paths[:, step - 1] * np.exp(
            (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)

    t_axis    = np.linspace(0, 1, T_mc + 1)
    pct5      = np.percentile(paths, 5,  axis=0)
    pct25     = np.percentile(paths, 25, axis=0)
    pct50     = np.percentile(paths, 50, axis=0)
    pct75     = np.percentile(paths, 75, axis=0)
    pct95     = np.percentile(paths, 95, axis=0)

    fig_mc = go.Figure()

    # sample paths
    sample_idx = np.random.choice(n_mc_paths, min(80, n_mc_paths), replace=False)
    for idx in sample_idx:
        fig_mc.add_trace(go.Scatter(
            x=t_axis, y=paths[idx],
            line=dict(color="rgba(92,107,255,0.12)", width=0.8),
            showlegend=False, hoverinfo="skip",
        ))

    # confidence bands
    fig_mc.add_trace(go.Scatter(x=np.concatenate([t_axis, t_axis[::-1]]),
        y=np.concatenate([pct95, pct5[::-1]]),
        fill="toself", fillcolor="rgba(0,229,160,0.07)",
        line=dict(color="rgba(0,0,0,0)"), name="5–95%"))
    fig_mc.add_trace(go.Scatter(x=np.concatenate([t_axis, t_axis[::-1]]),
        y=np.concatenate([pct75, pct25[::-1]]),
        fill="toself", fillcolor="rgba(0,229,160,0.15)",
        line=dict(color="rgba(0,0,0,0)"), name="25–75%"))
    fig_mc.add_trace(go.Scatter(x=t_axis, y=pct50,
        line=dict(color="#00e5a0", width=2), name="Median"))
    fig_mc.add_trace(go.Scatter(x=t_axis, y=pct5,
        line=dict(color="#ff6b6b", width=1, dash="dash"), name="5th pct"))
    fig_mc.add_trace(go.Scatter(x=t_axis, y=pct95,
        line=dict(color="#5c6bff", width=1, dash="dash"), name="95th pct"))

    fig_mc.update_layout(**PLOTLY_THEME,
        title=f"Monte Carlo GBM — {n_mc_paths} paths, 1-year horizon",
        xaxis_title="Time (years)", yaxis_title="Price ($)", height=420)
    st.plotly_chart(fig_mc, use_container_width=True)

    # MC stats
    end_prices   = paths[:, -1]
    prob_above   = (end_prices > S0).mean() * 100
    expected_end = end_prices.mean()
    var_95       = np.percentile(end_prices, 5)
    cvar_95      = end_prices[end_prices <= var_95].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Expected Price (1yr)",  f"${expected_end:.2f}",
                f"{(expected_end/S0-1)*100:+.1f}%")
    col2.metric("Median Price (1yr)",    f"${pct50[-1]:.2f}",
                f"{(pct50[-1]/S0-1)*100:+.1f}%")
    col3.metric("Prob. Above Current",  f"{prob_above:.1f}%")
    col4.metric("VaR 95% (1yr)",        f"${var_95:.2f}",
                f"{(var_95/S0-1)*100:+.1f}%")

    # Return distribution
    st.markdown('<div class="section-title">Return Distribution (1-yr MC)</div>',
                unsafe_allow_html=True)
    ret_dist = (end_prices / S0 - 1) * 100
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=ret_dist, nbinsx=80,
        marker_color="#5c6bff", opacity=0.7, name="Returns",
    ))
    fig_dist.add_vline(x=0, line_dash="dash",
                       line_color="#ff6b6b", annotation_text="Break-even")
    fig_dist.add_vline(x=ret_dist.mean(), line_dash="dot",
                       line_color="#00e5a0", annotation_text=f"Mean {ret_dist.mean():.1f}%")
    fig_dist.update_layout(**PLOTLY_THEME,
        title="1-Year Return Distribution", xaxis_title="Return (%)",
        yaxis_title="Frequency", height=300)
    st.plotly_chart(fig_dist, use_container_width=True)


# ══════════════════════════════════════════════
#  TAB 2 – PROPHET FORECAST
# ══════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="section-title">Prophet Time-Series Forecast</div>',
                unsafe_allow_html=True)

    with st.spinner("Training Prophet model…"):
        try:
            forecast, model = run_prophet(df, forecast_days, ci_level)
        except Exception as e:
            st.error(f"Prophet error: {e}")
            st.stop()

    # Align historical
    hist_prophet = df[["Close"]].reset_index()
    hist_prophet.columns = ["ds", "y"]
    hist_prophet["ds"] = pd.to_datetime(hist_prophet["ds"]).dt.tz_localize(None)

    split   = len(hist_prophet)
    fut_fc  = forecast[forecast["ds"] > hist_prophet["ds"].max()]

    fig_fc = go.Figure()

    # CI band
    fig_fc.add_trace(go.Scatter(
        x=pd.concat([fut_fc["ds"], fut_fc["ds"][::-1]]),
        y=pd.concat([fut_fc["yhat_upper"], fut_fc["yhat_lower"][::-1]]),
        fill="toself", fillcolor="rgba(92,107,255,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name=f"{int(ci_level*100)}% CI",
    ))
    fig_fc.add_trace(go.Scatter(
        x=hist_prophet["ds"], y=hist_prophet["y"],
        line=dict(color="#00e5a0", width=1.5), name="Historical",
    ))
    fig_fc.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["yhat"],
        line=dict(color="#5c6bff", width=2, dash="dot"), name="Forecast (yhat)",
    ))
    fig_fc.add_trace(go.Scatter(
        x=fut_fc["ds"], y=fut_fc["yhat_upper"],
        line=dict(color="rgba(92,107,255,0.5)", width=1, dash="dash"),
        name="Upper bound",
    ))
    fig_fc.add_trace(go.Scatter(
        x=fut_fc["ds"], y=fut_fc["yhat_lower"],
        line=dict(color="rgba(255,107,107,0.5)", width=1, dash="dash"),
        name="Lower bound",
    ))
    fig_fc.update_layout(**PLOTLY_THEME,
        title=f"Prophet Forecast — {forecast_days}d horizon, {int(ci_level*100)}% CI",
        xaxis_title="Date", yaxis_title="Price ($)", height=440)
    st.plotly_chart(fig_fc, use_container_width=True)

    # Forecast summary
    last_fc    = fut_fc.iloc[-1]
    mid_fc     = fut_fc.iloc[len(fut_fc)//2]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Forecast End Price",   f"${last_fc['yhat']:.2f}",
                f"{(last_fc['yhat']/last_price-1)*100:+.1f}%")
    col2.metric(f"Upper ({int(ci_level*100)}% CI)", f"${last_fc['yhat_upper']:.2f}")
    col3.metric(f"Lower ({int(ci_level*100)}% CI)", f"${last_fc['yhat_lower']:.2f}")
    col4.metric("Mid-horizon Forecast", f"${mid_fc['yhat']:.2f}",
                f"{(mid_fc['yhat']/last_price-1)*100:+.1f}%")

    # Trend components
    st.markdown('<div class="section-title">Trend Decomposition</div>',
                unsafe_allow_html=True)

    comp_fig = make_subplots(rows=3, cols=1,
        subplot_titles=["Trend", "Weekly Seasonality", "Yearly Seasonality"],
        shared_xaxes=False)
    comp_fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["trend"],
        line=dict(color="#00e5a0", width=1.5)), row=1, col=1)
    comp_fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast.get("weekly", pd.Series(0, index=forecast.index)),
        line=dict(color="#5c6bff", width=1.5)), row=2, col=1)
    comp_fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast.get("yearly", pd.Series(0, index=forecast.index)),
        line=dict(color="#ffd166", width=1.5)), row=3, col=1)
    comp_fig.update_layout(**PLOTLY_THEME, height=550, showlegend=False,
                           title="Prophet Component Decomposition")
    st.plotly_chart(comp_fig, use_container_width=True)

    # Forecast table
    with st.expander("📋 Forecast data table"):
        show_fc = fut_fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        show_fc.columns = ["Date", "Forecast", "Lower CI", "Upper CI"]
        show_fc["Date"] = show_fc["Date"].dt.strftime("%Y-%m-%d")
        for c in ["Forecast", "Lower CI", "Upper CI"]:
            show_fc[c] = show_fc[c].map("${:.2f}".format)
        st.dataframe(show_fc.reset_index(drop=True), use_container_width=True,
                     height=300)


# ══════════════════════════════════════════════
#  TAB 3 – CAPM
# ══════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-title">Capital Asset Pricing Model (CAPM)</div>',
                unsafe_allow_html=True)

    with st.spinner("Fetching market data…"):
        mkt_df = fetch_data("SPY", hist_period)
        mkt_ret = mkt_df["Close"].pct_change().dropna()

    common_idx = returns.index.intersection(mkt_ret.index)
    capm = compute_capm(returns.loc[common_idx], mkt_ret.loc[common_idx])

    # CAPM metric row
    beta_color = "pos" if 0.5 <= capm["beta"] <= 1.5 else "neg"
    st.markdown(f"""
    <div class="metric-grid">
      <div class="metric-card">
        <div class="metric-label">Beta (β)</div>
        <div class="metric-value {beta_color}">{capm['beta']:.3f}</div>
        <div class="metric-delta">vs S&P 500</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Alpha (α) Ann.</div>
        <div class="metric-value {label_color(capm['alpha'])}">{capm['alpha']*100:.2f}%</div>
        <div class="metric-delta">Jensen's Alpha</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">R²</div>
        <div class="metric-value">{capm['r_squared']:.3f}</div>
        <div class="metric-delta">Explained variance</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Sharpe Ratio</div>
        <div class="metric-value {label_color(capm['sharpe'])}">{capm['sharpe']:.3f}</div>
        <div class="metric-delta">Risk-adj. return</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Treynor Ratio</div>
        <div class="metric-value {label_color(capm['treynor'])}">{capm['treynor']:.4f}</div>
        <div class="metric-delta">Return per β unit</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Expected Return</div>
        <div class="metric-value {label_color(capm['expected_return'])}">{capm['expected_return']*100:.2f}%</div>
        <div class="metric-delta">CAPM E[R]</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # SML scatter
    st.markdown('<div class="section-title">Security Market Line (SML)</div>',
                unsafe_allow_html=True)
    betas_range = np.linspace(-0.5, 2.5, 100)
    rf          = 0.05
    mkt_premium = mkt_ret.mean() * 252 - rf
    sml_ret     = rf + betas_range * mkt_premium

    fig_sml = go.Figure()
    fig_sml.add_trace(go.Scatter(x=betas_range, y=sml_ret * 100,
        line=dict(color="#64748b", width=1.5, dash="dash"), name="SML"))
    fig_sml.add_trace(go.Scatter(
        x=[capm["beta"]], y=[capm["expected_return"] * 100],
        mode="markers", marker=dict(color="#5c6bff", size=14, symbol="diamond"),
        name=f"{ticker} CAPM E[R]"))
    fig_sml.add_trace(go.Scatter(
        x=[capm["beta"]], y=[ann_ret],
        mode="markers", marker=dict(color="#00e5a0", size=14, symbol="star"),
        name=f"{ticker} Actual Return"))
    fig_sml.update_layout(**PLOTLY_THEME,
        title="Security Market Line — CAPM",
        xaxis_title="Beta (β)", yaxis_title="Return (%)", height=380)
    st.plotly_chart(fig_sml, use_container_width=True)

    # Rolling beta
    st.markdown('<div class="section-title">Rolling Beta (252-day window)</div>',
                unsafe_allow_html=True)

    roll_beta = []
    ret_df = pd.DataFrame({"stock": returns, "mkt": mkt_ret}).dropna()
    window = 252
    for i in range(window, len(ret_df)):
        slice_  = ret_df.iloc[i - window:i]
        slope, *_ = stats.linregress(slice_["mkt"], slice_["stock"])
        roll_beta.append({"date": ret_df.index[i], "beta": slope})
    rb_df = pd.DataFrame(roll_beta)

    fig_rb = go.Figure()
    fig_rb.add_trace(go.Scatter(x=rb_df["date"], y=rb_df["beta"],
        line=dict(color="#5c6bff", width=1.5), fill="tozeroy",
        fillcolor="rgba(92,107,255,0.07)", name="Rolling β"))
    fig_rb.add_hline(y=1, line_dash="dot", line_color="#64748b",
                     annotation_text="β = 1")
    fig_rb.update_layout(**PLOTLY_THEME, title="Rolling 252-day Beta",
                         xaxis_title="Date", yaxis_title="Beta", height=320)
    st.plotly_chart(fig_rb, use_container_width=True)

    # Scatter excess returns
    st.markdown('<div class="section-title">Excess Return Scatter</div>',
                unsafe_allow_html=True)
    fig_sc = go.Figure()
    fig_sc.add_trace(go.Scatter(
        x=capm["excess_market"] * 100, y=capm["excess_stock"] * 100,
        mode="markers", marker=dict(color="#5c6bff", size=3, opacity=0.4),
        name="Daily returns"))

    xs = np.array([capm["excess_market"].min(), capm["excess_market"].max()])
    ys = capm["alpha"] / 252 + capm["beta"] * xs
    fig_sc.add_trace(go.Scatter(x=xs * 100, y=ys * 100,
        line=dict(color="#00e5a0", width=2), name=f"β={capm['beta']:.2f}"))
    fig_sc.update_layout(**PLOTLY_THEME,
        xaxis_title="Market Excess Return (%)", yaxis_title=f"{ticker} Excess Return (%)",
        title="CAPM Regression", height=380)
    st.plotly_chart(fig_sc, use_container_width=True)


# ══════════════════════════════════════════════
#  TAB 4 – FAMA-FRENCH 3F
# ══════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-title">Fama-French 3-Factor Model</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <strong>Model:</strong> Rᵢ − Rƒ = α + β₁(Mkt−RF) + β₂·SMB + β₃·HML + ε<br>
    <strong>Mkt-RF</strong> — Market excess return &nbsp;|&nbsp;
    <strong>SMB</strong> — Small Minus Big (size premium) &nbsp;|&nbsp;
    <strong>HML</strong> — High Minus Low (value premium)
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Loading Fama-French factors…"):
        try:
            ff = fetch_ff3_factors()
        except Exception as e:
            st.error(f"Could not load FF3 factors: {e}")
            st.stop()

    ff3 = compute_ff3(returns, ff)

    # Factor loading cards
    def pval_star(p):
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        if p < 0.1:   return "†"
        return ""

    a_star   = pval_star(ff3["p"]["Alpha"])
    bm_star  = pval_star(ff3["p"]["Beta_Mkt"])
    bs_star  = pval_star(ff3["p"]["Beta_SMB"])
    bh_star  = pval_star(ff3["p"]["Beta_HML"])

    smb_label = "Small-cap tilt" if ff3["beta_smb"] > 0 else "Large-cap tilt"
    hml_label = "Value tilt"     if ff3["beta_hml"] > 0 else "Growth tilt"

    st.markdown(f"""
    <div class="metric-grid">
      <div class="metric-card">
        <div class="metric-label">Alpha (α) Ann. {a_star}</div>
        <div class="metric-value {label_color(ff3['alpha'])}">{ff3['alpha']*100:.2f}%</div>
        <div class="metric-delta">p={ff3['p']['Alpha']:.4f}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">β Market {bm_star}</div>
        <div class="metric-value">{ff3['beta_mkt']:.3f}</div>
        <div class="metric-delta">p={ff3['p']['Beta_Mkt']:.4f}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">β SMB (Size) {bs_star}</div>
        <div class="metric-value {label_color(ff3['beta_smb'])}">{ff3['beta_smb']:.3f}</div>
        <div class="metric-delta">{smb_label}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">β HML (Value) {bh_star}</div>
        <div class="metric-value {label_color(ff3['beta_hml'])}">{ff3['beta_hml']:.3f}</div>
        <div class="metric-delta">{hml_label}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">R² (3-Factor)</div>
        <div class="metric-value">{ff3['r_squared']:.3f}</div>
        <div class="metric-delta">{ff3['n']} observations</div>
      </div>
    </div>
    <div class="info-box" style="margin-top:0.5rem">
    * p&lt;0.05 &nbsp; ** p&lt;0.01 &nbsp; *** p&lt;0.001 &nbsp; † p&lt;0.10
    </div>
    """, unsafe_allow_html=True)

    # Factor loadings bar chart
    st.markdown('<div class="section-title">Factor Loadings</div>',
                unsafe_allow_html=True)

    factors  = ["β Market", "β SMB", "β HML"]
    loadings = [ff3["beta_mkt"], ff3["beta_smb"], ff3["beta_hml"]]
    colors   = ["#5c6bff", "#00e5a0" if ff3["beta_smb"] > 0 else "#ff6b6b",
                "#ffd166" if ff3["beta_hml"] > 0 else "#ff6b6b"]

    fig_bar = go.Figure(go.Bar(
        x=factors, y=loadings, marker_color=colors,
        text=[f"{v:.3f}" for v in loadings], textposition="auto",
    ))
    fig_bar.add_hline(y=0, line_color="#64748b")
    fig_bar.update_layout(**PLOTLY_THEME,
        title=f"Fama-French Factor Loadings — {ticker}",
        yaxis_title="Loading (β)", height=350)
    st.plotly_chart(fig_bar, use_container_width=True)

    # Rolling FF3 betas
    st.markdown('<div class="section-title">Rolling Factor Exposures (126-day)</div>',
                unsafe_allow_html=True)

    common2 = returns.index.intersection(ff.index)
    if len(common2) < 60:
        # use monthly alignment
        st_monthly = returns.resample("ME").apply(lambda x: (1+x).prod()-1)
        common2    = st_monthly.index.intersection(ff.index)
        roll_src   = pd.DataFrame({
            "excess": st_monthly.loc[common2] - ff.loc[common2, "RF"],
            "mkt":  ff.loc[common2, "Mkt_RF"],
            "smb":  ff.loc[common2, "SMB"],
            "hml":  ff.loc[common2, "HML"],
        }).dropna()
        win = 24
    else:
        roll_src = pd.DataFrame({
            "excess": returns.loc[common2] - ff.loc[common2, "RF"],
            "mkt":  ff.loc[common2, "Mkt_RF"],
            "smb":  ff.loc[common2, "SMB"],
            "hml":  ff.loc[common2, "HML"],
        }).dropna()
        win = 126

    roll_records = []
    for i in range(win, len(roll_src)):
        sl = roll_src.iloc[i-win:i]
        r  = sm_ols_fit(sl[["mkt","smb","hml"]], sl["excess"])
        roll_records.append({
            "date": roll_src.index[i],
            "beta_mkt": r["beta_mkt"],
            "beta_smb": r["beta_smb"],
            "beta_hml": r["beta_hml"],
        })
    rr_df = pd.DataFrame(roll_records)

    fig_roll = go.Figure()
    for col, color, label in [
        ("beta_mkt","#5c6bff","β Market"),
        ("beta_smb","#00e5a0","β SMB"),
        ("beta_hml","#ffd166","β HML"),
    ]:
        fig_roll.add_trace(go.Scatter(x=rr_df["date"], y=rr_df[col],
            line=dict(color=color, width=1.5), name=label))
    fig_roll.add_hline(y=0, line_dash="dot", line_color="#64748b")
    fig_roll.update_layout(**PLOTLY_THEME,
        title=f"Rolling {win}-period Factor Betas",
        xaxis_title="Date", yaxis_title="Beta", height=360)
    st.plotly_chart(fig_roll, use_container_width=True)

    # Interpretation
    with st.expander("📖 Factor Interpretation"):
        st.markdown(f"""
**α (Alpha) = {ff3['alpha']*100:.2f}%/yr** — {'Positive alpha suggests the stock generates returns above what the 3-factor model predicts.' if ff3['alpha']>0 else 'Negative alpha suggests underperformance vs the 3-factor model.'}

**β Market = {ff3['beta_mkt']:.3f}** — {'Above 1: amplifies market moves (aggressive).' if ff3['beta_mkt']>1 else 'Below 1: dampens market moves (defensive).'}

**β SMB = {ff3['beta_smb']:.3f}** — {'Positive: behaves more like a small-cap stock.' if ff3['beta_smb']>0 else 'Negative: behaves more like a large-cap stock.'}

**β HML = {ff3['beta_hml']:.3f}** — {'Positive: value tilt (high book-to-market).' if ff3['beta_hml']>0 else 'Negative: growth tilt (low book-to-market).'}

**R² = {ff3['r_squared']:.3f}** — {ff3['r_squared']*100:.0f}% of the stock's excess return variation is explained by these three factors.
        """)


# ══════════════════════════════════════════════
#  TAB 5 – SUMMARY
# ══════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-title">Full Analysis Summary</div>',
                unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### 📈 Price Statistics")
        stats_df = pd.DataFrame({
            "Metric": ["Last Price", "Ann. Return", "Ann. Volatility",
                       "YTD Return", "Max Drawdown", "Skewness", "Kurtosis"],
            "Value": [
                f"${last_price:.2f}",
                f"{ann_ret:.2f}%",
                f"{ann_vol:.2f}%",
                f"{ytd_ret:.2f}%",
                f"{((df['Close'] / df['Close'].cummax()) - 1).min()*100:.2f}%",
                f"{float(returns.skew()):.3f}",
                f"{float(returns.kurtosis()):.3f}",
            ]
        })
        st.dataframe(stats_df, hide_index=True, use_container_width=True)

        st.markdown("#### 📐 CAPM Summary")
        capm_df = pd.DataFrame({
            "Metric": ["Beta", "Alpha (ann.)", "R²", "Sharpe", "Treynor", "E[R] CAPM"],
            "Value": [
                f"{capm['beta']:.3f}",
                f"{capm['alpha']*100:.2f}%",
                f"{capm['r_squared']:.3f}",
                f"{capm['sharpe']:.3f}",
                f"{capm['treynor']:.4f}",
                f"{capm['expected_return']*100:.2f}%",
            ]
        })
        st.dataframe(capm_df, hide_index=True, use_container_width=True)

    with col_b:
        st.markdown("#### 🔮 Prophet Forecast")
        fc_end = fut_fc.iloc[-1]
        fc_df = pd.DataFrame({
            "Metric": ["Forecast End Price", "Upper CI", "Lower CI",
                       "CI Width", "Expected Change"],
            "Value": [
                f"${fc_end['yhat']:.2f}",
                f"${fc_end['yhat_upper']:.2f}",
                f"${fc_end['yhat_lower']:.2f}",
                f"${fc_end['yhat_upper']-fc_end['yhat_lower']:.2f}",
                f"{(fc_end['yhat']/last_price-1)*100:+.1f}%",
            ]
        })
        st.dataframe(fc_df, hide_index=True, use_container_width=True)

        st.markdown("#### 🧮 Fama-French 3F Summary")
        ff_df = pd.DataFrame({
            "Factor": ["Alpha (ann.)", "β Market", "β SMB", "β HML", "R²"],
            "Loading": [
                f"{ff3['alpha']*100:.2f}%",
                f"{ff3['beta_mkt']:.3f}",
                f"{ff3['beta_smb']:.3f}",
                f"{ff3['beta_hml']:.3f}",
                f"{ff3['r_squared']:.3f}",
            ],
            "p-value": [
                f"{ff3['p']['Alpha']:.4f}",
                f"{ff3['p']['Beta_Mkt']:.4f}",
                f"{ff3['p']['Beta_SMB']:.4f}",
                f"{ff3['p']['Beta_HML']:.4f}",
                "—",
            ]
        })
        st.dataframe(ff_df, hide_index=True, use_container_width=True)

    # Monte Carlo summary
    st.markdown("#### 💹 Monte Carlo (1yr GBM)")
    mc_df = pd.DataFrame({
        "Metric": ["Expected Price", "Median Price", "5th Percentile",
                   "95th Percentile", "Prob. Above Entry", "CVaR (95%)"],
        "Value": [
            f"${end_prices.mean():.2f}",
            f"${np.median(end_prices):.2f}",
            f"${np.percentile(end_prices,5):.2f}",
            f"${np.percentile(end_prices,95):.2f}",
            f"{(end_prices > S0).mean()*100:.1f}%",
            f"${end_prices[end_prices<=np.percentile(end_prices,5)].mean():.2f}",
        ]
    })
    st.dataframe(mc_df, hide_index=True, use_container_width=True)

    st.markdown("""
    <div class="info-box" style="margin-top:1rem">
    <strong>⚠️ Disclaimer:</strong> This tool is for educational and research purposes only.
    It does not constitute financial advice. Past performance is not indicative of future results.
    All models contain assumptions and limitations. Consult a licensed financial advisor before making investment decisions.
    </div>
    """, unsafe_allow_html=True)
