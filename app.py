# ============================================================
# AI NIFTY 500 TRADING SYSTEM — STREAMLIT DASHBOARD  v2.0
# ============================================================
# Run:  streamlit run app.py
# ============================================================

import hmac, json, os, sys, warnings
from datetime import datetime
from pathlib import Path

# Python 3.14 workaround
if "warnings" not in sys.modules:
    sys.modules["warnings"] = warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="AI NIFTY 500 Trader — Prasad R. Paranjape",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# PASSWORD GATE  (runs before any other UI)
# ============================================================
def _check_password() -> bool:
    """Returns True once the user has entered the correct password."""
    if st.session_state.get("_auth"):
        return True

    # Resolve password: secrets.toml locally, env var on Cloud Run
    _correct = st.secrets.get("app_password",
                  os.environ.get("APP_PASSWORD", "nifty500"))

    # ── Full-page login layout ────────────────────────────
    st.markdown("""
    <style>
    .stApp,[data-testid="stSidebar"]{background:#080e1c !important;}
    header[data-testid="stHeader"]{background:#080e1c !important;border-bottom:none !important;}
    [data-testid="stDecoration"]{display:none !important;}
    </style>
    """, unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 1.05, 1])
    with mid:
        st.markdown("""
        <div style="text-align:center;padding:56px 0 32px;">
          <div style="font-size:3.5rem;line-height:1;filter:drop-shadow(0 0 24px rgba(56,189,248,0.35));">📈</div>
          <div style="color:#38bdf8;font-weight:800;font-size:1.5rem;
                      letter-spacing:0.05em;margin-top:14px;">NIFTY 500 AI</div>
          <div style="color:#1f3b56;font-size:0.70rem;letter-spacing:0.14em;
                      text-transform:uppercase;margin-top:5px;">Delivery Trader · v2.0</div>
          <div style="color:#162840;font-size:0.67rem;margin-top:5px;
                      font-style:italic;">Prasad R. Paranjape</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background:linear-gradient(135deg,#0c1a2e,#091220);
                    border:1px solid #1a3352;border-radius:18px;
                    padding:34px 36px 28px;
                    box-shadow:0 12px 48px rgba(0,0,0,0.6),
                               0 0 0 1px rgba(56,189,248,0.06);">
          <p style="color:#4a7a99;font-size:0.70rem;font-weight:700;
                    text-transform:uppercase;letter-spacing:0.10em;margin:0 0 8px;">
            Access Password
          </p>
        </div>
        """, unsafe_allow_html=True)

        pwd = st.text_input(
            "Password",
            type="password",
            placeholder="Enter your password …",
            label_visibility="collapsed",
            key="_pwd_input",
        )

        login_clicked = st.button(
            "🔓  Unlock Dashboard",
            type="primary",
            use_container_width=True,
        )

        if login_clicked:
            if pwd and hmac.compare_digest(pwd.strip(), _correct.strip()):
                st.session_state["_auth"] = True
                st.rerun()
            else:
                st.markdown("""
                <div style="background:#1c0a0a;border:1px solid #7f1d1d;border-radius:9px;
                            padding:10px 16px;margin-top:12px;text-align:center;
                            color:#fca5a5;font-size:0.84rem;font-weight:600;">
                  ✕ &nbsp; Incorrect password — please try again.
                </div>
                """, unsafe_allow_html=True)

        st.markdown("""
        <p style="text-align:center;color:#162840;font-size:0.66rem;margin-top:24px;">
          Protected · AI NIFTY 500 Trader · Prasad R. Paranjape
        </p>
        """, unsafe_allow_html=True)

    return False


if not _check_password():
    st.stop()

# ── Core engine ──────────────────────────────────────────────
from nifty500_ai_trader import (
    CONFIG, NIFTY_500,
    DataLoader, ModelTrainer, RiskManager, Backtester, Scanner,
    add_indicators, add_candle_patterns, add_weekly_ema,
    market_regime, ml_predict, relative_strength, generate_signal,
    monthly_pnl_matrix, get_sector,
)

# ============================================================
# CSS — Premium dark trading terminal theme
# ============================================================
st.markdown("""
<style>
/* ═══ BASE ═══════════════════════════════════════════════════ */
* { box-sizing:border-box; }
.stApp { background:#080e1c !important; color:#e2e8f0; }

/* ── Kill the white Streamlit top header bar ── */
header[data-testid="stHeader"] {
    background:#080e1c !important;
    border-bottom:1px solid #1a3352 !important;
}
header[data-testid="stHeader"]::before,
header[data-testid="stHeader"]::after { display:none !important; }

/* ── Toolbar icons inside the header ── */
[data-testid="stToolbar"]  { right:1rem !important; }
[data-testid="stDecoration"] { display:none !important; }

.block-container {
    padding-top:3.5rem !important;
    padding-bottom:2rem !important;
    background:#080e1c !important;
    max-width:1600px !important;
}

/* ── Animated glow keyframes ── */
@keyframes pulse-glow {
  0%,100% { box-shadow:0 4px 18px rgba(14,165,233,0.40); }
  50%      { box-shadow:0 6px 32px rgba(56,189,248,0.65); }
}
@keyframes border-pulse {
  0%,100% { border-color:#1a3352; }
  50%      { border-color:#1e4976; }
}

/* ═══ SIDEBAR ════════════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background:linear-gradient(180deg,#080e1c 0%,#060c18 100%) !important;
    border-right:1px solid #112236 !important;
}
[data-testid="stSidebar"] > div { padding-top:0.5rem !important; }
[data-testid="stSidebar"] * { color:#e2e8f0 !important; }
[data-testid="stSidebar"] hr { border:none !important; border-top:1px solid #1a3352 !important; margin:14px 0 !important; }

/* ═══ INPUTS ═════════════════════════════════════════════════ */
input[type="number"], input[type="text"] {
    background:#0d1b2e !important;
    color:#f1f5f9 !important;
    border:1px solid #1f3b56 !important;
    border-radius:8px !important;
    font-size:0.95rem !important;
    transition:border-color 0.2s, box-shadow 0.2s !important;
}
input[type="number"]:focus, input[type="text"]:focus {
    border:1px solid #38bdf8 !important;
    box-shadow:0 0 0 3px rgba(56,189,248,0.15) !important;
    outline:none !important;
}
[data-testid="stNumberInput"] label,
[data-testid="stTextInput"]   label {
    color:#6b9ab8 !important;
    font-size:0.80rem !important;
    font-weight:600 !important;
    text-transform:uppercase !important;
    letter-spacing:0.06em !important;
}
[data-testid="stNumberInput"] button {
    background:#0d1b2e !important;
    color:#6b9ab8 !important;
    border:1px solid #1f3b56 !important;
    border-radius:6px !important;
    transition:all 0.15s !important;
}
[data-testid="stNumberInput"] button:hover {
    background:#1a3352 !important;
    color:#e2e8f0 !important;
}

/* ═══ SLIDERS ════════════════════════════════════════════════ */
[data-testid="stSlider"] label {
    color:#6b9ab8 !important;
    font-size:0.80rem !important;
    font-weight:600 !important;
    text-transform:uppercase !important;
    letter-spacing:0.06em !important;
}
[data-baseweb="slider"] [role="slider"] {
    background:#38bdf8 !important;
    box-shadow:0 0 0 4px rgba(56,189,248,0.20) !important;
    width:17px !important; height:17px !important;
}
[data-baseweb="slider"] div[data-baseweb="slider-track-fill"] {
    background:linear-gradient(90deg,#0ea5e9,#38bdf8) !important;
}
[data-baseweb="slider"] div[data-baseweb="slider-track"] {
    background:#1a3352 !important;
    height:4px !important;
}

/* ═══ SELECTBOX / MULTISELECT ════════════════════════════════ */
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
    background:#0d1b2e !important;
    color:#e2e8f0 !important;
    border:1px solid #1f3b56 !important;
    border-radius:8px !important;
}
[data-testid="stSelectbox"] label,
[data-testid="stMultiSelect"] label {
    color:#6b9ab8 !important;
    font-size:0.80rem !important;
    font-weight:600 !important;
    text-transform:uppercase !important;
    letter-spacing:0.06em !important;
}
[data-baseweb="popover"] {
    background:#0d1b2e !important;
    border:1px solid #1f3b56 !important;
    border-radius:10px !important;
}
[data-baseweb="popover"] * { background:#0d1b2e !important; color:#e2e8f0 !important; }
[data-baseweb="tag"] {
    background:#1a3352 !important;
    color:#7dd3fc !important;
    border-radius:6px !important;
}
[data-testid="stMultiSelect"] [data-baseweb="select"] > div { background:#0d1b2e !important; }
[data-testid="stMultiSelect"] [data-baseweb="select"] span { color:#e2e8f0 !important; }

/* ═══ TABS ═══════════════════════════════════════════════════ */
[data-baseweb="tab-list"] {
    background:#0d1b2e !important;
    border:1px solid #1a3352 !important;
    border-radius:12px !important;
    padding:5px 6px !important;
    gap:3px !important;
}
[data-baseweb="tab"] {
    color:#4a7a99 !important;
    background:transparent !important;
    border-radius:8px !important;
    padding:8px 20px !important;
    font-weight:500 !important;
    font-size:0.88rem !important;
    transition:all 0.18s !important;
    border:none !important;
}
[data-baseweb="tab"]:hover {
    color:#94a3b8 !important;
    background:#111e33 !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    color:#f1f5f9 !important;
    background:linear-gradient(135deg,#1a3352,#1d3f66) !important;
    border-radius:8px !important;
    box-shadow:0 2px 10px rgba(56,189,248,0.18) !important;
    font-weight:700 !important;
}

/* ═══ METRICS (st.metric) ════════════════════════════════════ */
[data-testid="stMetric"] {
    background:linear-gradient(160deg,#0c1929 0%,#0e1e30 60%,#0a1520 100%) !important;
    border:1px solid #162d46 !important;
    border-top:3px solid #1e6090 !important;
    border-radius:13px !important;
    padding:14px 18px !important;
    box-shadow:0 4px 20px rgba(0,0,0,0.4),
               inset 0 1px 0 rgba(56,189,248,0.05) !important;
    transition:transform 0.18s, box-shadow 0.18s !important;
}
[data-testid="stMetric"]:hover {
    transform:translateY(-3px) !important;
    box-shadow:0 10px 32px rgba(0,0,0,0.5),
               0 0 0 1px rgba(56,189,248,0.12) !important;
}
[data-testid="stMetricLabel"] {
    color:#3a6a8a !important;
    font-size:0.70rem !important;
    font-weight:700 !important;
    text-transform:uppercase !important;
    letter-spacing:0.09em !important;
}
[data-testid="stMetricValue"] {
    color:#e8f4fd !important;
    font-size:1.5rem !important;
    font-weight:800 !important;
    letter-spacing:-0.01em !important;
}
[data-testid="stMetricDelta"] { font-size:0.75rem !important; }
[data-testid="stMetricDelta"] svg { fill:#22c55e !important; }

/* ═══ DATAFRAME ══════════════════════════════════════════════ */
[data-testid="stDataFrame"] {
    background:#080e1c !important;
    border:1px solid #1a3352 !important;
    border-radius:12px !important;
    overflow:hidden !important;
}

/* ═══ BUTTONS ════════════════════════════════════════════════ */
[data-testid="stButton"] button[kind="primary"] {
    background:linear-gradient(135deg,#0ea5e9,#0369a1) !important;
    color:#fff !important;
    border:none !important;
    border-radius:10px !important;
    font-weight:700 !important;
    font-size:0.92rem !important;
    letter-spacing:0.03em !important;
    animation:pulse-glow 2.8s ease-in-out infinite !important;
    transition:all 0.2s !important;
    padding:10px 24px !important;
}
[data-testid="stButton"] button[kind="primary"]:hover {
    background:linear-gradient(135deg,#38bdf8,#0ea5e9) !important;
    box-shadow:0 8px 32px rgba(56,189,248,0.65) !important;
    transform:translateY(-2px) !important;
    animation:none !important;
}
[data-testid="stButton"] button[kind="primary"]:active {
    transform:translateY(0) !important;
    animation:none !important;
}
[data-testid="stButton"] button:not([kind="primary"]) {
    background:#0d1b2e !important;
    color:#6b9ab8 !important;
    border:1px solid #1f3b56 !important;
    border-radius:10px !important;
    font-weight:500 !important;
    transition:all 0.18s !important;
}
[data-testid="stButton"] button:not([kind="primary"]):hover {
    background:#1a3352 !important;
    color:#e2e8f0 !important;
    border-color:#38bdf8 !important;
}
[data-testid="stDownloadButton"] button {
    background:#0d1b2e !important;
    color:#38bdf8 !important;
    border:1px solid #1a3352 !important;
    border-radius:8px !important;
    font-weight:600 !important;
    font-size:0.84rem !important;
    transition:all 0.18s !important;
}
[data-testid="stDownloadButton"] button:hover {
    background:#1a3352 !important;
    border-color:#38bdf8 !important;
    box-shadow:0 0 12px rgba(56,189,248,0.20) !important;
}

/* ═══ CHECKBOX / RADIO ═══════════════════════════════════════ */
[data-testid="stCheckbox"] label { color:#94a3b8 !important; font-size:0.86rem !important; }
[data-testid="stRadio"] label    { color:#94a3b8 !important; font-size:0.85rem !important; }
[data-testid="stRadio"] > label  {
    color:#6b9ab8 !important;
    font-size:0.78rem !important;
    font-weight:700 !important;
    text-transform:uppercase !important;
    letter-spacing:0.06em !important;
}

/* ═══ PROGRESS BAR ═══════════════════════════════════════════ */
[data-testid="stProgressBar"] > div > div {
    background:linear-gradient(90deg,#0ea5e9,#22c55e) !important;
    border-radius:4px !important;
}
[data-testid="stProgressBar"] > div { background:#0d1b2e !important; border-radius:4px !important; }

/* ═══ DIVIDER ════════════════════════════════════════════════ */
hr { border:none !important; border-top:1px solid #1a3352 !important; margin:1rem 0 !important; }

/* ═══ EXPANDER ═══════════════════════════════════════════════ */
[data-testid="stExpander"] {
    background:linear-gradient(135deg,#0d1b2e,#0e1c30) !important;
    border:1px solid #1a3352 !important;
    border-radius:12px !important;
    box-shadow:0 2px 10px rgba(0,0,0,0.25) !important;
}
[data-testid="stExpander"] summary {
    color:#94a3b8 !important;
    font-weight:600 !important;
    font-size:0.88rem !important;
}
[data-testid="stExpander"] summary:hover { color:#e2e8f0 !important; }

/* ═══ ALERTS ═════════════════════════════════════════════════ */
[data-testid="stAlert"] { border-radius:10px !important; }
[data-testid="stAlert"] p { color:#e2e8f0 !important; font-size:0.87rem !important; }

/* ═══ REGIME BADGES ══════════════════════════════════════════ */
.regime-bull {
    background:linear-gradient(135deg,#041a0c,#083d1a);
    color:#4ade80; padding:7px 24px; border-radius:22px;
    font-weight:800; display:inline-block; font-size:0.90rem;
    border:1px solid rgba(74,222,128,0.50);
    box-shadow:0 0 20px rgba(34,197,94,0.30),
               inset 0 1px 0 rgba(74,222,128,0.10);
    letter-spacing:0.07em;
}
.regime-side {
    background:linear-gradient(135deg,#1e0e00,#3a1a00);
    color:#fbbf24; padding:7px 24px; border-radius:22px;
    font-weight:800; display:inline-block; font-size:0.90rem;
    border:1px solid rgba(251,191,36,0.50);
    box-shadow:0 0 20px rgba(245,158,11,0.25),
               inset 0 1px 0 rgba(251,191,36,0.08);
    letter-spacing:0.07em;
}
.regime-bear {
    background:linear-gradient(135deg,#140404,#380808);
    color:#f87171; padding:7px 24px; border-radius:22px;
    font-weight:800; display:inline-block; font-size:0.90rem;
    border:1px solid rgba(248,113,113,0.50);
    box-shadow:0 0 20px rgba(239,68,68,0.28),
               inset 0 1px 0 rgba(248,113,113,0.08);
    letter-spacing:0.07em;
}

/* ═══ CUSTOM COMPONENT CLASSES ═══════════════════════════════ */
.kpi-card {
    background:linear-gradient(160deg,#0b1826 0%,#0e1e32 60%,#091420 100%);
    border:1px solid #162d46;
    border-radius:14px;
    padding:18px 20px;
    text-align:center;
    height:100%;
    transition:transform 0.18s, box-shadow 0.18s;
    box-shadow:0 4px 20px rgba(0,0,0,0.35),
               inset 0 1px 0 rgba(255,255,255,0.03);
}
.kpi-card:hover {
    transform:translateY(-3px);
    box-shadow:0 12px 32px rgba(0,0,0,0.5),
               0 0 0 1px rgba(56,189,248,0.10);
}
.kpi-label {
    color:#2e6080; font-size:0.67rem; font-weight:700;
    text-transform:uppercase; letter-spacing:0.10em; margin-bottom:8px;
}
.kpi-value {
    color:#e8f4fd; font-size:1.80rem; font-weight:800;
    line-height:1.1; letter-spacing:-0.01em;
}
.kpi-sub   { color:#1e4060; font-size:0.68rem; margin-top:6px; }

.sig-strong {
    background:linear-gradient(135deg,#041f10,#0a3d1e);
    color:#4ade80;
    padding:4px 14px; border-radius:20px; font-size:0.76rem; font-weight:700;
    border:1px solid rgba(74,222,128,0.40);
    box-shadow:0 0 10px rgba(34,197,94,0.15);
    white-space:nowrap;
}
.sig-buy {
    background:linear-gradient(135deg,#081828,#0f2d4a);
    color:#38bdf8;
    padding:4px 14px; border-radius:20px; font-size:0.76rem; font-weight:700;
    border:1px solid rgba(56,189,248,0.40);
    box-shadow:0 0 10px rgba(56,189,248,0.12);
    white-space:nowrap;
}
.sig-watch {
    background:linear-gradient(135deg,#130c2e,#221550);
    color:#c084fc;
    padding:4px 14px; border-radius:20px; font-size:0.76rem; font-weight:700;
    border:1px solid rgba(192,132,252,0.40);
    box-shadow:0 0 10px rgba(167,139,250,0.12);
    white-space:nowrap;
}

.sec-label {
    color:#38bdf8; font-size:0.68rem; font-weight:700;
    text-transform:uppercase; letter-spacing:0.10em;
    border-left:3px solid #38bdf8; padding-left:9px;
    margin:16px 0 10px;
}
.sec-label-amber {
    color:#f59e0b; font-size:0.68rem; font-weight:700;
    text-transform:uppercase; letter-spacing:0.10em;
    border-left:3px solid #f59e0b; padding-left:9px;
    margin:16px 0 10px;
}
.sec-label-green {
    color:#22c55e; font-size:0.68rem; font-weight:700;
    text-transform:uppercase; letter-spacing:0.10em;
    border-left:3px solid #22c55e; padding-left:9px;
    margin:16px 0 10px;
}

/* ═══ CUSTOM SCROLLBAR ═══════════════════════════════════════ */
::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background:#080e1c; }
::-webkit-scrollbar-thumb { background:#1a3352; border-radius:3px; }
::-webkit-scrollbar-thumb:hover { background:#1f3b56; }

/* ═══ TYPOGRAPHY ════════════════════════════════════════════ */
h1,h2,h3      { color:#f1f5f9 !important; }
h4,h5,h6      { color:#e2e8f0 !important; }
p,span,label  { color:#e2e8f0; }
.stMarkdown p { color:#94a3b8; line-height:1.65; }
.stCaption p  { color:#334d65 !important; font-size:0.77rem !important; }
.stCaption    { color:#334d65 !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================
_defaults = {
    "scan_results": [], "signal_stocks": [], "regime": "—", "scan_ts": None,
    "bt_summary": None, "bt_all_trades": [],
    "model_trained": Path(CONFIG["model_path"]).exists(),
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
# CACHED RESOURCES
# ============================================================
@st.cache_resource(show_spinner=False)
def get_loader():
    return DataLoader(CONFIG["cache_dir"])

@st.cache_resource(show_spinner=False)
def get_trainer():
    return ModelTrainer(CONFIG)

@st.cache_data(ttl=43200, show_spinner=False)
def cached_load(sym, period):
    loader = get_loader()
    df = loader.load(sym, period=period)
    df = add_indicators(df)
    df = add_candle_patterns(df)
    try:
        wk = loader.load_weekly(sym, period=period)
        df = add_weekly_ema(df, wk, CONFIG["weekly_ema_period"])
    except Exception:
        pass
    return df

@st.cache_data(ttl=43200, show_spinner=False)
def cached_index(period="1y"):
    return get_loader().load(CONFIG["index_symbol"], period=period)

@st.cache_resource(show_spinner=False)
def load_model_cached():
    if not Path(CONFIG["model_path"]).exists():
        return None, None
    return get_trainer().load()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    # ── Brand header ─────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;padding:12px 0 18px;">
        <div style="font-size:2.2rem;line-height:1;">📈</div>
        <div style="color:#38bdf8;font-weight:800;font-size:1.05rem;
                    letter-spacing:0.04em;margin-top:6px;">NIFTY 500 AI</div>
        <div style="color:#1f3b56;font-size:0.68rem;letter-spacing:0.12em;
                    text-transform:uppercase;margin-top:2px;">Delivery Trader · v2.0</div>
        <div style="color:#22344d;font-size:0.65rem;margin-top:6px;
                    letter-spacing:0.04em;">Prasad R. Paranjape</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # ── Section: Capital & Risk ───────────────────────────
    st.markdown('<div class="sec-label">💰 Capital &amp; Risk</div>', unsafe_allow_html=True)

    capital = st.number_input(
        "Trading Capital (₹)",
        min_value=10_000,
        max_value=100_000_000,
        value=max(10_000, int(CONFIG["capital"])),
        step=10_000,
        format="%d",
    )
    st.caption(f"₹{capital:,.0f}  ·  range ₹10 K – ₹1 Cr")

    risk_pct = st.slider("Risk per trade (%)", 0.25, 3.0,
                          float(CONFIG["risk_per_trade"]*100), 0.25) / 100
    max_pos  = st.slider("Max open positions", 3, 20, int(CONFIG["max_positions"]))

    st.divider()

    # ── Section: Signal Filters ───────────────────────────
    st.markdown('<div class="sec-label-amber">🎯 Signal Filters</div>', unsafe_allow_html=True)

    min_prob = st.slider("Min ML probability", 0.45, 0.90,
                          float(CONFIG["min_probability"]), 0.01)
    min_adx  = st.slider("Min ADX", 10, 40, int(CONFIG["min_adx"]))
    min_rs   = st.slider("Min Rel. Strength", 0.70, 2.0,
                          float(CONFIG["min_rel_strength"]), 0.05)

    CONFIG.update({
        "capital": capital, "risk_per_trade": risk_pct,
        "max_positions": max_pos, "min_probability": min_prob,
        "min_adx": min_adx, "min_rel_strength": min_rs,
    })

    st.divider()

    # ── Section: ML Model ─────────────────────────────────
    st.markdown('<div class="sec-label-green">🤖 ML Model</div>', unsafe_allow_html=True)

    _model_exists = Path(CONFIG["model_path"]).exists()
    _status_color = "#22c55e" if _model_exists else "#ef4444"
    _status_text  = "Ready" if _model_exists else "Not trained"
    st.markdown(
        f'<div style="background:#0d1b2e;border:1px solid #1f3b56;border-radius:8px;'
        f'padding:8px 12px;margin-bottom:10px;display:flex;align-items:center;gap:8px;">'
        f'<span style="width:8px;height:8px;border-radius:50%;background:{_status_color};'
        f'display:inline-block;box-shadow:0 0 6px {_status_color};"></span>'
        f'<span style="color:{_status_color};font-size:0.80rem;font-weight:700;">{_status_text}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    _total_syms = len(NIFTY_500)
    train_all   = st.checkbox("Train on full population", value=False,
                              help=f"Use all {_total_syms} symbols. Takes 20–40 min.")
    if train_all:
        train_n = _total_syms
        st.caption(f"All **{_total_syms} symbols** — allow 20–40 min.")
    else:
        train_n = st.slider("Train on top N symbols", 20, _total_syms, 150, 10,
                            help="More symbols = richer model, longer training.")
        est_min = max(2, int(train_n * 0.07))
        st.caption(f"**{train_n} / {_total_syms}** symbols  ·  ~{est_min}–{est_min+5} min")

    if st.button("🔁 Train / Retrain Model", use_container_width=True, type="primary"):
        with st.spinner(f"Training on {train_n} symbols …"):
            try:
                get_trainer().train(NIFTY_500[:train_n], get_loader())
                st.session_state["model_trained"] = True
                st.cache_resource.clear()
                st.success(f"Model trained on {train_n} symbols!")
            except Exception as ex:
                st.error(f"Error: {ex}")

model, scaler = load_model_cached()

# ── Model-not-trained banner ──────────────────────────────────
if model is None:
    st.error(
        "⚠️  **No trained model found.**  "
        "Click **🔁 Train / Retrain Model** in the sidebar before running a scan.  "
        "Training takes 3–8 minutes and only needs to be done once "
        "(then weekly to refresh).",
        icon="🤖",
    )

# ============================================================
# HEADER BANNER
# ============================================================
_r       = st.session_state["regime"]
_scan_ts = st.session_state["scan_ts"]

# Regime badge HTML
if _r == "—":
    _regime_html = (
        '<span style="background:#0d1b2e;color:#334d65;padding:5px 18px;'
        'border-radius:20px;font-weight:700;border:1px solid #1a3352;'
        'font-size:0.85rem;letter-spacing:0.05em;">— NOT SCANNED</span>'
    )
else:
    _cls = {"BULL": "regime-bull", "BEAR": "regime-bear"}.get(_r, "regime-side")
    _regime_html = f'<span class="{_cls}">▲ {_r}</span>'

_ts_html = (
    f'<span style="color:#334d65;font-size:0.73rem;">Last scan&nbsp;·&nbsp;{_scan_ts}</span>'
    if _scan_ts else ""
)

st.markdown(f"""
<div style="background:linear-gradient(135deg,#0a1828 0%,#080e1c 55%,#0c1830 100%);
            border:1px solid #1a3a5c;border-radius:16px;
            padding:20px 30px;margin-bottom:14px;
            box-shadow:0 6px 32px rgba(0,0,0,0.6),
                       0 0 0 1px rgba(56,189,248,0.06),
                       inset 0 1px 0 rgba(56,189,248,0.05);
            display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;">
  <div>
    <div style="color:#1f4a6b;font-size:0.65rem;font-weight:700;
                text-transform:uppercase;letter-spacing:0.14em;margin-bottom:5px;">
      AI-Powered · Delivery Strategy · XGBoost + Supertrend + CMF
    </div>
    <div style="color:#f1f5f9;font-size:1.55rem;font-weight:800;
                letter-spacing:-0.01em;line-height:1.15;">
      📈 NIFTY 500 Positional Trader
    </div>
    <div style="color:#1f4a6b;font-size:0.72rem;margin-top:4px;">
      14-gate signal engine &nbsp;·&nbsp; Kelly sizing &nbsp;·&nbsp; ATR trailing stop
    </div>
    <div style="color:#1a3352;font-size:0.68rem;margin-top:6px;font-style:italic;">
      Prasad R. Paranjape
    </div>
  </div>
  <div style="display:flex;flex-direction:column;align-items:flex-end;gap:8px;">
    <div>
      <span style="color:#334d65;font-size:0.65rem;font-weight:700;
                   text-transform:uppercase;letter-spacing:0.08em;margin-right:10px;">Market Regime</span>
      {_regime_html}
    </div>
    {_ts_html}
  </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# TABS
# ============================================================
tabs = st.tabs([
    "🔍  Scanner",
    "📊  Chart",
    "📉  Backtest",
    "📅  Monthly P&L",
    "💼  Portfolio",
])
tab_scan, tab_chart, tab_bt, tab_monthly, tab_port = tabs

# ╔══════════════════════════════════════╗
# ║  TAB 1 — SCANNER                    ║
# ╚══════════════════════════════════════╝
with tab_scan:
    rc1, rc2, rc3 = st.columns([2, 3, 2])
    run_scan = rc1.button("▶  Run Full Scan", type="primary",
                          use_container_width=True, disabled=(model is None))
    if model is None:
        rc2.warning("Train the model first (sidebar ▶ ML Model section).")
    rc3.markdown(f"""
    <div class="kpi-card" style="border-top:3px solid #1f3b56;padding:10px 14px;">
      <div class="kpi-label">Universe</div>
      <div class="kpi-value" style="font-size:1.35rem;">{len(NIFTY_500)}</div>
      <div class="kpi-sub">NIFTY 500 symbols</div>
    </div>""", unsafe_allow_html=True)

    if run_scan and model:
        with st.spinner(f"Scanning {len(NIFTY_500)} symbols …"):
            scanner = Scanner(NIFTY_500, CONFIG)
            results, regime = scanner.run(model, scaler)
            _sig_tiers = ["STRONG_BUY", "BUY", "WATCHLIST"]
            signal_stocks = [r for r in results if r.get("signal") in _sig_tiers]
            st.session_state.update({
                "scan_results":  results,
                "signal_stocks": signal_stocks,   # shared source for Scanner + Portfolio
                "regime":        regime,
                "scan_ts":       datetime.now().strftime("%Y-%m-%d %H:%M"),
            })
        st.rerun()  # Re-runs from top so header re-renders with updated regime

    results       = st.session_state["scan_results"]
    signal_stocks = st.session_state["signal_stocks"]   # STRONG_BUY/BUY/WATCHLIST only

    if results:
        df_res       = pd.DataFrame(results)
        df_signals   = pd.DataFrame(signal_stocks) if signal_stocks else pd.DataFrame()
        df_prospects = (df_res[df_res["signal"] == "NO TRADE"]
                        .nlargest(20, "probability").copy())

        # ── Summary KPI cards ─────────────────────────────
        n_sb     = int((df_signals["signal"]=="STRONG_BUY").sum()) if not df_signals.empty else 0
        n_b      = int((df_signals["signal"]=="BUY").sum())        if not df_signals.empty else 0
        n_wl     = int((df_signals["signal"]=="WATCHLIST").sum())  if not df_signals.empty else 0
        avg_rr   = df_signals["risk_reward"].mean()  if not df_signals.empty else 0.0
        avg_prob = df_signals["probability"].mean()  if not df_signals.empty else 0.0
        total_sig = n_sb + n_b + n_wl

        kc = st.columns(5)
        kc[0].markdown(f"""
        <div class="kpi-card" style="border-top:3px solid #22c55e;">
          <div class="kpi-label">🟢 Strong Buy</div>
          <div class="kpi-value" style="color:#86efac;">{n_sb}</div>
          <div class="kpi-sub">{"Act now" if n_sb else "—"}</div>
        </div>""", unsafe_allow_html=True)
        kc[1].markdown(f"""
        <div class="kpi-card" style="border-top:3px solid #38bdf8;">
          <div class="kpi-label">🔵 Buy</div>
          <div class="kpi-value" style="color:#7dd3fc;">{n_b}</div>
          <div class="kpi-sub">{"Act now" if n_b else "—"}</div>
        </div>""", unsafe_allow_html=True)
        kc[2].markdown(f"""
        <div class="kpi-card" style="border-top:3px solid #a78bfa;">
          <div class="kpi-label">🟣 Watchlist</div>
          <div class="kpi-value" style="color:#c4b5fd;">{n_wl}</div>
          <div class="kpi-sub">Monitor</div>
        </div>""", unsafe_allow_html=True)
        kc[3].markdown(f"""
        <div class="kpi-card" style="border-top:3px solid #f59e0b;">
          <div class="kpi-label">Avg R:R</div>
          <div class="kpi-value" style="color:#fcd34d;">{avg_rr:.1f}×</div>
          <div class="kpi-sub">Risk : Reward</div>
        </div>""", unsafe_allow_html=True)
        kc[4].markdown(f"""
        <div class="kpi-card" style="border-top:3px solid #6366f1;">
          <div class="kpi-label">Avg Probability</div>
          <div class="kpi-value" style="color:#a5b4fc;">{avg_prob:.3f}</div>
          <div class="kpi-sub">{total_sig} total signals</div>
        </div>""", unsafe_allow_html=True)
        st.write("")   # spacing
        st.divider()

        # ── Context box ───────────────────────────────────
        regime_now = st.session_state.get("regime","—")
        if n_sb == 0 and n_b == 0 and n_wl == 0:
            st.warning(
                f"**Market Regime: {regime_now}** — No signals today.  "
                "Check **🔭 Top AI Prospects** below for stocks nearest the threshold.  "
                "Try lowering **Min ML probability** / **Min ADX** in the sidebar, "
                "or wait for better market conditions.",
                icon="⚠️",
            )
        elif regime_now == "SIDEWAYS":
            st.info(
                f"**Market Regime: SIDEWAYS** — Prefer STRONG_BUY only "
                "with Probability > 0.65 and R:R > 1.5× in sideways markets.",
                icon="ℹ️",
            )

        # ── Signal table (STRONG_BUY / BUY / WATCHLIST only) ──
        if not df_signals.empty:
            st.markdown("""
            <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:12px;">
              <span style="color:#4a7a99;font-size:0.70rem;font-weight:700;
                           text-transform:uppercase;letter-spacing:0.09em;">Signal key</span>
              <span class="sig-strong">🟢 STRONG BUY</span>
              <span class="sig-buy">🔵 BUY</span>
              <span class="sig-watch">🟣 WATCHLIST</span>
            </div>
            """, unsafe_allow_html=True)

            fc1,fc2,fc3 = st.columns(3)
            sig_f = fc1.selectbox("Filter by signal",
                                  ["All","STRONG_BUY","BUY","WATCHLIST"],
                                  format_func=lambda x: {
                                      "All": "All signals",
                                      "STRONG_BUY": "🟢 Strong Buy",
                                      "BUY":         "🔵 Buy",
                                      "WATCHLIST":   "🟣 Watchlist",
                                  }.get(x, x))
            _all_sectors = sorted(df_signals["sector"].unique())
            sec_f = fc2.multiselect("Sectors", _all_sectors,
                                    default=_all_sectors,
                                    key=f"sec_filter_{st.session_state['scan_ts']}")
            min_prob_f = fc3.slider("Min Probability", 0.45, 0.95,
                                    0.50, 0.01)

            filt = df_signals.reset_index(drop=True).copy()
            if sig_f != "All":
                filt = filt[filt["signal"] == sig_f]
            filt = filt[filt["probability"] >= min_prob_f].reset_index(drop=True)
            if sec_f:
                filt = filt[filt["sector"].isin(sec_f)].reset_index(drop=True)

            # Replace signal text with emoji-prefixed label — same source as legend
            _SIG_LABEL = {
                "STRONG_BUY": "🟢 STRONG BUY",
                "BUY":        "🔵 BUY",
                "WATCHLIST":  "🟣 WATCHLIST",
            }
            disp_cols = ["sector","signal","price","stop_loss","target1","target2",
                         "quantity","risk_amt","probability","rsi","adx",
                         "rel_strength","vol_ratio","cmf","mfi","supertrend_bull",
                         "risk_reward","candle_score","pct_from_52h","vwap_ratio"]
            disp_cols = [c for c in disp_cols if c in filt.columns]

            if not filt.empty:
                display_df = filt.copy()
                if "symbol" in display_df.columns:
                    display_df = display_df.set_index("symbol")
                display_df["signal"] = display_df["signal"].map(_SIG_LABEL).fillna(display_df["signal"])
                styled = (
                    display_df[disp_cols].style
                    .format({
                        "price":"Rs {:.2f}","stop_loss":"Rs {:.2f}",
                        "target1":"Rs {:.2f}","target2":"Rs {:.2f}",
                        "risk_amt":"Rs {:,.0f}","probability":"{:.3f}",
                        "rel_strength":"{:.2f}","vol_ratio":"{:.2f}",
                        "cmf":"{:.3f}","mfi":"{:.1f}","vwap_ratio":"{:.2f}%",
                        "risk_reward":"{:.1f}x","candle_score":"{:.2f}",
                        "rsi":"{:.1f}","adx":"{:.1f}","pct_from_52h":"{:.1f}%",
                    })
                )
                st.caption(f"Showing {len(filt)} of {len(df_signals)} signals")
                st.dataframe(styled, width='stretch', height=500)
                st.download_button("⬇ Download CSV",
                    filt[disp_cols].to_csv(index=False),
                    file_name=f"signals_{datetime.now():%Y%m%d_%H%M}.csv",
                    mime="text/csv")
            else:
                st.warning("No signals match the current filters. Try loosening the R:R or Rel Strength sliders.")

        # ── Top 20 AI Prospects (NO TRADE — near threshold) ───
        if not df_prospects.empty:
            st.divider()
            with st.expander(
                f"🔭 Top 20 AI Prospects  ({len(df_prospects)} stocks near signal threshold — click to expand)",
                expanded=(df_signals.empty),
            ):
                st.caption(
                    "These stocks passed the ML model filter but didn't satisfy enough "
                    "gate conditions today.  Monitor them — a volume spike or RSI move "
                    "could trigger a signal on the next scan."
                )
                p_disp = ["symbol","sector","probability","rsi","adx",
                          "rel_strength","vol_ratio","candle_score","pct_from_52h"]
                p_disp = [c for c in p_disp if c in df_prospects.columns]
                st.dataframe(
                    df_prospects[p_disp].style
                    .format({
                        "probability":"{:.3f}","rel_strength":"{:.2f}",
                        "vol_ratio":"{:.2f}","candle_score":"{:.2f}",
                        "rsi":"{:.1f}","adx":"{:.1f}","pct_from_52h":"{:.1f}%",
                    })
                    .background_gradient(subset=["probability"], cmap="YlOrRd"),
                    width='stretch',
                )
                st.download_button("⬇ Download Prospects CSV",
                    df_prospects[p_disp].to_csv(index=False),
                    file_name=f"prospects_{datetime.now():%Y%m%d_%H%M}.csv",
                    mime="text/csv",
                    key="dl_prospects",
                )

    elif st.session_state["scan_ts"]:
        # Scan ran but returned a completely empty list (should not happen normally)
        st.error(
            "Scan completed but found **0 results**.  \n"
            "**Try these fixes in the sidebar:**  \n"
            "- Lower **Min ML probability** to 0.55  \n"
            "- Lower **Min ADX** to 15  \n"
            "- Lower **Min Rel. Strength** to 0.90  \n"
            "Then click **▶ Run Full Scan** again.",
            icon="🔧",
        )
    else:
        st.markdown("""
        <div style="background:linear-gradient(135deg,#0c1a2e,#0a1525);
                    border:1px solid #1a3352;border-radius:14px;
                    padding:40px 32px;text-align:center;margin-top:16px;">
          <div style="font-size:2.5rem;margin-bottom:12px;">🔍</div>
          <div style="color:#f1f5f9;font-size:1.1rem;font-weight:700;margin-bottom:8px;">
            Ready to scan the market
          </div>
          <div style="color:#334d65;font-size:0.85rem;max-width:400px;margin:0 auto;">
            Click <strong style="color:#38bdf8;">▶ Run Full Scan</strong> above to screen
            all NIFTY 500 symbols through the 14-gate delivery signal engine.
          </div>
        </div>
        """, unsafe_allow_html=True)

# ╔══════════════════════════════════════╗
# ║  TAB 2 — CHART                      ║
# ╚══════════════════════════════════════╝
with tab_chart:
    results       = st.session_state["scan_results"]
    scan_ts       = st.session_state["scan_ts"]
    _sig_syms     = [r["symbol"] for r in results
                     if r.get("signal") in ("STRONG_BUY","BUY","WATCHLIST")]
    _all_syms     = [r["symbol"] for r in results]
    _rest_syms    = [s for s in _all_syms if s not in _sig_syms]
    _has_signals  = bool(_sig_syms)

    cc1, cc2, cc3 = st.columns([2, 3, 1])

    # Source selector — key tied to scan_ts so it resets after every fresh scan
    chart_src = cc1.radio(
        "Show symbols from",
        ["Signal stocks only", "All scanned symbols", "Search full list"],
        horizontal=False,
        key=f"chart_src_{scan_ts}",
        index=0 if _has_signals else 2,
    )

    if chart_src == "Signal stocks only":
        sym_opts = _sig_syms if _sig_syms else NIFTY_500
        if not _sig_syms:
            cc1.caption("No signals yet — run a scan first.")
    elif chart_src == "All scanned symbols":
        sym_opts = (_sig_syms + _rest_syms) if results else NIFTY_500
        if not results:
            cc1.caption("No scan run yet — showing full list.")
    else:
        sym_opts = NIFTY_500

    # Key tied to scan_ts+source so selection resets to top stock after each new scan
    chart_sym    = cc2.selectbox("Symbol", sym_opts,
                                 key=f"chart_sym_{scan_ts}_{chart_src}")
    chart_period = cc3.selectbox("Period", ["3mo","6mo","1y","2y"], index=2)

    if chart_sym:
        with st.spinner(f"Loading {chart_sym} …"):
            try:
                df_c = cached_load(chart_sym, chart_period)
                sig_info = next((r for r in results if r["symbol"]==chart_sym), None)

                fig = make_subplots(
                    rows=5, cols=1, shared_xaxes=True,
                    row_heights=[0.42,0.14,0.14,0.15,0.15],
                    vertical_spacing=0.02,
                    subplot_titles=("Price · EMAs · BB · Supertrend",
                                    "Volume","RSI (14) · MFI (14)","ADX / DI","CMF (20)"),
                )

                # Candlestick
                fig.add_trace(go.Candlestick(
                    x=df_c.index,
                    open=df_c["Open"], high=df_c["High"],
                    low=df_c["Low"],   close=df_c["Close"],
                    name="Price",
                    increasing_line_color="#22c55e",
                    decreasing_line_color="#ef4444",
                ), row=1, col=1)

                # EMAs
                for p,col in [(9,"#38bdf8"),(20,"#60a5fa"),(50,"#f59e0b"),(200,"#a78bfa")]:
                    k = f"ema{p}"
                    if k in df_c.columns:
                        fig.add_trace(go.Scatter(
                            x=df_c.index, y=df_c[k],
                            name=f"EMA{p}", line=dict(color=col,width=1.2), opacity=0.85,
                        ), row=1, col=1)

                # Supertrend line
                if "supertrend" in df_c.columns and "supertrend_bull" in df_c.columns:
                    bull_mask = df_c["supertrend_bull"] == 1
                    bear_mask = ~bull_mask
                    for mask, color, name in [
                        (bull_mask, "#22c55e", "ST Bull"),
                        (bear_mask, "#ef4444", "ST Bear"),
                    ]:
                        seg = df_c[mask]
                        if not seg.empty:
                            fig.add_trace(go.Scatter(
                                x=seg.index, y=seg["supertrend"],
                                name=name, line=dict(color=color, width=1.5, dash="dot"),
                                opacity=0.8, showlegend=True,
                            ), row=1, col=1)

                # Bollinger Bands
                if "bb_upper" in df_c.columns:
                    fig.add_trace(go.Scatter(
                        x=df_c.index, y=df_c["bb_upper"],
                        name="BB Upper", line=dict(color="#475569",width=1,dash="dot"),
                        showlegend=False,
                    ), row=1, col=1)
                    fig.add_trace(go.Scatter(
                        x=df_c.index, y=df_c["bb_lower"],
                        name="BB Lower", line=dict(color="#475569",width=1,dash="dot"),
                        fill="tonexty", fillcolor="rgba(71,85,105,0.08)",
                        showlegend=False,
                    ), row=1, col=1)

                # Signal SL / Target lines
                if sig_info:
                    last_d = df_c.index[-1]
                    fig.add_trace(go.Scatter(
                        x=[last_d], y=[sig_info["price"]],
                        mode="markers+text",
                        marker=dict(symbol="triangle-up",color="#22c55e",size=16),
                        text=[sig_info["signal"]], textposition="top center",
                        name="Signal",
                    ), row=1, col=1)
                    for val,lbl,color in [
                        (sig_info["stop_loss"],"SL","#ef4444"),
                        (sig_info["target1"],"T1","#fbbf24"),
                        (sig_info["target2"],"T2","#22c55e"),
                    ]:
                        fig.add_hline(y=val, line_color=color,
                            line_dash="dash", line_width=1.5,
                            annotation_text=f"  {lbl}: ₹{val:.2f}",
                            annotation_font_color=color, row=1, col=1)

                # Volume
                bar_colors = ["#22c55e" if c >= o else "#ef4444"
                              for c,o in zip(df_c["Close"], df_c["Open"])]
                fig.add_trace(go.Bar(
                    x=df_c.index, y=df_c["Volume"],
                    name="Volume", marker_color=bar_colors, opacity=0.7,
                ), row=2, col=1)
                if "vol_ma20" in df_c.columns:
                    fig.add_trace(go.Scatter(
                        x=df_c.index, y=df_c["vol_ma20"],
                        name="Vol MA20", line=dict(color="#f59e0b",width=1.2),
                    ), row=2, col=1)

                # RSI
                if "rsi" in df_c.columns:
                    fig.add_trace(go.Scatter(
                        x=df_c.index, y=df_c["rsi"],
                        name="RSI", line=dict(color="#a78bfa",width=1.5),
                    ), row=3, col=1)
                    for lvl,col in [(70,"#ef4444"),(50,"#64748b"),(30,"#22c55e")]:
                        fig.add_hline(y=lvl, line_dash="dot",
                                      line_color=col, line_width=1, row=3, col=1)
                # MFI overlaid on RSI panel
                if "mfi" in df_c.columns:
                    fig.add_trace(go.Scatter(
                        x=df_c.index, y=df_c["mfi"],
                        name="MFI", line=dict(color="#fbbf24",width=1.2,dash="dash"),
                        opacity=0.85,
                    ), row=3, col=1)

                # ADX + DI
                if "adx" in df_c.columns:
                    fig.add_trace(go.Scatter(
                        x=df_c.index, y=df_c["adx"],
                        name="ADX", line=dict(color="#fb923c",width=1.8),
                    ), row=4, col=1)
                if "adx_pos" in df_c.columns:
                    fig.add_trace(go.Scatter(
                        x=df_c.index, y=df_c["adx_pos"],
                        name="+DI", line=dict(color="#22c55e",width=1,dash="dot"),
                    ), row=4, col=1)
                if "adx_neg" in df_c.columns:
                    fig.add_trace(go.Scatter(
                        x=df_c.index, y=df_c["adx_neg"],
                        name="-DI", line=dict(color="#ef4444",width=1,dash="dot"),
                    ), row=4, col=1)
                    fig.add_hline(y=22, line_dash="dot", line_color="#475569",
                                  line_width=1, row=4, col=1)

                # CMF — Chaikin Money Flow (delivery volume panel)
                if "cmf" in df_c.columns:
                    cmf_colors = ["#22c55e" if v >= 0 else "#ef4444"
                                  for v in df_c["cmf"].fillna(0)]
                    fig.add_trace(go.Bar(
                        x=df_c.index, y=df_c["cmf"],
                        name="CMF", marker_color=cmf_colors, opacity=0.75,
                    ), row=5, col=1)
                    fig.add_hline(y=0, line_color="#64748b",
                                  line_width=1, row=5, col=1)
                    for lvl in [0.1, -0.1]:
                        fig.add_hline(y=lvl, line_dash="dot",
                                      line_color="#475569", line_width=1, row=5, col=1)

                fig.update_layout(
                    height=950, paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                    font=dict(color="#e2e8f0",size=12),
                    xaxis_rangeslider_visible=False,
                    legend=dict(orientation="h", y=1.02, bgcolor="rgba(0,0,0,0)"),
                    margin=dict(l=8,r=8,t=30,b=8),
                )
                fig.update_xaxes(gridcolor="#1e293b")
                fig.update_yaxes(gridcolor="#1e293b")

                st.plotly_chart(fig, width='stretch')

                # Info card
                if sig_info:
                    st.divider()
                    c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
                    c1.metric("Signal",    sig_info["signal"])
                    c2.metric("Price",     f"₹{sig_info['price']:,.2f}")
                    c3.metric("Stop Loss", f"₹{sig_info['stop_loss']:,.2f}")
                    c4.metric("Target 1",  f"₹{sig_info['target1']:,.2f}")
                    c5.metric("Target 2",  f"₹{sig_info['target2']:,.2f}")
                    c6.metric("Qty",       sig_info["quantity"])
                    c7.metric("R:R",       f"{sig_info['risk_reward']:.1f}x")
                    d1,d2,d3,d4 = st.columns(4)
                    d1.metric("CMF",            f"{sig_info.get('cmf', 0):.3f}")
                    d2.metric("MFI",            f"{sig_info.get('mfi', 0):.1f}")
                    d3.metric("Supertrend",     "Bull ✅" if sig_info.get("supertrend_bull") else "Bear ❌")
                    d4.metric("VWAP vs Price",  f"{sig_info.get('vwap_ratio', 0):.2f}%")

            except Exception as e:
                st.error(f"Chart error: {e}")

# ╔══════════════════════════════════════╗
# ║  TAB 3 — BACKTEST                   ║
# ╚══════════════════════════════════════╝
with tab_bt:
    if model is None:
        st.warning("Train the model first via the sidebar.")
    else:
        # ── Symbol source selector ─────────────────────────
        _scan_syms   = [r["symbol"] for r in st.session_state["scan_results"]]
        _signal_syms = [r["symbol"] for r in st.session_state["signal_stocks"]]
        _has_scan    = bool(_scan_syms)
        _has_signals = bool(_signal_syms)

        bt_src = st.radio(
            "Symbols to backtest",
            ["Signal stocks (Scanner results)", "All scanned symbols", "Top N from full list"],
            index=0 if _has_signals else 2,
            horizontal=True,
            help="'Signal stocks' = only STRONG_BUY / BUY / WATCHLIST from last scan. "
                 "'All scanned' = every symbol the scanner evaluated. "
                 "'Top N' = first N stocks in the NIFTY 500 list.",
        )

        bc1, bc2, bc3 = st.columns([2, 2, 2])
        bt_period = bc1.selectbox("Period", ["1y","2y","3y"], index=1)

        if bt_src == "Top N from full list":
            bt_n = bc2.slider("N symbols", 5, 50, 20)
            bt_syms_chosen = NIFTY_500[:bt_n]
        elif bt_src == "All scanned symbols":
            bt_syms_chosen = _scan_syms if _has_scan else NIFTY_500[:20]
            bc2.caption(f"{len(bt_syms_chosen)} scanned symbols")
        else:  # Signal stocks
            bt_syms_chosen = _signal_syms if _has_signals else NIFTY_500[:20]
            bc2.caption(f"{len(bt_syms_chosen)} signal stocks from last scan")

        run_bt = bc3.button("▶ Run Backtest", type="primary", use_container_width=True)

        if not _has_scan and bt_src != "Top N from full list":
            st.info("Run a scan first (Scanner tab) to backtest scanner results. "
                    "Using Top N as fallback.")

        if run_bt:
            bt_syms = bt_syms_chosen
            loader   = get_loader()
            idx_df   = loader.load(CONFIG["index_symbol"], period=bt_period)
            bt_eng   = Backtester(CONFIG)
            summary, all_trades = [], []
            prog = st.progress(0, text="Running backtest …")

            for idx, sym in enumerate(bt_syms):
                try:
                    df_bt = loader.load(sym, period=bt_period)
                    df_bt = add_indicators(df_bt)
                    df_bt = add_candle_patterns(df_bt)
                    try:
                        wk = loader.load_weekly(sym, period=bt_period)
                        df_bt = add_weekly_ema(df_bt, wk, CONFIG["weekly_ema_period"])
                    except Exception:
                        pass
                    m = bt_eng.run(df_bt, model, scaler, idx_df, sym)
                    m["symbol"] = sym
                    summary.append(m)
                    all_trades.extend(m.get("trades", []))
                except Exception as ex:
                    pass
                prog.progress((idx+1)/len(bt_syms),
                              text=f"{sym} … {idx+1}/{len(bt_syms)}")

            prog.empty()
            st.session_state["bt_summary"]   = summary
            st.session_state["bt_all_trades"] = all_trades

        summary    = st.session_state.get("bt_summary")
        all_trades = st.session_state.get("bt_all_trades", [])

        if summary:
            mdf = pd.DataFrame(summary)

            # ── Aggregate KPI cards ────────────────────
            _avg_cagr = mdf['cagr_pct'].mean()
            _avg_wr   = mdf['win_rate'].mean()
            _avg_dd   = mdf['max_drawdown_pct'].mean()
            _cagr_col = "#22c55e" if _avg_cagr >= 0 else "#ef4444"
            kc = st.columns(6)
            kc[0].markdown(f"""<div class="kpi-card" style="border-top:3px solid {_cagr_col};">
              <div class="kpi-label">Avg CAGR</div>
              <div class="kpi-value" style="color:{_cagr_col};">{_avg_cagr:.1f}%</div>
              <div class="kpi-sub">annualised return</div></div>""", unsafe_allow_html=True)
            kc[1].markdown(f"""<div class="kpi-card" style="border-top:3px solid #22c55e;">
              <div class="kpi-label">Win Rate</div>
              <div class="kpi-value" style="color:#86efac;">{_avg_wr:.0%}</div>
              <div class="kpi-sub">avg across symbols</div></div>""", unsafe_allow_html=True)
            kc[2].markdown(f"""<div class="kpi-card" style="border-top:3px solid #ef4444;">
              <div class="kpi-label">Avg Max DD</div>
              <div class="kpi-value" style="color:#fca5a5;">{_avg_dd:.1f}%</div>
              <div class="kpi-sub">peak-to-trough</div></div>""", unsafe_allow_html=True)
            kc[3].markdown(f"""<div class="kpi-card" style="border-top:3px solid #38bdf8;">
              <div class="kpi-label">Avg Sharpe</div>
              <div class="kpi-value" style="color:#7dd3fc;">{mdf['sharpe'].mean():.2f}</div>
              <div class="kpi-sub">risk-adjusted</div></div>""", unsafe_allow_html=True)
            kc[4].markdown(f"""<div class="kpi-card" style="border-top:3px solid #f59e0b;">
              <div class="kpi-label">Profit Factor</div>
              <div class="kpi-value" style="color:#fcd34d;">{mdf['profit_factor'].mean():.2f}</div>
              <div class="kpi-sub">gross win / loss</div></div>""", unsafe_allow_html=True)
            kc[5].markdown(f"""<div class="kpi-card" style="border-top:3px solid #a78bfa;">
              <div class="kpi-label">Total Trades</div>
              <div class="kpi-value" style="color:#c4b5fd;">{int(mdf['total_trades'].sum())}</div>
              <div class="kpi-sub">across all symbols</div></div>""", unsafe_allow_html=True)
            st.write("")
            st.divider()

            st.markdown('<div class="sec-label">Per-Symbol Results</div>', unsafe_allow_html=True)
            disp_bt = ["symbol","cagr_pct","win_rate","max_drawdown_pct",
                       "sharpe","sortino","calmar","profit_factor",
                       "avg_win_pct","avg_loss_pct","avg_hold_days","total_trades"]
            disp_bt = [c for c in disp_bt if c in mdf.columns]
            styled_bt = (
                mdf[disp_bt].style
                .format({
                    "cagr_pct":"{:.1f}%","win_rate":"{:.0%}",
                    "max_drawdown_pct":"{:.1f}%","sharpe":"{:.2f}",
                    "sortino":"{:.2f}","calmar":"{:.2f}",
                    "profit_factor":"{:.2f}","avg_win_pct":"{:.1f}%",
                    "avg_loss_pct":"{:.1f}%",
                })
                .background_gradient(subset=["cagr_pct"],         cmap="RdYlGn")
                .background_gradient(subset=["win_rate"],          cmap="Greens")
                .background_gradient(subset=["max_drawdown_pct"],  cmap="Reds_r")
                .background_gradient(subset=["sharpe"],            cmap="Blues")
            )
            st.dataframe(styled_bt, width='stretch')

            # ── Equity curves ─────────────────────────
            st.divider()
            st.markdown("**Equity Curves**")
            fig_eq = go.Figure()
            for m in summary:
                if m.get("equity_curve"):
                    norm = [v / m["equity_curve"][0] * 100
                            for v in m["equity_curve"]]
                    fig_eq.add_trace(go.Scatter(
                        y=norm, mode="lines", name=m["symbol"],
                        line=dict(width=1.2), opacity=0.75,
                    ))
            fig_eq.update_layout(
                title="Normalised Equity (start=100)",
                paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                font=dict(color="#e2e8f0"), height=380,
                margin=dict(l=8,r=8,t=40,b=8),
                xaxis=dict(gridcolor="#1e293b"),
                yaxis=dict(gridcolor="#1e293b", ticksuffix=""),
            )
            st.plotly_chart(fig_eq, width='stretch')

            # ── Drawdown bar ───────────────────────────
            fig_dd = go.Figure(go.Bar(
                x=mdf["symbol"],
                y=mdf["max_drawdown_pct"].abs(),
                marker_color="#ef4444", opacity=0.8,
            ))
            fig_dd.update_layout(
                title="Max Drawdown % by Symbol",
                paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                font=dict(color="#e2e8f0"), height=320,
                margin=dict(l=8,r=8,t=40,b=8),
                yaxis=dict(gridcolor="#1e293b", ticksuffix="%"),
                xaxis=dict(gridcolor="#1e293b"),
            )
            st.plotly_chart(fig_dd, width='stretch')

            # ── Trade exit reason distribution ─────────
            if all_trades:
                t_df   = pd.DataFrame(all_trades)
                reason = t_df["reason"].value_counts().reset_index()
                reason.columns = ["Reason","Count"]
                fig_r = px.pie(reason, names="Reason", values="Count",
                               title="Exit Reason Distribution",
                               color_discrete_sequence=["#22c55e","#ef4444","#f59e0b","#60a5fa"],
                               hole=0.4)
                fig_r.update_layout(
                    paper_bgcolor="#0f172a", font=dict(color="#e2e8f0"),
                    height=340, margin=dict(l=8,r=8,t=40,b=8),
                )
                cl1, cl2 = st.columns(2)
                with cl1:
                    st.plotly_chart(fig_r, width='stretch')
                with cl2:
                    # Win rate by exit reason
                    if "pnl" in t_df.columns:
                        t_df["win"] = (t_df["pnl"] > 0).astype(int)
                        wr_r = t_df.groupby("reason")["win"].mean().reset_index()
                        wr_r.columns = ["Reason","WinRate"]
                        fig_wr = px.bar(wr_r, x="Reason", y="WinRate",
                                        title="Win Rate by Exit Type",
                                        color="WinRate",
                                        color_continuous_scale="RdYlGn",
                                        range_color=[0,1])
                        fig_wr.update_layout(
                            paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                            font=dict(color="#e2e8f0"),
                            height=340, margin=dict(l=8,r=8,t=40,b=8),
                            yaxis=dict(gridcolor="#1e293b", tickformat=".0%"),
                            xaxis=dict(gridcolor="#1e293b"),
                        )
                        st.plotly_chart(fig_wr, width='stretch')
        else:
            st.info("Configure and click **▶ Run Backtest**.")

# ╔══════════════════════════════════════╗
# ║  TAB 4 — MONTHLY P&L HEATMAP        ║
# ╚══════════════════════════════════════╝
with tab_monthly:
    all_trades  = st.session_state.get("bt_all_trades", [])
    bt_summary  = st.session_state.get("bt_summary")
    if bt_summary is not None and not all_trades:
        # Backtest ran but generated 0 trades (model had 0% probabilities before fix)
        st.warning(
            "The backtest completed but found **0 trades**.  \n"
            "This usually happens when the ML model was trained before the probability fix.  \n"
            "**Re-run the backtest** on the Backtest tab — it will now generate real entries.",
            icon="🔄",
        )
    elif not all_trades:
        st.info("Run the backtest first (Backtest tab) to see monthly P&L.")
    else:
        pivot = monthly_pnl_matrix(all_trades)
        if pivot.empty:
            st.warning("Not enough trade data for monthly matrix.")
        else:
            st.markdown("### Monthly P&L Heatmap  (₹)")
            # Plotly heatmap
            month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                           "Jul","Aug","Sep","Oct","Nov","Dec"]
            cols_present = [c for c in month_names if c in pivot.columns]
            z_vals = pivot[cols_present].values.tolist()
            fig_h = go.Figure(go.Heatmap(
                z=z_vals,
                x=cols_present,
                y=[str(y) for y in pivot.index],
                colorscale="RdYlGn",
                zmid=0,
                text=[[f"₹{v:,.0f}" for v in row] for row in z_vals],
                texttemplate="%{text}",
                textfont=dict(size=11),
                colorbar=dict(title="P&L (₹)"),
            ))
            fig_h.update_layout(
                paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                font=dict(color="#e2e8f0"), height=380,
                margin=dict(l=8,r=8,t=20,b=8),
            )
            st.plotly_chart(fig_h, width='stretch')

            # Annual totals
            t_df = pd.DataFrame(all_trades)
            t_df["exit_dt"] = pd.to_datetime(t_df["exit_date"])
            t_df["year"]    = t_df["exit_dt"].dt.year
            ann = t_df.groupby("year")["pnl"].agg(["sum","count"]).reset_index()
            ann.columns = ["Year", "Total_PnL", "Trades"]
            ann["Avg_PnL"] = ann["Total_PnL"] / ann["Trades"]
            st.dataframe(
                ann.style.format({
                    "Total_PnL": "Rs {:,.0f}",
                    "Avg_PnL":   "Rs {:,.0f}",
                }).background_gradient(subset=["Total_PnL"], cmap="RdYlGn"),
                width='stretch',
            )

# ╔══════════════════════════════════════╗
# ║  TAB 5 — PORTFOLIO SIZING           ║
# ╚══════════════════════════════════════╝
with tab_port:
    # Use the same signal_stocks list that the Scanner tab shows — keeps both in sync
    sig_results = st.session_state["signal_stocks"]
    if not sig_results:
        st.info("Run a scan first (Scanner tab) to see portfolio sizing.")
    else:
        rm   = RiskManager(CONFIG)
        rows, heat = [], 0.0

        for r in sig_results:
            if heat >= CONFIG["max_portfolio_risk"]:
                break
            qty    = rm.position_size(CONFIG["capital"], r["price"],
                                       r["stop_loss"], r["probability"])
            invest = r["price"] * qty
            risk_r = (r["price"] - r["stop_loss"]) * qty
            rp     = risk_r / CONFIG["capital"]
            heat  += rp
            rows.append({
                "Symbol":    r["symbol"],
                "Sector":    r["sector"],
                "Signal":    r["signal"],
                "Price":     r["price"],
                "StopLoss":  r["stop_loss"],
                "Target2":   r["target2"],
                "Qty":       qty,
                "Invested":  round(invest, 0),
                "Risk":      round(risk_r, 0),
                "RiskPct":   round(rp * 100, 2),
                "RR":        r["risk_reward"],
            })

        p_df = pd.DataFrame(rows)

        # Apply same emoji signal labels as Scanner tab
        _SIG_LABEL = {
            "STRONG_BUY": "🟢 STRONG BUY",
            "BUY":        "🔵 BUY",
            "WATCHLIST":  "🟣 WATCHLIST",
        }
        p_df["Signal"] = p_df["Signal"].map(_SIG_LABEL).fillna(p_df["Signal"])

        tot_inv   = p_df["Invested"].sum()
        tot_risk  = p_df["Risk"].sum()
        heat_pct  = tot_risk / CONFIG["capital"] * 100
        heat_col  = "#22c55e" if heat_pct < 5 else "#f59e0b" if heat_pct < 7 else "#ef4444"

        pc = st.columns(4)
        pc[0].markdown(f"""<div class="kpi-card" style="border-top:3px solid #38bdf8;">
          <div class="kpi-label">Open Positions</div>
          <div class="kpi-value" style="color:#7dd3fc;">{len(p_df)}</div>
          <div class="kpi-sub">max {CONFIG['max_positions']}</div></div>""", unsafe_allow_html=True)
        pc[1].markdown(f"""<div class="kpi-card" style="border-top:3px solid #6366f1;">
          <div class="kpi-label">Total Invested</div>
          <div class="kpi-value" style="color:#a5b4fc;font-size:1.2rem;">₹{tot_inv:,.0f}</div>
          <div class="kpi-sub">of ₹{CONFIG['capital']:,.0f} capital</div></div>""", unsafe_allow_html=True)
        pc[2].markdown(f"""<div class="kpi-card" style="border-top:3px solid #ef4444;">
          <div class="kpi-label">Total Risk</div>
          <div class="kpi-value" style="color:#fca5a5;font-size:1.2rem;">₹{tot_risk:,.0f}</div>
          <div class="kpi-sub">if all SLs hit</div></div>""", unsafe_allow_html=True)
        pc[3].markdown(f"""<div class="kpi-card" style="border-top:3px solid {heat_col};">
          <div class="kpi-label">Portfolio Heat</div>
          <div class="kpi-value" style="color:{heat_col};">{heat_pct:.1f}%</div>
          <div class="kpi-sub">max {CONFIG['max_portfolio_risk']*100:.0f}% allowed</div></div>""",
          unsafe_allow_html=True)
        st.write("")
        st.divider()

        st.markdown("""
        <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:12px;">
          <span style="color:#4a7a99;font-size:0.70rem;font-weight:700;
                       text-transform:uppercase;letter-spacing:0.09em;">Signal key</span>
          <span class="sig-strong">🟢 STRONG BUY</span>
          <span class="sig-buy">🔵 BUY</span>
          <span class="sig-watch">🟣 WATCHLIST</span>
        </div>
        """, unsafe_allow_html=True)

        st.dataframe(
            p_df.style.format({
                "Price":    "Rs {:.2f}",
                "StopLoss": "Rs {:.2f}",
                "Target2":  "Rs {:.2f}",
                "Invested": "Rs {:,.0f}",
                "Risk":     "Rs {:,.0f}",
                "RiskPct":  "{:.2f}%",
                "RR":       "{:.1f}x",
            }),
            width='stretch',
        )

        # Allocation charts side by side
        ca1, ca2 = st.columns(2)
        with ca1:
            fig_pie = go.Figure(go.Pie(
                labels=p_df["Symbol"], values=p_df["Invested"],
                hole=0.45, textinfo="label+percent",
                marker=dict(colors=px.colors.qualitative.Vivid),
            ))
            fig_pie.update_layout(
                title="Capital Allocation by Symbol",
                paper_bgcolor="#0f172a", font=dict(color="#e2e8f0"),
                height=400, margin=dict(l=8,r=8,t=40,b=8),
                showlegend=False,
            )
            st.plotly_chart(fig_pie, width='stretch')
        with ca2:
            sec_grp = p_df.groupby("Sector")["Invested"].sum().reset_index()
            fig_sec = go.Figure(go.Pie(
                labels=sec_grp["Sector"], values=sec_grp["Invested"],
                hole=0.45, textinfo="label+percent",
                marker=dict(colors=px.colors.qualitative.Safe),
            ))
            fig_sec.update_layout(
                title="Capital Allocation by Sector",
                paper_bgcolor="#0f172a", font=dict(color="#e2e8f0"),
                height=400, margin=dict(l=8,r=8,t=40,b=8),
                showlegend=False,
            )
            st.plotly_chart(fig_sec, width='stretch')

        st.download_button(
            "⬇ Download Portfolio Sheet",
            p_df.to_csv(index=False),
            file_name=f"portfolio_{datetime.now():%Y%m%d_%H%M}.csv",
            mime="text/csv",
        )

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div style="margin-top:48px;padding:20px 28px;
            border-top:1px solid #1a3352;
            display:flex;align-items:center;justify-content:space-between;
            flex-wrap:wrap;gap:8px;">
  <div style="display:flex;align-items:center;gap:10px;">
    <span style="font-size:1.2rem;">📈</span>
    <span style="color:#1f3b56;font-size:0.78rem;font-weight:700;letter-spacing:0.04em;">
      AI NIFTY 500 Positional Trader &nbsp;·&nbsp; v2.0
    </span>
  </div>
  <div style="color:#1a3352;font-size:0.73rem;text-align:right;">
    Built by &nbsp;<strong style="color:#334d65;font-weight:700;">Prasad R. Paranjape</strong>
    &nbsp;·&nbsp; XGBoost + Supertrend + CMF
    &nbsp;·&nbsp; 14-gate delivery signal engine
  </div>
</div>
""", unsafe_allow_html=True)
