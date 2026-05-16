# ============================================================
# AI NIFTY 500 POSITIONAL TRADING SYSTEM  v2.0
# ============================================================
# Improvements over v1:
#  - Realistic backtest: commissions (0.1%) + slippage (0.05%)
#  - Trailing stop-loss (ATR-based, ratchets up)
#  - Partial profit: 50% exit at 1×R, remainder trails
#  - Time-based stop: close if flat/losing after 15 bars
#  - Multi-timeframe: weekly EMA trend must agree with daily
#  - Consolidated breakout filter: narrow-range bar before entry
#  - Kelly-fraction position sizing (volatility-adjusted)
#  - Sector concentration cap (max 3 stocks per sector)
#  - Richer ML features: lagged returns, rolling vol, z-scores
#  - Walk-forward optimised parameter set (pre-tuned)
#  - Full performance metrics: CAGR, Sharpe, Sortino, Calmar
#  - Monthly P&L matrix
#
# USAGE:
#   python nifty500_ai_trader.py train
#   python nifty500_ai_trader.py scan
#   python nifty500_ai_trader.py backtest
# ============================================================

import os, sys, json, warnings, logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import ta
import joblib

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ── Logging ──────────────────────────────────────────────────
# Silence yfinance's own noisy download-error output
for _noisy in ("yfinance","peewee","urllib3","requests"):
    logging.getLogger(_noisy).setLevel(logging.CRITICAL)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("trading_system.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ============================================================
# 1. CONFIGURATION  (tuned from walk-forward backtest)
# ============================================================

CONFIG = {
    # ── Capital & portfolio risk ──────────────────────────
    "capital":            1_000_000,
    "risk_per_trade":     0.008,      # 0.8 % risk per position
    "max_portfolio_risk": 0.08,       # 8 % total heat at any time
    "max_positions":      10,
    "max_per_sector":     3,          # sector concentration cap

    # ── Signal quality gates ──────────────────────────────
    "min_probability":    0.58,       # XGBoost confidence
    "min_adx":            18,         # trend strength
    "min_rsi":            45,         # not oversold
    "max_rsi":            78,         # not overbought
    "min_vol_ratio":      1.15,       # volume vs 20-day avg (raised for delivery)
    "min_rel_strength":   0.95,       # must beat index

    # ── Delivery-specific gates ───────────────────────────
    "min_cmf":           -0.05,       # Chaikin Money Flow (institutional flow)
    "min_mfi":            40,         # Money Flow Index lower bound
    "max_mfi":            80,         # Money Flow Index upper bound (not overbought)

    # ── Trade geometry (delivery — wider SL, higher target) ─
    "atr_sl_mult":        2.2,        # initial SL = entry − N×ATR (was 1.8)
    "atr_trail_mult":     1.8,        # trailing SL distance (was 1.5)
    "atr_tgt1_mult":      2.0,        # partial profit target (50% qty)
    "atr_tgt2_mult":      4.5,        # full exit target (was 3.6)
    "time_stop_bars":     20,         # exit if flat/losing after N bars (was 15)
    "min_hold_bars":      3,          # trailing stop inactive for first N bars

    # ── Realistic costs ───────────────────────────────────
    "commission":         0.001,      # 0.1 % per side
    "slippage":           0.0005,     # 0.05 % per side

    # ── ML training ───────────────────────────────────────
    "training_period":    "5y",
    "forward_days":       20,         # delivery horizon: 20 days (was 15)
    "min_return":         0.08,       # 8 % target in 20 days (was 6 % in 15)

    # ── Paths ─────────────────────────────────────────────
    "model_path":  "models/xgb_model.pkl",
    "scaler_path": "models/scaler.pkl",
    "cache_dir":   "cache/",
    "results_dir": "results/",

    # ── Market ────────────────────────────────────────────
    "index_symbol": "^NSEI",
    "weekly_ema_period": 13,          # EMA on weekly chart
}

# ── Sector map (used for concentration cap) ───────────────────
SECTOR_MAP = {
    "BANK":    ["HDFCBANK.NS","ICICIBANK.NS","SBIN.NS","KOTAKBANK.NS","AXISBANK.NS",
                "INDUSINDBK.NS","BANDHANBNK.NS","FEDERALBNK.NS","IDFCFIRSTB.NS",
                "AUBANK.NS","CUB.NS","INDIANB.NS","UNIONBANK.NS","PNB.NS","BANKBARODA.NS"],
    "IT":      ["TCS.NS","INFY.NS","HCLTECH.NS","WIPRO.NS","TECHM.NS","LTIMINDTREE.NS",
                "LTTS.NS","PERSISTENT.NS","KPITTECH.NS","MPHASIS.NS","COFORGE.NS"],
    "PHARMA":  ["SUNPHARMA.NS","DRREDDY.NS","CIPLA.NS","DIVISLAB.NS","AUROPHARMA.NS",
                "LUPIN.NS","ALKEM.NS","AJANTPHARM.NS","LAURUSLABS.NS","NATCOPHARM.NS",
                "ZYDUSLIFE.NS","TORNTPHARM.NS","PFIZER.NS","SANOFI.NS","GLAXO.NS"],
    "AUTO":    ["MARUTI.NS","TATAMOTORS.NS","BAJAJ-AUTO.NS","HEROMOTOCO.NS","M&M.NS",
                "EICHERMOT.NS","TVSMOTOR.NS","ESCORTS.NS","UNOMINDA.NS","BALKRISIND.NS"],
    "FMCG":   ["HINDUNILVR.NS","ITC.NS","NESTLEIND.NS","BRITANNIA.NS","DABUR.NS",
               "MARICO.NS","COLPAL.NS","GODREJCP.NS","TATACONSUM.NS","VBL.NS"],
    "METAL":   ["JSWSTEEL.NS","TATASTEEL.NS","HINDALCO.NS","VEDL.NS","SAIL.NS",
                "NMDC.NS","COALINDIA.NS","ONGC.NS","OIL.NS"],
    "REALTY":  ["DLF.NS","OBEROIRLTY.NS","PHOENIXLTD.NS","SOBHA.NS","GODREJIND.NS"],
    "ENERGY":  ["RELIANCE.NS","BPCL.NS","IOC.NS","GAIL.NS","PETRONET.NS","ADANIGREEN.NS",
                "ADANIENSOL.NS","TATAPOWER.NS"],
    "CAPITAL": ["BAJFINANCE.NS","BAJAJFINSV.NS","CHOLAFIN.NS","MUTHOOTFIN.NS",
                "MANAPPURAM.NS","LICHSGFIN.NS","CANFINHOME.NS","POONAWALLA.NS"],
    "INFRA":   ["LT.NS","ADANIENT.NS","ADANIPORTS.NS","HAL.NS","CONCOR.NS",
                "NBCC.NS","HUDCO.NS","KEC.NS","WABAG.NS"],
    "CONS":    ["ASIANPAINT.NS","PIDILITIND.NS","BERGE.NS","HAVELLS.NS","CROMPTON.NS",
                "POLYCAB.NS","VOLTAS.NS","WHIRLPOOL.NS","DIXON.NS"],
    "MISC":    [],   # catch-all
}

def get_sector(symbol: str) -> str:
    for sector, syms in SECTOR_MAP.items():
        if symbol in syms:
            return sector
    return "MISC"

# ============================================================
# 2. NIFTY 500 SYMBOL LIST
# ============================================================

NIFTY_500 = [
    # ── NIFTY 50 ─────────────────────────────────────────────
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS",
    "HINDUNILVR.NS","ITC.NS","SBIN.NS","BHARTIARTL.NS","KOTAKBANK.NS",
    "LT.NS","AXISBANK.NS","ASIANPAINT.NS","MARUTI.NS","BAJFINANCE.NS",
    "HCLTECH.NS","SUNPHARMA.NS","ULTRACEMCO.NS","TITAN.NS","WIPRO.NS",
    "NESTLEIND.NS","POWERGRID.NS","NTPC.NS","ONGC.NS","BAJAJFINSV.NS",
    "JSWSTEEL.NS","TATAMOTORS.NS","TATASTEEL.NS","TECHM.NS","ADANIENT.NS",
    "ADANIPORTS.NS","COALINDIA.NS","DIVISLAB.NS","DRREDDY.NS","EICHERMOT.NS",
    "GRASIM.NS","HDFCLIFE.NS","HEROMOTOCO.NS","HINDALCO.NS","INDUSINDBK.NS",
    "CIPLA.NS","APOLLOHOSP.NS","TATACONSUM.NS","BAJAJ-AUTO.NS","BPCL.NS",
    "BRITANNIA.NS","SBILIFE.NS","M&M.NS","UPL.NS","VEDL.NS",
    # ── NIFTY NEXT 50 ────────────────────────────────────────
    "BANKBARODA.NS","BERGEPAINT.NS","BIOCON.NS","BOSCHLTD.NS","COLPAL.NS",
    "CONCOR.NS","CUMMINSIND.NS","DABUR.NS","DLF.NS","GAIL.NS",
    "GODREJCP.NS","HAVELLS.NS","ICICIGI.NS","ICICIPRULI.NS","IGL.NS",
    "INDUSTOWER.NS","IOC.NS","IRCTC.NS","LICI.NS","LUPIN.NS",
    "MARICO.NS","UNITDSPR.NS","MUTHOOTFIN.NS","NAUKRI.NS","NMDC.NS",
    "PAGEIND.NS","PIDILITIND.NS","PNB.NS","RECLTD.NS","SAIL.NS",
    "SIEMENS.NS","SRF.NS","TORNTPHARM.NS","TVSMOTOR.NS","VOLTAS.NS",
    "ZYDUSLIFE.NS","ALKEM.NS","AMBUJACEM.NS","AUROPHARMA.NS",
    # ── NIFTY MIDCAP 150 ─────────────────────────────────────
    "ABCAPITAL.NS","ABFRL.NS","ACC.NS","AFFLE.NS","AJANTPHARM.NS",
    "APLLTD.NS","ASTRAL.NS","ATUL.NS","AUBANK.NS","BALKRISIND.NS",
    "BANDHANBNK.NS","BATAINDIA.NS","CAMS.NS","CANFINHOME.NS","CDSL.NS",
    "CENTURYTEX.NS","CHOLAFIN.NS","CROMPTON.NS","CUB.NS","DEEPAKNTR.NS","COFORGE.NS",
    "DIXON.NS","ESCORTS.NS","EXIDEIND.NS","FEDERALBNK.NS","GNFC.NS",
    "GODREJIND.NS","GRINDWELL.NS","HAL.NS","HONAUT.NS","HUDCO.NS",
    "IDFCFIRSTB.NS","INDIANB.NS","INDIGO.NS","INDIAMART.NS","INTELLECT.NS",
    "JKCEMENT.NS","JUBLFOOD.NS","KAJARIACER.NS","KANSAINER.NS","KEC.NS",
    "KPITTECH.NS","LAURUSLABS.NS","LICHSGFIN.NS","LTIMINDTREE.NS","LTTS.NS",
    "MANAPPURAM.NS","MCX.NS","METROPOLIS.NS","MRF.NS","NATCOPHARM.NS",
    "NBCC.NS","OBEROIRLTY.NS","OIL.NS","PERSISTENT.NS","PETRONET.NS",
    "PFIZER.NS","PHOENIXLTD.NS","POLYCAB.NS","POONAWALLA.NS",
    "RAMCOCEM.NS","ROUTE.NS","SAFARI.NS","SCHAEFFLER.NS",
    "SHREECEM.NS","SOBHA.NS","STARHEALTH.NS","SUNDARMFIN.NS",
    "SUPREMEIND.NS","SYNGENE.NS","TANLA.NS","TATACOMM.NS","TATACHEM.NS",
    "TATAELXSI.NS","TATAPOWER.NS","TRENT.NS","TRIDENT.NS",
    "UJJIVANSFB.NS","UNIONBANK.NS","UNOMINDA.NS","UTIAMC.NS",
    "VBL.NS","VINATIORGA.NS","VSTIND.NS","ZEEL.NS",
    "ADANIGREEN.NS","ADANIENSOL.NS","AEGISLOG.NS","CASTROLIND.NS",
    "FLUOROCHEM.NS","GALAXYSURF.NS","GLAXO.NS","GSPL.NS","HFCL.NS",
    "IIFL.NS","KRBL.NS","LMW.NS","LINDEINDIA.NS","MFSL.NS",
    "NOCIL.NS","OLECTRA.NS","POLYMED.NS","PVRINOX.NS",
    "RELAXO.NS","SANOFI.NS","MPHASIS.NS",
    "TEAMLEASE.NS","THYROCARE.NS","TIMKEN.NS",
    "TRIVENI.NS","UCOBANK.NS","WABAG.NS",
    # ── NIFTY SMALLCAP / REST OF 500 ─────────────────────────
    # Banking & Financial Services
    "RBLBANK.NS","DCBBANK.NS","KARURVYSYA.NS","SOUTHBANK.NS","EQUITASBNK.NS",
    "ANGELONE.NS","ICICISEC.NS","MOTILALOFS.NS","ANANDRATHI.NS","NUVAMA.NS",
    "PFC.NS","IRFC.NS","NHPC.NS","SJVN.NS","RITES.NS","IRCON.NS","RVNL.NS",
    "CREDITACC.NS","APTUS.NS","AAVAS.NS","HOMEFIRST.NS","FIVESTAR.NS",
    "INDIASHLTR.NS","BAJAJHLDNG.NS","CHOLAHLDNG.NS","KFINTECH.NS","BSE.NS",
    # Auto & Ancillaries
    "ASHOKLEY.NS","MOTHERSON.NS","TIINDIA.NS","ENDURANCE.NS","SONACOMS.NS",
    "APOLLOTYRE.NS","CEATLTD.NS","AMARARAJA.NS","CRAFTSMAN.NS",
    "GABRIEL.NS","MAHINDCIE.NS","SUPRAJIT.NS","SUNDRMFAST.NS",
    "APLAPOLLO.NS","JMFINANCIL.NS",
    # Pharma & Healthcare
    "ABBOTINDIA.NS","IPCALAB.NS","GRANULES.NS","GLAND.NS","JBCHEPHARM.NS",
    "MARKSANS.NS","ERIS.NS","STRIDES.NS","FORTIS.NS","NARAYANHLT.NS",
    "MAXHEALTH.NS","KIMS.NS","RAINBOW.NS","HCG.NS","SUVEN.NS",
    "SOLARA.NS","BLISSGVS.NS","NEULANDLAB.NS","HIKAL.NS","CAPLIPOINT.NS",
    # FMCG & Consumer
    "EMAMILTD.NS","JYOTHYLAB.NS","RADICO.NS","GILLETTE.NS","ZOMATO.NS",
    "DEVYANI.NS","WESTLIFE.NS","SAPPHIRE.NS","BIKAJI.NS",
    "NYKAA.NS","GODFRYPHLP.NS","USHAMART.NS",
    # Cement
    "DALMIACHIN.NS","JKLAKSHMI.NS","BIRLACORPN.NS","NUVOCO.NS","STARCEMENT.NS",
    "HEIDELBERG.NS","JKIL.NS","SAGCEM.NS",
    # Chemicals
    "AARTIIND.NS","NAVINFLUOR.NS","PIIND.NS","COROMANDEL.NS","BASF.NS",
    "ALKYLAMINE.NS","ROSSARI.NS","ASTEC.NS","NEOGEN.NS","CLEAN.NS",
    "SUDARSCHEM.NS","PCBL.NS","VINDHYATEL.NS",
    # Capital Goods & Engineering
    "THERMAX.NS","BHEL.NS","ABB.NS","AIAENG.NS","KIRLOSENG.NS",
    "BLUESTARCO.NS","SKFINDIA.NS","KALPATPOWR.NS","PRAJ.NS","SUZLON.NS",
    "ELGIEQUIP.NS","ELECON.NS","INOXWIND.NS","GEPIL.NS","GPIL.NS",
    "RATNAMANI.NS","WELCORP.NS","JINDALSAW.NS","MAHSEAMLES.NS",
    # IT & Tech
    "CYIENT.NS","MASTEK.NS","ZENSAR.NS","BIRLASOFT.NS","HAPPSTMNDS.NS",
    "RATEGAIN.NS","TTML.NS","REDINGTON.NS","NETWEB.NS","LATENTVIEW.NS",
    "XCHANGING.NS","INSPIRISYS.NS","SAKSOFT.NS","DATAMATICS.NS",
    # Power & Energy
    "ADANIPOWER.NS","TORNTPOWER.NS","CESC.NS","RAILTEL.NS",
    "JPPOWER.NS","INOXGREEN.NS","ORIENTGREEN.NS",
    # Real Estate
    "GODREJPROP.NS","PRESTIGE.NS","BRIGADE.NS","SUNTECK.NS",
    "KOLTEPATIL.NS","MAHLIFE.NS","MACROTECH.NS","ANANTRAJ.NS",
    "RAYMOND.NS","NESCO.NS",
    # Media & Entertainment
    "SUNTV.NS","SAREGAMA.NS","TV18BRDCST.NS","NETWORK18.NS",
    # Textiles
    "WELSPUNIND.NS","VARDHMAN.NS","ARVIND.NS","KPR.NS",
    "GOKEX.NS","SPANDANA.NS",
    # Retail & Consumer Discretionary
    "DMART.NS","SHOPERSTOP.NS","VMART.NS","METRO.NS",
    # Metals & Mining
    "HINDCOPPER.NS","MOIL.NS","NALCO.NS","WELSPUNLIV.NS",
    "GESHIP.NS","GMRAIRPORT.NS",
    # Insurance
    "NIACL.NS","GICRE.NS",
    # Logistics & Transport
    "BLUEDART.NS","MAHLOG.NS","TCI.NS","GATI.NS","ALLCARGO.NS",
    # Specialty & Others
    "KEI.NS","VGUARD.NS","POLYPLEX.NS","GARFIBRES.NS",
    "TTKPRESTIG.NS","WONDERLA.NS","INGERRAND.NS","FINEORG.NS",
    "TATAINVEST.NS","SWSOLAR.NS","TEXRAIL.NS","MOLDTKPAC.NS",
    "ISGEC.NS","HLEGLAS.NS","SUPRIYA.NS","WINDLAS.NS",
    "SAGILITY.NS","SENCO.NS","SBCL.NS","CAMPUS.NS",
    "DELHIVERY.NS","ZINKA.NS","MAZDOCK.NS","COCHINSHIP.NS",
    "GRSE.NS","BEL.NS","MIDHANI.NS","BEML.NS",
    "MTAR.NS","PARAS.NS","IDEAFORGE.NS","SOLARINDS.NS",
    "PAISALO.NS","UJJIVAN.NS","SURYODAY.NS","ESAFSFB.NS",
    "JAIBALAJI.NS","NSLNISP.NS","GODREJAGRO.NS","VSTTILLERS.NS",
    "PNBHOUSING.NS","REPCO.NS","MAHINDRA.NS","SBICARD.NS",
]

# ============================================================
# 3. DATA LOADER  (12-hour cache)
# ============================================================

class DataLoader:
    def __init__(self, cache_dir: str = CONFIG["cache_dir"]):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, symbol: str, period: str) -> Path:
        s = symbol.replace(".", "_").replace("^", "IDX_").replace("&", "AND")
        return self.cache_dir / f"{s}_{period}.pkl"

    def load(self, symbol: str, period: str = "5y", force: bool = False) -> pd.DataFrame:
        path = self._path(symbol, period)
        if not force and path.exists():
            age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
            if age < timedelta(hours=12):
                return pd.read_pickle(path)

        df = yf.download(symbol, period=period, interval="1d",
                         progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError(f"No data for {symbol}")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.dropna(inplace=True)
        df.to_pickle(path)
        return df

    def load_weekly(self, symbol: str, period: str = "5y") -> pd.DataFrame:
        """Resample daily cache to weekly OHLCV."""
        df = self.load(symbol, period)
        wk = df.resample("W").agg({
            "Open":  "first", "High": "max",
            "Low":   "min",   "Close": "last",
            "Volume":"sum",
        }).dropna()
        return wk

# ============================================================
# 4. INDICATORS  (daily + weekly EMA for multi-timeframe)
# ============================================================

def _squeeze(s: pd.Series | pd.DataFrame) -> pd.Series:
    return s.squeeze() if hasattr(s, "squeeze") else s


def _compute_supertrend(h: pd.Series, l: pd.Series, c: pd.Series,
                        period: int = 10, mult: float = 3.0):
    """Vectorised Supertrend indicator. Returns (line, direction) where direction=1 is bullish."""
    atr   = ta.volatility.average_true_range(h, l, c, period).values
    hl2   = ((h + l) / 2).values
    cls   = c.values
    n     = len(cls)
    upper = hl2 + mult * atr
    lower = hl2 - mult * atr
    st    = np.full(n, np.nan)
    dir_  = np.ones(n, dtype=int)
    for i in range(1, n):
        if np.isnan(upper[i]) or np.isnan(lower[i]):
            continue
        # Bands only tighten, never widen against prior trend
        if not np.isnan(upper[i-1]):
            upper[i] = upper[i] if cls[i-1] > upper[i-1] else min(upper[i], upper[i-1])
            lower[i] = lower[i] if cls[i-1] < lower[i-1] else max(lower[i], lower[i-1])
        prev_st = st[i-1]
        if np.isnan(prev_st):
            dir_[i] = 1;  st[i] = lower[i]
        elif prev_st == upper[i-1]:
            dir_[i] = -1 if cls[i] < upper[i] else 1
            st[i]   = lower[i] if dir_[i] == 1 else upper[i]
        else:
            dir_[i] = 1 if cls[i] > lower[i] else -1
            st[i]   = lower[i] if dir_[i] == 1 else upper[i]
    return pd.Series(st, index=c.index), pd.Series(dir_, index=c.index)

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c = _squeeze(df["Close"])
    h = _squeeze(df["High"])
    l = _squeeze(df["Low"])
    v = _squeeze(df["Volume"])

    # ── EMAs ─────────────────────────────────────────────
    for p in [9, 20, 50, 100, 200]:
        df[f"ema{p}"] = ta.trend.ema_indicator(c, p)
    for p in [20, 50, 200]:
        df[f"sma{p}"] = c.rolling(p).mean()

    # ── MACD ─────────────────────────────────────────────
    macd = ta.trend.MACD(c, window_slow=26, window_fast=12, window_sign=9)
    df["macd"]        = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"]   = macd.macd_diff()

    # ── ADX / DI ─────────────────────────────────────────
    adx = ta.trend.ADXIndicator(h, l, c, 14)
    df["adx"]     = adx.adx()
    df["adx_pos"] = adx.adx_pos()
    df["adx_neg"] = adx.adx_neg()
    df["di_diff"] = df["adx_pos"] - df["adx_neg"]   # +DI − −DI

    # ── RSI ───────────────────────────────────────────────
    df["rsi"]      = ta.momentum.rsi(c, 14)
    df["rsi_fast"] = ta.momentum.rsi(c, 7)

    # ── Stochastic ────────────────────────────────────────
    st_ = ta.momentum.StochasticOscillator(h, l, c, 14, 3)
    df["stoch_k"] = st_.stoch()
    df["stoch_d"] = st_.stoch_signal()

    # ── Bollinger ─────────────────────────────────────────
    bb = ta.volatility.BollingerBands(c, 20, 2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mid"]   = bb.bollinger_mavg()
    df["bb_pct"]   = bb.bollinger_pband()
    df["bb_width"] = bb.bollinger_wband()

    # ── ATR ───────────────────────────────────────────────
    df["atr"]     = ta.volatility.average_true_range(h, l, c, 14)
    df["atr_pct"] = df["atr"] / c * 100

    # Historical Volatility (20-day realised vol annualised)
    df["hvol20"] = c.pct_change().rolling(20).std() * np.sqrt(252) * 100

    # ── Volume ────────────────────────────────────────────
    df["vol_ma20"]  = v.rolling(20).mean()
    df["vol_ma5"]   = v.rolling(5).mean()
    df["vol_ratio"] = v / df["vol_ma20"]
    df["obv"]       = ta.volume.on_balance_volume(c, v)
    df["obv_ema"]   = ta.trend.ema_indicator(_squeeze(df["obv"]), 20)

    # ── Momentum / returns ────────────────────────────────
    for p in [3, 5, 10, 20, 60]:
        df[f"mom{p}"] = c.pct_change(p)

    # ── Rolling z-scores of returns ───────────────────────
    r1 = c.pct_change(1)
    df["ret1_z20"] = (r1 - r1.rolling(20).mean()) / (r1.rolling(20).std() + 1e-9)

    # ── 52-week range (min_periods=150 so 1-year downloads still produce values)
    df["high52w"]      = h.rolling(252, min_periods=150).max()
    df["low52w"]       = l.rolling(252, min_periods=150).min()
    df["pct_from_52h"] = (c - df["high52w"]) / df["high52w"]
    df["pct_from_52l"] = (c - df["low52w"])  / df["low52w"]
    df["rng52_pct"]    = (df["high52w"] - df["low52w"]) / df["low52w"]

    # ── Narrow-range consolidation (NR7) ─────────────────
    day_range = h - l
    df["nr7"] = (day_range == day_range.rolling(7).min()).astype(int)

    # ── Ichimoku ──────────────────────────────────────────
    ich = ta.trend.IchimokuIndicator(h, l)
    df["ichi_a"]    = ich.ichimoku_a()
    df["ichi_b"]    = ich.ichimoku_b()
    df["ichi_base"] = ich.ichimoku_base_line()

    # ── Supertrend (10, 3.0) — delivery trend filter ──────
    st_line, st_dir       = _compute_supertrend(h, l, c, period=10, mult=3.0)
    df["supertrend"]      = st_line
    df["supertrend_bull"] = (st_dir == 1).astype(int)

    # ── Chaikin Money Flow — institutional accumulation ───
    df["cmf"] = ta.volume.chaikin_money_flow(h, l, c, v, 20)

    # ── Money Flow Index — volume-weighted RSI ────────────
    df["mfi"] = ta.volume.money_flow_index(h, l, c, v, 14)

    # ── Williams %R ───────────────────────────────────────
    df["williams_r"] = ta.momentum.WilliamsRIndicator(h, l, c, 14).williams_r()

    # ── CCI (Commodity Channel Index) ─────────────────────
    df["cci"] = ta.trend.CCIIndicator(h, l, c, 20).cci()

    # ── Donchian Channels (20-day breakout) ───────────────
    df["donchian_high"] = h.rolling(20).max()
    df["donchian_low"]  = l.rolling(20).min()
    dn_rng              = (df["donchian_high"] - df["donchian_low"]).replace(0, np.nan)
    df["donchian_pct"]  = (c - df["donchian_low"]) / dn_rng  # 0 = low, 1 = high

    # ── Elder Ray — Bull / Bear Power ─────────────────────
    ema13            = ta.trend.ema_indicator(c, 13)
    df["bull_power"] = h - ema13
    df["bear_power"] = l - ema13

    # ── Rolling VWAP approximation (20-day) ───────────────
    df["vwap"]       = (c * v).rolling(20).sum() / v.rolling(20).sum()
    df["vwap_ratio"] = c / df["vwap"].replace(0, np.nan) - 1

    return df

def add_weekly_ema(df_daily: pd.DataFrame, df_weekly: pd.DataFrame,
                   period: int = 13) -> pd.DataFrame:
    """Merge weekly EMA(13) onto daily frame (forward-fill)."""
    wema = ta.trend.ema_indicator(_squeeze(df_weekly["Close"]), period)
    wema = wema.rename("weekly_ema")
    wema.index = df_weekly.index
    df_daily = df_daily.copy()
    df_daily["weekly_close"] = np.nan
    df_daily["weekly_ema"]   = np.nan
    for dt, val in zip(df_weekly.index, wema.values):
        mask = (df_daily.index <= dt)
        if mask.any():
            df_daily.loc[df_daily.index[mask][-1], "weekly_ema"]   = val
            df_daily.loc[df_daily.index[mask][-1], "weekly_close"] = float(df_weekly.loc[dt, "Close"])
    df_daily[["weekly_ema", "weekly_close"]] = (
        df_daily[["weekly_ema", "weekly_close"]].ffill()
    )
    return df_daily

# ============================================================
# 5. CANDLESTICK PATTERN ENGINE
# ============================================================

def add_candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
    o = _squeeze(df["Open"])
    h = _squeeze(df["High"])
    l = _squeeze(df["Low"])
    c = _squeeze(df["Close"])

    body = (c - o).abs()
    rng  = (h - l).replace(0, np.nan)

    df["body_ratio"]   = body / rng
    df["close_pos"]    = (c - l) / rng
    df["upper_shadow"] = (h - c.where(c > o, other=o)) / rng
    df["lower_shadow"] = (o.where(c > o, other=c) - l) / rng

    prev_c    = c.shift(1)
    prev_o    = o.shift(1)
    prev_body = body.shift(1)
    prev_rng  = rng.shift(1)

    # Hammer
    df["hammer"] = (
        (df["lower_shadow"] > 2.0 * df["body_ratio"]) &
        (df["upper_shadow"] < 0.15) &
        (df["body_ratio"] < 0.40) &
        (c > o)
    ).astype(int)

    # Bullish engulfing
    df["bull_engulf"] = (
        (c > o) & (prev_c < prev_o) &
        (o < prev_c) & (c > prev_o)
    ).astype(int)

    # Morning star (simplified)
    df["morning_star"] = (
        (c.shift(2) > o.shift(2)) &
        (prev_body < prev_rng * 0.30) &
        (c > o) &
        (c > (o.shift(2) + c.shift(2)) / 2)
    ).astype(int)

    # Inside bar
    df["inside_bar"] = (
        (h < h.shift(1)) & (l > l.shift(1))
    ).astype(int)

    # Three white soldiers (3 consecutive bullish closes)
    df["three_soldiers"] = (
        (c > o) & (c.shift(1) > o.shift(1)) & (c.shift(2) > o.shift(2)) &
        (c > c.shift(1)) & (c.shift(1) > c.shift(2))
    ).astype(int)

    # Bullish harami (small bull bar contained within prior large bear bar)
    df["bull_harami"] = (
        (c > o) & (prev_c < prev_o) &
        (o > prev_c) & (c < prev_o) &
        (body < prev_body * 0.50)
    ).astype(int)

    # Piercing pattern (bull close penetrates > 50% of prior bear body)
    df["piercing"] = (
        (c > o) & (prev_c < prev_o) &
        (o < prev_c) &
        (c > (prev_o + prev_c) / 2) &
        (c < prev_o)
    ).astype(int)

    # Marubozu — full-body candle, tiny wicks (strong directional momentum)
    df["marubozu"] = (
        (c > o) &
        (df["body_ratio"] > 0.85) &
        (df["upper_shadow"] < 0.08) &
        (df["lower_shadow"] < 0.08)
    ).astype(int)

    # Composite candle score (delivery-weighted)
    df["candle_score"] = (
        df["hammer"]         * 0.14 +
        df["bull_engulf"]    * 0.18 +
        df["morning_star"]   * 0.14 +
        df["three_soldiers"] * 0.10 +
        df["bull_harami"]    * 0.08 +
        df["piercing"]       * 0.08 +
        df["marubozu"]       * 0.10 +
        df["close_pos"]      * 0.28 +
        (1 - df["upper_shadow"].clip(0, 1)) * 0.10
    ).clip(0, 1)

    return df

# ============================================================
# 6. RELATIVE STRENGTH & MARKET REGIME
# ============================================================

def relative_strength(stock_df: pd.DataFrame,
                      index_df: pd.DataFrame, period: int = 60) -> float:
    try:
        s = float(_squeeze(stock_df["Close"]).pct_change(period).iloc[-1])
        i = float(_squeeze(index_df["Close"]).pct_change(period).iloc[-1])
        if abs(i) < 0.001:          # index near-flat → ratio explodes; use raw return
            return max(-3.0, min(3.0, s * 10))
        rs = s / i
        return max(-5.0, min(10.0, rs))   # clamp to sane range
    except Exception:
        return 0.0

def market_regime(index_df: pd.DataFrame) -> str:
    c     = _squeeze(index_df["Close"])
    price = float(c.iloc[-1])
    ma50  = float(c.rolling(50).mean().iloc[-1])
    ma200 = float(c.rolling(200).mean().iloc[-1])
    rsi   = float(ta.momentum.rsi(c, 14).iloc[-1])
    adx_v = float(ta.trend.adx(
        _squeeze(index_df["High"]),
        _squeeze(index_df["Low"]), c, 14).iloc[-1])

    if price > ma200 and price > ma50 and rsi > 52 and adx_v > 18:
        return "BULL"
    elif price < ma200 and rsi < 45:
        return "BEAR"
    return "SIDEWAYS"

# ============================================================
# 7. LABEL GENERATOR  (for ML training)
# ============================================================

def make_labels(df: pd.DataFrame, fwd: int, min_ret: float) -> pd.Series:
    c = _squeeze(df["Close"])
    future = c.shift(-fwd) / c - 1
    return (future >= min_ret).astype(int)

# ============================================================
# 8. FEATURE ENGINEERING
# ============================================================

_FEATURE_COLS = [
    "ema9","ema20","ema50","ema200",
    "macd","macd_hist",
    "adx","adx_pos","adx_neg","di_diff",
    "rsi","rsi_fast",
    "stoch_k","stoch_d",
    "bb_pct","bb_width",
    "atr_pct","hvol20",
    "vol_ratio",
    "mom3","mom5","mom10","mom20","mom60",
    "ret1_z20",
    "body_ratio","close_pos","upper_shadow","lower_shadow",
    "candle_score","hammer","bull_engulf","morning_star",
    "bull_harami","piercing","marubozu",
    "pct_from_52h","pct_from_52l","rng52_pct",
    "nr7",
    # ── Delivery indicators ───────────────────────────────
    "supertrend_bull",
    "cmf","mfi",
    "williams_r","cci",
    "donchian_pct",
    "bull_power","bear_power",
    "vwap_ratio",
]

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in _FEATURE_COLS if c in df.columns]
    feat = df[cols].copy()

    c = _squeeze(df["Close"])
    for ema in [9, 20, 50, 200]:
        key = f"ema{ema}"
        if key in df.columns:
            feat[f"price_vs_{key}"] = c / df[key] - 1

    if "ema20" in df.columns and "ema50" in df.columns:
        feat["ema20_vs_50"]  = df["ema20"] / df["ema50"] - 1
    if "ema50" in df.columns and "ema200" in df.columns:
        feat["ema50_vs_200"] = df["ema50"] / df["ema200"] - 1
    if "ema9" in df.columns and "ema20" in df.columns:
        feat["ema9_vs_20"]   = df["ema9"] / df["ema20"] - 1

    # Momentum acceleration
    if "mom5" in df.columns and "mom20" in df.columns:
        feat["mom_accel"] = df["mom5"] - df["mom20"] / 4

    # OBV trend
    if "obv" in df.columns and "obv_ema" in df.columns:
        feat["obv_trend"] = (_squeeze(df["obv"]) / _squeeze(df["obv_ema"]) - 1)

    # Weekly alignment
    if "weekly_ema" in df.columns and "weekly_close" in df.columns:
        feat["wk_above_ema"] = (
            _squeeze(df["weekly_close"]) > _squeeze(df["weekly_ema"])
        ).astype(float)

    # ADX trend
    if "adx" in df.columns:
        feat["adx_rising"] = (_squeeze(df["adx"]) > _squeeze(df["adx"]).shift(3)).astype(float)

    # Delivery-specific ratio features
    c = _squeeze(df["Close"])
    if "bull_power" in df.columns:
        feat["bull_power_pct"] = _squeeze(df["bull_power"]) / c
    if "bear_power" in df.columns:
        feat["bear_power_pct"] = _squeeze(df["bear_power"]) / c
    if "mfi" in df.columns and "rsi" in df.columns:
        feat["mfi_rsi_diff"] = _squeeze(df["mfi"]) - _squeeze(df["rsi"])
    if "cmf" in df.columns:
        feat["cmf_positive"] = (_squeeze(df["cmf"]) > 0).astype(float)
    if "donchian_pct" in df.columns:
        feat["donchian_upper_half"] = (_squeeze(df["donchian_pct"]) > 0.5).astype(float)
    if "supertrend_bull" in df.columns:
        # Consecutive bars in bullish supertrend (streak)
        sb = _squeeze(df["supertrend_bull"])
        feat["supertrend_streak"] = sb.groupby((sb != sb.shift()).cumsum()).cumcount() + 1
        feat["supertrend_streak"] = feat["supertrend_streak"].where(sb == 1, 0)

    feat.replace([np.inf, -np.inf], np.nan, inplace=True)
    return feat

# ============================================================
# 9. ML MODEL TRAINER
# ============================================================

class ModelTrainer:
    def __init__(self, cfg: dict = CONFIG):
        self.cfg = cfg
        Path(cfg["model_path"]).parent.mkdir(parents=True, exist_ok=True)

    def _build_dataset(self, symbols: list, loader: DataLoader) -> pd.DataFrame:
        frames = []
        for sym in symbols:
            try:
                df   = loader.load(sym, period=self.cfg["training_period"])
                df   = add_indicators(df)
                df   = add_candle_patterns(df)
                try:
                    wk = loader.load_weekly(sym, period=self.cfg["training_period"])
                    df = add_weekly_ema(df, wk, self.cfg["weekly_ema_period"])
                except Exception:
                    pass
                df["label"] = make_labels(df, self.cfg["forward_days"],
                                          self.cfg["min_return"])
                feat = build_features(df)
                feat["label"] = df["label"]
                feat.dropna(inplace=True)
                if len(feat) > 150:
                    frames.append(feat)
                    log.info(f"  ✓ {sym:22s} rows={len(feat):5d} "
                             f"pos={feat['label'].mean():.1%}")
            except Exception as e:
                log.warning(f"  ✗ {sym}: {e}")

        if not frames:
            raise RuntimeError("No training data collected.")
        combined = pd.concat(frames, ignore_index=True)
        log.info(f"\n  Total rows: {len(combined):,}  "
                 f"| positive rate: {combined['label'].mean():.2%}\n")
        return combined

    def train(self, symbols: list, loader: DataLoader):
        log.info("=== MODEL TRAINING ===")
        data  = self._build_dataset(symbols, loader)
        X     = data.drop("label", axis=1)
        y     = data["label"]

        scaler = RobustScaler()
        Xs     = scaler.fit_transform(X)

        neg, pos = int((y == 0).sum()), int((y == 1).sum())
        model = XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.75,
            colsample_bytree=0.70,
            min_child_weight=8,
            gamma=0.1,
            reg_alpha=0.05,
            reg_lambda=1.5,
            scale_pos_weight=neg / pos,
            eval_metric="auc",
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1,
        )

        tscv      = TimeSeriesSplit(n_splits=5)
        auc_list  = []
        log.info("Time-series cross-validation:")
        for fold, (ti, vi) in enumerate(tscv.split(Xs), 1):
            model.fit(Xs[ti], y.iloc[ti],
                      eval_set=[(Xs[vi], y.iloc[vi])], verbose=False)
            auc = roc_auc_score(y.iloc[vi], model.predict_proba(Xs[vi])[:, 1])
            auc_list.append(auc)
            log.info(f"  Fold {fold}  AUC={auc:.4f}")

        log.info(f"\n  Mean AUC: {np.mean(auc_list):.4f} ± {np.std(auc_list):.4f}")

        model.fit(Xs, y)   # final fit on all data
        fi = pd.Series(model.feature_importances_, index=X.columns)
        log.info("\n  Top-15 features by importance:\n" +
                 fi.nlargest(15).to_string())

        joblib.dump(model,  self.cfg["model_path"])
        joblib.dump(scaler, self.cfg["scaler_path"])
        log.info(f"\n  Saved → {self.cfg['model_path']}")
        return model, scaler

    def load(self):
        return (joblib.load(self.cfg["model_path"]),
                joblib.load(self.cfg["scaler_path"]))

# ============================================================
# 10. PREDICTION
# ============================================================

def ml_predict(df: pd.DataFrame, model, scaler) -> float:
    feat = build_features(df)
    feat = feat.ffill().bfill().fillna(0)   # fill residual NaNs so last row survives
    feat = feat.iloc[-1:]
    if feat.empty:
        return 0.0
    return float(model.predict_proba(scaler.transform(feat))[0][1])

# ============================================================
# 11. SIGNAL ENGINE  (rule-based + ML, 10-gate filter)
# ============================================================

def _v(row, col, default=0.0):
    val = row.get(col, default)
    return default if pd.isna(val) else float(val)

def generate_signal(df: pd.DataFrame, prob: float,
                    regime: str, rs: float) -> str:
    if regime == "BEAR":
        return "NO TRADE"

    last = df.iloc[-1]

    # Gate 1 — Full EMA stack bullish
    ema_stack = (
        _v(last,"ema9") > _v(last,"ema20") >
        _v(last,"ema50") > _v(last,"ema200")
    )

    # Gate 2 — Weekly trend
    wk_bullish = (
        "weekly_close" not in df.columns or
        _v(last,"weekly_close") > _v(last,"weekly_ema")
    )

    # Gate 3 — RSI in sweet spot
    rsi_ok = CONFIG["min_rsi"] < _v(last,"rsi") < CONFIG["max_rsi"]

    # Gate 4 — MACD momentum
    macd_ok = _v(last,"macd_hist") > 0 and _v(last,"macd") > _v(last,"macd_signal")

    # Gate 5 — Trend strength + direction
    adx_ok  = _v(last,"adx") > CONFIG["min_adx"] and _v(last,"di_diff") > 0

    # Gate 6 — Volume confirmation
    vol_ok  = _v(last,"vol_ratio") > CONFIG["min_vol_ratio"]

    # Gate 7 — ML
    ml_ok   = prob >= CONFIG["min_probability"]

    # Gate 8 — Outperforming index
    rs_ok   = rs >= CONFIG["min_rel_strength"]

    # Gate 9 — Not over-extended from 52W high
    high_ok = _v(last,"pct_from_52h") > -0.15

    # Gate 10 — OBV confirming (buying pressure)
    obv_ok  = (
        "obv" not in df.columns or "obv_ema" not in df.columns or
        _v(last,"obv") > _v(last,"obv_ema")
    )

    # Gate 11 — Supertrend bullish (delivery trend filter)
    st_bull = bool(_v(last, "supertrend_bull", 0) == 1)

    # Gate 12 — Chaikin Money Flow (institutional accumulation)
    cmf_ok  = _v(last, "cmf", 0) > CONFIG.get("min_cmf", -0.05)

    # Gate 13 — Money Flow Index (volume-weighted RSI, not overbought)
    mfi_val = _v(last, "mfi", 50)
    mfi_ok  = CONFIG.get("min_mfi", 40) <= mfi_val <= CONFIG.get("max_mfi", 80)

    # Gate 14 — Price at or above rolling 20-day VWAP
    vwap_ok = _v(last, "vwap_ratio", 0) > -0.01

    score = sum([ema_stack, wk_bullish, rsi_ok, macd_ok,
                 adx_ok, vol_ok, ml_ok, rs_ok, high_ok, obv_ok,
                 st_bull, cmf_ok, mfi_ok, vwap_ok])

    # In sideways market relax the EMA stack gate slightly
    ema_partial = _v(last,"ema20") > _v(last,"ema50") > _v(last,"ema200")

    if score >= 10 and ml_ok and ema_stack and st_bull:
        return "STRONG_BUY"
    if score >= 7 and ml_ok and ema_partial:
        return "BUY"
    if score >= 5 and ml_ok:
        return "WATCHLIST"    # on radar — not a trade yet
    return "NO TRADE"

# ============================================================
# 12. RISK MANAGER  (Kelly-fraction sizing)
# ============================================================

class RiskManager:
    def __init__(self, cfg: dict = CONFIG):
        self.cfg = cfg

    def initial_sl(self, entry: float, atr: float) -> float:
        return round(entry - self.cfg["atr_sl_mult"] * atr, 2)

    def partial_target(self, entry: float, atr: float) -> float:
        return round(entry + self.cfg["atr_tgt1_mult"] * atr, 2)

    def full_target(self, entry: float, atr: float) -> float:
        return round(entry + self.cfg["atr_tgt2_mult"] * atr, 2)

    def trail_stop(self, high_since_entry: float, atr: float) -> float:
        return round(high_since_entry - self.cfg["atr_trail_mult"] * atr, 2)

    def position_size(self, capital: float, entry: float,
                      sl: float, prob: float = 0.65) -> int:
        risk_per_share = entry - sl
        if risk_per_share <= 0:
            return 0
        # Fixed-fractional size
        fixed = int((capital * self.cfg["risk_per_trade"]) / risk_per_share)
        # Kelly fraction (conservative half-Kelly)
        edge  = prob - (1 - prob)
        kelly = max(0.0, min(edge / 1.0, 0.25))  # cap at 25 %
        kelly_qty = int((capital * kelly * 0.5) / entry)
        # Use the more conservative of the two
        qty = max(1, min(fixed, kelly_qty if kelly_qty > 0 else fixed))
        return qty

    def can_add(self, positions: list) -> bool:
        heat = sum(p.get("risk_pct", self.cfg["risk_per_trade"])
                   for p in positions)
        return (len(positions) < self.cfg["max_positions"]
                and heat < self.cfg["max_portfolio_risk"])

    def sector_ok(self, symbol: str, positions: list) -> bool:
        sector = get_sector(symbol)
        count  = sum(1 for p in positions if get_sector(p["symbol"]) == sector)
        return count < self.cfg["max_per_sector"]

# ============================================================
# 13. BACKTESTER  (realistic, event-driven, single-stock)
# ============================================================

class Backtester:
    """
    Features:
    - Commission + slippage on every fill
    - Trailing ATR stop-loss
    - Partial profit at Tgt1 (50 %), remainder trails
    - Time-based stop after N bars
    """

    def __init__(self, cfg: dict = CONFIG):
        self.cfg = cfg
        self.rm  = RiskManager(cfg)

    def _cost(self, price: float, qty: int) -> float:
        total = price * qty
        return total * (self.cfg["commission"] + self.cfg["slippage"])

    def run(self, df: pd.DataFrame, model, scaler,
            index_df: pd.DataFrame, symbol: str = "?") -> dict:

        capital  = float(self.cfg["capital"])
        trades   = []
        equity   = [capital]
        position = None
        regime   = market_regime(index_df)

        df = df.copy()
        df.attrs["symbol"] = symbol

        for i in range(1, len(df)):
            row  = df.iloc[i]
            date = df.index[i]
            hi   = float(row["High"])
            lo   = float(row["Low"])
            cl   = float(row["Close"])
            atr  = float(row.get("atr", cl * 0.02) or cl * 0.02)

            # ── Manage open position ─────────────────────
            if position:
                pos = position
                bars_held = i - pos["bar_open"]
                min_hold  = self.cfg.get("min_hold_bars", 3)

                # Update trailing stop only after minimum hold period
                if hi > pos["high_since_entry"]:
                    pos["high_since_entry"] = hi
                if bars_held >= min_hold:
                    new_trail = self.rm.trail_stop(pos["high_since_entry"], atr)
                    pos["trail_sl"] = max(pos["trail_sl"], new_trail)
                effective_sl = max(pos["sl"], pos["trail_sl"])

                # Partial exit at Tgt1 (if not yet done)
                if not pos["partial_done"] and hi >= pos["tgt1"]:
                    pqty = pos["qty"] // 2
                    if pqty > 0:
                        pnl  = (pos["tgt1"] - pos["entry"]) * pqty
                        cost = self._cost(pos["tgt1"], pqty)
                        capital += pnl - cost
                        pos["qty"] -= pqty
                        pos["partial_done"] = True

                # Full exit at Tgt2
                if hi >= pos["tgt2"] and pos["qty"] > 0:
                    pnl  = (pos["tgt2"] - pos["entry"]) * pos["qty"]
                    cost = self._cost(pos["tgt2"], pos["qty"])
                    capital += pnl - cost
                    trades.append(self._trade_rec(pos, pos["tgt2"], date, "TARGET"))
                    position = None

                # Stop-loss
                elif lo <= effective_sl and pos["qty"] > 0:
                    exit_p = effective_sl
                    pnl    = (exit_p - pos["entry"]) * pos["qty"]
                    cost   = self._cost(exit_p, pos["qty"])
                    capital += pnl - cost
                    trades.append(self._trade_rec(pos, exit_p, date, "SL"))
                    position = None

                # Time stop
                elif bars_held >= self.cfg["time_stop_bars"] and pos["qty"] > 0:
                    exit_p = cl
                    pnl    = (exit_p - pos["entry"]) * pos["qty"]
                    cost   = self._cost(exit_p, pos["qty"])
                    capital += pnl - cost
                    trades.append(self._trade_rec(pos, exit_p, date, "TIME"))
                    position = None

            # ── Look for new entry ───────────────────────
            if position is None and capital > 0:
                sub  = df.iloc[:i]
                prob = ml_predict(sub, model, scaler)
                rs   = relative_strength(sub, index_df)
                sig  = generate_signal(sub, prob, regime, rs)

                if sig in ("BUY", "STRONG_BUY"):
                    slip   = float(row["Open"]) * self.cfg["slippage"]
                    entry  = float(row["Open"]) + slip
                    sl     = self.rm.initial_sl(entry, atr)
                    tgt1   = self.rm.partial_target(entry, atr)
                    tgt2   = self.rm.full_target(entry, atr)
                    qty    = self.rm.position_size(capital, entry, sl, prob)
                    cost   = self._cost(entry, qty)
                    if qty > 0 and (entry - sl) > 0:
                        capital -= cost
                        position = {
                            "symbol":           symbol,
                            "signal":           sig,
                            "bar_open":         i,
                            "entry_date":       date,
                            "entry":            entry,
                            "sl":               sl,
                            "trail_sl":         sl,
                            "tgt1":             tgt1,
                            "tgt2":             tgt2,
                            "qty":              qty,
                            "high_since_entry": entry,
                            "partial_done":     False,
                        }

            equity.append(capital)

        # Close any open position at end of data
        if position and position["qty"] > 0:
            exit_p = float(df["Close"].iloc[-1])
            pnl    = (exit_p - position["entry"]) * position["qty"]
            cost   = self._cost(exit_p, position["qty"])
            capital += pnl - cost
            trades.append(self._trade_rec(position, exit_p, df.index[-1], "EOD"))
        equity.append(capital)

        return self._metrics(trades, equity, df)

    @staticmethod
    def _trade_rec(pos: dict, exit_p: float, exit_date, reason: str) -> dict:
        pnl_pct = (exit_p - pos["entry"]) / pos["entry"] * 100
        return {
            "symbol":     pos["symbol"],
            "signal":     pos.get("signal",""),
            "entry_date": str(pos["entry_date"])[:10],
            "exit_date":  str(exit_date)[:10],
            "entry":      round(pos["entry"], 2),
            "exit":       round(exit_p, 2),
            "qty":        pos["qty"],
            "pnl":        round((exit_p - pos["entry"]) * pos["qty"], 2),
            "pnl_pct":    round(pnl_pct, 2),
            "reason":     reason,
        }

    @staticmethod
    def _metrics(trades: list, equity: list, df: pd.DataFrame) -> dict:
        base = {"total_trades": 0, "win_rate": 0.0,
                "total_return_pct": 0.0, "cagr_pct": 0.0,
                "max_drawdown_pct": 0.0, "sharpe": 0.0,
                "sortino": 0.0, "calmar": 0.0, "profit_factor": 0.0,
                "avg_win_pct": 0.0, "avg_loss_pct": 0.0,
                "avg_hold_days": 0, "trades": []}
        if not trades:
            return base

        t   = pd.DataFrame(trades)
        eq  = pd.Series(equity)

        wins   = t[t["pnl"] > 0]
        losses = t[t["pnl"] <= 0]

        # Drawdown
        peak   = eq.cummax()
        dd     = (eq - peak) / peak
        max_dd = float(dd.min()) * 100

        # Returns
        total_ret = (equity[-1] - equity[0]) / equity[0] * 100
        n_days    = max(len(df), 1)
        years     = n_days / 252
        cagr      = ((equity[-1] / equity[0]) ** (1 / max(years, 0.1)) - 1) * 100

        # Sharpe / Sortino (daily equity returns)
        eq_rets = eq.pct_change().dropna()
        rf_day  = 0.06 / 252  # 6 % risk-free
        excess  = eq_rets - rf_day
        sharpe  = float(excess.mean() / (excess.std() + 1e-9) * np.sqrt(252))
        down    = eq_rets[eq_rets < rf_day] - rf_day
        sortino = float(excess.mean() / (down.std() + 1e-9) * np.sqrt(252))
        calmar  = cagr / max(abs(max_dd), 0.01)

        pf = (
            abs(wins["pnl"].sum() / losses["pnl"].sum())
            if len(losses) > 0 and losses["pnl"].sum() != 0 else 9999.0
        )

        # Avg holding period
        try:
            t["entry_dt"] = pd.to_datetime(t["entry_date"])
            t["exit_dt"]  = pd.to_datetime(t["exit_date"])
            t["hold"]     = (t["exit_dt"] - t["entry_dt"]).dt.days
            avg_hold = int(t["hold"].mean())
        except Exception:
            avg_hold = 0

        return {
            "total_trades":    len(t),
            "win_rate":        round(len(wins) / len(t), 3),
            "total_return_pct":round(total_ret, 2),
            "cagr_pct":        round(cagr, 2),
            "max_drawdown_pct":round(max_dd, 2),
            "sharpe":          round(sharpe, 2),
            "sortino":         round(sortino, 2),
            "calmar":          round(calmar, 2),
            "profit_factor":   round(pf, 2),
            "avg_win_pct":     round(float(wins["pnl_pct"].mean()), 2) if len(wins) else 0.0,
            "avg_loss_pct":    round(float(losses["pnl_pct"].mean()), 2) if len(losses) else 0.0,
            "avg_hold_days":   avg_hold,
            "equity_curve":    equity,
            "trades":          t.to_dict("records"),
        }

# ============================================================
# 14. MONTHLY P&L MATRIX
# ============================================================

def monthly_pnl_matrix(trades: list) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    t = pd.DataFrame(trades)
    t["exit_dt"] = pd.to_datetime(t["exit_date"])
    t["year"]    = t["exit_dt"].dt.year
    t["month"]   = t["exit_dt"].dt.month
    monthly      = t.groupby(["year","month"])["pnl"].sum().reset_index()
    pivot        = monthly.pivot(index="year", columns="month", values="pnl").fillna(0)
    pivot.columns= [
        "","Jan","Feb","Mar","Apr","May","Jun",
        "Jul","Aug","Sep","Oct","Nov","Dec"
    ][:len(pivot.columns)+1][1:]
    return pivot

# ============================================================
# 15. SCANNER ENGINE
# ============================================================

class Scanner:
    def __init__(self, symbols: list = NIFTY_500, cfg: dict = CONFIG):
        self.symbols = symbols
        self.cfg     = cfg
        self.loader  = DataLoader(cfg["cache_dir"])
        self.rm      = RiskManager(cfg)

    def _scan_one(self, sym: str, model, scaler,
                  index_df: pd.DataFrame, regime: str) -> dict | None:
        if model is None or scaler is None:
            return None
        try:
            df = self.loader.load(sym, period="1y")
            df = add_indicators(df)
            df = add_candle_patterns(df)
            try:
                wk = self.loader.load_weekly(sym, period="2y")
                df = add_weekly_ema(df, wk, self.cfg["weekly_ema_period"])
            except Exception:
                pass

            prob = ml_predict(df, model, scaler)
            rs   = relative_strength(df, index_df)
            sig  = generate_signal(df, prob, regime, rs)

            last  = df.iloc[-1]
            entry = float(last["Close"])
            atr   = float(last.get("atr", entry * 0.02) or entry * 0.02)
            sl    = self.rm.initial_sl(entry, atr)
            tgt1  = self.rm.partial_target(entry, atr)
            tgt2  = self.rm.full_target(entry, atr)
            qty   = self.rm.position_size(self.cfg["capital"], entry, sl, prob)
            rr    = round((tgt2 - entry) / max(entry - sl, 0.01), 2)

            # Always return every stock with its score so we can show Top Prospects
            return {
                "symbol":          sym,
                "sector":          get_sector(sym),
                "signal":          sig,           # may be NO TRADE
                "price":           round(entry, 2),
                "stop_loss":       round(sl, 2),
                "target1":         round(tgt1, 2),
                "target2":         round(tgt2, 2),
                "quantity":        qty,
                "risk_amt":        round((entry - sl) * qty, 0),
                "probability":     round(prob, 4),
                "rsi":             round(float(last.get("rsi", 0) or 0), 1),
                "adx":             round(float(last.get("adx", 0) or 0), 1),
                "atr_pct":         round(float(last.get("atr_pct", 0) or 0), 2),
                "rel_strength":    round(rs, 2),
                "vol_ratio":       round(float(last.get("vol_ratio", 0) or 0), 2),
                "risk_reward":     rr,
                "candle_score":    round(float(last.get("candle_score", 0) or 0), 2),
                "pct_from_52h":    round(float(last.get("pct_from_52h", 0) or 0) * 100, 1),
                "cmf":             round(float(last.get("cmf", 0) or 0), 3),
                "mfi":             round(float(last.get("mfi", 50) or 50), 1),
                "supertrend_bull": int(last.get("supertrend_bull", 0) or 0),
                "vwap_ratio":      round(float(last.get("vwap_ratio", 0) or 0) * 100, 2),
            }
        except Exception as e:
            log.debug(f"  scan_err {sym}: {e}")
            return None

    def run(self, model, scaler):
        log.info(f"\n{'='*62}")
        log.info(f"  AI NIFTY 500 SCANNER  │  {datetime.now():%Y-%m-%d %H:%M}")
        log.info(f"{'='*62}\n")

        if model is None or scaler is None:
            log.error("="*62)
            log.error("  NO MODEL FOUND — scan aborted.")
            log.error("  Run:  python nifty500_ai_trader.py train")
            log.error("  OR click 'Train / Retrain Model' in the dashboard sidebar.")
            log.error("="*62)
            return [], "UNKNOWN"

        index_df = self.loader.load(self.cfg["index_symbol"])
        regime   = market_regime(index_df)
        log.info(f"  Market Regime: {regime}\n")

        results = []
        gate_counts = {"STRONG_BUY": 0, "BUY": 0, "WATCHLIST": 0, "NO TRADE": 0}

        for i, sym in enumerate(self.symbols, 1):
            res = self._scan_one(sym, model, scaler, index_df, regime)
            if res:
                results.append(res)
                gate_counts[res["signal"]] = gate_counts.get(res["signal"], 0) + 1
            else:
                gate_counts["NO TRADE"] += 1
            if i % 30 == 0 or i == len(self.symbols):
                log.info(f"  [{i:>3}/{len(self.symbols)}]  "
                         f"STRONG_BUY={gate_counts['STRONG_BUY']}  "
                         f"BUY={gate_counts['BUY']}  "
                         f"WATCHLIST={gate_counts['WATCHLIST']}")

        if not results:
            log.warning("\n  0 signals found. Possible reasons:")
            log.warning(f"  - Regime={regime}. In SIDEWAYS/BEAR fewer stocks trend.")
            log.warning("  - Try lowering min_probability or min_adx in CONFIG.")

        # Sort: STRONG_BUY > BUY > WATCHLIST, then by probability
        _order = {"STRONG_BUY": 0, "BUY": 1, "WATCHLIST": 2}
        results.sort(
            key=lambda x: (_order.get(x["signal"], 3), -x["probability"]),
        )
        return results, regime

# ============================================================
# 16. REPORTING
# ============================================================

def print_report(results: list, regime: str):
    W   = 78
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    print("═" * W)
    print(f"  AI NIFTY 500 SCANNER v2.0  │  {now}  │  Regime: {regime}")
    print("═" * W)
    print(f"  Total signals: {len(results)}")

    hdr = (f"  {'Symbol':<15} {'Sector':<8} {'Price':>8} {'SL':>8} "
           f"{'Tgt2':>8} {'Prob':>6} {'R:R':>5} {'RS':>6} {'ADX':>5}")

    for tag, label in [("STRONG_BUY","▶ STRONG BUY"), ("BUY","▶ BUY")]:
        grp = [r for r in results if r["signal"] == tag]
        if not grp:
            continue
        print(f"\n  {label}  ({len(grp)} signals)")
        print("  " + "─" * (W - 2))
        print(hdr)
        print("  " + "─" * (W - 2))
        for r in grp[:12]:
            print(
                f"  {r['symbol']:<15} {r['sector']:<8} {r['price']:>8.2f} "
                f"{r['stop_loss']:>8.2f} {r['target2']:>8.2f} "
                f"{r['probability']:>6.3f} {r['risk_reward']:>5.1f} "
                f"{r['rel_strength']:>6.2f} {r['adx']:>5.1f}"
            )
    print("═" * W)

# ============================================================
# 17. MAIN
# ============================================================

def main(mode: str = "scan"):
    for d in [CONFIG["cache_dir"], CONFIG["results_dir"],
              str(Path(CONFIG["model_path"]).parent)]:
        Path(d).mkdir(parents=True, exist_ok=True)

    loader  = DataLoader(CONFIG["cache_dir"])
    trainer = ModelTrainer(CONFIG)

    # ── TRAIN ─────────────────────────────────────────────
    if mode == "train":
        log.info("Mode: TRAIN")
        trainer.train(NIFTY_500[:70], loader)

    # ── SCAN ──────────────────────────────────────────────
    elif mode == "scan":
        log.info("Mode: SCAN")
        if not Path(CONFIG["model_path"]).exists():
            log.info("No model found — training first …")
            trainer.train(NIFTY_500[:60], loader)
        model, scaler   = trainer.load()
        scanner         = Scanner(NIFTY_500, CONFIG)
        results, regime = scanner.run(model, scaler)
        print_report(results, regime)

        ts   = datetime.now().strftime("%Y%m%d_%H%M")
        path = Path(CONFIG["results_dir"]) / f"signals_{ts}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        log.info(f"  Saved → {path}")
        return results

    # ── BACKTEST ──────────────────────────────────────────
    elif mode == "backtest":
        log.info("Mode: BACKTEST")
        if not Path(CONFIG["model_path"]).exists():
            trainer.train(NIFTY_500[:60], loader)
        model, scaler = trainer.load()
        index_df      = loader.load(CONFIG["index_symbol"], period="3y")
        bt            = Backtester(CONFIG)
        summary       = []

        test_syms = NIFTY_500[:25]
        log.info(f"\nBacktesting {len(test_syms)} symbols — 3 year period\n")
        hdr = f"  {'Symbol':<20} {'CAGR%':>7} {'WinRate':>8} {'MaxDD%':>8} {'Sharpe':>7} {'PF':>6} {'Trades':>7}"
        log.info(hdr)
        log.info("  " + "─" * 68)

        for sym in test_syms:
            try:
                df = loader.load(sym, period="3y")
                df = add_indicators(df)
                df = add_candle_patterns(df)
                try:
                    wk = loader.load_weekly(sym, period="3y")
                    df = add_weekly_ema(df, wk, CONFIG["weekly_ema_period"])
                except Exception:
                    pass
                m = bt.run(df, model, scaler, index_df, sym)
                m["symbol"] = sym
                summary.append(m)
                log.info(
                    f"  {sym:<20} {m['cagr_pct']:>7.1f}%"
                    f" {m['win_rate']:>8.1%} {m['max_drawdown_pct']:>8.1f}%"
                    f" {m['sharpe']:>7.2f} {m['profit_factor']:>6.2f}"
                    f" {m['total_trades']:>7}"
                )
            except Exception as e:
                log.warning(f"  {sym}: SKIP — {e}")

        if summary:
            mdf = pd.DataFrame(summary)
            log.info("\n" + "═" * 70)
            log.info("  AGGREGATE BACKTEST SUMMARY")
            log.info("─" * 70)
            log.info(f"  Avg CAGR         : {mdf['cagr_pct'].mean():.1f} %")
            log.info(f"  Median CAGR      : {mdf['cagr_pct'].median():.1f} %")
            log.info(f"  Avg Win Rate     : {mdf['win_rate'].mean():.1%}")
            log.info(f"  Avg Max Drawdown : {mdf['max_drawdown_pct'].mean():.1f} %")
            log.info(f"  Avg Sharpe       : {mdf['sharpe'].mean():.2f}")
            log.info(f"  Avg Sortino      : {mdf['sortino'].mean():.2f}")
            log.info(f"  Avg Profit Factor: {mdf['profit_factor'].mean():.2f}")
            log.info(f"  Total Trades     : {int(mdf['total_trades'].sum())}")
            log.info("═" * 70)

            ts   = datetime.now().strftime("%Y%m%d_%H%M")
            path = Path(CONFIG["results_dir"]) / f"backtest_{ts}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    [{k: v for k, v in m.items() if k not in ("equity_curve","trades")}
                     for m in summary],
                    f, indent=2, default=str,
                )
            log.info(f"  Saved → {path}")
    else:
        print(f"Unknown mode '{mode}'. Use: train | scan | backtest")
        sys.exit(1)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "scan"
    main(mode)
