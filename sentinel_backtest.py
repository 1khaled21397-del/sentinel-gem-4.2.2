"""
Sentinel-EGX v3.7 — Backtest Engine with EODHD Integration
Standalone walk-forward backtester + Streamlit-compatible module.

Usage (CLI):
    python sentinel_backtest.py

Usage (Streamlit import):
    from sentinel_backtest import BacktestEngine, BacktestConfig
    engine = BacktestConfig(synthetic_mode=False, eodhd_api_key="xxx")
    portfolio = engine.run()

Defaults (configurable in BacktestConfig):
    - Universe: EGX30 (30 tickers)
    - Capital: EGP 100,000
    - Risk per trade: 2%
    - Max positions: 10
    - Stop: ATR 2×
    - Target: VAMP forecast
    - Commission: 0.125% per side + 0.1% slippage
    - Benchmark: Equal-weight EGX30 buy-and-hold
    - Period: 2 years (synthetic or real EODHD data)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from pathlib import Path
import json
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
import warnings
import tempfile

# FIX: Scoped warning suppression for pandas/numpy deprecations.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Cloud-safe output directory
OUTPUT_DIR = Path(os.environ.get("SENTINEL_BACKTEST_DIR", tempfile.gettempdir())) / "sentinel_backtest_output"
try:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    OUTPUT_DIR = Path(tempfile.gettempdir()) / "sentinel_backtest_output"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# EODHD INTEGRATION (NEW v3.7)
# ═══════════════════════════════════════════════════════════════════════════════

def _fetch_eodhd_data(api_key: str, symbol: str, period: str = "d", days: int = 730) -> Optional[pd.DataFrame]:
    """Fetch historical data from EODHD. Returns DataFrame with DatetimeIndex."""
    try:
        from eodhd import APIClient
        api = APIClient(api_key)
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        raw = api.get_eod_historical_stock_market_data(symbol=symbol, period=period, from_date=start_date)
        df = pd.DataFrame(raw)
        if df.empty or len(df) < 20:
            return None
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(subset=["close"], inplace=True)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
        elif "Date" in df.columns:
            df["date"] = pd.to_datetime(df["Date"])
            df.set_index("date", inplace=True)
            df.drop(columns=["Date"], inplace=True, errors="ignore")
        return df
    except Exception as e:
        print(f"⚠️ EODHD fetch failed for {symbol}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BacktestConfig:
    capital: float = 100_000.0
    risk_per_trade_pct: float = 2.0
    max_positions: int = 10
    commission_pct: float = 0.125
    slippage_pct: float = 0.1
    stop_loss_atr_mult: float = 2.0
    take_profit_method: str = "vamp_target"
    horizon_days: int = 7
    min_history_days: int = 50
    benchmark: str = "egx30_buyhold"
    synthetic_mode: bool = True
    eodhd_api_key: Optional[str] = None
    data_start: str = "2024-01-01"
    data_end: str = "2026-01-01"
    universe: List[str] = field(default_factory=lambda: [
        "COMI.EGX", "TMGH.EGX", "FWRY.EGX", "HRHO.EGX", "PHDC.EGX",
        "ETEL.EGX", "EFIH.EGX", "ABUK.EGX", "AMOC.EGX", "ORAS.EGX",
        "CCAP.EGX", "RAYA.EGX", "MNHD.EGX", "EAST.EGX", "ORWE.EGX",
        "BTFH.EGX", "ADIB.EGX", "EMFD.EGX", "ISPH.EGX", "RMDA.EGX",
        "VLMR.EGX", "GBCO.EGX", "MCQE.EGX", "EGAL.EGX", "EGCH.EGX",
        "ESRS.EGX", "MOIL.EGX", "SUGR.EGX", "DOMT.EGX", "JUFO.EGX"
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Trade:
    ticker: str
    entry_date: datetime
    entry_price: float
    shares: int
    stop_price: float
    target_price: float
    risk_amount: float
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    holding_days: int = 0


@dataclass
class DailySnapshot:
    date: datetime
    equity: float
    cash: float
    open_positions: int
    benchmark_equity: float


# ═══════════════════════════════════════════════════════════════════════════════
# EGX CALENDAR
# ═══════════════════════════════════════════════════════════════════════════════

def is_egx_trading_day(date: datetime) -> bool:
    if date.weekday() in (4, 5):
        return False
    fixed = [(1, 1), (1, 7), (1, 25), (4, 25), (5, 1), (6, 30), (7, 23), (10, 6)]
    if (date.month, date.day) in fixed:
        return False
    return True


def generate_egx_calendar(start: str, end: str) -> List[datetime]:
    dates = pd.date_range(start, end, freq="D")
    return [d for d in dates if is_egx_trading_day(d)]


# ═══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC DATA GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def generate_synthetic_ohlc(ticker: str, calendar: List[datetime], seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed + hash(ticker) % 10000)
    n = len(calendar)
    base_price = np.random.uniform(5, 150)
    annual_vol = np.random.uniform(0.22, 0.38)
    daily_vol = annual_vol / np.sqrt(252)
    drift = np.random.uniform(0.0001, 0.0008)
    returns = np.zeros(n)
    vol = daily_vol
    for i in range(1, n):
        vol = 0.1 * daily_vol + 0.85 * vol + 0.05 * (returns[i-1]**2)
        vol = max(vol, daily_vol * 0.3)
        mom = 0.15 * returns[i-1] if i > 1 else 0
        returns[i] = np.random.normal(drift + mom, vol)
    prices = base_price * np.exp(np.cumsum(returns))
    daily_range = np.abs(np.random.normal(0, daily_vol * prices, n))
    high = prices + daily_range * np.random.uniform(0.3, 1.0, n)
    low = prices - daily_range * np.random.uniform(0.3, 1.0, n)
    open_p = prices + np.random.normal(0, daily_vol * prices * 0.5, n)
    base_vol = np.random.uniform(50_000, 2_000_000)
    volume = base_vol * (1 + 5 * (daily_range / prices)) * np.random.uniform(0.5, 2.0, n)
    df = pd.DataFrame({
        "open": np.round(open_p, 2), "high": np.round(high, 2),
        "low": np.round(low, 2), "close": np.round(prices, 2),
        "volume": np.round(volume, 0).astype(int),
    }, index=calendar)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# NATIVE TA INDICATORS (zero external dependency)
# ═══════════════════════════════════════════════════════════════════════════════

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(window=length).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/length, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1/length, min_periods=length).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift()).abs()
    tr3 = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(span=length, adjust=False).mean()


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line, macd_line - signal_line


def bbands(series: pd.Series, length: int = 20, std: float = 2.0):
    middle = sma(series, length)
    sigma = series.rolling(window=length).std()
    return middle + std * sigma, middle, middle - std * sigma


def vwap(df: pd.DataFrame) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3
    return (typical * df["volume"]).cumsum() / df["volume"].cumsum()


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    obv = pd.Series(index=close.index, dtype=float)
    obv.iloc[0] = volume.iloc[0]
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    return obv


def adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
    plus_dm = df["high"].diff()
    minus_dm = -df["low"].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift()).abs()
    tr3 = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_val = tr.ewm(span=length, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=length, adjust=False).mean() / atr_val)
    minus_di = 100 * (minus_dm.ewm(span=length, adjust=False).mean() / atr_val)
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di)) * 100
    return dx.ewm(span=length, adjust=False).mean()


# ═══════════════════════════════════════════════════════════════════════════════
# INDICATOR CALCULATOR
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["EMA20"] = ema(df["close"], 20)
    df["EMA50"] = ema(df["close"], 50)
    df["MACD"], df["MACD_signal"], df["MACD_hist"] = macd(df["close"])
    df["BB_upper"], df["BB_middle"], df["BB_lower"] = bbands(df["close"])
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / df["BB_middle"]
    df["VWAP"] = vwap(df)
    df["ADX"] = adx(df)
    df["RSI"] = rsi(df["close"])
    df["ATR"] = atr(df)
    df["Vol_EMA20"] = ema(df["volume"], 20)
    df["OBV"] = obv(df["close"], df["volume"])
    df["20D_high"] = df["high"].rolling(20).max()
    df["20D_low"] = df["low"].rolling(20).min()
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# S/R LEVELS
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_sr_levels(df: pd.DataFrame) -> Dict:
    if len(df) < 20:
        return {}
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    d20_high = float(df["high"].iloc[-20:].max())
    d20_low = float(df["low"].iloc[-20:].min())
    pp = (prev["high"] + prev["low"] + prev["close"]) / 3
    r1 = 2 * pp - prev["low"]
    s1 = 2 * pp - prev["high"]
    r2 = pp + (prev["high"] - prev["low"])
    s2 = pp - (prev["high"] - prev["low"])
    d20_range = d20_high - d20_low
    return {
        "d20_high": round(d20_high, 2), "d20_low": round(d20_low, 2),
        "pp": round(float(pp), 2), "r1": round(float(r1), 2), "r2": round(float(r2), 2),
        "s1": round(float(s1), 2), "s2": round(float(s2), 2),
        "fib_382": round(d20_low + 0.382 * d20_range, 2),
        "fib_618": round(d20_low + 0.618 * d20_range, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# WEEKLY CONFLUENCE
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_weekly_confluence(df_daily: pd.DataFrame) -> Dict:
    if len(df_daily) < 100:
        return {"aligned": False, "trend": "Unknown"}
    weekly = df_daily.resample("W-SAT").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()
    if len(weekly) < 20:
        return {"aligned": False, "trend": "Unknown"}
    weekly["W_EMA20"] = ema(weekly["close"], 20)
    w_macd, w_sig, w_hist = macd(weekly["close"])
    weekly["W_MACD_hist"] = w_hist
    latest = weekly.iloc[-1]
    ema20_bull = not pd.isna(latest.get("W_EMA20")) and latest["close"] > latest["W_EMA20"]
    macd_bull = not pd.isna(latest.get("W_MACD_hist")) and latest["W_MACD_hist"] > 0
    aligned = ema20_bull and macd_bull
    return {"aligned": aligned, "trend": "Bullish" if aligned else "Bearish"}


# ═══════════════════════════════════════════════════════════════════════════════
# SKILL DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_skills(df: pd.DataFrame, weekly_conf: Optional[Dict] = None) -> List[Dict]:
    if len(df) < 50:
        return []
    weekly_aligned = weekly_conf.get("aligned", False) if weekly_conf else False
    skills = []
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    if (not pd.isna(latest.get("BB_upper")) and latest["close"] > latest["BB_upper"] and
        not pd.isna(latest.get("MACD_hist")) and latest["MACD_hist"] > 0 and prev["MACD_hist"] <= 0 and
        not pd.isna(latest.get("Vol_EMA20")) and latest["volume"] > latest["Vol_EMA20"] * 1.5):
        skills.append({"id": "breakout", "name": "Breakout", "emoji": "🔥",
                       "confidence": min(95, int(70 + (latest["volume"] / max(latest["Vol_EMA20"], 1) - 1) * 15)),
                       "weekly_aligned": weekly_aligned})

    if (not pd.isna(latest.get("BB_lower")) and latest["close"] < latest["BB_lower"] and
        not pd.isna(latest.get("RSI")) and latest["RSI"] < 30):
        skills.append({"id": "mean_reversion", "name": "Mean Reversion", "emoji": "🔄",
                       "confidence": min(95, int(75 + (30 - latest["RSI"]) * 1.5)),
                       "weekly_aligned": weekly_aligned})

    if (not pd.isna(latest.get("VWAP")) and latest["close"] > latest["VWAP"] and
        not pd.isna(latest.get("ADX")) and latest["ADX"] > 25 and
        not pd.isna(latest.get("EMA20")) and not pd.isna(latest.get("EMA50")) and
        latest["EMA20"] > latest["EMA50"]):
        skills.append({"id": "trend_following", "name": "Trend Following", "emoji": "📈",
                       "confidence": min(95, int(60 + latest["ADX"] * 1.2)),
                       "weekly_aligned": weekly_aligned})

    if (not pd.isna(latest.get("Vol_EMA20")) and latest["Vol_EMA20"] > 0 and
        latest["volume"] > latest["Vol_EMA20"] * 2.0 and "OBV" in df.columns and len(df) > 1):
        if latest["OBV"] > prev["OBV"]:
            skills.append({"id": "volume_spike", "name": "Volume Spike", "emoji": "💥",
                           "confidence": min(95, int(65 + (latest["volume"] / latest["Vol_EMA20"] - 2) * 10)),
                           "weekly_aligned": weekly_aligned})

    if (not pd.isna(latest.get("20D_high")) and not pd.isna(latest.get("20D_low")) and
        not pd.isna(latest.get("BB_width"))):
        near_high = abs(latest["close"] - latest["20D_high"]) / latest["close"] < 0.03
        near_low = abs(latest["close"] - latest["20D_low"]) / latest["close"] < 0.03
        if (near_high or near_low) and latest["BB_width"] < 0.05:
            skills.append({"id": "support_resistance", "name": "S/R Compression", "emoji": "🛡️",
                           "confidence": min(90, int(70 + (0.05 - latest["BB_width"]) * 500)),
                           "weekly_aligned": weekly_aligned})
    return sorted(skills, key=lambda x: x["confidence"], reverse=True)


# ═══════════════════════════════════════════════════════════════════════════════
# VAMP PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════

def vamp_prediction(df: pd.DataFrame, days: int = 7) -> Optional[Dict]:
    # --- VAMP Prediction Constants ---
    W_WEEKLY = 0.15
    W_VOL = 0.125
    ATR_NORMALIZATION_BASE = 5.0
    W_EMA_BASE = 0.55
    W_EMA_VOL_ADJUST = 0.25
    W_TREND_BASE = 0.45
    DAYS_PER_WEEK = 5
    PCT_DIVISOR = 100
    MIN_SCORE_DENOM = 0.01
    # ---------------------------------
    if len(df) < 50:
        return None
    current = float(df["close"].iloc[-1])
    y = df["close"].values
    x = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    trend_forecast = float(model.predict([[len(y) + days]])[0])
    ema20 = float(df["EMA20"].iloc[-1]) if "EMA20" in df.columns and not df["EMA20"].isna().all() else current
    if pd.isna(ema20):
        ema20 = current

    volume_implied = current
    obv_signal = 0.0
    if "OBV" in df.columns and not df["OBV"].isna().all() and len(df) >= 14:
        lookback = 14
        obv_series = df["OBV"].iloc[-lookback:].values
        price_series = df["close"].iloc[-lookback:].values
        xv = np.arange(len(obv_series))
        obv_detrended = obv_series - np.polyval(np.polyfit(xv, obv_series, 1), xv)
        price_detrended = price_series - np.polyval(np.polyfit(xv, price_series, 1), xv)
        std_obv, std_price = np.std(obv_detrended), np.std(price_detrended)
        if std_obv > 0 and std_price > 0:
            obv_signal = float(np.corrcoef(obv_detrended, price_detrended)[0, 1])
            if not np.isnan(obv_signal):
                volume_implied = current * (1.0 + np.clip(obv_signal, -1, 1) * 0.20)

    w_weekly = W_WEEKLY
    w_vol = 0.125
    remaining = 1.0 - w_weekly - w_vol
    atr = float(df["ATR"].iloc[-1]) if "ATR" in df.columns and not df["ATR"].isna().all() else 0.0
    atr_pct = (atr / current * 100) if current else 0
    vol_ratio = min(atr_pct / ATR_NORMALIZATION_BASE, 1.0)
    w_ema = remaining * (W_EMA_BASE + W_EMA_VOL_ADJUST * vol_ratio)
    w_trend = remaining * (W_TREND_BASE - 0.25 * vol_ratio)

    weekly_trend = current
    if len(df) >= 100:
        weekly = df.resample("W-SAT")["close"].last().dropna()
        if len(weekly) >= 20:
            w_y = weekly.values
            w_x = np.arange(len(w_y)).reshape(-1, 1)
            w_model = LinearRegression().fit(w_x, w_y)
            weeks_fwd = max(1, round(days  / DAYS_PER_WEEK))
            weekly_trend = float(w_model.predict([[len(w_y) + weeks_fwd]])[0])

    target = (trend_forecast * w_trend) + (ema20 * w_ema) + (volume_implied * w_vol) + (weekly_trend * w_weekly)
    return {
        "current": round(current, 2), "target": round(target, 2),
        "growth": round((target - current) / current * 100, 2),
        "atr": round(atr, 2), "atr_pct": round(atr_pct, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO
# ═══════════════════════════════════════════════════════════════════════════════

class Portfolio:
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.cash = config.capital
        self.equity = config.capital
        self.positions: Dict[str, Trade] = {}
        self.trades: List[Trade] = []
        self.snapshots: List[DailySnapshot] = []
        self.benchmark_shares: Dict[str, float] = {}
        self.benchmark_cash = config.capital

    def update_equity(self, date: datetime, prices: Dict[str, float]):
        open_val = self.cash
        for ticker, trade in self.positions.items():
            if ticker in prices:
                open_val += trade.shares * prices[ticker]
        self.equity = open_val
        self.snapshots.append(DailySnapshot(
            date=date, equity=self.equity, cash=self.cash,
            open_positions=len(self.positions),
            benchmark_equity=self.benchmark_cash
        ))

    def can_open_new_position(self) -> bool:
        return len(self.positions) < self.config.max_positions

    def enter_position(self, ticker: str, date: datetime, price: float,
                       stop: float, target: float, risk_amount: float, shares: int) -> Optional[Trade]:
        exec_price = price * (1 + self.config.slippage_pct / 100)
        cost = shares * exec_price * (1 + self.config.commission_pct  / PCT_DIVISOR)
        if cost > self.cash:
            max_shares = int(self.cash / (exec_price * (1 + self.config.commission_pct / 100)))
            shares = min(shares, max_shares)
            cost = shares * exec_price * (1 + self.config.commission_pct  / PCT_DIVISOR)
            if shares <= 0:
                return None
        self.cash -= cost
        trade = Trade(ticker=ticker, entry_date=date, entry_price=exec_price,
                      shares=shares, stop_price=stop, target_price=target, risk_amount=risk_amount)
        self.positions[ticker] = trade
        return trade

    def exit_position(self, ticker: str, date: datetime, price: float, reason: str):
        if ticker not in self.positions:
            return
        trade = self.positions.pop(ticker)
        exec_price = price * (1 - self.config.slippage_pct / 100)
        gross = trade.shares * exec_price
        commission = gross * (self.config.commission_pct  / PCT_DIVISOR)
        net = gross - commission
        self.cash += net
        trade.exit_date = date
        trade.exit_price = exec_price
        trade.exit_reason = reason
        trade.pnl = net - (trade.shares * trade.entry_price)
        trade.pnl_pct = (trade.pnl / (trade.shares * trade.entry_price)) * 100
        trade.holding_days = (date - trade.entry_date).days
        self.trades.append(trade)


# ═══════════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class BacktestEngine:
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.data: Dict[str, pd.DataFrame] = {}
        self.calendar: List[datetime] = []
        self.portfolio = Portfolio(config)
        self.fetch_errors: List[str] = []

    def load_data(self):
        self.calendar = generate_egx_calendar(self.config.data_start, self.config.data_end)

        for ticker in self.config.universe:
            if self.config.synthetic_mode:
                df = generate_synthetic_ohlc(ticker, self.calendar, seed=hash(ticker) % 10000)
                self.data[ticker] = df
            else:
                # REAL EODHD DATA MODE
                if not self.config.eodhd_api_key:
                    raise ValueError("eodhd_api_key required when synthetic_mode=False")
                df = _fetch_eodhd_data(self.config.eodhd_api_key, ticker, period="d", days=730)
                if df is not None and len(df) >= self.config.min_history_days:
                    self.data[ticker] = df
                else:
                    self.fetch_errors.append(f"{ticker}: insufficient data or fetch failed")
                    # Fallback to synthetic for this ticker
                    df = generate_synthetic_ohlc(ticker, self.calendar, seed=hash(ticker) % 10000)
                    self.data[ticker] = df

        print(f"Loaded {len(self.data)} tickers, {len(self.calendar)} trading days.")
        if self.fetch_errors:
            print(f"⚠️ {len(self.fetch_errors)} tickers used synthetic fallback:")
            for err in self.fetch_errors[:5]:
                print(f"   {err}")

    def run(self):
        self.load_data()
        for ticker in self.data:
            self.data[ticker] = calculate_indicators(self.data[ticker])

        benchmark_allocation = self.config.capital / len(self.config.universe) if self.config.universe else 1
        for ticker in self.config.universe:
            if ticker in self.data:
                price = self.data[ticker]["close"].iloc[0]
                self.portfolio.benchmark_shares[ticker] = benchmark_allocation / price

        for i, date in enumerate(self.calendar):
            if i < self.config.min_history_days:
                continue
            today_prices = {t: df.loc[date, "close"] for t, df in self.data.items() if date in df.index}

            # Exit checks
            for ticker in list(self.portfolio.positions.keys()):
                trade = self.portfolio.positions[ticker]
                if ticker in today_prices:
                    price = today_prices[ticker]
                    row = self.data[ticker].loc[date] if date in self.data[ticker].index else None
                    low = row["low"] if row is not None else price
                    high = row["high"] if row is not None else price
                    if low <= trade.stop_price:
                        self.portfolio.exit_position(ticker, date, trade.stop_price, "Stop Loss")
                    elif high >= trade.target_price:
                        self.portfolio.exit_position(ticker, date, trade.target_price, "Target Hit")
                    elif (date - trade.entry_date).days >= self.config.horizon_days * 2:
                        self.portfolio.exit_position(ticker, date, price, "Time Exit")

            # Entry signals
            if self.portfolio.can_open_new_position():
                signals = []
                for ticker, df in self.data.items():
                    if ticker in self.portfolio.positions or date not in df.index:
                        continue
                    hist = df.loc[:date].copy()
                    if len(hist) < self.config.min_history_days:
                        continue
                    w_conf = calculate_weekly_confluence(hist)
                    if not w_conf.get("aligned", False):
                        continue
                    skills = detect_skills(hist, w_conf)
                    if not skills:
                        continue
                    vamp = vamp_prediction(hist, days=self.config.horizon_days)
                    if not vamp or vamp["growth"] <= 0:
                        continue

                    entry_price = hist["close"].iloc[-1]
                    stop_price = entry_price - (self.config.stop_loss_atr_mult * vamp["atr"])
                    if stop_price <= 0:
                        continue
                    target_price = vamp["target"]
                    risk_amount = self.portfolio.equity * (self.config.risk_per_trade_pct / 100)
                    risk_per_share = entry_price - stop_price
                    if risk_per_share <= 0:
                        continue
                    shares = int(risk_amount / risk_per_share)
                    if shares <= 0:
                        continue

                    best_skill = skills[0]
                    score = best_skill["confidence"] * vamp["growth"] / max(risk_per_share / entry_price * PCT_DIVISOR, MIN_SCORE_DENOM)
                    signals.append({"ticker": ticker, "score": score, "entry": entry_price,
                                    "stop": stop_price, "target": target_price,
                                    "risk_amount": risk_amount, "shares": shares})

                signals.sort(key=lambda x: x["score"], reverse=True)
                slots = self.config.max_positions - len(self.portfolio.positions)
                for sig in signals[:slots]:
                    self.portfolio.enter_position(sig["ticker"], date, sig["entry"], sig["stop"],
                                                  sig["target"], sig["risk_amount"], sig["shares"])

            benchmark_val = sum(self.portfolio.benchmark_shares.get(t, 0) * today_prices.get(t, 0)
                                for t in self.portfolio.benchmark_shares)
            self.portfolio.benchmark_cash = benchmark_val
            self.portfolio.update_equity(date, today_prices)

        return self.portfolio


# ═══════════════════════════════════════════════════════════════════════════════
# REPORT GENERATION (NEW v3.7)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_backtest_report(portfolio, config: BacktestConfig, output_dir: Path = OUTPUT_DIR):
    """Generate CSV reports and Plotly equity curve chart."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Trade log
    trade_df = pd.DataFrame([{
        "ticker": t.ticker, "entry_date": t.entry_date, "entry_price": t.entry_price,
        "exit_date": t.exit_date, "exit_price": t.exit_price, "exit_reason": t.exit_reason,
        "pnl": t.pnl, "pnl_pct": t.pnl_pct, "holding_days": t.holding_days,
        "shares": t.shares, "risk_amount": t.risk_amount
    } for t in portfolio.trades])
    trade_path = output_dir / "trade_log.csv"
    trade_df.to_csv(trade_path, index=False)

    # Equity curve
    eq_df = pd.DataFrame([{"date": s.date, "equity": s.equity, "benchmark": s.benchmark_equity,
                           "cash": s.cash, "open_positions": s.open_positions}
                          for s in portfolio.snapshots])
    eq_path = output_dir / "equity_curve.csv"
    eq_df.to_csv(eq_path, index=False)

    # Performance metrics
    total_return = (portfolio.equity - config.capital) / config.capital * PCT_DIVISOR if config.capital != 0 else 0
    benchmark_return = (portfolio.benchmark_cash - config.capital) / config.capital * PCT_DIVISOR if config.capital != 0 else 0

    winning_trades = [t for t in portfolio.trades if t.pnl and t.pnl > 0]
    losing_trades = [t for t in portfolio.trades if t.pnl and t.pnl <= 0]

    win_rate = len(winning_trades) / len(portfolio.trades) * 100 if portfolio.trades else 0
    avg_win = np.mean([t.pnl_pct for t in winning_trades]) if winning_trades else 0
    avg_loss = np.mean([t.pnl_pct for t in losing_trades]) if losing_trades else 0
    profit_factor = abs(sum(t.pnl for t in winning_trades) / sum(t.pnl for t in losing_trades)) if losing_trades and sum(t.pnl for t in losing_trades) != 0 else float('inf')
    max_drawdown = 0
    peak = config.capital
    for s in portfolio.snapshots:
        if s.equity > peak:
            peak = s.equity
        dd = (peak - s.equity) / peak * PCT_DIVISOR if peak != 0 else 0
        if dd > max_drawdown:
            max_drawdown = dd

    metrics = {
        "total_return_pct": round(total_return, 2),
        "benchmark_return_pct": round(benchmark_return, 2),
        "alpha": round(total_return - benchmark_return, 2),
        "total_trades": len(portfolio.trades),
        "win_rate_pct": round(win_rate, 1),
        "avg_win_pct": round(avg_win, 2),
        "avg_loss_pct": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2),
        "max_drawdown_pct": round(max_drawdown, 2),
        "final_equity": round(portfolio.equity, 2),
        "final_benchmark": round(portfolio.benchmark_cash, 2),
    }

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Plotly chart
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Equity Curve vs Benchmark", "Drawdown %"),
                        row_heights=[0.7, 0.3])

    fig.add_trace(go.Scatter(x=eq_df["date"], y=eq_df["equity"], mode="lines",
                               name="Strategy", line=dict(color="#10b981", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=eq_df["date"], y=eq_df["benchmark"], mode="lines",
                               name="Benchmark (EGX30 BH)", line=dict(color="#64748b", width=1.5, dash="dash")), row=1, col=1)

    # Drawdown calculation
    dd_series = []
    peak = config.capital
    for equity in eq_df["equity"]:
        if equity > peak:
            peak = equity
        dd_series.append((peak - equity) / peak * 100)

    fig.add_trace(go.Scatter(x=eq_df["date"], y=dd_series, mode="lines",
                               name="Drawdown", fill="tozeroy", line=dict(color="#ef4444", width=1),
                               fillcolor="rgba(239,68,68,0.2)"), row=2, col=1)

    fig.update_layout(template="plotly_dark", height=700,
                      title_text=f"Sentinel Backtest Results | Return: {total_return:.1f}% | Alpha: {metrics['alpha']:.1f}% | Max DD: {max_drawdown:.1f}%",
                      paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                      font=dict(color="#e2e8f0", family="Inter, sans-serif"))
    fig.update_xaxes(showgrid=True, gridcolor="#1e293b")
    fig.update_yaxes(showgrid=True, gridcolor="#1e293b")

    chart_path = output_dir / "equity_curve.html"
    fig.write_html(str(chart_path))

    print(f"\n📊 Reports saved to {output_dir}")
    print(f"   Trade log: {trade_path.name} ({len(trade_df)} trades)")
    print(f"   Equity curve: {eq_path.name}")
    print(f"   Metrics: {metrics_path.name}")
    print(f"   Chart: {chart_path.name}")

    return metrics, fig


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    # Parse CLI args
    api_key = os.environ.get("EODHD_API_KEY", "")
    use_real = "--real" in sys.argv or os.environ.get("SENTINEL_REAL_DATA", "") == "1"

    config = BacktestConfig(
        synthetic_mode=not use_real,
        eodhd_api_key=api_key if use_real else None,
    )

    print("=" * 60)
    print("  🛡️ Sentinel-EGX v3.7 Backtest Engine")
    print(f"  Mode: {'REAL EODHD DATA' if use_real else 'SYNTHETIC DATA'}")
    print(f"  Universe: {len(config.universe)} tickers")
    print(f"  Capital: EGP {config.capital:,.0f}")
    print(f"  Risk/Trade: {config.risk_per_trade_pct}%")
    print(f"  Max Positions: {config.max_positions}")
    print("=" * 60)

    engine = BacktestEngine(config)
    portfolio = engine.run()

    metrics, fig = generate_backtest_report(portfolio, config)

    print(f"\n📈 RESULTS")
    print(f"   Strategy Return: {metrics['total_return_pct']:.2f}%")
    print(f"   Benchmark Return: {metrics['benchmark_return_pct']:.2f}%")
    print(f"   Alpha: {metrics['alpha']:.2f}%")
    print(f"   Total Trades: {metrics['total_trades']}")
    print(f"   Win Rate: {metrics['win_rate_pct']:.1f}%")
    print(f"   Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"   Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
    print(f"   Final Equity: EGP {metrics['final_equity']:,.2f}")
