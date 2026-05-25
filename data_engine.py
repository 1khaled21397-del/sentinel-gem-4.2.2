"""
Sentinel-EGX v4.2.2 — Data Engine (Delta-Update Enhanced)
==========================================================
EODHD fetcher + 9 indicators + T+0 segment awareness + NEW DeltaCache integration.
FIXED v4.2.1 → v4.2.2:
  - Double exchange suffix bug (COMI.EGX.EGX → COMI.EGX)
  - Synthetic calendar: Sun-Thu (was incorrectly Mon-Fri)
  - Debug logging for URL, HTTP status, cache hits
  - DataCache.clear() for cache invalidation
  - Export API keys to os.environ for child modules
NEW v4.2.2:
  - DeltaCache integration: fetches only missing date ranges
  - Backward-compatible: falls back to legacy DataCache if DeltaCache unavailable
  - EGX trading-day aware gap detection
"""

import pandas as pd
import numpy as np
import requests
import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List
import os
import time

SCRIPT_DIR = Path(__file__).parent.resolve()
CACHE_DB = SCRIPT_DIR / "sentinel_cache.db"
CONFIG_FILE = SCRIPT_DIR / "sentinel_config.json"

with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

# API key resolution: Streamlit Secrets → .env → sentinel_config.json → os.environ
EODHD_API_KEY = ""
try:
    import streamlit as st
    EODHD_API_KEY = st.secrets.get("EODHD_API_KEY", "").strip()
except Exception:
    pass

if not EODHD_API_KEY:
    try:
        from dotenv import load_dotenv
        load_dotenv()
        EODHD_API_KEY = os.getenv("EODHD_API_KEY", "").strip()
    except Exception:
        pass

if not EODHD_API_KEY:
    EODHD_API_KEY = CONFIG.get("eodhd_api_key", "").strip()

# Export to environment for child modules
os.environ["EODHD_API_KEY"] = EODHD_API_KEY

CACHE_TTL_HOURS = CONFIG.get("rules", {}).get("cache_ttl_hours", CONFIG.get("cache_ttl_hours", 6))
EGX_HOLIDAYS = set(CONFIG.get("trading_calendar", {}).get("holidays_2024_2025", []))


def is_egx_trading_day(date: datetime) -> bool:
    """EGX trades Sunday(6) through Thursday(3). Friday(4) and Saturday(5) are weekend."""
    if date.weekday() in (4, 5):  # Friday, Saturday
        return False
    dstr = date.strftime("%Y-%m-%d")
    if dstr in EGX_HOLIDAYS:
        return False
    return True


def get_segment(ticker: str) -> str:
    for seg, tickers in CONFIG.get("market_segments", {}).items():
        if ticker in tickers:
            return seg
    return "moderate_activity"


class EGXCalendar:
    def __init__(self, start: str, end: str):
        self.dates = pd.date_range(start, end, freq="D")
        self.trading_days = [d for d in self.dates if is_egx_trading_day(d)]

    def trading_days_between(self, start: datetime, end: datetime) -> int:
        return sum(1 for d in self.trading_days if start <= d <= end)


# ── LEGACY CACHE (backward compatible) ──

class DataCache:
    """Legacy full-DataFrame blob cache (kept for backward compatibility)."""

    def __init__(self, db_path: str = str(CACHE_DB)):
        self.conn = sqlite3.connect(db_path)
        self._init_tables()

    def _init_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS eod_cache (
                symbol TEXT PRIMARY KEY,
                data TEXT,
                fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def get(self, symbol: str, max_age_hours: int = CACHE_TTL_HOURS) -> Optional[pd.DataFrame]:
        cursor = self.conn.execute(
            "SELECT data, fetched_at FROM eod_cache WHERE symbol = ? AND " +
            "datetime(fetched_at) > datetime('now', '-" + str(max_age_hours) + " hours')",
            (symbol,)
        )
        row = cursor.fetchone()
        if row:
            from io import StringIO
            return pd.read_json(StringIO(row[0]), orient="split")
        return None

    def set(self, symbol: str, df: pd.DataFrame):
        json_str = df.to_json(orient="split")
        self.conn.execute(
            "INSERT OR REPLACE INTO eod_cache (symbol, data) VALUES (?, ?)",
            (symbol, json_str)
        )
        self.conn.commit()

    def clear(self):
        """Clear all cached EOD data."""
        self.conn.execute("DELETE FROM eod_cache")
        self.conn.commit()


# ── DELTA CACHE INTEGRATION ──

try:
    from delta_cache import DeltaCache
    DELTA_CACHE_AVAILABLE = True
except ImportError:
    DELTA_CACHE_AVAILABLE = False


def _build_url(symbol: str, exchange: str, period: str, from_date: str, to_date: str) -> str:
    """Build EODHD API URL. FIX v4.2.1: strip duplicate exchange suffix if present in symbol."""
    base = "https://eodhd.com/api/eod"
    suffix = f".{exchange}"
    if symbol.upper().endswith(suffix.upper()):
        symbol_clean = symbol[:-len(suffix)]
    else:
        symbol_clean = symbol

    url = f"{base}/{symbol_clean}.{exchange}?from={from_date}&to={to_date}&period={period}&api_token={EODHD_API_KEY}&fmt=json"
    return url


def _parse_eodhd_response(data: list, symbol: str) -> Optional[pd.DataFrame]:
    """Parse raw EODHD JSON into clean DataFrame."""
    if not data:
        return None
    df = pd.DataFrame(data)
    if df.empty:
        return None

    col_map = {}
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ["date", "datetime", "timestamp"]:
            col_map[col] = "date"
        elif col_lower in ["open", "o"]:
            col_map[col] = "open"
        elif col_lower in ["high", "h"]:
            col_map[col] = "high"
        elif col_lower in ["low", "l"]:
            col_map[col] = "low"
        elif col_lower in ["close", "c", "adjusted_close", "adj_close"]:
            col_map[col] = "close"
        elif col_lower in ["volume", "vol", "v"]:
            col_map[col] = "volume"

    df = df.rename(columns=col_map)
    required = ["date", "open", "high", "low", "close", "volume"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        return None

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["close"])
    return df


def _fetch_eodhd_range(symbol: str, exchange: str, period: str,
                       from_date: str, to_date: str) -> Optional[pd.DataFrame]:
    """Fetch a specific date range from EODHD (used by delta fetcher)."""
    if not EODHD_API_KEY or EODHD_API_KEY == "YOUR_EODHD_API_KEY":
        return None

    url = _build_url(symbol, exchange, period, from_date, to_date)
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 403:
                print(f"[DataEngine] ⚠️ EODHD quota exceeded (403) for {symbol}")
                return None
            resp.raise_for_status()
            data = resp.json()
            df = _parse_eodhd_response(data, symbol)
            if df is not None and len(df) >= 1:
                return df
            time.sleep(1)
        except Exception as e:
            time.sleep(1)
            continue
    return None


def fetch_eod(symbol: str, exchange: str = "EGX", period: str = "d", days: int = 500,
              use_synthetic_fallback: bool = True, use_delta_cache: bool = True) -> Optional[pd.DataFrame]:
    """
    Fetch EOD data with delta-update caching.

    Strategy:
      1. If DeltaCache available and enabled: check gaps, fetch only missing ranges, merge
      2. If DeltaCache unavailable/disabled: fall back to legacy DataCache (full blob)
      3. If all cache misses: full EODHD fetch
      4. Synthetic fallback if EODHD fails
    """
    end = datetime.now()
    start = end - timedelta(days=days + 30)
    req_start = start.strftime("%Y-%m-%d")
    req_end = end.strftime("%Y-%m-%d")

    # ── PATH A: DeltaCache (intelligent) ──
    if use_delta_cache and DELTA_CACHE_AVAILABLE:
        dc = DeltaCache()
        meta = dc.get_meta(symbol)

        # Check if we have ANY data and it's fresh enough
        if meta and not dc.needs_refresh(symbol, req_end, ttl_hours=CACHE_TTL_HOURS):
            # Fully cached and fresh — just return it
            full = dc.get_data(symbol, req_start, req_end)
            if full is not None and len(full) >= 20:
                print(f"[DataEngine] DeltaCache hit: {len(full)} bars for {symbol} (fully cached)")
                return full

        # Partial or stale — detect gaps
        # Build trading day list for gap detection
        cal = EGXCalendar(req_start, req_end)
        trading_day_strs = [d.strftime("%Y-%m-%d") for d in cal.trading_days]
        missing_ranges = dc.get_missing_ranges(symbol, req_start, req_end, trading_day_strs)

        if not missing_ranges:
            # All trading days cached — just return
            full = dc.get_data(symbol, req_start, req_end)
            if full is not None and len(full) >= 20:
                dc._update_meta(symbol, full)
                print(f"[DataEngine] DeltaCache: all {len(full)} bars present for {symbol}")
                return full

        print(f"[DataEngine] DeltaCache gaps for {symbol}: {len(missing_ranges)} range(s) to fetch")

        # Fetch each missing range
        fetched_frames = []
        for from_d, to_d in missing_ranges:
            print(f"[DataEngine] Fetching delta: {symbol} [{from_d} → {to_d}]")
            df_chunk = _fetch_eodhd_range(symbol, exchange, period, from_d, to_d)
            if df_chunk is not None and not df_chunk.empty:
                fetched_frames.append(df_chunk)
                dc.insert_data(symbol, df_chunk, source="eodhd")
            else:
                print(f"[DataEngine] ⚠️ Failed to fetch delta range [{from_d} → {to_d}]")

        # Merge all cached + new data
        full = dc.get_data(symbol, req_start, req_end)
        if full is not None and len(full) >= 20:
            print(f"[DataEngine] DeltaCache merged: {len(full)} bars for {symbol}")
            return full

        # If delta fetch failed, fall through to full fetch
        print(f"[DataEngine] Delta fetch incomplete, falling back to full fetch for {symbol}")

    # ── PATH B: Legacy DataCache (full blob) ──
    legacy = DataCache()
    cached = legacy.get(symbol)
    if cached is not None:
        print(f"[DataEngine] Legacy cache hit for {symbol}")
        return cached

    # ── PATH C: Full EODHD fetch ──
    if not EODHD_API_KEY or EODHD_API_KEY == "YOUR_EODHD_API_KEY":
        if use_synthetic_fallback:
            print(f"[DataEngine] No API key, using synthetic data for {symbol}")
            return _generate_synthetic_data(symbol, days)
        raise ValueError("EODHD_API_KEY not found. Set it in: 1) Streamlit Secrets, 2) .env file, or 3) sentinel_config.json")

    df = _fetch_eodhd_range(symbol, exchange, period, req_start, req_end)
    if df is not None and len(df) >= 20:
        legacy.set(symbol, df)
        if DELTA_CACHE_AVAILABLE:
            dc = DeltaCache()
            dc.insert_data(symbol, df, source="eodhd")
        print(f"[DataEngine] Full fetch: {len(df)} bars for {symbol}")
        return df

    # ── PATH D: Synthetic fallback ──
    print(f"[DataEngine] EODHD failed for {symbol}, using synthetic fallback")
    if use_synthetic_fallback:
        return _generate_synthetic_data(symbol, days)
    return None


def _generate_synthetic_data(symbol: str, days: int = 500) -> pd.DataFrame:
    """Generate realistic synthetic OHLCV data for testing when EODHD is unavailable.
    EGX trades Sun-Thu.
    """
    import numpy as np
    np.random.seed(hash(symbol) % 2**32)

    end = datetime.now()
    dates = []
    current = end - timedelta(days=int(days * 1.5))
    while current <= end:
        if current.weekday() in (0, 1, 2, 3, 6):
            dates.append(current)
        current += timedelta(days=1)

    dates = dates[-days:] if len(dates) > days else dates
    n = len(dates)

    base_price = np.random.uniform(5, 200)
    annual_vol = np.random.uniform(0.20, 0.45)
    daily_vol = annual_vol / np.sqrt(252)
    drift = np.random.uniform(-0.0002, 0.0005)

    returns = np.random.normal(drift, daily_vol, n)
    prices = base_price * np.exp(np.cumsum(returns))

    daily_range = prices * daily_vol * np.random.uniform(0.5, 2.0, n)
    high = prices + daily_range * np.random.uniform(0.3, 0.7, n)
    low = prices - daily_range * np.random.uniform(0.3, 0.7, n)
    open_p = prices + np.random.normal(0, daily_vol * prices * 0.3, n)

    high = np.maximum(high, np.maximum(open_p, prices))
    low = np.minimum(low, np.minimum(open_p, prices))

    base_vol = np.random.uniform(100_000, 5_000_000)
    volume = base_vol * (1 + 3 * (daily_range / prices)) * np.random.uniform(0.3, 3.0, n)

    df = pd.DataFrame({
        "date": dates,
        "open": np.round(open_p, 4),
        "high": np.round(high, 4),
        "low": np.round(low, 4),
        "close": np.round(prices, 4),
        "volume": np.round(volume, 0).astype(int)
    })

    df.attrs["synthetic"] = True
    df.attrs["ticker"] = symbol
    return df


def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    df["ema_20"] = close.ewm(span=20, adjust=False).mean()
    df["ema_50"] = close.ewm(span=50, adjust=False).mean()
    df["sma_200"] = close.rolling(window=200).mean()

    delta = close.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    rsi = df["rsi_14"]
    min_rsi = rsi.rolling(14).min()
    max_rsi = rsi.rolling(14).max()
    stoch_rsi = (rsi - min_rsi) / (max_rsi - min_rsi).replace(0, np.nan)
    df["stoch_rsi_k"] = stoch_rsi.rolling(3).mean() * 100
    df["stoch_rsi_d"] = df["stoch_rsi_k"].rolling(3).mean()

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd_line"] = ema12 - ema26
    df["macd_signal"] = df["macd_line"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd_line"] - df["macd_signal"]

    def _macd_state(row):
        if pd.isna(row["macd_line"]) or pd.isna(row["macd_signal"]):
            return "neutral"
        if row["macd_line"] > row["macd_signal"] and row["macd_hist"] > 0:
            return "bullish"
        elif row["macd_line"] < row["macd_signal"] and row["macd_hist"] < 0:
            return "bearish"
        elif row["macd_line"] > row["macd_signal"] and row["macd_hist"] < 0:
            return "bullish_cross"
        elif row["macd_line"] < row["macd_signal"] and row["macd_hist"] > 0:
            return "bearish_cross"
        return "neutral"
    df["macd_state"] = df.apply(_macd_state, axis=1)

    obv = [0]
    for i in range(1, len(df)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.append(obv[-1] + volume.iloc[i])
        elif close.iloc[i] < close.iloc[i-1]:
            obv.append(obv[-1] - volume.iloc[i])
        else:
            obv.append(obv[-1])
    df["obv"] = obv
    df["obv_slope_5d"] = df["obv"] - df["obv"].shift(5)

    mfm = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    mfv = mfm * volume
    df["cmf_20"] = mfv.rolling(20).sum() / volume.rolling(20).sum()

    typical = (high + low + close) / 3
    df["vwap_20d"] = (typical * volume).rolling(20).sum() / volume.rolling(20).sum()

    avwap_vals = []
    for i in range(len(df)):
        if i < 60:
            avwap_vals.append(np.nan)
            continue
        window = df.iloc[i-60:i+1]
        swing_idx = (window["low"] * window["volume"]).idxmin()
        anchor_slice = window.loc[swing_idx:]
        t = (anchor_slice["high"] + anchor_slice["low"] + anchor_slice["close"]) / 3
        avwap = (t * anchor_slice["volume"]).sum() / anchor_slice["volume"].sum()
        avwap_vals.append(avwap)
    df["anchored_vwap"] = avwap_vals

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()

    prev = df.shift(1)
    pp = (prev["high"] + prev["low"] + prev["close"]) / 3
    df["pivot"] = pp
    df["r1"] = 2 * pp - prev["low"]
    df["r2"] = pp + (prev["high"] - prev["low"])
    df["r3"] = df["r1"] + (prev["high"] - prev["low"])
    df["s1"] = 2 * pp - prev["high"]
    df["s2"] = pp - (prev["high"] - prev["low"])
    df["s3"] = df["s1"] - (prev["high"] - prev["low"])

    d20_high = df["high"].rolling(20).max()
    d20_low = df["low"].rolling(20).min()
    d20_range = d20_high - d20_low
    df["fib_382"] = d20_low + 0.382 * d20_range
    df["fib_500"] = d20_low + 0.500 * d20_range
    df["fib_618"] = d20_low + 0.618 * d20_range

    df["volume_vs_avg20"] = volume / volume.rolling(20).mean()

    ema50 = df["ema_50"]
    sma200 = df["sma_200"]
    obv_s = df["obv"]
    cmf = df["cmf_20"]
    rsi = df["rsi_14"]
    ema20 = df["ema_20"]

    df["gemini_trend_score"] = (ema50 > sma200).astype(float).fillna(0)
    df["gemini_volume_score"] = ((obv_s > obv_s.shift(5)) & (cmf > 0)).astype(float).fillna(0)
    df["gemini_timing_score"] = (((rsi >= 40) & (rsi <= 50)) & (np.abs(close - ema20) / ema20 < 0.02)).astype(float).fillna(0)
    df["gemini_composite"] = (df["gemini_trend_score"] + df["gemini_volume_score"] + df["gemini_timing_score"]) / 3

    weekly_ema20 = close.ewm(span=20, adjust=False).mean()
    weekly_ema50 = close.ewm(span=50, adjust=False).mean()
    weekly_sma200 = close.rolling(200).mean()
    df["daily_trend"] = np.where((weekly_ema20 > weekly_ema50) & (weekly_ema50 > weekly_sma200), "bullish",
                          np.where((weekly_ema20 < weekly_ema50) & (weekly_ema50 < weekly_sma200), "bearish", "neutral"))
    df["weekly_trend"] = np.where(weekly_ema50 > weekly_sma200, "bullish",
                          np.where(weekly_ema50 < weekly_sma200, "bearish", "neutral"))
    df["monthly_trend"] = df["weekly_trend"]

    def score_row(row):
        s = 0
        for t in [row["daily_trend"], row["weekly_trend"], row["monthly_trend"]]:
            if t == "bullish": s += 1
            elif t == "bearish": s -= 1
        return s / 3
    df["confluence_score"] = df.apply(score_row, axis=1)

    segment = get_segment(df.attrs.get("ticker", ""))
    t0_eligible = segment in CONFIG.get("t0_rules", {}).get("t0_enabled_segments", [])
    vol_avg = volume.rolling(20).mean()
    df["liquidity_score"] = np.where(t0_eligible, 0.5, 0.0) + np.minimum(volume / vol_avg * 0.5, 0.5)
    df["t0_eligible"] = t0_eligible
    df["market_segment"] = segment

    return df


def fetch_and_build(symbol: str, exchange: str = "EGX", lookback: int = 400,
                    use_cache: bool = True, allow_synthetic: bool = True,
                    use_delta_cache: bool = True) -> pd.DataFrame:
    """
    Fetch and build full indicator-enriched dataframe.
    NEW v4.2.2: use_delta_cache=True triggers intelligent gap fetching.
    """
    df = fetch_eod(symbol, exchange, "d", lookback,
                   use_synthetic_fallback=allow_synthetic,
                   use_delta_cache=use_delta_cache)

    if df is None:
        raise ValueError(
            f"Unable to fetch data for {symbol}. "
            f"EODHD API may be down or ticker invalid. "
            f"Try: 1) Check API key, 2) Verify ticker exists, 3) Enable synthetic fallback"
        )

    if len(df) < 50:
        if len(df) < 20:
            raise ValueError(
                f"Insufficient data for {symbol}: only {len(df)} bars. "
                f"Minimum required: 50 bars (≈ 2.5 months). "
                f"This ticker may be newly listed or delisted."
            )
        print(f"[DataEngine] Warning: Only {len(df)} bars for {symbol}. SMA200 will be NaN.")

    df.attrs["ticker"] = symbol
    df = _compute_indicators(df)
    return df


def get_cache_stats() -> Dict:
    """Return stats for both legacy and delta caches."""
    stats = {"legacy": {}, "delta": {}}
    try:
        legacy = DataCache()
        conn = legacy.conn
        cursor = conn.execute("SELECT COUNT(*), MAX(fetched_at) FROM eod_cache")
        row = cursor.fetchone()
        stats["legacy"] = {"symbols": row[0], "latest_fetch": row[1]}
    except Exception as e:
        stats["legacy"] = {"error": str(e)}

    if DELTA_CACHE_AVAILABLE:
        try:
            dc = DeltaCache()
            stats["delta"] = dc.get_stats()
            stats["delta"]["symbols_list"] = dc.list_symbols()
        except Exception as e:
            stats["delta"] = {"error": str(e)}
    else:
        stats["delta"] = {"status": "unavailable"}

    return stats


def clear_all_caches():
    """Clear both legacy and delta caches."""
    try:
        legacy = DataCache()
        legacy.clear()
    except Exception as e:
        print(f"[DataEngine] Legacy cache clear error: {e}")

    if DELTA_CACHE_AVAILABLE:
        try:
            dc = DeltaCache()
            dc.clear_all()
        except Exception as e:
            print(f"[DataEngine] Delta cache clear error: {e}")

    print("[DataEngine] All caches cleared.")


if __name__ == "__main__":
    print("DataEngine v4.2.2 ready: EODHD fetch + DeltaCache + 9 indicators + T+0 awareness")
    print(f"DeltaCache available: {DELTA_CACHE_AVAILABLE}")
    print(f"EODHD API key configured: {'Yes' if EODHD_API_KEY and EODHD_API_KEY != 'YOUR_EODHD_API_KEY' else 'No'}")
