"""
Sentinel-EGX v4.2 — Technical Analysis Engine
================================================
All v3.7 indicators + NEW: VWAP, Anchored VWAP, CMF, OBV, RSI, 
Stochastic RSI, MACD, EMA 20/50, SMA 200 + EGX T+0/T+1 Liquidity Filter
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import sqlite3
from datetime import datetime, timedelta
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return str(obj)
        return super().default(obj)


with open("sentinel_config.json", "r") as f:
    CONFIG = json.load(f)

TA_CONFIG = CONFIG.get("technical_analysis", {})

# ── INDICATOR ENGINE CONSTANTS (module-level so methods can reference bare names) ──
RSI_MAX                = 100
EMA_PROXIMITY_PCT      = 0.02   # 2% proximity band around EMA20
GEMINI_COMPONENT_COUNT = 3
T0_LIQUIDITY_BASE      = 0.5
VOLUME_LIQUIDITY_CAP   = 0.5
SLOPE_LOOKBACK         = 5
# ────────────────────────────────────────────────────────────────────────────────

# EGX T+0 / T+1 Market Segments (per EGX Feb 2024 restructure)
T0_SEGMENTS = {
    "high_activity": "EGX100",           # T+0 enabled
    "moderate_activity": "Moderate",      # T+0 enabled  
    "tamayuz": "Tamayuz",                 # T+0 enabled
    "low_activity": "Low",                # T+1 ONLY
    "nile": "Nile",                       # T+1 ONLY
}

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class IndicatorSnapshot:
    ticker: str
    timestamp: str

    # Trend
    ema_20: float
    ema_50: float
    sma_200: float
    ema_20_slope: float
    ema_50_slope: float
    trend_direction: str  # "bullish", "bearish", "neutral"

    # Momentum
    rsi_14: float
    stoch_rsi_k: float
    stoch_rsi_d: float
    macd_line: float
    macd_signal: float
    macd_histogram: float
    macd_state: str  # "bullish_cross", "bearish_cross", "bullish", "bearish", "neutral"

    # Volume
    obv: float
    obv_slope: float
    cmf_20: float
    vwap_20d: float
    anchored_vwap: float
    anchor_date: str
    volume_vs_avg: float  # today's volume / 20-day avg

    # Gemini Flash Framework Scores
    trend_score: float      # EMA50 > SMA200 ? 1 : 0
    volume_score: float     # OBV rising & CMF > 0 ? 1 : 0
    timing_score: float    # RSI 40-50 & price near EMA20 ? 1 : 0
    gemini_framework_score: float  # composite 0-1

    # T+0 Liquidity Filter
    t0_eligible: bool
    market_segment: str
    liquidity_score: float  # 0-1 based on volume and T+0 status

    # S/R (ported from v3.7)
    pivot: float
    r1: float; r2: float; r3: float
    s1: float; s2: float; s3: float
    fib_382: float; fib_500: float; fib_618: float

    # ATR Stop Loss (ported from v3.7)
    atr_14: float
    stop_loss: float
    take_profit: float
    risk_reward: float

    # Weekly Confluence (ported from v3.7)
    daily_trend: str
    weekly_trend: str
    monthly_trend: str
    confluence_score: float


@dataclass
class SetupQuality:
    ticker: str
    setup_type: str
    quality_score: float  # 0-100
    entry_zone: Tuple[float, float]
    stop_loss: float
    targets: List[float]
    confidence: str
    rationale: List[str]
    t0_recommended: bool
    best_session: str  # "first_hour", "last_hour", "mid_session"


# ============================================================================
# INDICATOR CALCULATIONS
# ============================================================================

class IndicatorEngine:
    """Compute all technical indicators from EOD data."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df = self.df.sort_values("date").reset_index(drop=True)

    # --- TREND INDICATORS ---

    def compute_ema(self, period: int, column: str = "close") -> pd.Series:
        return self.df[column].ewm(span=period, adjust=False).mean()

    def compute_sma(self, period: int, column: str = "close") -> pd.Series:
        return self.df[column].rolling(window=period).mean()

    def compute_trend_direction(self) -> pd.Series:
        ema20 = self.compute_ema(20)
        ema50 = self.compute_ema(50)
        sma200 = self.compute_sma(200)

        conditions = [
            (ema20 > ema50) & (ema50 > sma200),
            (ema20 < ema50) & (ema50 < sma200),
        ]
        choices = ["bullish", "bearish"]
        return pd.Series(np.select(conditions, choices, default="neutral"), index=self.df.index)

    # --- MOMENTUM INDICATORS ---

    def compute_rsi(self, period: int = 14) -> pd.Series:
        delta = self.df["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)

        avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = RSI_MAX - (RSI_MAX / (1 + rs))
        return rsi

    def compute_stochastic_rsi(self, rsi_period: int = 14, 
                             stoch_period: int = 14, 
                             k_period: int = 3, 
                             d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        rsi = self.compute_rsi(rsi_period)

        stoch_rsi = (rsi - rsi.rolling(stoch_period).min()) / (rsi.rolling(stoch_period).max() - rsi.rolling(stoch_period).min()).replace(0, np.nan)

        k = stoch_rsi.rolling(k_period).mean() * 100
        d = k.rolling(d_period).mean()
        return k, d

    def compute_macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        ema_fast = self.compute_ema(fast)
        ema_slow = self.compute_ema(slow)
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def compute_macd_state(self) -> pd.Series:
        macd, signal, hist = self.compute_macd()
        prev_macd = macd.shift(1)
        prev_signal = signal.shift(1)

        states = []
        for i in range(len(self.df)):
            if pd.isna(macd.iloc[i]) or pd.isna(signal.iloc[i]):
                states.append("neutral")
                continue

            m, s, pm, ps = macd.iloc[i], signal.iloc[i], prev_macd.iloc[i], prev_signal.iloc[i]

            if pm <= ps and m > s:
                states.append("bullish_cross")
            elif pm >= ps and m < s:
                states.append("bearish_cross")
            elif m > s:
                states.append("bullish")
            elif m < s:
                states.append("bearish")
            else:
                states.append("neutral")
        return pd.Series(states, index=self.df.index)

    # --- VOLUME INDICATORS ---

    def compute_obv(self) -> pd.Series:
        obv = [0]
        for i in range(1, len(self.df)):
            if self.df["close"].iloc[i] > self.df["close"].iloc[i-1]:
                obv.append(obv[-1] + self.df["volume"].iloc[i])
            elif self.df["close"].iloc[i] < self.df["close"].iloc[i-1]:
                obv.append(obv[-1] - self.df["volume"].iloc[i])
            else:
                obv.append(obv[-1])
        return pd.Series(obv, index=self.df.index)

    def compute_cmf(self, period: int = 20) -> pd.Series:
        mfm = ((self.df["close"] - self.df["low"]) - (self.df["high"] - self.df["close"])) / (self.df["high"] - self.df["low"]).replace(0, np.nan)
        mfv = mfm * self.df["volume"]

        vol_sum = self.df["volume"].rolling(window=period).sum()
        cmf = mfv.rolling(window=period).sum() / vol_sum.replace(0, np.nan)
        return cmf

    def compute_vwap(self, period: int = 20) -> pd.Series:
        """Rolling VWAP over N days using daily typical price * volume."""
        typical = (self.df["high"] + self.df["low"] + self.df["close"]) / 3
        tp_vol = typical * self.df["volume"]

        vol_sum = self.df["volume"].rolling(window=period).sum()
        vwap = tp_vol.rolling(window=period).sum() / vol_sum.replace(0, np.nan)
        return vwap

    def compute_anchored_vwap(self, lookback: int = 60) -> Tuple[pd.Series, pd.Series]:
        """
        Anchored VWAP from the most significant swing low/high in the lookback period.
        Returns (anchored_vwap, anchor_date).
        """
        avwap_values = []
        anchor_dates = []

        for i in range(len(self.df)):
            if i < lookback:
                avwap_values.append(np.nan)
                anchor_dates.append(None)
                continue

            window = self.df.iloc[i-lookback:i+1]

            # Find significant swing low (highest volume + lowest price)
            swing_idx = (window["low"] * window["volume"]).idxmin()
            anchor_idx = window.index.get_loc(swing_idx)

            anchor_slice = window.iloc[anchor_idx:]
            typical = (anchor_slice["high"] + anchor_slice["low"] + anchor_slice["close"]) / 3
            tp_vol = typical * anchor_slice["volume"]

            anchor_vol_cumsum = anchor_slice["volume"].cumsum().iloc[-1]
            avwap = tp_vol.cumsum().iloc[-1] / anchor_vol_cumsum if anchor_vol_cumsum != 0 else np.nan
            avwap_values.append(avwap)
            anchor_dates.append(anchor_slice["date"].iloc[0] if "date" in anchor_slice.columns else None)

        return pd.Series(avwap_values, index=self.df.index), pd.Series(anchor_dates, index=self.df.index)

    # --- S/R LEVELS (ported from v3.7) ---

    def compute_pivot_points(self) -> Dict[str, pd.Series]:
        pivot = (self.df["high"].shift(1) + self.df["low"].shift(1) + self.df["close"].shift(1)) / 3
        r1 = 2 * pivot - self.df["low"].shift(1)
        r2 = pivot + (self.df["high"].shift(1) - self.df["low"].shift(1))
        r3 = r1 + (self.df["high"].shift(1) - self.df["low"].shift(1))
        s1 = 2 * pivot - self.df["high"].shift(1)
        s2 = pivot - (self.df["high"].shift(1) - self.df["low"].shift(1))
        s3 = s1 - (self.df["high"].shift(1) - self.df["low"].shift(1))

        return {"pivot": pivot, "r1": r1, "r2": r2, "r3": r3, "s1": s1, "s2": s2, "s3": s3}

    def compute_fibonacci(self, lookback: int = 20) -> Dict[str, pd.Series]:
        high = self.df["high"].rolling(lookback).max()
        low = self.df["low"].rolling(lookback).min()
        diff = high - low

        return {
            "fib_382": high - 0.382 * diff,
            "fib_500": high - 0.500 * diff,
            "fib_618": high - 0.618 * diff,
        }

    # --- ATR & STOP LOSS (ported from v3.7) ---

    def compute_atr(self, period: int = 14) -> pd.Series:
        high_low = self.df["high"] - self.df["low"]
        high_close = np.abs(self.df["high"] - self.df["close"].shift())
        low_close = np.abs(self.df["low"] - self.df["close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def compute_stop_loss(self, atr_multiplier: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        atr = self.compute_atr()
        stop = self.df["close"] - atr * atr_multiplier
        target = self.df["close"] + atr * atr_multiplier * 2  # 1:2 R/R
        risk = self.df["close"] - stop
        rr = (target - self.df["close"]) / risk.replace(0, np.nan)
        return stop, target, rr

    # --- WEEKLY CONFLUENCE (ported from v3.7) ---

    def compute_weekly_confluence(self) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """Daily, Weekly, Monthly trend alignment."""
        daily = self.compute_trend_direction()

        # Weekly approximation: 5-day rolling trend
        weekly_ema20 = self.compute_ema(20)  # ~4 weeks
        weekly_ema50 = self.compute_ema(50)    # ~10 weeks
        weekly_sma200 = self.compute_sma(200) # ~40 weeks
        weekly = pd.Series(np.where(
            (weekly_ema20 > weekly_ema50) & (weekly_ema50 > weekly_sma200), "bullish",
            np.where((weekly_ema20 < weekly_ema50) & (weekly_ema50 < weekly_sma200), "bearish", "neutral"
        )), index=self.df.index)

        # Monthly approximation: 20-day rolling trend  
        monthly_ema50 = self.compute_ema(50)
        monthly = pd.Series(np.where(monthly_ema50 > weekly_sma200, "bullish",
                           np.where(monthly_ema50 < weekly_sma200, "bearish", "neutral")), index=self.df.index)

        # Confluence score: +1 for each bullish, -1 for each bearish
        def score(d, w, m):
            s = 0
            for t in [d, w, m]:
                if t == "bullish": s += 1
                elif t == "bearish": s -= 1
            return s / 3  # normalized -1 to +1

        confluence = pd.Series([score(d, w, m) for d, w, m in zip(daily, weekly, monthly)], index=self.df.index)
        return daily, weekly, monthly, confluence

    # --- GEMINI FLASH FRAMEWORK ---

    def compute_gemini_framework(self) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Gemini Flash Recommended Framework:
        1. Trend Confirmation: Daily EMA 50 > SMA 200
        2. Volume & Liquidity: Rising OBV / CMF > 0
        3. Execution Timing: RSI Pullback to 40-50 / Price touching EMA 20
        """
        ema50 = self.compute_ema(50)
        sma200 = self.compute_sma(200)
        obv = self.compute_obv()
        cmf = self.compute_cmf(20)
        rsi = self.compute_rsi(14)
        ema20 = self.compute_ema(20)

        # 1. Trend Score
        trend_score = ((ema50 > sma200).astype(float) * 1.0).fillna(0)

        # 2. Volume Score
        obv_rising = (obv > obv.shift(5)).astype(float)
        cmf_positive = (cmf > 0).astype(float)
        volume_score = ((obv_rising + cmf_positive) / 2).fillna(0)

        # 3. Timing Score
        rsi_pullback = ((rsi >= 40) & (rsi <= 50)).astype(float)
        price_near_ema20 = (np.abs(self.df["close"] - ema20) / ema20.replace(0, np.nan) < EMA_PROXIMITY_PCT).astype(float)  # within 2%
        timing_score = ((rsi_pullback + price_near_ema20) / 2).fillna(0)

        # Composite (equal weight)
        composite = (trend_score + volume_score + timing_score)  / GEMINI_COMPONENT_COUNT

        return trend_score, volume_score, timing_score, composite

    # --- T+0 / T+1 LIQUIDITY FILTER ---

    def compute_t0_liquidity(self, ticker: str, market_segment: str) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        EGX T+0 Eligibility & Liquidity Scoring.
        High-momentum T+0 stocks exhibit higher volatility and sharper 
        indicator reactions during first and last hours.
        """
        t0_eligible = market_segment in ["high_activity", "moderate_activity", "tamayuz"]

        # Volume vs average
        vol_avg = self.df["volume"].rolling(20).mean()
        vol_ratio = self.df["volume"] / vol_avg

        # Liquidity score: T+0 gets boost, high volume gets boost
        liquidity = np.where(t0_eligible, T0_LIQUIDITY_BASE, 0.0)  # T+1 gets no base boost + np.minimum(vol_ratio * VOLUME_LIQUIDITY_CAP, 0.5)
        liquidity = pd.Series(liquidity, index=self.df.index).fillna(0)

        return pd.Series([t0_eligible] * len(self.df), index=self.df.index),                pd.Series([market_segment] * len(self.df), index=self.df.index),                liquidity

    # --- MASTER COMPUTE ---

    def compute_all(self, ticker: str, market_segment: str = "high_activity") -> IndicatorSnapshot:
        """Compute complete indicator snapshot for latest bar."""
        if len(self.df) < 200:
            # Graceful fallback: compute what we can, warn but don't crash
            print(f"[TA] Warning: Only {len(self.df)} bars available for {ticker}. SMA200 will be NaN.")
            # Continue with available data — SMA200 will naturally be NaN
            pass

        i = len(self.df) - 1  # latest bar

        # Trend
        ema20 = self.compute_ema(20).iloc[i]
        ema50 = self.compute_ema(50).iloc[i]
        sma200 = self.compute_sma(200).iloc[i]
        ema20_slope = (self.compute_ema(20).iloc[i] - self.compute_ema(20).iloc[i-SLOPE_LOOKBACK])  / SLOPE_LOOKBACK
        ema50_slope = (self.compute_ema(50).iloc[i] - self.compute_ema(50).iloc[i-SLOPE_LOOKBACK])  / SLOPE_LOOKBACK
        trend_dir = self.compute_trend_direction().iloc[i]

        # Momentum
        rsi = self.compute_rsi().iloc[i]
        stoch_k, stoch_d = self.compute_stochastic_rsi()
        stoch_k = stoch_k.iloc[i]
        stoch_d = stoch_d.iloc[i]
        macd, signal, hist = self.compute_macd()
        macd_state = self.compute_macd_state().iloc[i]

        # Volume
        obv = self.compute_obv().iloc[i]
        obv_slope = self.compute_obv().iloc[i] - self.compute_obv().iloc[i-5]
        cmf = self.compute_cmf(20).iloc[i]
        vwap = self.compute_vwap(20).iloc[i]
        avwap, anchor_date = self.compute_anchored_vwap(60)
        avwap = avwap.iloc[i]
        anchor = anchor_date.iloc[i]
        vol_ratio = (self.df["volume"].iloc[i] / self.df["volume"].rolling(20).mean().iloc[i])

        # Gemini Framework
        trend_score, vol_score, timing_score, gemini_score = self.compute_gemini_framework()
        trend_score = trend_score.iloc[i]
        vol_score = vol_score.iloc[i]
        timing_score = timing_score.iloc[i]
        gemini_score = gemini_score.iloc[i]

        # T+0
        t0_eligible, segment, liquidity = self.compute_t0_liquidity(ticker, market_segment)
        t0_eligible = t0_eligible.iloc[i]
        segment = segment.iloc[i]
        liquidity = liquidity.iloc[i]

        # S/R
        pivots = self.compute_pivot_points()
        fibs = self.compute_fibonacci(20)

        # ATR
        atr = self.compute_atr().iloc[i]
        stop, target, rr = self.compute_stop_loss()
        stop = stop.iloc[i]
        target = target.iloc[i]
        rr = rr.iloc[i]

        # Weekly Confluence
        daily_t, weekly_t, monthly_t, conf = self.compute_weekly_confluence()

        return IndicatorSnapshot(
            ticker=ticker,
            timestamp=str(self.df["date"].iloc[i]) if "date" in self.df.columns else str(datetime.now()),
            ema_20=round(ema20, 4),
            ema_50=round(ema50, 4),
            sma_200=round(sma200, 4),
            ema_20_slope=round(ema20_slope, 6),
            ema_50_slope=round(ema50_slope, 6),
            trend_direction=trend_dir,
            rsi_14=round(rsi, 2),
            stoch_rsi_k=round(stoch_k, 2),
            stoch_rsi_d=round(stoch_d, 2),
            macd_line=round(macd.iloc[i], 4),
            macd_signal=round(signal.iloc[i], 4),
            macd_histogram=round(hist.iloc[i], 4),
            macd_state=macd_state,
            obv=round(obv, 0),
            obv_slope=round(obv_slope, 0),
            cmf_20=round(cmf, 4),
            vwap_20d=round(vwap, 4),
            anchored_vwap=round(avwap, 4) if not pd.isna(avwap) else None,
            anchor_date=str(anchor) if anchor else None,
            volume_vs_avg=round(vol_ratio, 2),
            trend_score=round(trend_score, 2),
            volume_score=round(vol_score, 2),
            timing_score=round(timing_score, 2),
            gemini_framework_score=round(gemini_score, 2),
            t0_eligible=t0_eligible,
            market_segment=segment,
            liquidity_score=round(liquidity, 2),
            pivot=round(pivots["pivot"].iloc[i], 4),
            r1=round(pivots["r1"].iloc[i], 4),
            r2=round(pivots["r2"].iloc[i], 4),
            r3=round(pivots["r3"].iloc[i], 4),
            s1=round(pivots["s1"].iloc[i], 4),
            s2=round(pivots["s2"].iloc[i], 4),
            s3=round(pivots["s3"].iloc[i], 4),
            fib_382=round(fibs["fib_382"].iloc[i], 4),
            fib_500=round(fibs["fib_500"].iloc[i], 4),
            fib_618=round(fibs["fib_618"].iloc[i], 4),
            atr_14=round(atr, 4),
            stop_loss=round(stop, 4),
            take_profit=round(target, 4),
            risk_reward=round(rr, 2),
            daily_trend=daily_t.iloc[i],
            weekly_trend=weekly_t.iloc[i],
            monthly_trend=monthly_t.iloc[i],
            confluence_score=round(conf.iloc[i], 2)
        )


# ============================================================================
# SETUP QUALITY ENGINE
# ============================================================================

class SetupQualityEngine:
    """Generate actionable setups using all indicators + EGX T+0 nuance."""

    def __init__(self, snapshot: IndicatorSnapshot):
        self.s = snapshot
        self.rationale = []
        self.score = 0

    def evaluate(self) -> SetupQuality:
        """Full setup evaluation with Gemini Flash framework + T+0 nuance."""

        # --- GEMINI FLASH FRAMEWORK CHECK ---
        if self.s.trend_score > 0.5:
            self.rationale.append("✅ Trend: EMA50 > SMA200 (bullish bias confirmed)")
            self.score += 25
        else:
            self.rationale.append("❌ Trend: EMA50 < SMA200 (counter-trend risk)")

        if self.s.volume_score > 0.5:
            self.rationale.append("✅ Volume: OBV rising + CMF > 0 (institutional accumulation)")
            self.score += 25
        else:
            self.rationale.append("⚠️ Volume: Weak OBV/CMF (confirm with price action)")
            self.score += 10

        if self.s.timing_score > 0.5:
            self.rationale.append("✅ Timing: RSI pullback to 40-50 zone + price near EMA20 (optimal entry)")
            self.score += 25
        else:
            rsi_zone = "overbought" if self.s.rsi_14 > 70 else "oversold" if self.s.rsi_14 < 30 else "neutral"
            self.rationale.append(f"⚠️ Timing: RSI {self.s.rsi_14:.1f} ({rsi_zone}) — wait for 40-50 pullback")
            self.score += 10

        # --- T+0 LIQUIDITY NUANCE ---
        if self.s.t0_eligible:
            self.rationale.append(f"🔥 T+0 ELIGIBLE ({self.s.market_segment}): Higher volatility expected in first/last hour")
            self.score += 15
            best_session = "first_hour" if self.s.liquidity_score > 0.8 else "last_hour"
        else:
            self.rationale.append(f"⏳ T+1 ONLY ({self.s.market_segment}): Lower intraday volatility, plan for overnight hold")
            best_session = "mid_session"

        # --- VWAP CONFLUENCE ---
        price = self.s.pivot  # approximate current price
        if self.s.vwap_20d and price > self.s.vwap_20d:
            self.rationale.append("✅ Price above 20-day VWAP (bullish volume-weighted sentiment)")
            self.score += 10
        elif self.s.vwap_20d and price < self.s.vwap_20d:
            self.rationale.append("⚠️ Price below 20-day VWAP (caution on longs)")

        if self.s.anchored_vwap and abs(price - self.s.anchored_vwap) / price < 0.02:
            self.rationale.append(f"🎯 Price at Anchored VWAP (anchor: {self.s.anchor_date}) — key decision level")
            self.score += 10

        # --- MACD MOMENTUM ---
        if self.s.macd_state == "bullish_cross":
            self.rationale.append("🚀 MACD bullish crossover — momentum shift confirmed")
            self.score += 15
        elif self.s.macd_state == "bearish_cross":
            self.rationale.append("🔻 MACD bearish crossover — avoid longs")
            self.score -= 10
        elif self.s.macd_state == "bullish":
            self.rationale.append("✅ MACD bullish (above signal)")
            self.score += 5

        # --- STOCHASTIC RSI ---
        if self.s.stoch_rsi_k < 20 and self.s.stoch_rsi_d < 20:
            self.rationale.append("🔄 StochRSI deeply oversold — mean reversion potential")
            self.score += 10
        elif self.s.stoch_rsi_k > 80 and self.s.stoch_rsi_d > 80:
            self.rationale.append("⚠️ StochRSI overbought — pullback likely")
            self.score -= 5

        # --- WEEKLY CONFLUENCE BONUS ---
        if self.s.confluence_score > 0.5:
            self.rationale.append(f"✅ Multi-timeframe bullish ({self.s.daily_trend}/{self.s.weekly_trend}/{self.s.monthly_trend})")
            self.score += 10
        elif self.s.confluence_score < -0.5:
            self.rationale.append(f"❌ Multi-timeframe bearish ({self.s.daily_trend}/{self.s.weekly_trend}/{self.s.monthly_trend})")
            self.score -= 10

        # --- RISK CHECK ---
        if self.s.risk_reward >= 2.0:
            self.rationale.append(f"✅ R/R {self.s.risk_reward:.1f}:1 meets minimum 2:1 threshold")
            self.score += 10
        else:
            self.rationale.append(f"❌ R/R {self.s.risk_reward:.1f}:1 below 2:1 — skip or widen target")
            self.score -= 15

        # Clamp score
        self.score = max(0, min(100, self.score))

        # Determine setup type
        if self.score >= 75:
            setup_type = "STRONG_BUY"
            confidence = "HIGH"
        elif self.score >= 60:
            setup_type = "BUY"
            confidence = "MEDIUM"
        elif self.score >= 45:
            setup_type = "WATCH"
            confidence = "LOW"
        else:
            setup_type = "AVOID"
            confidence = "NONE"

        # Entry zone: between EMA20 and VWAP (tight) or S1/R1 (wider)
        if self.s.ema_20 and self.s.vwap_20d:
            entry_low = min(self.s.ema_20, self.s.vwap_20d)
            entry_high = max(self.s.ema_20, self.s.vwap_20d)
        else:
            entry_low = self.s.s1
            entry_high = self.s.pivot

        return SetupQuality(
            ticker=self.s.ticker,
            setup_type=setup_type,
            quality_score=round(self.score, 1),
            entry_zone=(round(entry_low, 4), round(entry_high, 4)),
            stop_loss=self.s.stop_loss,
            targets=[self.s.take_profit, round(self.s.take_profit * 1.5, 4)],
            confidence=confidence,
            rationale=self.rationale,
            t0_recommended=self.s.t0_eligible and self.s.liquidity_score > 0.7,
            best_session=best_session
        )


# ============================================================================
# CACHE & PERSISTENCE
# ============================================================================

class TechnicalCache:
    """SQLite cache for technical indicators (6h TTL, ported from v3.7)."""

    def __init__(self, db_path: str = "sentinel_cache.db"):
        self.conn = sqlite3.connect(db_path)
        self._init_tables()

    def _init_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS technical_indicators (
                ticker TEXT,
                timestamp TEXT,
                data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (ticker, timestamp)
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS setup_quality (
                ticker TEXT PRIMARY KEY,
                data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def save_indicators(self, snapshot: IndicatorSnapshot):
        self.conn.execute(
            "INSERT OR REPLACE INTO technical_indicators VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
            (snapshot.ticker, snapshot.timestamp, json.dumps(snapshot.__dict__, cls=NumpyEncoder))
        )
        self.conn.commit()

    def load_indicators(self, ticker: str, max_age_hours: int = 6) -> Optional[IndicatorSnapshot]:
        cursor = self.conn.execute(
            "SELECT data FROM technical_indicators WHERE ticker = ? AND datetime(created_at) > datetime('now', '-{} hours')".format(max_age_hours),
            (ticker,)
        )
        row = cursor.fetchone()
        if row:
            data = json.loads(row[0])
            return IndicatorSnapshot(**data)
        return None

    def save_setup(self, setup: SetupQuality):
        self.conn.execute(
            "INSERT OR REPLACE INTO setup_quality VALUES (?, ?, CURRENT_TIMESTAMP)",
            (setup.ticker, json.dumps(setup.__dict__, cls=NumpyEncoder))
        )
        self.conn.commit()


# ============================================================================
# PUBLIC API
# ============================================================================

def analyze_ticker(df: pd.DataFrame, ticker: str, market_segment: str = "high_activity") -> Tuple[IndicatorSnapshot, SetupQuality]:
    """Full technical analysis pipeline for a single ticker."""
    engine = IndicatorEngine(df)
    snapshot = engine.compute_all(ticker, market_segment)

    quality_engine = SetupQualityEngine(snapshot)
    setup = quality_engine.evaluate()

    # Cache
    cache = TechnicalCache()
    cache.save_indicators(snapshot)
    cache.save_setup(setup)

    return snapshot, setup


def batch_analyze(dataframes: Dict[str, pd.DataFrame], 
                  segments: Dict[str, str]) -> Dict[str, Tuple[IndicatorSnapshot, SetupQuality]]:
    """Analyze multiple tickers."""
    results = {}
    for ticker, df in dataframes.items():
        try:
            seg = segments.get(ticker, "high_activity")
            results[ticker] = analyze_ticker(df, ticker, seg)
        except Exception as e:
            print(f"[TA] Error analyzing {ticker}: {e}")
            continue
    return results


def get_indicator_summary(snapshot: IndicatorSnapshot) -> Dict:
    """Human-readable summary for UI / Telegram."""
    return {
        "ticker": snapshot.ticker,
        "trend": {
            "direction": snapshot.trend_direction,
            "ema20": snapshot.ema_20,
            "ema50": snapshot.ema_50,
            "sma200": snapshot.sma_200,
            "gemini_trend": "✅ EMA50 > SMA200" if snapshot.trend_score > 0.5 else "❌ EMA50 < SMA200"
        },
        "momentum": {
            "rsi": snapshot.rsi_14,
            "stoch_rsi_k": snapshot.stoch_rsi_k,
            "macd_state": snapshot.macd_state,
            "gemini_timing": "✅ RSI 40-50 + near EMA20" if snapshot.timing_score > 0.5 else f"RSI {snapshot.rsi_14} (wait for 40-50)"
        },
        "volume": {
            "obv": snapshot.obv,
            "cmf": snapshot.cmf_20,
            "vwap": snapshot.vwap_20d,
            "volume_vs_avg": snapshot.volume_vs_avg,
            "gemini_volume": "✅ OBV rising + CMF>0" if snapshot.volume_score > 0.5 else "⚠️ Weak volume"
        },
        "gemini_framework": {
            "trend_score": snapshot.trend_score,
            "volume_score": snapshot.volume_score,
            "timing_score": snapshot.timing_score,
            "composite": snapshot.gemini_framework_score,
            "signal": "STRONG" if snapshot.gemini_framework_score > 0.7 else 
                     "MODERATE" if snapshot.gemini_framework_score > 0.5 else "WEAK"
        },
        "egx_nuance": {
            "t0_eligible": snapshot.t0_eligible,
            "segment": snapshot.market_segment,
            "liquidity": snapshot.liquidity_score,
            "note": "First/last hour volatility expected" if snapshot.t0_eligible else "Plan for overnight hold (T+1)"
        },
        "risk": {
            "stop": snapshot.stop_loss,
            "target": snapshot.take_profit,
            "rr": snapshot.risk_reward,
            "atr": snapshot.atr_14
        },
        "confluence": {
            "daily": snapshot.daily_trend,
            "weekly": snapshot.weekly_trend,
            "monthly": snapshot.monthly_trend,
            "score": snapshot.confluence_score
        }
    }


if __name__ == "__main__":
    # Demo with synthetic data
    dates = pd.date_range(end=datetime.now(), periods=250, freq="B")
    np.random.seed(42)
    noise = np.random.randn(250).cumsum() * 0.5

    demo_df = pd.DataFrame({
        "date": dates,
        "open": 100 + noise + np.random.randn(250) * 0.3,
        "high": 100 + noise + abs(np.random.randn(250)) * 0.5 + 0.3,
        "low": 100 + noise - abs(np.random.randn(250)) * 0.5 - 0.3,
        "close": 100 + noise + np.random.randn(250) * 0.2,
        "volume": np.random.randint(100000, 500000, 250)
    })

    snap, setup = analyze_ticker(demo_df, "DEMO.CA", "high_activity")
    print("=" * 60)
    print(f"Ticker: {snap.ticker} | T+0: {snap.t0_eligible} | Segment: {snap.market_segment}")
    print(f"Gemini Score: {snap.gemini_framework_score} | Setup: {setup.setup_type} ({setup.quality_score})")
    print(f"RSI: {snap.rsi_14} | StochRSI: {snap.stoch_rsi_k}/{snap.stoch_rsi_d} | MACD: {snap.macd_state}")
    print(f"VWAP: {snap.vwap_20d} | CMF: {snap.cmf_20} | OBV slope: {snap.obv_slope}")
    print(f"Best Session: {setup.best_session} | T0 Rec: {setup.t0_recommended}")
    print("Rationale:")
    for r in setup.rationale:
        print(f"  {r}")
