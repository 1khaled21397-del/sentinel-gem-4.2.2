"""
Sentinel-EGX v4.2 — Regime Detector
======================================
Market regime detection: bull / sideways / bear
Uses heuristic fallback (no hmmlearn dependency required)
"""

import numpy as np
import pandas as pd
from typing import Dict


def detect_regime(df: pd.DataFrame) -> Dict:
    """
    Detect market regime from price action.
    Returns: {"regime": str, "position_size": float, "confidence": float}
    """
    if len(df) < 50:
        return {"regime": "unknown", "position_size": 1.0, "confidence": 0.0}

    close = df["close"]
    returns = close.pct_change().dropna()

    # Trend: slope of 50-day linear regression
    x = np.arange(len(close))
    slope = np.polyfit(x[-50:], close.iloc[-50:].values, 1)[0]
    slope_pct = slope / close.iloc[-1] * 100

    # Volatility regime
    vol_20 = returns.iloc[-20:].std() * np.sqrt(252)
    vol_50 = returns.iloc[-50:].std() * np.sqrt(252)
    vol_trend = vol_20 / vol_50 if vol_50 > 0 else 1.0

    # ADX proxy using price range
    high_low = (df["high"] - df["low"]) / df["close"]
    adx_proxy = high_low.iloc[-14:].mean() * 100

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    latest_rsi = rsi.iloc[-1] if not rsi.empty else 50

    # Scoring
    trend_score = 0
    if slope_pct > 0.05:
        trend_score = 2  # strong bull
    elif slope_pct > 0.01:
        trend_score = 1  # weak bull
    elif slope_pct < -0.05:
        trend_score = -2  # strong bear
    elif slope_pct < -0.01:
        trend_score = -1  # weak bear

    vol_score = 0
    if vol_trend > 1.2:
        vol_score = 1  # expanding vol
    elif vol_trend < 0.8:
        vol_score = -1  # contracting vol

    # Regime classification
    total_score = trend_score + vol_score

    if total_score >= 2 and latest_rsi > 50:
        regime = "bull"
        position_size = 1.0
    elif total_score <= -2 and latest_rsi < 50:
        regime = "bear"
        position_size = 0.2
    else:
        regime = "sideways"
        position_size = 0.6

    confidence = min(1.0, abs(total_score) / 3.0 + 0.3)

    return {
        "regime": regime,
        "position_size": position_size,
        "confidence": round(confidence, 2),
        "slope_pct": round(slope_pct, 4),
        "volatility_annual": round(vol_20, 4),
        "rsi": round(latest_rsi, 2),
        "adx_proxy": round(adx_proxy, 2)
    }


class RegimeDetector:
    """Wrapper class for consistent API."""

    def detect(self, df: pd.DataFrame) -> Dict:
        return detect_regime(df)


if __name__ == "__main__":
    print("RegimeDetector v4.2 ready: heuristic bull/sideways/bear classifier")
