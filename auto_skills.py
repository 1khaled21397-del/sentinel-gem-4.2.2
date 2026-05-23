"""
Sentinel-EGX v4.2.1 — Auto Skills Engine
==========================================
7 pattern recognition skills for alpha scoring.
Breakout, Mean Reversion, Trend Following, Volume Spike,
Gap Fill, Support Bounce, Resistance Rejection.
Aligned with sentinel_config.json v4.2 specs.
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

with open("sentinel_config.json", "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

SKILLS_CFG = CONFIG.get("auto_skills", {})
WEIGHTS = SKILLS_CFG.get("weights", {})
T0_BONUS_SKILLS = SKILLS_CFG.get("t0_bonus_skills", [])
GEMINI_BONUS = SKILLS_CFG.get("gemini_alignment_bonus", 0.15)


@dataclass
class SkillResult:
    skill: str
    direction: str  # "bullish", "bearish", "neutral"
    confidence: float  # 0.0 to 1.0
    gemini_aligned: bool
    details: str


def _detect_breakout(df: pd.DataFrame) -> Optional[SkillResult]:
    """Price breaks above 20-day high with volume confirmation."""
    if len(df) < 20:
        return None
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    high20 = df["high"].rolling(20).max().iloc[-1]
    vol_avg = df["volume"].rolling(20).mean().iloc[-1]

    if latest["close"] > high20 and latest["volume"] > vol_avg * 1.5:
        return SkillResult(
            skill="breakout",
            direction="bullish",
            confidence=min(0.95, 0.7 + (latest["volume"] / vol_avg - 1.5) * 0.1),
            gemini_aligned=latest.get("gemini_composite", 0) > 0.6,
            details=f"Broke 20D high {high20:.2f} with {latest['volume']/vol_avg:.1f}x volume"
        )
    return None


def _detect_mean_reversion(df: pd.DataFrame) -> Optional[SkillResult]:
    """Price hits lower Bollinger Band + RSI oversold."""
    if len(df) < 20:
        return None
    latest = df.iloc[-1]
    close = df["close"]
    sma20 = close.rolling(20).mean().iloc[-1]
    std20 = close.rolling(20).std().iloc[-1]
    bb_lower = sma20 - 2 * std20

    delta = close.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    latest_rsi = rsi.iloc[-1]

    if latest["close"] < bb_lower and latest_rsi < 30:
        return SkillResult(
            skill="mean_reversion",
            direction="bullish",
            confidence=min(0.9, 0.6 + (30 - latest_rsi) / 30 * 0.3),
            gemini_aligned=latest.get("gemini_composite", 0) > 0.5,
            details=f"Below BB lower {bb_lower:.2f}, RSI {latest_rsi:.1f}"
        )
    return None


def _detect_trend_following(df: pd.DataFrame) -> Optional[SkillResult]:
    """EMA20 > EMA50 with ADX > 25."""
    if len(df) < 50:
        return None
    ema20 = df["close"].ewm(span=20, adjust=False).mean().iloc[-1]
    ema50 = df["close"].ewm(span=50, adjust=False).mean().iloc[-1]

    # ADX proxy
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift()).abs()
    tr3 = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False).mean()
    adx_proxy = (tr / df["close"]).rolling(14).mean().iloc[-1] * 100

    if ema20 > ema50 and adx_proxy > 25:
        return SkillResult(
            skill="trend_following",
            direction="bullish",
            confidence=min(0.9, 0.6 + (adx_proxy - 25) / 50 * 0.3),
            gemini_aligned=ema20 > ema50,
            details=f"EMA20>{ema50:.2f} EMA50, ADX proxy {adx_proxy:.1f}"
        )
    elif ema20 < ema50 and adx_proxy > 25:
        return SkillResult(
            skill="trend_following",
            direction="bearish",
            confidence=min(0.9, 0.6 + (adx_proxy - 25) / 50 * 0.3),
            gemini_aligned=ema20 < ema50,
            details=f"EMA20<{ema50:.2f} EMA50, ADX proxy {adx_proxy:.1f}"
        )
    return None


def _detect_volume_spike(df: pd.DataFrame) -> Optional[SkillResult]:
    """Volume > 2x 20-day average with price direction."""
    if len(df) < 20:
        return None
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    vol_avg = df["volume"].rolling(20).mean().iloc[-1]

    if latest["volume"] > vol_avg * 2.0:
        direction = "bullish" if latest["close"] > prev["close"] else "bearish"
        return SkillResult(
            skill="volume_spike",
            direction=direction,
            confidence=min(0.95, 0.65 + (latest["volume"] / vol_avg - 2) * 0.1),
            gemini_aligned=latest.get("gemini_volume_score", 0) > 0.5,
            details=f"{latest['volume']/vol_avg:.1f}x avg volume, {direction} close"
        )
    return None


def _detect_gap_fill(df: pd.DataFrame) -> Optional[SkillResult]:
    """Detect overnight gap and probability of fill."""
    if len(df) < 5:
        return None
    latest = df.iloc[-1]
    prev_close = df.iloc[-2]["close"] if len(df) > 1 else latest["open"]
    gap = (latest["open"] - prev_close) / prev_close

    if abs(gap) > 0.01:  # > 1% gap
        direction = "bearish" if gap > 0 else "bullish"  # gap up = likely fill down
        # Gap fill probability: larger gaps less likely to fill same day
        fill_prob = max(0.3, 1.0 - abs(gap) * 10)
        return SkillResult(
            skill="gap_fill",
            direction=direction,
            confidence=fill_prob,
            gemini_aligned=abs(gap) < 0.03,
            details=f"{gap*100:.1f}% gap, fill probability {fill_prob:.0%}"
        )
    return None


def _detect_support_bounce(df: pd.DataFrame) -> Optional[SkillResult]:
    """Price bounces off S1 or 20-day low."""
    if len(df) < 20:
        return None
    latest = df.iloc[-1]
    low20 = df["low"].rolling(20).min().iloc[-1]

    # Calculate pivot S1
    prev = df.iloc[-2] if len(df) > 1 else latest
    pp = (prev["high"] + prev["low"] + prev["close"]) / 3
    s1 = 2 * pp - prev["high"]

    near_support = abs(latest["close"] - s1) / latest["close"] < 0.02 or                    abs(latest["close"] - low20) / latest["close"] < 0.02

    if near_support and latest["close"] > latest["open"]:  # bullish candle
        return SkillResult(
            skill="support_bounce",
            direction="bullish",
            confidence=0.7,
            gemini_aligned=latest.get("gemini_composite", 0) > 0.5,
            details=f"Bounce off support S1={s1:.2f} or 20D low={low20:.2f}"
        )
    return None


def _detect_resistance_rejection(df: pd.DataFrame) -> Optional[SkillResult]:
    """Price rejected at R1 or 20-day high."""
    if len(df) < 20:
        return None
    latest = df.iloc[-1]
    high20 = df["high"].rolling(20).max().iloc[-1]

    prev = df.iloc[-2] if len(df) > 1 else latest
    pp = (prev["high"] + prev["low"] + prev["close"]) / 3
    r1 = 2 * pp - prev["low"]

    near_resistance = abs(latest["high"] - r1) / latest["close"] < 0.02 or                       abs(latest["high"] - high20) / latest["close"] < 0.02

    if near_resistance and latest["close"] < latest["open"]:  # bearish candle
        return SkillResult(
            skill="resistance_rejection",
            direction="bearish",
            confidence=0.7,
            gemini_aligned=latest.get("gemini_composite", 0) < 0.5,
            details=f"Rejection at resistance R1={r1:.2f} or 20D high={high20:.2f}"
        )
    return None


def analyze_skills(df: pd.DataFrame, ticker: str, segment: str = "high_activity") -> Dict:
    """
    Run all 7 skills on a ticker and return composite score.
    T+0 eligible tickers get bonus on volume_spike, breakout, gap_fill.
    """
    skills = [
        _detect_breakout(df),
        _detect_mean_reversion(df),
        _detect_trend_following(df),
        _detect_volume_spike(df),
        _detect_gap_fill(df),
        _detect_support_bounce(df),
        _detect_resistance_rejection(df),
    ]

    triggered = [s for s in skills if s is not None]
    t0_eligible = segment in CONFIG.get("t0_rules", {}).get("t0_enabled_segments", [])

    total_score = 0.0
    details = []

    for skill in triggered:
        weight = WEIGHTS.get(skill.skill, 0.1)

        # T+0 bonus for high-momentum skills
        if t0_eligible and skill.skill in T0_BONUS_SKILLS:
            weight *= 1.3

        # Gemini alignment bonus
        if skill.gemini_aligned:
            weight += GEMINI_BONUS

        score = skill.confidence * weight
        if skill.direction == "bearish":
            score *= -1

        total_score += score
        details.append({
            "skill": skill.skill,
            "direction": skill.direction,
            "confidence": round(skill.confidence, 2),
            "gemini_aligned": skill.gemini_aligned,
            "details": skill.details,
            "weight": round(weight, 3),
            "score": round(score, 3)
        })

    # Normalize to 0-1 range
    composite = np.clip((total_score + 1) / 2, 0, 1)

    return {
        "ticker": ticker,
        "segment": segment,
        "t0_eligible": t0_eligible,
        "composite_score": round(composite, 3),
        "skills_triggered": len(triggered),
        "total_possible": 7,
        "triggered_details": details,
        "dominant_direction": "bullish" if total_score > 0 else "bearish" if total_score < 0 else "neutral",
        "raw_score": round(total_score, 3)
    }


def get_skill_summary(skills_result: Dict) -> str:
    """Human-readable summary of skill analysis."""
    lines = [
        f"Skills: {skills_result['skills_triggered']}/7 triggered | Composite: {skills_result['composite_score']}",
        f"Direction: {skills_result['dominant_direction']} | T+0: {'Yes' if skills_result['t0_eligible'] else 'No'}"
    ]
    for d in skills_result.get("triggered_details", []):
        emoji = "🔥" if d["skill"] == "breakout" else "🔄" if d["skill"] == "mean_reversion" else                 "📈" if d["skill"] == "trend_following" else "💥" if d["skill"] == "volume_spike" else                 "🌙" if d["skill"] == "gap_fill" else "🛡️" if d["skill"] == "support_bounce" else "🚫"
        lines.append(f"  {emoji} {d['skill']}: {d['direction']} ({d['confidence']:.0%}) — {d['details']}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("AutoSkills v4.2.1 ready: 7 pattern recognition skills")
    print("Skills: breakout, mean_reversion, trend_following, volume_spike,")
    print("        gap_fill, support_bounce, resistance_rejection")
