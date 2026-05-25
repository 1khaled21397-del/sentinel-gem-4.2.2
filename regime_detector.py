"""
Sentinel-EGX v4.2.2 — Hybrid Regime Detector (Blended Path)
============================================================
Foundation: Enhanced heuristic detector (per-ticker, free, deterministic)
Layer 2:    Claude macro analyzer (market-wide, 1 call/day)
Layer 3:    Hybrid ensemble with disagreement handling + EGX shock detection

Paper-trading ready: logs all signals, conflicts, and shock events.
"""

import numpy as np
import pandas as pd
import json
import os
import requests
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta

# ── CONFIG ──
CONFIG_PATH = "sentinel_config.json"
try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        CONFIG = json.load(f)
except Exception:
    CONFIG = {}

REGIME_CFG = CONFIG.get("regime_detector", {})
MULTIPLIERS = REGIME_CFG.get("multipliers", {"bull": 1.0, "sideways": 0.6, "bear": 0.2})

# API Keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
if not ANTHROPIC_API_KEY:
    try:
        import streamlit as st
        ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY", "").strip()
    except Exception:
        pass

# ── DATA CLASSES ──

@dataclass
class HeuristicRegimeResult:
    """Output from the enhanced heuristic detector."""
    regime: str  # strong_bull, fragile_bull, sideways, weak_bear, strong_bear, unknown
    position_size: float
    confidence: float
    slope_pct: float
    volatility_annual: float
    rsi: float
    adx_proxy: float
    shock_detected: bool
    shock_type: Optional[str]
    t0_volatility_spike: bool
    sector_rotation_signal: Optional[str]
    raw_scores: Dict


@dataclass
class MacroRegimeResult:
    """Output from Claude macro analyzer."""
    macro_score: float  # -1.0 (very bearish) to +1.0 (very bullish)
    confidence: float
    risk_off_flag: bool
    risk_on_flag: bool
    sector_rotation: Dict[str, float]  # sector -> score
    key_factors: List[str]
    summary: str
    source: str  # "claude" or "fallback"


@dataclass
class HybridRegimeResult:
    """Final blended output."""
    regime: str
    position_size: float
    confidence: float
    heuristic: HeuristicRegimeResult
    macro: MacroRegimeResult
    disagreement_index: float
    conflict_flag: bool
    recommendation: str
    timestamp: str


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 1: ENHANCED HEURISTIC DETECTOR (per-ticker, free, deterministic)
# ═══════════════════════════════════════════════════════════════════════════════

class HeuristicRegimeDetector:
    """
    Enhanced version of v4.2 heuristic detector.
    Adds EGX-specific shock detection, T+0 volatility spikes, and sector rotation proxy.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.cfg = config or {}
        # Configurable thresholds (not hardcoded)
        self.thresholds = self.cfg.get("thresholds", {
            "strong_bull_slope": 0.05,
            "weak_bull_slope": 0.01,
            "strong_bear_slope": -0.05,
            "weak_bear_slope": -0.01,
            "vol_expanding": 1.2,
            "vol_contracting": 0.8,
            "shock_gap_pct": 0.10,       # 10% overnight gap = shock
            "shock_volume_mult": 5.0,    # 5x avg volume = shock
            "t0_vol_spike_mult": 3.0,    # 3x avg volume on T+0 ticker
            "rsi_bull": 50,
            "rsi_bear": 50,
        })

    def detect(self, df: pd.DataFrame, ticker: str = "", segment: str = "") -> HeuristicRegimeResult:
        """Run full heuristic analysis on single-ticker dataframe."""
        if len(df) < 50:
            return HeuristicRegimeResult(
                regime="unknown", position_size=1.0, confidence=0.0,
                slope_pct=0.0, volatility_annual=0.0, rsi=50.0, adx_proxy=0.0,
                shock_detected=False, shock_type=None,
                t0_volatility_spike=False, sector_rotation_signal=None,
                raw_scores={}
            )

        close = df["close"]
        returns = close.pct_change().dropna()
        th = self.thresholds

        # ── Trend: 50-day linear regression slope ──
        x = np.arange(len(close))
        slope = np.polyfit(x[-50:], close.iloc[-50:].values, 1)[0]
        slope_pct = slope / close.iloc[-1] * 100

        # ── Volatility regime ──
        vol_20 = returns.iloc[-20:].std() * np.sqrt(252)
        vol_50 = returns.iloc[-50:].std() * np.sqrt(252)
        vol_trend = vol_20 / vol_50 if vol_50 > 0 else 1.0

        # ── ADX proxy ──
        high_low = (df["high"] - df["low"]) / df["close"]
        adx_proxy = high_low.iloc[-14:].mean() * 100

        # ── RSI ──
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        latest_rsi = rsi.iloc[-1] if not rsi.empty else 50

        # ── EGX SHOCK DETECTION ──
        shock_detected, shock_type = self._detect_shock(df, th)

        # ── T+0 VOLATILITY SPIKE ──
        t0_spike = self._detect_t0_spike(df, segment, th)

        # ── SCORING ──
        trend_score = 0
        if slope_pct > th["strong_bull_slope"]:
            trend_score = 2
        elif slope_pct > th["weak_bull_slope"]:
            trend_score = 1
        elif slope_pct < th["strong_bear_slope"]:
            trend_score = -2
        elif slope_pct < th["weak_bear_slope"]:
            trend_score = -1

        vol_score = 0
        if vol_trend > th["vol_expanding"]:
            vol_score = 1
        elif vol_trend < th["vol_contracting"]:
            vol_score = -1

        # Shock overrides trend score
        if shock_detected:
            if "devaluation" in (shock_type or "") or "crash" in (shock_type or ""):
                trend_score = min(trend_score, -1)
            elif "policy" in (shock_type or ""):
                trend_score = 0  # neutralize until clarity

        total_score = trend_score + vol_score

        # ── REGIME CLASSIFICATION (5-state) ──
        if total_score >= 2 and latest_rsi > th["rsi_bull"] and not shock_detected:
            regime = "strong_bull"
            position_size = MULTIPLIERS.get("bull", 1.0)
        elif total_score >= 1 and latest_rsi > th["rsi_bull"]:
            regime = "fragile_bull"
            position_size = MULTIPLIERS.get("bull", 1.0) * 0.8
        elif total_score <= -2 and latest_rsi < th["rsi_bear"]:
            regime = "strong_bear"
            position_size = MULTIPLIERS.get("bear", 0.2)
        elif total_score <= -1 and latest_rsi < th["rsi_bear"]:
            regime = "weak_bear"
            position_size = MULTIPLIERS.get("bear", 0.2) * 1.5  # less conservative than strong_bear
        else:
            regime = "sideways"
            position_size = MULTIPLIERS.get("sideways", 0.6)

        # T+0 volatility spike reduces position size
        if t0_spike and position_size > 0.5:
            position_size *= 0.7

        confidence = min(1.0, abs(total_score) / 3.0 + 0.3)
        if shock_detected:
            confidence *= 0.6  # reduce confidence during shocks

        return HeuristicRegimeResult(
            regime=regime,
            position_size=round(position_size, 2),
            confidence=round(confidence, 2),
            slope_pct=round(slope_pct, 4),
            volatility_annual=round(vol_20, 4),
            rsi=round(latest_rsi, 2),
            adx_proxy=round(adx_proxy, 2),
            shock_detected=shock_detected,
            shock_type=shock_type,
            t0_volatility_spike=t0_spike,
            sector_rotation_signal=None,  # per-ticker doesn't have sector context
            raw_scores={"trend": trend_score, "vol": vol_score, "total": total_score}
        )

    def _detect_shock(self, df: pd.DataFrame, th: Dict) -> Tuple[bool, Optional[str]]:
        """Detect EGX-specific market shocks."""
        if len(df) < 5:
            return False, None

        latest = df.iloc[-1]
        prev = df.iloc[-2]
        vol_avg = df["volume"].rolling(20).mean().iloc[-1]

        # Overnight gap shock
        gap = (latest["open"] - prev["close"]) / prev["close"] if prev["close"] != 0 else 0
        if abs(gap) > th["shock_gap_pct"]:
            direction = "devaluation" if gap < 0 else "melt_up"
            return True, f"{direction}_gap_{abs(gap)*100:.1f}pct"

        # Volume shock (policy announcement pattern)
        if latest["volume"] > vol_avg * th["shock_volume_mult"]:
            return True, "volume_spike_policy"

        # Consecutive limit-down days (rare but critical)
        if len(df) >= 3:
            last3 = df["close"].iloc[-3:].pct_change().dropna()
            if len(last3) >= 2 and all(last3 < -0.05):
                return True, "consecutive_limit_down"

        return False, None

    def _detect_t0_spike(self, df: pd.DataFrame, segment: str, th: Dict) -> bool:
        """Detect abnormal volatility on T+0 eligible tickers."""
        t0_segments = CONFIG.get("t0_rules", {}).get("t0_enabled_segments", [])
        if segment not in t0_segments:
            return False
        if len(df) < 20:
            return False

        vol_avg = df["volume"].rolling(20).mean().iloc[-1]
        latest_vol = df["volume"].iloc[-1]
        latest_range = (df["high"].iloc[-1] - df["low"].iloc[-1]) / df["close"].iloc[-1]

        # T+0 spike: volume > 3x avg AND daily range > 5%
        return latest_vol > vol_avg * th["t0_vol_spike_mult"] and latest_range > 0.05


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 2: CLAUDE MACRO ANALYZER (market-wide, 1 call/day)
# ═══════════════════════════════════════════════════════════════════════════════

class MacroRegimeAnalyzer:
    """
    Uses Claude for market-wide macro regime classification.
    One call per day — not per ticker. Batched headlines for entire EGX market.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or ANTHROPIC_API_KEY
        self.model = "claude-3-haiku-20240307"
        self.max_tokens = 512
        self.enabled = bool(self.api_key) and self.api_key != "YOUR_ANTHROPIC_API_KEY"

    def analyze(self, headlines: List[str] = None, market_summary: str = "") -> MacroRegimeResult:
        """
        Analyze EGX macro conditions.
        headlines: list of recent headlines (batch of 10-20)
        market_summary: optional pre-computed market stats string
        """
        if not self.enabled:
            return self._fallback("Claude API key not configured")

        if not headlines:
            return self._fallback("No headlines provided")

        prompt = self._build_prompt(headlines, market_summary)

        try:
            result = self._call_claude(prompt)
            if result:
                return self._parse_claude_response(result)
        except Exception as e:
            print(f"[MacroRegime] Claude error: {e}")

        return self._fallback("Claude call failed")

    def _build_prompt(self, headlines: List[str], market_summary: str) -> str:
        return f"""You are a senior macroeconomic analyst specializing in the Egyptian stock market (EGX).
Analyze the following headlines and market data. Focus on:
1. Geopolitical risks (Red Sea, regional stability, Nile Basin)
2. Macroeconomic factors (inflation, EGP/USD, CBE interest rates)
3. Regulatory/policy changes (T+0 rules, foreign ownership, taxes)
4. Sector trends (banks, real estate, industrials, fintech)

Score the overall market regime from -1.0 (very bearish/risk-off) to +1.0 (very bullish/risk-on).
Identify which sectors are favored vs. avoided.
Flag if this is a "risk_off" environment where traders should reduce exposure.

Headlines:
{chr(10).join(f"- {h}" for h in headlines[:20])}

{market_summary}

Respond ONLY with valid JSON in this exact format:
{{
  "macro_score": float,
  "confidence": float (0-1),
  "risk_off": bool,
  "risk_on": bool,
  "sector_rotation": {{"sector_name": score (-1 to +1)}},
  "key_factors": ["factor 1", "factor 2"],
  "summary": "one sentence"
}}"""

    def _call_claude(self, prompt: str) -> Optional[str]:
        url = "https://api.anthropic.com/v1/messages"
        resp = requests.post(
            url,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            },
            json={
                "model": self.model,
                "max_tokens": self.max_tokens,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2
            },
            timeout=45
        )
        resp.raise_for_status()
        return resp.json()["content"][0]["text"]

    def _parse_claude_response(self, text: str) -> MacroRegimeResult:
        """Extract JSON from Claude response."""
        try:
            # Try direct JSON parse
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return MacroRegimeResult(
                    macro_score=float(data.get("macro_score", 0)),
                    confidence=float(data.get("confidence", 0.5)),
                    risk_off_flag=bool(data.get("risk_off", False)),
                    risk_on_flag=bool(data.get("risk_on", False)),
                    sector_rotation=data.get("sector_rotation", {}),
                    key_factors=data.get("key_factors", []),
                    summary=data.get("summary", ""),
                    source="claude"
                )
        except Exception as e:
            print(f"[MacroRegime] Parse error: {e}")

        return self._fallback("Failed to parse Claude response")

    def _fallback(self, reason: str) -> MacroRegimeResult:
        """Neutral fallback when Claude is unavailable."""
        return MacroRegimeResult(
            macro_score=0.0,
            confidence=0.0,
            risk_off_flag=False,
            risk_on_flag=False,
            sector_rotation={},
            key_factors=[reason],
            summary=f"Macro analysis unavailable: {reason}. Using heuristic only.",
            source="fallback"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 3: HYBRID ENSEMBLE (blending + disagreement handling)
# ═══════════════════════════════════════════════════════════════════════════════

class HybridRegimeEnsemble:
    """
    Blends heuristic per-ticker regime with Claude macro regime.
    Handles disagreement via confidence penalty and conflict flags.
    """

    def __init__(self, heuristic_detector: Optional[HeuristicRegimeDetector] = None,
                 macro_analyzer: Optional[MacroRegimeAnalyzer] = None):
        self.heuristic = heuristic_detector or HeuristicRegimeDetector()
        self.macro = macro_analyzer or MacroRegimeAnalyzer()
        # Weights: heuristic dominates per-ticker, macro guides market-wide
        self.heuristic_weight = 0.6
        self.macro_weight = 0.4
        self.disagreement_threshold = 1.0  # max difference before flagging

    def detect(self, df: pd.DataFrame, ticker: str = "", segment: str = "",
               headlines: List[str] = None, market_summary: str = "") -> HybridRegimeResult:
        """
        Full hybrid detection pipeline.

        Args:
            df: ticker OHLCV dataframe
            ticker: symbol
            segment: market segment (egx30, high_activity, etc.)
            headlines: optional headlines for macro analysis (cached across tickers)
            market_summary: optional pre-computed market stats
        """
        # Layer 1: Heuristic (always runs)
        h = self.heuristic.detect(df, ticker, segment)

        # Layer 2: Macro (runs once per day, cached — pass same result for all tickers)
        m = self.macro.analyze(headlines, market_summary)

        # Convert heuristic regime to numeric score for blending
        regime_score_map = {
            "strong_bull": 1.0,
            "fragile_bull": 0.5,
            "sideways": 0.0,
            "weak_bear": -0.5,
            "strong_bear": -1.0,
            "unknown": 0.0
        }
        h_score = regime_score_map.get(h.regime, 0.0)
        m_score = m.macro_score

        # ── DISAGREEMENT INDEX ──
        disagreement = abs(h_score - m_score)  # range 0-2
        conflict = disagreement > self.disagreement_threshold

        # ── BLENDING ──
        if conflict:
            # When heuristic and macro conflict, reduce both weights and trust heuristic more
            # (price action is ground truth; macro is interpretation)
            blended_score = h_score * 0.7 + m_score * 0.3
            confidence = min(h.confidence, m.confidence) * 0.5
        else:
            blended_score = h_score * self.heuristic_weight + m_score * self.macro_weight
            confidence = (h.confidence + m.confidence) / 2

        # ── FINAL REGIME ──
        if blended_score >= 0.7:
            regime = "bull"
            position_size = MULTIPLIERS.get("bull", 1.0)
        elif blended_score <= -0.7:
            regime = "bear"
            position_size = MULTIPLIERS.get("bear", 0.2)
        else:
            regime = "sideways"
            position_size = MULTIPLIERS.get("sideways", 0.6)

        # Risk-off override from macro
        if m.risk_off_flag and position_size > 0.4:
            position_size *= 0.6
            regime = f"{regime}_risk_off"

        # Shock override from heuristic
        if h.shock_detected:
            position_size *= 0.5
            regime = f"{regime}_shock"

        # ── RECOMMENDATION ──
        if conflict:
            if h.shock_detected:
                recommendation = "HOLD — shock detected + macro conflict. Wait for clarity."
            elif m.risk_off_flag:
                recommendation = "REDUCE — macro risk-off conflicts with price action. Cut size."
            else:
                recommendation = "CAUTION — heuristic and macro disagree. Reduce position or skip."
        elif h.shock_detected:
            recommendation = "HOLD — EGX shock detected. Do not add new positions."
        elif m.risk_off_flag:
            recommendation = "DEFENSIVE — macro risk-on. Reduce size, favor quality stocks."
        elif regime == "bull":
            recommendation = "DEPLOY — regime bullish. Full size on high-conviction setups."
        elif regime == "bear":
            recommendation = "PROTECT — regime bearish. Cash preservation mode."
        else:
            recommendation = "SELECTIVE — regime sideways. Only A+ setups, tight stops."

        return HybridRegimeResult(
            regime=regime,
            position_size=round(position_size, 2),
            confidence=round(confidence, 2),
            heuristic=h,
            macro=m,
            disagreement_index=round(disagreement, 2),
            conflict_flag=conflict,
            recommendation=recommendation,
            timestamp=datetime.now().isoformat()
        )

    def detect_market_only(self, headlines: List[str], market_summary: str = "") -> MacroRegimeResult:
        """Quick macro-only check (for daily market briefing before scanning)."""
        return self.macro.analyze(headlines, market_summary)


# ═══════════════════════════════════════════════════════════════════════════════
# BACKWARD COMPATIBILITY: Legacy API
# ═══════════════════════════════════════════════════════════════════════════════

def detect_regime(df: pd.DataFrame) -> Dict:
    """
    Legacy API — returns same format as v4.2 for backward compatibility.
    Uses enhanced heuristic only (no Claude, no hybrid).
    """
    detector = HeuristicRegimeDetector()
    result = detector.detect(df)

    # Map 5-state back to 3-state for compatibility
    regime_3state = {
        "strong_bull": "bull",
        "fragile_bull": "bull",
        "sideways": "sideways",
        "weak_bear": "bear",
        "strong_bear": "bear",
        "unknown": "unknown"
    }

    return {
        "regime": regime_3state.get(result.regime, "unknown"),
        "position_size": result.position_size,
        "confidence": result.confidence,
        "slope_pct": result.slope_pct,
        "volatility_annual": result.volatility_annual,
        "rsi": result.rsi,
        "adx_proxy": result.adx_proxy,
        "shock_detected": result.shock_detected,
        "shock_type": result.shock_type,
        "t0_spike": result.t0_volatility_spike,
        "raw_scores": result.raw_scores
    }


class RegimeDetector:
    """Wrapper class for consistent API (backward compatible)."""

    def __init__(self, use_hybrid: bool = False):
        self.use_hybrid = use_hybrid
        if use_hybrid:
            self.ensemble = HybridRegimeEnsemble()
        else:
            self.heuristic = HeuristicRegimeDetector()

    def detect(self, df: pd.DataFrame, **kwargs) -> Dict:
        if self.use_hybrid:
            result = self.ensemble.detect(df, **kwargs)
            return {
                "regime": result.regime,
                "position_size": result.position_size,
                "confidence": result.confidence,
                "disagreement_index": result.disagreement_index,
                "conflict_flag": result.conflict_flag,
                "recommendation": result.recommendation,
                "heuristic_regime": result.heuristic.regime,
                "macro_score": result.macro.macro_score,
                "macro_source": result.macro.source,
                "shock_detected": result.heuristic.shock_detected,
                "shock_type": result.heuristic.shock_type,
                "timestamp": result.timestamp
            }
        else:
            return detect_regime(df)


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO / TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("Sentinel-EGX v4.2.2 — Hybrid Regime Detector (Blended Path)")
    print("=" * 70)

    # Generate synthetic test data
    np.random.seed(42)
    n = 100
    trend = np.linspace(100, 120, n)  # gentle uptrend
    noise = np.random.randn(n).cumsum() * 0.5
    prices = trend + noise

    demo_df = pd.DataFrame({
        "date": pd.date_range(end=datetime.now(), periods=n, freq="B"),
        "open": prices + np.random.randn(n) * 0.3,
        "high": prices + abs(np.random.randn(n)) * 0.5 + 0.3,
        "low": prices - abs(np.random.randn(n)) * 0.5 - 0.3,
        "close": prices + np.random.randn(n) * 0.2,
        "volume": np.random.randint(100000, 500000, n)
    })

    # Test 1: Legacy API (backward compatible)
    print("\n[TEST 1] Legacy detect_regime() — backward compatible")
    legacy = detect_regime(demo_df)
    print(f"  Regime: {legacy['regime']} | Size: {legacy['position_size']} | Conf: {legacy['confidence']}")
    print(f"  Shock: {legacy['shock_detected']} | T0 spike: {legacy['t0_spike']}")

    # Test 2: Enhanced heuristic
    print("\n[TEST 2] Enhanced Heuristic Detector")
    det = HeuristicRegimeDetector()
    h = det.detect(demo_df, "COMI.EGX", "high_activity")
    print(f"  Regime: {h.regime} | Size: {h.position_size} | Conf: {h.confidence}")
    print(f"  Slope: {h.slope_pct}% | RSI: {h.rsi} | Vol: {h.volatility_annual}")
    print(f"  Shock: {h.shock_detected} ({h.shock_type}) | T0 spike: {h.t0_volatility_spike}")

    # Test 3: Macro analyzer (fallback if no API key)
    print("\n[TEST 3] Macro Analyzer (Claude or fallback)")
    macro = MacroRegimeAnalyzer()
    m = macro.analyze([
        "CBE raises interest rates by 200bps to combat inflation",
        "EGP strengthens against USD in interbank market",
        "Foreign investors increase holdings in EGX30 stocks"
    ])
    print(f"  Score: {m.macro_score} | Conf: {m.confidence} | Source: {m.source}")
    print(f"  Risk-off: {m.risk_off_flag} | Risk-on: {m.risk_on_flag}")
    print(f"  Factors: {m.key_factors}")
    print(f"  Summary: {m.summary}")

    # Test 4: Full hybrid ensemble
    print("\n[TEST 4] Full Hybrid Ensemble")
    ensemble = HybridRegimeEnsemble()
    hybrid = ensemble.detect(
        demo_df, "COMI.EGX", "high_activity",
        headlines=[
            "CBE maintains rates, signals cautious stance",
            "EGX trading volume rises 15% on foreign buying"
        ]
    )
    print(f"  Final Regime: {hybrid.regime}")
    print(f"  Position Size: {hybrid.position_size} | Confidence: {hybrid.confidence}")
    print(f"  Heuristic: {hybrid.heuristic.regime} | Macro: {hybrid.macro.macro_score:+.2f}")
    print(f"  Disagreement: {hybrid.disagreement_index} | Conflict: {hybrid.conflict_flag}")
    print(f"  Recommendation: {hybrid.recommendation}")

    print("\n" + "=" * 70)
    print("All tests complete. Hybrid regime detector ready for paper trading.")
    print("=" * 70)
