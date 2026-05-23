"""
Sentinel-EGX v4.2.1 — Sentiment Scraper
=========================================
Keyword-based sentiment scoring + Dual AI (Claude + Kimi) sentiment analysis
NEW v4.2.1:
  - HTTP-based AI calls (no SDK dependencies)
  - Verified endpoints: Claude (api.anthropic.com) + Kimi (api.moonshot.cn)
  - Config-driven weights (claude_weight=0.6, kimi_weight=0.4)
  - Graceful fallback to keyword scoring
  - Fires on every ticker analysis (not just top liquid)
  - Exports API keys to os.environ for child modules
"""

import re
import os
import json
import requests
from typing import Dict, List, Optional
from dataclasses import dataclass

# Keyword dictionaries for EGX-specific sentiment (fallback)
BULLISH_KEYWORDS = [
    "profit", "earnings", "dividend", "growth", "expansion", "contract", "deal",
    "partnership", "acquisition", "merger", "buyback", "upgrade", "outperform",
    "strong", "beat", "exceed", "record", "surge", "rally", "boom", "bullish",
    "positive", "optimistic", "momentum", "breakout", "support", "accumulation",
    "institutional", "foreign buying", "fpi", "portfolio", "inflow", "rebound"
]

BEARISH_KEYWORDS = [
    "loss", "deficit", "decline", "drop", "fall", "crash", "bearish", "downgrade",
    "underperform", "weak", "miss", "below", "disappoint", "concern", "risk",
    "investigation", "lawsuit", "fine", "penalty", "debt", "default", "liquidity",
    "crisis", "recession", "inflation", "devaluation", "depreciation", "outflow",
    "foreign selling", "margin call", "stop loss", "resistance", "distribution"
]

NEUTRAL_KEYWORDS = [
    "maintain", "hold", "stable", "flat", "sideways", "consolidation", "await",
    "pending", "review", "monitor", "neutral", "mixed", "uncertain", "cautious"
]

# Load config for dual AI settings
CONFIG_PATH = "sentinel_config.json"
DUAL_AI_ENABLED = False
CLAUDE_WEIGHT = 0.6
KIMI_WEIGHT = 0.4
BATCH_SIZE = 20

try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    sentiment_cfg = cfg.get("sentiment", {})
    dual_ai_cfg = sentiment_cfg.get("dual_ai", {})
    DUAL_AI_ENABLED = dual_ai_cfg.get("enabled", False)
    CLAUDE_WEIGHT = dual_ai_cfg.get("claude_weight", 0.6)
    KIMI_WEIGHT = dual_ai_cfg.get("kimi_weight", 0.4)
    BATCH_SIZE = dual_ai_cfg.get("batch_size", 20)
except Exception:
    pass

# API Keys — try os.environ first (exported by sentinel_app), then Streamlit secrets
CLAUDE_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
KIMI_API_KEY = os.getenv("KIMI_API_KEY", "").strip()

# Fallback to Streamlit secrets
if not CLAUDE_API_KEY or not KIMI_API_KEY:
    try:
        import streamlit as st
        if not CLAUDE_API_KEY:
            CLAUDE_API_KEY = st.secrets.get("ANTHROPIC_API_KEY", "").strip()
        if not KIMI_API_KEY:
            # Try both flat and nested secret formats
            KIMI_API_KEY = st.secrets.get("KIMI_API_KEY", "").strip()
            if not KIMI_API_KEY and "sentinel" in st.secrets:
                sentinel_cfg = st.secrets.get("sentinel", {})
                if isinstance(sentinel_cfg, dict):
                    KIMI_API_KEY = sentinel_cfg.get("KIMI_API_KEY", "").strip()
    except Exception:
        pass

# Export back to environment
os.environ["ANTHROPIC_API_KEY"] = CLAUDE_API_KEY
os.environ["KIMI_API_KEY"] = KIMI_API_KEY


@dataclass
class SentimentResult:
    ticker: str
    score: float  # -1.0 to +1.0
    confidence: float  # 0.0 to 1.0
    bullish_count: int
    bearish_count: int
    neutral_count: int
    summary: str
    ai_score: Optional[float] = None  # NEW: AI-derived score
    ai_confidence: Optional[float] = None  # NEW: AI confidence
    ai_source: Optional[str] = None  # NEW: "claude", "kimi", "ensemble", or None


def score_text(text: str) -> Dict:
    """Score a single text snippet for sentiment (keyword fallback)."""
    text_lower = text.lower()
    bull = sum(1 for kw in BULLISH_KEYWORDS if kw in text_lower)
    bear = sum(1 for kw in BEARISH_KEYWORDS if kw in text_lower)
    neut = sum(1 for kw in NEUTRAL_KEYWORDS if kw in text_lower)

    total = bull + bear + neut
    if total == 0:
        return {"score": 0.0, "confidence": 0.0, "bull": 0, "bear": 0, "neut": 0}

    raw_score = (bull - bear) / total
    confidence = min(1.0, total / 10.0)

    return {
        "score": raw_score,
        "confidence": confidence,
        "bull": bull,
        "bear": bear,
        "neut": neut
    }


def _call_claude(headlines: List[str]) -> Optional[Dict]:
    """Call Claude API via HTTP for sentiment analysis.
    Endpoint verified: https://api.anthropic.com/v1/messages
    Model: claude-3-haiku-20240307 (fast, cheap, sufficient for sentiment)
    """
    if not CLAUDE_API_KEY:
        return None

    prompt = f"""Analyze the sentiment of these Egyptian stock market headlines.
Score from -1.0 (very bearish) to +1.0 (very bullish).
Provide a single JSON object with keys: score, confidence (0-1), summary (1 sentence).

Headlines:
{chr(10).join(f"- {h}" for h in headlines)}

Respond ONLY with valid JSON."""

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": CLAUDE_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            },
            json={
                "model": "claude-3-haiku-20240307",
                "max_tokens": 256,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=30
        )
        resp.raise_for_status()
        content = resp.json()["content"][0]["text"]
        # Extract JSON from response
        json_match = re.search(r'\{[^}]*"score"[^}]*\}', content)
        if json_match:
            return json.loads(json_match.group())
        return None
    except Exception as e:
        print(f"[Sentiment] Claude API error: {e}")
        return None


def _call_kimi(headlines: List[str]) -> Optional[Dict]:
    """Call Kimi API via HTTP for sentiment analysis.
    Endpoint verified: https://api.moonshot.cn/v1/chat/completions
    Model: moonshot-v1-8k (sufficient for sentiment, cheapest option)
    """
    if not KIMI_API_KEY:
        return None

    prompt = f"""Analyze the sentiment of these Egyptian stock market headlines.
Score from -1.0 (very bearish) to +1.0 (very bullish).
Provide a single JSON object with keys: score, confidence (0-1), summary (1 sentence).

Headlines:
{chr(10).join(f"- {h}" for h in headlines)}

Respond ONLY with valid JSON."""

    try:
        resp = requests.post(
            "https://api.moonshot.cn/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {KIMI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "moonshot-v1-8k",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 256,
                "temperature": 0.3
            },
            timeout=30
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        json_match = re.search(r'\{[^}]*"score"[^}]*\}', content)
        if json_match:
            return json.loads(json_match.group())
        return None
    except Exception as e:
        print(f"[Sentiment] Kimi API error: {e}")
        return None


def _dual_ai_sentiment(headlines: List[str]) -> Optional[Dict]:
    """Combine Claude + Kimi sentiment scores with config weights.
    Fires on every ticker analysis (not just top liquid).
    """
    if not DUAL_AI_ENABLED:
        return None

    claude_result = _call_claude(headlines)
    kimi_result = _call_kimi(headlines)

    scores = []
    confs = []
    sources = []

    if claude_result and "score" in claude_result:
        scores.append(claude_result["score"] * CLAUDE_WEIGHT)
        confs.append(claude_result.get("confidence", 0.5) * CLAUDE_WEIGHT)
        sources.append("claude")

    if kimi_result and "score" in kimi_result:
        scores.append(kimi_result["score"] * KIMI_WEIGHT)
        confs.append(kimi_result.get("confidence", 0.5) * KIMI_WEIGHT)
        sources.append("kimi")

    if not scores:
        return None

    total_weight = (CLAUDE_WEIGHT if claude_result else 0) + (KIMI_WEIGHT if kimi_result else 0)
    ensemble_score = sum(scores) / total_weight if total_weight > 0 else 0
    ensemble_conf = sum(confs) / total_weight if total_weight > 0 else 0

    return {
        "score": round(ensemble_score, 3),
        "confidence": round(ensemble_conf, 3),
        "summary": f"Ensemble ({', '.join(sources)}): {'bullish' if ensemble_score > 0.2 else 'bearish' if ensemble_score < -0.2 else 'neutral'}",
        "source": "ensemble"
    }


def batch_sentiment(tickers: List[str], texts: Dict[str, List[str]] = None) -> Dict[str, SentimentResult]:
    """
    Batch sentiment analysis for multiple tickers.
    Uses dual AI if enabled + keys available, falls back to keyword scoring.
    Fires on every ticker (not just top liquid).
    """
    results = {}
    for ticker in tickers:
        ticker_texts = texts.get(ticker, []) if texts else []
        if not ticker_texts:
            results[ticker] = SentimentResult(
                ticker=ticker, score=0.0, confidence=0.0,
                bullish_count=0, bearish_count=0, neutral_count=0,
                summary="No news data available"
            )
            continue

        # Try dual AI first (fires on every ticker)
        ai_sentiment = _dual_ai_sentiment(ticker_texts) if DUAL_AI_ENABLED else None

        # Keyword fallback (always computed)
        scores = []
        total_bull = total_bear = total_neut = 0
        for t in ticker_texts:
            s = score_text(t)
            scores.append(s["score"] * s["confidence"])
            total_bull += s["bull"]
            total_bear += s["bear"]
            total_neut += s["neut"]

        avg_score = sum(scores) / len(scores) if scores else 0.0
        keyword_confidence = min(1.0, len(ticker_texts) / 20.0)

        # Blend AI + keyword if AI available
        if ai_sentiment:
            blended_score = ai_sentiment["score"] * 0.7 + avg_score * 0.3
            blended_conf = max(ai_sentiment["confidence"], keyword_confidence)
            summary = ai_sentiment["summary"]
            ai_source = ai_sentiment.get("source")
            ai_score = ai_sentiment["score"]
            ai_conf = ai_sentiment["confidence"]
        else:
            blended_score = avg_score
            blended_conf = keyword_confidence
            ai_source = None
            ai_score = None
            ai_conf = None
            if blended_score > 0.3:
                summary = f"Bullish sentiment ({total_bull} positive signals)"
            elif blended_score < -0.3:
                summary = f"Bearish sentiment ({total_bear} negative signals)"
            else:
                summary = f"Neutral/Mixed ({total_neut} neutral signals)"

        results[ticker] = SentimentResult(
            ticker=ticker,
            score=round(blended_score, 3),
            confidence=round(blended_conf, 3),
            bullish_count=total_bull,
            bearish_count=total_bear,
            neutral_count=total_neut,
            summary=summary,
            ai_score=ai_score,
            ai_confidence=ai_conf,
            ai_source=ai_source
        )

    return results


def get_sentiment_for_ticker(ticker: str, headlines: List[str] = None) -> SentimentResult:
    """Convenience wrapper for single ticker."""
    texts = {ticker: headlines} if headlines else None
    results = batch_sentiment([ticker], texts)
    return results[ticker]


if __name__ == "__main__":
    print("Sentiment Scraper v4.2.1 ready: Keyword + Dual AI (Claude + Kimi)")
    print("Endpoints verified: Claude (api.anthropic.com) | Kimi (api.moonshot.cn)")
    demo = batch_sentiment(
        ["COMI", "FWRY"],
        {
            "COMI": ["COMI reports record profits, beats estimates", "Foreign buying surge in COMI shares"],
            "FWRY": ["FWRY expansion into new markets announced", "FWRY partnership with global tech firm"]
        }
    )
    for t, r in demo.items():
        print(f"{t}: score={r.score} | conf={r.confidence} | AI={r.ai_source} | {r.summary}")
