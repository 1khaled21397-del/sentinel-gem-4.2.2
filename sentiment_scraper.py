"""
Sentinel-EGX v4.2.1 — Sentiment Scraper
========================================
Keyword-based sentiment scoring + Dual AI (Claude + Kimi) sentiment analysis
NEW: HTTP-based AI calls (no SDK dependencies), config-driven weights, graceful fallback
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

# --- Sentiment Scraper Constants ---
MAX_CONFIDENCE = 1.0
SENTIMENT_CONFIDENCE_DIVISOR = 10.0
KEYWORD_CONFIDENCE_DIVISOR = 20.0
AI_SCORE_WEIGHT = 0.7
KEYWORD_SCORE_WEIGHT = 0.3
# -----------------------------------
NEUTRAL_KEYWORDS = [
    "maintain", "hold", "stable", "flat", "sideways", "consolidation", "await",
    "pending", "review", "monitor", "neutral", "mixed", "uncertain", "cautious"
]

# Load config for triple AI settings
CONFIG_PATH = "sentinel_config.json"
TRIPLE_AI_ENABLED = False
CLAUDE_WEIGHT = 0.4
KIMI_WEIGHT = 0.3
GEMINI_WEIGHT = 0.3
BATCH_SIZE = 20

try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    sentiment_cfg = cfg.get("sentiment", {})
    triple_ai_cfg = sentiment_cfg.get("triple_ai", {})
    TRIPLE_AI_ENABLED = triple_ai_cfg.get("enabled", False)
    CLAUDE_WEIGHT = triple_ai_cfg.get("claude_weight", 0.4)
    KIMI_WEIGHT = triple_ai_cfg.get("kimi_weight", 0.3)
    GEMINI_WEIGHT = triple_ai_cfg.get("gemini_weight", 0.3)
    BATCH_SIZE = triple_ai_cfg.get("batch_size", 20)
except Exception:
    pass

# API Keys
CLAUDE_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
KIMI_API_KEY = os.getenv("KIMI_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# Try Streamlit secrets
if not CLAUDE_API_KEY or not KIMI_API_KEY or not GEMINI_API_KEY:
    try:
        import streamlit as st
        CLAUDE_API_KEY = CLAUDE_API_KEY or st.secrets.get("ANTHROPIC_API_KEY", "").strip()
        KIMI_API_KEY = KIMI_API_KEY or st.secrets.get("KIMI_API_KEY", "").strip()
        GEMINI_API_KEY = GEMINI_API_KEY or st.secrets.get("GEMINI_API_KEY", "").strip()
    except Exception:
        pass


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

    raw_score = (bull - bear) / total if total > 0 else 0.0
    confidence = min(MAX_CONFIDENCE, total / SENTIMENT_CONFIDENCE_DIVISOR)

    return {
        "score": raw_score,
        "confidence": confidence,
        "bull": bull,
        "bear": bear,
        "neut": neut
    }


def _call_claude(headlines: List[str]) -> Optional[Dict]:
    """Call Claude API via HTTP for specialized Geopolitical & Macroeconomic Risk analysis."""
    if not CLAUDE_API_KEY:
        return None

    prompt = f"""You are a senior macroeconomic analyst specializing in the Egyptian stock market (EGX).
Analyze the following headlines, focusing specifically on Geopolitical risks, Macroeconomic factors (inflation, devaluation, interest rates), and Regulatory/Policy changes.
Assess how these broader themes impact long-term corporate outlooks.
Score the net sentiment from -1.0 (very bearish) to +1.0 (very bullish).
Provide a single JSON object with keys: score, confidence (0-1), summary (1 sentence).

Headlines:
{chr(10).join(f"- {h}" for h in headlines)}

Respond ONLY with valid JSON."""

    try:
        url = "https://api.anthropic.com/v1/messages"
        resp = requests.post(
            url,
            headers={
                "x-api-key": CLAUDE_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            },
            json={
                "model": "claude-haiku-4-5-20251001",
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
    """Call Kimi API via HTTP for specialized Retail Momentum & Short-Term Flow analysis."""
    if not KIMI_API_KEY:
        return None

    prompt = f"""You are a specialized momentum trader analyzing EGX retail investor behavior and trading flows.
Analyze the following headlines, focusing specifically on retail excitement, short-term momentum triggers, volume surges, chart breakout news, and trading activity/liquidity.
Score the short-term sentiment from -1.0 (very bearish) to +1.0 (very bullish).
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


def _call_gemini(headlines: List[str]) -> Optional[Dict]:
    """Call Gemini API via HTTP for specialized Earnings & Fundamental Growth analysis."""
    if not GEMINI_API_KEY:
        return None

    prompt = f"""You are a fundamental equity analyst specializing in EGX corporate performance.
Analyze the following headlines, focusing specifically on hard fundamental figures: earnings surprises, quarterly profits/losses, revenue growth, profit margin adjustments, dividends, mergers, acquisitions, and core business expansions.
Score the fundamental sentiment from -1.0 (very bearish) to +1.0 (very bullish).
Provide a single JSON object with keys: score, confidence (0-1), summary (1 sentence).

Headlines:
{chr(10).join(f"- {h}" for h in headlines)}

Respond ONLY with valid JSON."""

    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
        resp = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "responseMimeType": "application/json"
                }
            },
            timeout=30
        )
        resp.raise_for_status()
        res_json = resp.json()
        content = res_json["candidates"][0]["content"]["parts"][0]["text"]
        
        # Extract JSON from response
        json_match = re.search(r'\{[^}]*"score"[^}]*\}', content)
        if json_match:
            return json.loads(json_match.group())
        return None
    except Exception as e:
        print(f"[Sentiment] Gemini API error: {e}")
        return None


def _triple_ai_sentiment(headlines: List[str]) -> Optional[Dict]:
    """Combine Claude + Kimi + Gemini sentiment scores with config weights."""
    if not TRIPLE_AI_ENABLED:
        return None

    claude_result = _call_claude(headlines)
    kimi_result = _call_kimi(headlines)
    gemini_result = _call_gemini(headlines)

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

    if gemini_result and "score" in gemini_result:
        scores.append(gemini_result["score"] * GEMINI_WEIGHT)
        confs.append(gemini_result.get("confidence", 0.5) * GEMINI_WEIGHT)
        sources.append("gemini")

    if not scores:
        return None

    actual_weight = 0.0
    if "claude" in sources:
        actual_weight += CLAUDE_WEIGHT
    if "kimi" in sources:
        actual_weight += KIMI_WEIGHT
    if "gemini" in sources:
        actual_weight += GEMINI_WEIGHT

    ensemble_score = sum(scores) / actual_weight if actual_weight > 0 else 0
    ensemble_conf = sum(confs) / actual_weight if actual_weight > 0 else 0

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

        # Try triple AI first
        ai_sentiment = _triple_ai_sentiment(ticker_texts) if TRIPLE_AI_ENABLED else None

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
        keyword_confidence = min(MAX_CONFIDENCE, len(ticker_texts) / KEYWORD_CONFIDENCE_DIVISOR)

        # Blend AI + keyword if AI available
        if ai_sentiment:
            blended_score = ai_sentiment["score"] * AI_SCORE_WEIGHT + avg_score * KEYWORD_SCORE_WEIGHT
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
    print("Sentiment Scraper v4.2.1 ready: Keyword + Triple AI (Claude + Kimi + Gemini)")
    demo = batch_sentiment(
        ["COMI", "FWRY"],
        {
            "COMI": ["COMI reports record profits, beats estimates", "Foreign buying surge in COMI shares"],
            "FWRY": ["FWRY expansion into new markets announced", "FWRY partnership with global tech firm"]
        }
    )
    for t, r in demo.items():
        print(f"{t}: score={r.score} | conf={r.confidence} | AI={r.ai_source} | {r.summary}")
