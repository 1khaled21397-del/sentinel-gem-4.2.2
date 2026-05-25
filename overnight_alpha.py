"""
Sentinel-EGX v4.2 — Overnight Alpha Pipeline
=============================================
Full overnight pipeline: Data → Sentiment → Gap → ML → Skills → Technical → Alpha
NEW v4.2.2: DeltaCache integration for faster, cheaper data fetches.
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

with open("sentinel_config.json", "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

ALPHA_CFG = CONFIG.get("alpha_scorer", {})
ALPHA_WEIGHTS = ALPHA_CFG.get("weights", {})
MIN_ALPHA = ALPHA_CFG.get("min_alpha_to_report", 0.55)
TOP_N = ALPHA_CFG.get("top_n_setups", 10)


@dataclass
class OvernightAlphaResult:
    ticker: str
    alpha: float
    t0_eligible: bool
    setup: Dict
    gap: Dict
    technical: Dict
    sentiment: Optional[Dict] = None
    ml_forecast: Optional[Dict] = None
    skills: Optional[Dict] = None
    flow_sentiment: Optional[Dict] = None
    hybrid_regime: Optional[Dict] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class OvernightAlphaPipeline:
    """Full overnight alpha pipeline with all layers."""

    def __init__(self, tickers: List[str] = None):
        self.tickers = tickers or CONFIG.get("tickers", [])
        self.results = []

    def run_ticker(self, ticker: str) -> Optional[OvernightAlphaResult]:
        """Run full pipeline for a single ticker."""
        try:
            from data_engine import fetch_and_build, get_segment
            from technical_analysis import analyze_ticker
            from gap_predictor import predict_overnight_gap
            from ml_forecast import MLForecastEngine
            from sentiment_scraper import get_sentiment_for_ticker
            from auto_skills import analyze_skills
            from regime_detector import HybridRegimeEnsemble
        except ImportError as e:
            print(f"[OvernightAlpha] Import error: {e}")
            return None

        segment = get_segment(ticker)
        t0_eligible = segment in CONFIG.get("t0_rules", {}).get("t0_enabled_segments", [])

        # 1. Data (uses DeltaCache automatically if available)
        try:
            df = fetch_and_build(ticker, "EGX", lookback=400, use_cache=True, use_delta_cache=True)
        except Exception as e:
            print(f"[OvernightAlpha] Data fetch failed for {ticker}: {e}")
            return None

        # 2. Technical Analysis
        try:
            snap, setup = analyze_ticker(df, ticker, segment)
            technical = {
                "gemini_framework": {
                    "trend_score": snap.trend_score,
                    "volume_score": snap.volume_score,
                    "timing_score": snap.timing_score,
                    "composite": snap.gemini_framework_score,
                    "signal": "STRONG" if snap.gemini_framework_score > 0.7 else 
                             "MODERATE" if snap.gemini_framework_score > 0.5 else "WEAK"
                },
                "rsi": snap.rsi_14,
                "macd_state": snap.macd_state,
                "trend_direction": snap.trend_direction,
                "confluence": snap.confluence_score
            }
        except Exception as e:
            print(f"[OvernightAlpha] Technical analysis failed for {ticker}: {e}")
            technical = {}
            setup = None

        # 3. Gap Prediction
        try:
            gap = predict_overnight_gap(df, ticker, segment)
            gap_dict = {
                "direction": gap.gap_direction,
                "probability": gap.gap_probability,
                "expected_magnitude": gap.expected_magnitude,
                "t0_boost": gap.t0_liquidity_boost,
                "confidence": gap.confidence
            }
        except Exception as e:
            gap_dict = {"direction": "unknown", "probability": 0, "confidence": 0}

        # 4. ML Forecast
        try:
            ml = MLForecastEngine()
            ml.train(df)
            ml_pred = ml.predict(df, ticker)
            ml_dict = ml_pred or {}
        except Exception as e:
            ml_dict = {}

        # 5. Sentiment
        try:
            demo_headlines = [f"{ticker.split('.')[0]} market update"]
            sent = get_sentiment_for_ticker(ticker, demo_headlines)
            sentiment = {
                "score": sent.score,
                "confidence": sent.confidence,
                "source": sent.ai_source,
                "summary": sent.summary
            }
        except Exception:
            sentiment = None

        # 6. Auto Skills
        try:
            skills = analyze_skills(df, ticker, segment)
            skills_dict = {
                "composite": skills["composite_score"],
                "triggered": skills["skills_triggered"],
                "direction": skills["dominant_direction"]
            }
        except Exception:
            skills_dict = None

        # 7. Hybrid Regime
        try:
            ensemble = HybridRegimeEnsemble()
            hybrid = ensemble.detect(df, ticker, segment)
            regime_dict = {
                "regime": hybrid.regime,
                "position_size": hybrid.position_size,
                "confidence": hybrid.confidence,
                "heuristic_regime": hybrid.heuristic.regime,
                "macro_score": hybrid.macro.macro_score,
                "disagreement_index": hybrid.disagreement_index,
                "conflict_flag": hybrid.conflict_flag,
                "shock_detected": hybrid.heuristic.shock_detected,
                "shock_type": hybrid.heuristic.shock_type,
                "recommendation": hybrid.recommendation
            }
        except Exception:
            regime_dict = None

        # 8. Alpha Scoring
        alpha = self._compute_alpha(
            gap_dict, ml_dict, sentiment, skills_dict, technical,
            setup.__dict__ if setup else {}, regime_dict
        )

        if alpha < MIN_ALPHA:
            return None

        return OvernightAlphaResult(
            ticker=ticker,
            alpha=round(alpha, 3),
            t0_eligible=t0_eligible,
            setup=setup.__dict__ if setup else {"type": "unknown", "rr": 0},
            gap=gap_dict,
            technical=technical,
            sentiment=sentiment,
            ml_forecast=ml_dict,
            skills=skills_dict,
            hybrid_regime=regime_dict
        )

    def _compute_alpha(self, gap, ml, sentiment, skills, technical, setup, regime) -> float:
        """Composite alpha score from all pipeline layers."""
        score = 0.0

        # Gap
        if gap.get("probability", 0) > 0.6:
            score += ALPHA_WEIGHTS.get("gap_magnitude", 0.2) * gap["probability"]
            score += ALPHA_WEIGHTS.get("gap_confidence", 0.15) * gap.get("confidence", 0)

        # ML
        if ml and ml.get("target_return", 0) > 0:
            score += ALPHA_WEIGHTS.get("ml_7d", 0.15) * ml["target_return"] * 5
            score += ALPHA_WEIGHTS.get("ml_7d", 0.15) * ml.get("confidence", 0)

        # Sentiment
        if sentiment and sentiment.get("confidence", 0) > 0.3:
            score += ALPHA_WEIGHTS.get("sentiment", 0.1) * abs(sentiment["score"]) * sentiment["confidence"]

        # Skills
        if skills:
            score += ALPHA_WEIGHTS.get("auto_skills", 0.1) * skills.get("composite", 0)

        # Technical
        gemini = technical.get("gemini_framework", {})
        score += ALPHA_WEIGHTS.get("technical", 0.1) * gemini.get("composite", 0)

        # Weekly confluence
        score += ALPHA_WEIGHTS.get("weekly_confluence", 0.1) * abs(technical.get("confluence", 0))

        # S/R bonus
        if setup.get("entry_zone") and setup.get("stop_loss"):
            rr = setup.get("rr", 0)
            if rr >= 2.0:
                score += ALPHA_WEIGHTS.get("sr_bonus", 0.05)

        # Regime penalty/bonus
        if regime:
            if regime.get("conflict_flag"):
                score *= 0.7
            if regime.get("shock_detected"):
                score *= 0.5
            if "bear" in regime.get("regime", ""):
                score *= 0.6

        return min(1.0, max(0.0, score))

    def run(self, tickers: List[str] = None) -> List[Dict]:
        """Run pipeline on all tickers and return top setups."""
        targets = tickers or self.tickers
        results = []
        for ticker in targets:
            res = self.run_ticker(ticker)
            if res:
                results.append(res.__dict__)
        results.sort(key=lambda x: x["alpha"], reverse=True)
        return results[:TOP_N]


def run_pipeline(tickers: List[str] = None) -> List[Dict]:
    """Convenience wrapper."""
    pipeline = OvernightAlphaPipeline(tickers)
    return pipeline.run()


if __name__ == "__main__":
    print("OvernightAlpha v4.2.2 ready: Full pipeline with DeltaCache support")
