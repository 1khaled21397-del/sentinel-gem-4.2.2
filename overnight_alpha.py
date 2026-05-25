"""
Sentinel-EGX v4.2.2 — Overnight Alpha Pipeline
================================================
Full integration: Data → Sentiment → Gap → ML → Skills → Technical → Alpha → Regime
NEW v4.2.2: Hybrid Regime Detector (heuristic + Claude macro + disagreement handling)
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass, field

with open("sentinel_config.json", "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

ALPHA_CFG = CONFIG.get("alpha_scorer", {})
WEIGHTS = ALPHA_CFG.get("weights", {})
MIN_ALPHA = ALPHA_CFG.get("min_alpha_to_report", 0.55)
TOP_N = ALPHA_CFG.get("top_n_setups", 10)
RISK_MULT = ALPHA_CFG.get("risk_penalty_multiplier", 1.0)
FLOW_CFG = CONFIG.get("flow_sentiment", {})


@dataclass
class AlphaResult:
    ticker: str
    alpha: float
    t0_eligible: bool
    gap: Dict
    ml: Dict
    sentiment: Dict
    flow_sentiment: Dict
    skills: Dict
    technical: Dict
    setup: Dict
    regime: str
    # NEW v4.2.2: Hybrid regime fields
    hybrid_regime: Dict = field(default_factory=dict)


class OvernightAlphaPipeline:
    """End-to-end overnight alpha scoring pipeline with flow sentiment + hybrid regime."""

    def __init__(self, tickers: List[str], config_path: str = "sentinel_config.json"):
        self.tickers = tickers
        self.config = CONFIG
        self.results: List[AlphaResult] = []

        # Pre-fetch market-wide flow sentiment (cached for all tickers)
        self.market_flow = self._fetch_market_flow()

        # NEW v4.2.2: Initialize hybrid regime ensemble
        try:
            from regime_detector_v2 import HybridRegimeEnsemble
            self.ensemble = HybridRegimeEnsemble()
            self.hybrid_regime_available = True
        except ImportError as e:
            print(f"[OvernightAlpha] regime_detector_v2 not available: {e}")
            self.ensemble = None
            self.hybrid_regime_available = False

        # NEW v4.2.2: Cache macro analysis across tickers (one Claude call/day)
        self._cached_macro = None
        self._cached_headlines = None

    def _fetch_market_flow(self) -> Dict:
        """Fetch EGX.com flow sentiment once per pipeline run."""
        if not FLOW_CFG.get("enabled", True):
            return {"score": 0, "confidence": 0, "note": "Flow sentiment disabled"}

        try:
            from egx_flow_scraper import get_flow_sentiment_for_alpha
            score, conf, meta = get_flow_sentiment_for_alpha()
            return {"score": score, "confidence": conf, "meta": meta}
        except Exception as e:
            if FLOW_CFG.get("fallback_if_down", True):
                return {"score": 0, "confidence": 0, "note": f"Flow fetch failed: {e}, using neutral"}
            return {"score": 0, "confidence": 0, "note": f"Flow fetch failed: {e}"}

    # NEW v4.2.2: Cache macro analysis across all tickers
    def _get_macro(self, headlines: List[str] = None) -> Optional[Dict]:
        """Cache macro analysis across all tickers (one Claude call per pipeline run)."""
        if not self.hybrid_regime_available or self.ensemble is None:
            return None

        # Use cached macro if headlines haven't changed
        if self._cached_macro is not None and headlines == self._cached_headlines:
            return self._cached_macro

        self._cached_headlines = headlines
        macro_result = self.ensemble.macro.analyze(headlines)
        self._cached_macro = {
            "macro_score": macro_result.macro_score,
            "confidence": macro_result.confidence,
            "risk_off": macro_result.risk_off_flag,
            "risk_on": macro_result.risk_on_flag,
            "sector_rotation": macro_result.sector_rotation,
            "key_factors": macro_result.key_factors,
            "summary": macro_result.summary,
            "source": macro_result.source
        }
        return self._cached_macro

    def run_ticker(self, ticker: str, headlines: List[str] = None) -> Optional[AlphaResult]:
        """Run full pipeline for a single ticker."""
        try:
            from data_engine import fetch_and_build, get_segment
            from technical_analysis import analyze_ticker, get_indicator_summary
            from auto_skills import analyze_skills
            from gap_predictor import predict_overnight_gap
            from ml_forecast import MLForecastEngine
            from sentiment_scraper import get_sentiment_for_ticker
            from regime_detector import detect_regime  # Legacy fallback
        except ImportError as e:
            print(f"[OvernightAlpha] Import error: {e}")
            return None

        segment = get_segment(ticker)
        t0_eligible = segment in self.config.get("t0_rules", {}).get("t0_enabled_segments", [])

        # 1. Data
        try:
            df = fetch_and_build(ticker, "EGX", lookback=400, use_cache=True)
        except Exception as e:
            print(f"[OvernightAlpha] Data fetch failed for {ticker}: {e}")
            return None

        # 2. Technical Analysis
        try:
            snap, setup = analyze_ticker(df, ticker, segment)
            tech_summary = get_indicator_summary(snap)
        except Exception as e:
            print(f"[OvernightAlpha] TA failed for {ticker}: {e}")
            return None

        # 3. Auto Skills
        try:
            skills = analyze_skills(df, ticker, segment)
        except Exception as e:
            skills = {"composite_score": 0, "skills_triggered": 0}

        # 4. Gap Prediction
        try:
            gap = predict_overnight_gap(df, ticker, segment)
            gap_dict = {
                "direction": gap.gap_direction,
                "probability": gap.gap_probability,
                "magnitude": gap.expected_magnitude,
                "t0_boost": gap.t0_liquidity_boost
            }
        except Exception as e:
            gap_dict = {"direction": "neutral", "probability": 0, "magnitude": 0, "t0_boost": 1.0}

        # 5. ML Forecast
        try:
            ml_engine = MLForecastEngine()
            ml_engine.train(df)
            ml_pred = ml_engine.predict(df)
            ml_dict = {
                "target_return": ml_pred.get("target_return", 0),
                "confidence": ml_pred.get("confidence", 0),
                "xgb": ml_pred.get("xgb_pred", 0),
                "rf": ml_pred.get("rf_pred", 0)
            }
        except Exception as e:
            ml_dict = {"target_return": 0, "confidence": 0, "xgb": 0, "rf": 0}

        # 6. News Sentiment
        try:
            sentiment = get_sentiment_for_ticker(ticker)
            sent_dict = {
                "score": sentiment.score,
                "confidence": sentiment.confidence,
                "summary": sentiment.summary
            }
        except Exception:
            sent_dict = {"score": 0, "confidence": 0, "summary": "No data"}

        # 7. EGX Flow Sentiment (market-wide, same for all tickers)
        flow_dict = self.market_flow.copy()
        flow_score = flow_dict.get("score", 0)
        flow_conf = flow_dict.get("confidence", 0)

        # ═══════════════════════════════════════════════════════════════════════
        # 8. Regime — NEW v4.2.2: Hybrid (heuristic + Claude macro)
        # ═══════════════════════════════════════════════════════════════════════
        regime_str = "unknown"
        position_mult = 1.0
        hybrid_regime_dict = {}

        if self.hybrid_regime_available and self.ensemble is not None:
            try:
                # Fetch cached macro (one call across all tickers)
                macro_cached = self._get_macro(headlines)

                # Run full hybrid detection
                hybrid = self.ensemble.detect(
                    df,
                    ticker=ticker,
                    segment=segment,
                    headlines=headlines  # triggers cache hit after first ticker
                )

                regime_str = hybrid.regime
                position_mult = hybrid.position_size

                # Build serializable hybrid regime dict
                hybrid_regime_dict = {
                    "regime": hybrid.regime,
                    "position_size": hybrid.position_size,
                    "confidence": hybrid.confidence,
                    "heuristic_regime": hybrid.heuristic.regime,
                    "macro_score": hybrid.macro.macro_score,
                    "macro_source": hybrid.macro.source,
                    "disagreement_index": hybrid.disagreement_index,
                    "conflict_flag": hybrid.conflict_flag,
                    "recommendation": hybrid.recommendation,
                    "shock_detected": hybrid.heuristic.shock_detected,
                    "shock_type": hybrid.heuristic.shock_type,
                    "t0_spike": hybrid.heuristic.t0_volatility_spike,
                    "timestamp": hybrid.timestamp
                }

                # Log for paper trading review
                print(f"[{ticker}] Regime: {regime_str} | "
                      f"H:{hybrid.heuristic.regime} M:{hybrid.macro.macro_score:+.2f} | "
                      f"Disagree: {hybrid.disagreement_index} | "
                      f"Rec: {hybrid.recommendation}")

                # If conflict flag, reduce position size further
                if hybrid.conflict_flag:
                    position_mult *= 0.5
                    print(f"[{ticker}] ⚠️ CONFLICT detected — position size reduced to {position_mult:.0%}")

                # If shock detected, log warning
                if hybrid.heuristic.shock_detected:
                    print(f"[{ticker}] 🚨 SHOCK: {hybrid.heuristic.shock_type} — defensive posture")

            except Exception as e:
                print(f"[OvernightAlpha] Hybrid regime failed for {ticker}: {e}")
                # Fallback to legacy regime detector
                try:
                    regime = detect_regime(df)
                    regime_str = regime.get("regime", "unknown")
                    position_mult = regime.get("position_size", 1.0)
                except Exception:
                    regime_str = "unknown"
                    position_mult = 1.0
        else:
            # Fallback to legacy regime detector
            try:
                regime = detect_regime(df)
                regime_str = regime.get("regime", "unknown")
                position_mult = regime.get("position_size", 1.0)
            except Exception:
                regime_str = "unknown"
                position_mult = 1.0

        # 9. Alpha Score (with flow sentiment)
        gap_score = gap_dict["magnitude"] * gap_dict["probability"]
        gap_conf = gap_dict["probability"]
        ml_score = max(0, ml_dict["target_return"]) * ml_dict["confidence"]
        sent_score = abs(sent_dict["score"]) * sent_dict["confidence"]
        skills_score = skills.get("composite_score", 0)
        tech_score = snap.gemini_framework_score
        weekly_score = max(0, snap.confluence_score)
        sr_bonus = 1.0 if abs(snap.pivot - snap.vwap_20d) / snap.pivot < 0.02 else 0.5

        # Flow sentiment: apply only if confidence is sufficient
        flow_alpha = 0
        if flow_conf > 0.3:
            flow_alpha = flow_score * flow_conf

        alpha = (
            WEIGHTS.get("gap_magnitude", 0.2) * gap_score +
            WEIGHTS.get("gap_confidence", 0.15) * gap_conf +
            WEIGHTS.get("ml_7d", 0.15) * ml_score +
            WEIGHTS.get("sentiment", 0.10) * sent_score +
            WEIGHTS.get("auto_skills", 0.1) * skills_score +
            WEIGHTS.get("technical", 0.1) * tech_score +
            WEIGHTS.get("weekly_confluence", 0.1) * weekly_score +
            WEIGHTS.get("sr_bonus", 0.05) * sr_bonus +
            WEIGHTS.get("flow_sentiment", 0.05) * flow_alpha
        ) * position_mult * RISK_MULT

        # Setup dict
        setup_dict = {
            "type": setup.setup_type,
            "entry": setup.entry_zone,
            "stop": setup.stop_loss,
            "targets": setup.targets,
            "rr": snap.risk_reward,
            "rationale": setup.rationale,
            "best_session": setup.best_session
        }

        return AlphaResult(
            ticker=ticker,
            alpha=round(alpha, 3),
            t0_eligible=t0_eligible,
            gap=gap_dict,
            ml=ml_dict,
            sentiment=sent_dict,
            flow_sentiment=flow_dict,
            skills=skills,
            technical=tech_summary,
            setup=setup_dict,
            regime=regime_str,
            hybrid_regime=hybrid_regime_dict  # NEW v4.2.2
        )

    def run(self, tickers: List[str] = None, headlines: List[str] = None) -> List[AlphaResult]:
        """Run pipeline across ticker universe."""
        universe = tickers or self.tickers
        results = []

        for ticker in universe:
            res = self.run_ticker(ticker, headlines=headlines)
            if res and res.alpha >= MIN_ALPHA:
                results.append(res)

        results.sort(key=lambda x: x.alpha, reverse=True)
        return results[:TOP_N]


def run_pipeline(tickers: List[str] = None, headlines: List[str] = None) -> List[Dict]:
    """Public API: run overnight alpha pipeline and return serializable results."""
    if tickers is None:
        tickers = CONFIG.get("tickers", [])[:50]

    pipeline = OvernightAlphaPipeline(tickers)
    results = pipeline.run(headlines=headlines)

    return [
        {
            "ticker": r.ticker,
            "alpha": r.alpha,
            "t0_eligible": r.t0_eligible,
            "gap": r.gap,
            "ml": r.ml,
            "sentiment": r.sentiment,
            "flow_sentiment": r.flow_sentiment,
            "skills": {
                "composite": r.skills.get("composite_score", 0),
                "triggered": r.skills.get("skills_triggered", 0)
            },
            "setup": r.setup,
            "regime": r.regime,
            "hybrid_regime": r.hybrid_regime  # NEW v4.2.2
        }
        for r in results
    ]


if __name__ == "__main__":
    print("OvernightAlpha v4.2.2 ready: Full pipeline + Hybrid Regime Detector")
    print("Features: Heuristic per-ticker + Claude macro (1 call/day) + Disagreement handling")
