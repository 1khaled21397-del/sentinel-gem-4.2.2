"""
Sentinel-EGX v4.2.2 — Overnight Alpha Pipeline
=============================================
Full overnight pipeline: Data → Sentiment → Gap → ML → Skills → Technical → Alpha
NEW v4.2.2: DeltaCache integration for faster, cheaper data fetches.
NEW v4.2.2: sentinel_learning integration — regime injection + forecast logging.
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

with open("sentinel_config.json", "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

ALPHA_CFG    = CONFIG.get("alpha_scorer", {})
ALPHA_WEIGHTS = ALPHA_CFG.get("weights", {})
# --- Alpha Scorer Constants ---
ML_RETURN_MULTIPLIER = 5
MIN_RR_FOR_BONUS     = 2.0
CONFLICT_PENALTY     = 0.7
SHOCK_PENALTY        = 0.5
BEAR_PENALTY         = 0.6
# ------------------------------
MIN_ALPHA = ALPHA_CFG.get("min_alpha_to_report", 0.55)
TOP_N     = ALPHA_CFG.get("top_n_setups", 10)

# --- Self-Learning integration (optional, fails silently if module absent) ---
try:
    import sentinel_learning as _learning
    LEARNING_AVAILABLE = True
except ImportError:
    LEARNING_AVAILABLE = False


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
    learning_regime: Optional[str] = None          # NEW: regime label from sentinel_learning
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class OvernightAlphaPipeline:
    """Full overnight alpha pipeline with all layers."""

    def __init__(self, tickers: List[str] = None):
        self.tickers = tickers or CONFIG.get("tickers", [])
        self.results = []

    # ------------------------------------------------------------------
    # SINGLE-TICKER PIPELINE
    # ------------------------------------------------------------------

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

        segment    = get_segment(ticker)
        t0_eligible = segment in CONFIG.get("t0_rules", {}).get("t0_enabled_segments", [])

        # 1. Data (DeltaCache if available)
        try:
            df = fetch_and_build(ticker, "EGX", lookback=400, use_cache=True, use_delta_cache=True)
        except Exception as e:
            print(f"[OvernightAlpha] Data fetch failed for {ticker}: {e}")
            return None

        # 2. Technical Analysis
        snap   = None
        setup  = None
        try:
            snap, setup = analyze_ticker(df, ticker, segment)
            technical = {
                "gemini_framework": {
                    "trend_score":  snap.trend_score,
                    "volume_score": snap.volume_score,
                    "timing_score": snap.timing_score,
                    "composite":    snap.gemini_framework_score,
                    "signal": ("STRONG"   if snap.gemini_framework_score > 0.7 else
                               "MODERATE" if snap.gemini_framework_score > 0.5 else "WEAK"),
                },
                "rsi":             snap.rsi_14,
                "macd_state":      snap.macd_state,
                "trend_direction": snap.trend_direction,
                "confluence":      snap.confluence_score,
            }
        except Exception as e:
            print(f"[OvernightAlpha] Technical analysis failed for {ticker}: {e}")
            technical = {}

        # 3. Gap Prediction
        try:
            gap    = predict_overnight_gap(df, ticker, segment)
            gap_dict = {
                "direction":        gap.gap_direction,
                "probability":      gap.gap_probability,
                "expected_magnitude": gap.expected_magnitude,
                "t0_boost":         gap.t0_liquidity_boost,
                "confidence":       gap.confidence,
            }
        except Exception:
            gap_dict = {"direction": "unknown", "probability": 0, "confidence": 0}

        # 4. ML Forecast
        try:
            ml     = MLForecastEngine()
            ml.train(df)
            ml_pred = ml.predict(df, ticker)
            ml_dict = ml_pred or {}
        except Exception:
            ml_dict = {}

        # 5. Sentiment
        try:
            demo_headlines = [f"{ticker.split('.')[0]} market update"]
            sent     = get_sentiment_for_ticker(ticker, demo_headlines)
            sentiment = {
                "score":      sent.score,
                "confidence": sent.confidence,
                "source":     sent.ai_source,
                "summary":    sent.summary,
            }
        except Exception:
            sentiment = None

        # 6. Auto Skills
        skills      = None
        skills_dict = None
        try:
            skills = analyze_skills(df, ticker, segment)
            skills_dict = {
                "composite":  skills["composite_score"],
                "triggered":  skills["skills_triggered"],
                "direction":  skills["dominant_direction"],
            }
        except Exception:
            pass

        # 6.5  Self-Learning — regime injection
        # inject_regime_weights() classifies the current market regime and returns
        # RULES updated with regime-specific adaptive weights accumulated over time.
        # Fails silently; learning_regime falls back to "unknown" if module absent.
        learning_regime  = "unknown"
        adaptive_rules   = None
        if LEARNING_AVAILABLE and snap is not None:
            try:
                close_val     = float(df["close"].iloc[-1])
                atr_pct_val   = (snap.atr_14 / close_val * 100) if close_val > 0 else 0.0
                obv_norm      = (float(snap.obv_slope) / max(abs(float(snap.obv or 1)), 1.0)
                                 if snap.obv else 0.0)
                skills_list   = ([{"id": d["skill"]}
                                   for d in (skills or {}).get("triggered_details", [])]
                                 if skills else [])
                w_conf        = {"aligned": snap.confluence_score > 0}

                learning_regime, adaptive_rules = _learning.inject_regime_weights(
                    skills_list, df, w_conf, atr_pct_val, obv_norm,
                    CONFIG.get("rules", {}),
                )
            except Exception:
                pass

        # 7. Hybrid Regime (HybridRegimeEnsemble — unchanged)
        regime_dict = None
        try:
            ensemble    = HybridRegimeEnsemble()
            hybrid      = ensemble.detect(df, ticker, segment)
            regime_dict = {
                "regime":             hybrid.regime,
                "position_size":      hybrid.position_size,
                "confidence":         hybrid.confidence,
                "heuristic_regime":   hybrid.heuristic.regime,
                "macro_score":        hybrid.macro.macro_score,
                "disagreement_index": hybrid.disagreement_index,
                "conflict_flag":      hybrid.conflict_flag,
                "shock_detected":     hybrid.heuristic.shock_detected,
                "shock_type":         hybrid.heuristic.shock_type,
                "recommendation":     hybrid.recommendation,
            }
        except Exception:
            pass

        # 8. Alpha Scoring
        # When adaptive_rules are available, use regime-specific weekly-confluence weight.
        effective_wc_weight = (
            float(adaptive_rules.get("_w_weekly", ALPHA_WEIGHTS.get("weekly_confluence", 0.1)))
            if adaptive_rules else ALPHA_WEIGHTS.get("weekly_confluence", 0.1)
        )
        alpha = self._compute_alpha(
            gap_dict, ml_dict, sentiment, skills_dict, technical,
            setup.__dict__ if setup else {}, regime_dict,
            wc_weight=effective_wc_weight,
        )

        if alpha < MIN_ALPHA:
            return None

        # 9. Log forecast for self-learning feedback loop
        if LEARNING_AVAILABLE and snap is not None:
            try:
                close_val = float(df["close"].iloc[-1])
                _learning.log_forecast({
                    "Symbol":       ticker,
                    "days":         CONFIG.get("rules", {}).get("prediction_days_default", 7),
                    "current":      close_val,
                    "target":       float(snap.take_profit) if setup else close_val,
                    "growth":       ((snap.take_profit - close_val) / close_val * 100
                                     if setup and close_val > 0 else 0.0),
                    "regime":       learning_regime,
                    "forecast_mode": ("ml" if ml_dict and ml_dict.get("confidence", 0) > 0.5
                                      else "classic"),
                    "active_skills": ([{"id": d["skill"]}
                                        for d in (skills or {}).get("triggered_details", [])]
                                      if skills else []),
                    "weekly_confluence": {"aligned": snap.confluence_score > 0},
                    "rsi":      snap.rsi_14,
                    "atr_pct":  (snap.atr_14 / close_val * 100) if close_val > 0 else 0.0,
                    "obv_signal": (float(snap.obv_slope) / max(abs(float(snap.obv or 1)), 1.0)
                                   if snap.obv else 0.0),
                    "indicators": {
                        "adx":   0,          # not computed in v4.2.2 default pipeline
                        "ema20": snap.ema_20,
                        "ema50": snap.ema_50,
                    },
                    # VAMP component scores (mapped from Gemini Flash Framework)
                    "w_trend":  snap.trend_score,
                    "w_ema":    snap.timing_score,
                    "w_vol":    snap.volume_score,
                    "w_weekly": snap.confluence_score,
                    # Component price levels used for contribution analysis
                    "trend_component":   snap.ema_50,
                    "ema20_component":   snap.ema_20,
                    "volume_component":  snap.vwap_20d if snap.vwap_20d else close_val,
                    "weekly_component":  snap.sma_200  if snap.sma_200  else close_val,
                })
            except Exception:
                pass

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
            hybrid_regime=regime_dict,
            learning_regime=learning_regime,
        )

    # ------------------------------------------------------------------
    # ALPHA SCORING
    # ------------------------------------------------------------------

    def _compute_alpha(self, gap, ml, sentiment, skills, technical, setup, regime,
                       wc_weight: float = None) -> float:
        """Composite alpha score from all pipeline layers.

        wc_weight: optional regime-adaptive weekly-confluence weight supplied
                   by sentinel_learning.inject_regime_weights(). Falls back to
                   the config value when None.
        """
        if wc_weight is None:
            wc_weight = ALPHA_WEIGHTS.get("weekly_confluence", 0.1)

        score = 0.0

        # Gap
        if gap.get("probability", 0) > 0.6:
            score += ALPHA_WEIGHTS.get("gap_magnitude", 0.2)  * gap["probability"]
            score += ALPHA_WEIGHTS.get("gap_confidence", 0.15) * gap.get("confidence", 0)

        # ML
        if ml and ml.get("target_return", 0) > 0:
            score += ALPHA_WEIGHTS.get("ml_7d", 0.15) * ml["target_return"] * ML_RETURN_MULTIPLIER
            score += ALPHA_WEIGHTS.get("ml_7d", 0.15) * ml.get("confidence", 0)

        # Sentiment
        if sentiment and sentiment.get("confidence", 0) > 0.3:
            score += (ALPHA_WEIGHTS.get("sentiment", 0.1)
                      * abs(sentiment["score"]) * sentiment["confidence"])

        # Skills
        if skills:
            score += ALPHA_WEIGHTS.get("auto_skills", 0.1) * skills.get("composite", 0)

        # Technical (Gemini Flash Framework composite)
        gemini = technical.get("gemini_framework", {})
        score += ALPHA_WEIGHTS.get("technical", 0.1) * gemini.get("composite", 0)

        # Weekly confluence (adaptive weight)
        score += wc_weight * abs(technical.get("confluence", 0))

        # S/R risk-reward bonus
        if setup.get("entry_zone") and setup.get("stop_loss"):
            if setup.get("rr", 0) >= MIN_RR_FOR_BONUS:
                score += ALPHA_WEIGHTS.get("sr_bonus", 0.05)

        # Regime penalties
        if regime:
            if regime.get("conflict_flag"):
                score *= CONFLICT_PENALTY
            if regime.get("shock_detected"):
                score *= SHOCK_PENALTY
            if "bear" in regime.get("regime", ""):
                score *= BEAR_PENALTY

        return min(1.0, max(0.0, score))

    # ------------------------------------------------------------------
    # BATCH RUN
    # ------------------------------------------------------------------

    def run(self, tickers: List[str] = None) -> List[Dict]:
        """Run pipeline on all tickers and return top N setups."""
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
    print("OvernightAlpha v4.2.2 ready: Full pipeline with DeltaCache + self-learning support")
    print(f"sentinel_learning available: {LEARNING_AVAILABLE}")
