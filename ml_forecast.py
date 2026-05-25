"""
Sentinel-EGX v4.2.2 — ML Forecast Engine
========================================
7-day return forecast using XGBoost + Random Forest ensemble.
Aligned with sentinel_config.json v4.2 ml_forecast specs.
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from dataclasses import dataclass

with open("sentinel_config.json", "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

ML_CFG = CONFIG.get("ml_forecast", {})
FEATURES = ML_CFG.get("features", [])
HORIZON = ML_CFG.get("target_horizon_days", 7)
MODELS = ML_CFG.get("models", ["xgboost", "random_forest"])
WEIGHTS = ML_CFG.get("ensemble_weight", {"xgboost": 0.6, "random_forest": 0.4})


@dataclass
class MLForecast:
    ticker: str
    target_return: float
    confidence: float
    xgb_pred: float
    rf_pred: float
    ensemble_pred: float
    features_used: int
    regime_adjusted: bool


class MLForecastEngine:
    """7-day return forecast using XGBoost + Random Forest."""

    def __init__(self):
        self.xgb_model = None
        self.rf_model = None
        self.is_trained = False
        self.feature_cols = FEATURES

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build ML forecast features from indicator-enriched dataframe."""
        feat = pd.DataFrame(index=df.index)
        close = df["close"]
        volume = df["volume"]

        # Returns
        feat["returns_1d"] = close.pct_change(1)
        feat["returns_5d"] = close.pct_change(5)
        feat["returns_10d"] = close.pct_change(10)
        feat["returns_20d"] = close.pct_change(20)

        # EMAs
        feat["ema20"] = df.get("ema_20", close)
        feat["ema50"] = df.get("ema_50", close)
        feat["sma200"] = df.get("sma_200", close)
        feat["ema_distance"] = (df.get("ema_20", close) - df.get("ema_50", close)) / close

        # RSI
        feat["rsi_14"] = df.get("rsi_14", 50) / 100
        feat["stoch_rsi_k"] = df.get("stoch_rsi_k", 50) / 100
        feat["stoch_rsi_d"] = df.get("stoch_rsi_d", 50) / 100

        # MACD
        feat["macd_hist"] = df.get("macd_hist", 0) / close
        feat["macd_signal"] = df.get("macd_signal", 0) / close

        # Volume
        feat["cmf_20"] = df.get("cmf_20", 0)
        feat["obv_slope_5d"] = df.get("obv_slope_5d", 0)
        feat["volume_vs_avg20"] = df.get("volume_vs_avg20", 1)
        feat["vwap_distance"] = (close - df.get("vwap_20d", close)) / close
        feat["anchored_vwap_distance"] = (close - df.get("anchored_vwap", close)) / close

        # Volatility
        feat["atr_14"] = df.get("atr_14", close * 0.02) / close
        feat["volatility_20d"] = close.pct_change().rolling(20).std() * np.sqrt(252)

        # Gemini
        feat["gemini_trend_score"] = df.get("gemini_trend_score", 0)
        feat["gemini_volume_score"] = df.get("gemini_volume_score", 0)
        feat["gemini_timing_score"] = df.get("gemini_timing_score", 0)
        feat["confluence_score"] = df.get("confluence_score", 0)
        feat["t0_liquidity"] = df.get("liquidity_score", 0)

        return feat.fillna(0)

    def train(self, df: pd.DataFrame) -> Dict:
        """Train ML models on historical data."""
        try:
            from xgboost import XGBRegressor
            from sklearn.ensemble import RandomForestRegressor
        except ImportError:
            print("[MLForecast] XGBoost/sklearn not installed. Using heuristic mode.")
            self.is_trained = False
            return {"status": "heuristic_mode"}

        feat = self._build_features(df)
        # Target: forward return over horizon days
        target = df["close"].shift(-HORIZON) / df["close"] - 1

        valid = feat.dropna().index.intersection(target.dropna().index)
        if len(valid) < 100:
            self.is_trained = False
            return {"status": "insufficient_data", "samples": len(valid)}

        X = feat.loc[valid].values
        y = target.loc[valid].values

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        self.xgb_model = XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42
        )
        self.xgb_model.fit(X_train, y_train)

        self.rf_model = RandomForestRegressor(
            n_estimators=200, max_depth=5, random_state=42
        )
        self.rf_model.fit(X_train, y_train)

        self.is_trained = True

        xgb_score = self.xgb_model.score(X_test, y_test)
        rf_score = self.rf_model.score(X_test, y_test)

        return {
            "status": "trained",
            "xgb_r2": round(xgb_score, 3),
            "rf_r2": round(rf_score, 3),
            "train_samples": len(X_train),
            "test_samples": len(X_test)
        }

    def predict(self, df: pd.DataFrame, ticker: str = "") -> Optional[Dict]:
        """Predict 7-day forward return."""
        feat = self._build_features(df)
        latest = feat.iloc[[-1]].values

        if not self.is_trained:
            # Heuristic fallback
            recent_return = df["close"].pct_change(5).iloc[-1]
            vol = df["close"].pct_change().rolling(20).std().iloc[-1]
            confidence = min(0.8, 1 - vol * 5)
            return {
                "ticker": ticker,
                "target_return": round(recent_return, 4),
                "confidence": round(confidence, 2),
                "xgb_pred": round(recent_return, 4),
                "rf_pred": round(recent_return, 4),
                "regime_adjusted": False
            }

        xgb_pred = float(self.xgb_model.predict(latest)[0])
        rf_pred = float(self.rf_model.predict(latest)[0])

        ensemble = xgb_pred * WEIGHTS.get("xgboost", 0.6) + rf_pred * WEIGHTS.get("random_forest", 0.4)

        # Confidence based on model agreement
        agreement = 1 - abs(xgb_pred - rf_pred) / (abs(ensemble) + 0.001)
        confidence = min(0.95, max(0.3, agreement))

        return {
            "ticker": ticker,
            "target_return": round(ensemble, 4),
            "confidence": round(confidence, 2),
            "xgb_pred": round(xgb_pred, 4),
            "rf_pred": round(rf_pred, 4),
            "regime_adjusted": False
        }

    def save(self, path: str):
        import pickle
        with open(path, "wb") as f:
            pickle.dump({"xgb": self.xgb_model, "rf": self.rf_model}, f)

    def load(self, path: str):
        import pickle
        with open(path, "rb") as f:
            models = pickle.load(f)
        self.xgb_model = models["xgb"]
        self.rf_model = models["rf"]
        self.is_trained = True


if __name__ == "__main__":
    print("MLForecastEngine v4.2 ready: XGBoost + Random Forest ensemble")
