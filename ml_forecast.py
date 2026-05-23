"""
Sentinel-EGX v4.2 — ML Forecast Engine
=========================================
XGBoost + Random Forest ensemble for 7-day return prediction.
Aligned with sentinel_config.json v4.2 specs.
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, Optional
from dataclasses import dataclass

with open("sentinel_config.json", "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

ML_CFG = CONFIG.get("ml_forecast", {})
FEATURES = ML_CFG.get("features", [])
HORIZON = ML_CFG.get("target_horizon_days", 7)
ENSEMBLE_W = ML_CFG.get("ensemble_weight", {"xgboost": 0.6, "random_forest": 0.4})
ALPHA_W = ML_CFG.get("alpha_weight", 0.15)


@dataclass
class MLForecastResult:
    target_return: float
    confidence: float
    xgb_pred: float
    rf_pred: float
    ensemble_pred: float
    feature_importance: Dict[str, float]


class MLForecastEngine:
    """XGBoost + Random Forest ensemble for 7-day return prediction."""

    def __init__(self, config_path: str = "sentinel_config.json"):
        self.xgb_model = None
        self.rf_model = None
        self.feature_cols = FEATURES
        self.horizon = HORIZON
        self.ensemble_weights = ENSEMBLE_W
        self.is_trained = False

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build feature matrix from indicator-enriched dataframe."""
        feat = pd.DataFrame(index=df.index)

        # Returns
        feat["returns_1d"] = df["close"].pct_change(1)
        feat["returns_5d"] = df["close"].pct_change(5)
        feat["returns_10d"] = df["close"].pct_change(10)
        feat["returns_20d"] = df["close"].pct_change(20)

        # EMA distances
        feat["ema20"] = (df["close"] - df["ema_20"]) / df["ema_20"]
        feat["ema50"] = (df["close"] - df["ema_50"]) / df["ema_50"]
        feat["sma200"] = (df["close"] - df["sma_200"]) / df["sma_200"]
        feat["ema_distance"] = (df["ema_20"] - df["ema_50"]) / df["ema_50"]

        # Momentum
        feat["rsi_14"] = df["rsi_14"] / 100.0
        feat["stoch_rsi_k"] = df.get("stoch_rsi_k", 50) / 100.0
        feat["stoch_rsi_d"] = df.get("stoch_rsi_d", 50) / 100.0
        feat["macd_hist"] = df.get("macd_hist", 0) / df["close"]
        feat["macd_signal"] = (df.get("macd_line", 0) - df.get("macd_signal", 0)) / df["close"]

        # Volume
        feat["cmf_20"] = df.get("cmf_20", 0)
        feat["obv_slope_5d"] = df.get("obv_slope_5d", 0) / (df.get("volume", 1).mean() + 1)
        feat["volume_vs_avg20"] = df.get("volume_vs_avg20", 1)
        feat["vwap_distance"] = (df["close"] - df.get("vwap_20d", df["close"])) / df["close"]
        feat["anchored_vwap_distance"] = (df["close"] - df.get("anchored_vwap", df["close"])) / df["close"]

        # Volatility
        feat["atr_14"] = df.get("atr_14", 0) / df["close"]
        feat["volatility_20d"] = df["close"].pct_change().rolling(20).std() * np.sqrt(252)

        # Gemini framework
        feat["gemini_trend_score"] = df.get("gemini_trend_score", 0)
        feat["gemini_volume_score"] = df.get("gemini_volume_score", 0)
        feat["gemini_timing_score"] = df.get("gemini_timing_score", 0)

        # Confluence
        feat["confluence_score"] = df.get("confluence_score", 0)
        feat["t0_liquidity"] = df.get("liquidity_score", 0)

        return feat.fillna(0)

    def train(self, df: pd.DataFrame) -> bool:
        """Train XGBoost + Random Forest ensemble."""
        try:
            from sklearn.ensemble import RandomForestRegressor
            from xgboost import XGBRegressor
        except ImportError:
            print("[MLForecast] xgboost or sklearn not installed. Using fallback heuristic.")
            self.is_trained = False
            return False

        if len(df) < 100:
            self.is_trained = False
            return False

        feat = self._build_features(df)
        # Target: forward return over horizon days
        target = df["close"].shift(-self.horizon) / df["close"] - 1

        valid = feat.dropna().index.intersection(target.dropna().index)
        X = feat.loc[valid].values
        y = target.loc[valid].values

        if len(X) < 50:
            self.is_trained = False
            return False

        # Train/validation split
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        self.xgb_model = XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )
        self.xgb_model.fit(X_train, y_train)

        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=6,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        )
        self.rf_model.fit(X_train, y_train)

        # Validation confidence
        xgb_val = self.xgb_model.predict(X_val)
        rf_val = self.rf_model.predict(X_val)
        ensemble_val = self.ensemble_weights["xgboost"] * xgb_val + self.ensemble_weights["random_forest"] * rf_val
        mse = np.mean((ensemble_val - y_val) ** 2)
        self.val_rmse = np.sqrt(mse)

        self.is_trained = True
        return True

    def predict(self, df: pd.DataFrame) -> Optional[Dict]:
        """Predict 7-day forward return."""
        feat = self._build_features(df)
        latest = feat.iloc[[-1]].values

        if not self.is_trained or self.xgb_model is None:
            # Fallback heuristic
            returns_5d = df["close"].pct_change(5).iloc[-1] if len(df) > 5 else 0
            returns_20d = df["close"].pct_change(20).iloc[-1] if len(df) > 20 else 0
            rsi = df.get("rsi_14", pd.Series([50])).iloc[-1] / 100
            heuristic = (returns_5d * 0.3 + returns_20d * 0.2 + (rsi - 0.5) * 0.1)
            return {
                "target_return": round(heuristic, 4),
                "confidence": 0.3,
                "xgb_pred": round(heuristic, 4),
                "rf_pred": round(heuristic, 4),
                "note": "Heuristic fallback (models not trained)"
            }

        xgb_pred = self.xgb_model.predict(latest)[0]
        rf_pred = self.rf_model.predict(latest)[0]
        ensemble = self.ensemble_weights["xgboost"] * xgb_pred + self.ensemble_weights["random_forest"] * rf_pred

        # Confidence based on historical RMSE
        confidence = max(0.1, min(0.95, 1.0 - self.val_rmse / (abs(ensemble) + 0.01)))

        # Feature importance
        importance = {}
        if hasattr(self.xgb_model, "feature_importances_"):
            for i, col in enumerate(self.feature_cols):
                if i < len(self.xgb_model.feature_importances_):
                    importance[col] = float(self.xgb_model.feature_importances_[i])

        return {
            "target_return": round(ensemble, 4),
            "confidence": round(confidence, 2),
            "xgb_pred": round(xgb_pred, 4),
            "rf_pred": round(rf_pred, 4),
            "feature_importance": importance,
            "val_rmse": round(self.val_rmse, 4)
        }


if __name__ == "__main__":
    print("MLForecastEngine v4.2 ready: XGBoost + Random Forest ensemble")
