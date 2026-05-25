"""
Sentinel-EGX v4.2.2 — Gap Predictor Engine
=========================================
Overnight gap prediction using 35 features + XGBoost/Random Forest.
Aligned with sentinel_config.json v4.2 gap_predictor specs.
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

with open("sentinel_config.json", "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

GAP_CFG = CONFIG.get("gap_predictor", {})
FEATURES = GAP_CFG.get("features", [])
GAP_THRESHOLD = GAP_CFG.get("gap_threshold_pct", 1.0) / 100.0
T0_BOOST = GAP_CFG.get("t0_magnitude_boost", 1.3)
TRAIN_CFG = GAP_CFG.get("training", {})


@dataclass
class GapPrediction:
    ticker: str
    gap_direction: str  # "up", "down", "neutral"
    gap_probability: float
    expected_magnitude: float
    t0_liquidity_boost: float
    confidence: float
    features_used: int


class GapPredictor:
    """Predict overnight gaps using 35 engineered features."""

    def __init__(self, config_path: str = "sentinel_config.json"):
        self.model = None
        self.is_trained = False
        self.feature_cols = FEATURES
        self.threshold = GAP_THRESHOLD
        self.t0_boost = T0_BOOST

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build 35 gap prediction features from indicator-enriched dataframe."""
        feat = pd.DataFrame(index=df.index)
        close = df["close"]
        volume = df["volume"]

        # Returns
        feat["returns_1d"] = close.pct_change(1)
        feat["returns_5d"] = close.pct_change(5)
        feat["returns_10d"] = close.pct_change(10)

        # Volatility
        feat["volatility_5d"] = close.pct_change().rolling(5).std() * np.sqrt(252)
        feat["volatility_20d"] = close.pct_change().rolling(20).std() * np.sqrt(252)

        # Volume
        feat["volume_ratio"] = volume / volume.rolling(20).mean()
        feat["volume_trend"] = volume.rolling(5).mean() / volume.rolling(20).mean()

        # Price position
        high20 = df["high"].rolling(20).max()
        low20 = df["low"].rolling(20).min()
        feat["price_vs_high_20d"] = (close - high20) / high20
        feat["price_vs_low_20d"] = (close - low20) / low20

        # ATR
        feat["atr_ratio"] = df.get("atr_14", close * 0.02) / close

        # S/R distance
        feat["dist_to_r1"] = (df.get("r1", close) - close) / close
        feat["dist_to_s1"] = (close - df.get("s1", close)) / close

        # EMA slopes
        feat["ema20_slope"] = df["ema_20"].diff(5) / df["ema_20"].shift(5)
        feat["ema50_slope"] = df["ema_50"].diff(5) / df["ema_50"].shift(5)

        # Sentiment placeholder
        feat["sentiment_score"] = 0.0

        # Historical gap stats
        gaps = close.pct_change().shift(-1)  # next day open/close gap proxy
        feat["avg_gap_20d"] = gaps.rolling(20).mean()
        feat["max_gap_20d"] = gaps.rolling(20).max()
        feat["gap_frequency"] = (gaps.abs() > self.threshold).rolling(20).mean()

        # VWAP/AVWAP
        feat["dist_to_vwap"] = (close - df.get("vwap_20d", close)) / close
        feat["dist_to_avwap"] = (close - df.get("anchored_vwap", close)) / close

        # CMF/OBV
        feat["cmf_level"] = df.get("cmf_20", 0)
        feat["cmf_trend"] = df.get("cmf_20", 0).diff(5)
        feat["obv_slope"] = df.get("obv", 0).diff(5)

        # RSI
        feat["rsi_level"] = df.get("rsi_14", 50) / 100
        feat["rsi_trend"] = df.get("rsi_14", 50).diff(5)

        # StochRSI
        feat["stochrsi_k"] = df.get("stoch_rsi_k", 50) / 100
        feat["stochrsi_d"] = df.get("stoch_rsi_d", 50) / 100

        # MACD
        feat["macd_hist"] = df.get("macd_hist", 0) / close
        feat["macd_bull"] = (df.get("macd_state", "neutral") == "bullish").astype(float)
        feat["macd_bear"] = (df.get("macd_state", "neutral") == "bearish").astype(float)

        # EMA alignment
        feat["ema_aligned_bull"] = (df["ema_20"] > df["ema_50"]).astype(float)
        feat["ema_aligned_bear"] = (df["ema_20"] < df["ema_50"]).astype(float)

        # Gemini
        feat["gemini_composite"] = df.get("gemini_composite", 0)
        feat["confluence_score"] = df.get("confluence_score", 0)

        # T+0
        feat["t0_liquidity"] = df.get("liquidity_score", 0)

        return feat.fillna(0)

    def train(self, historical_data: Dict[str, pd.DataFrame], segments: Dict[str, str]) -> Tuple['GapPredictor', Dict]:
        """Train gap prediction model on historical EOD data."""
        try:
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        except ImportError:
            print("[GapPredictor] sklearn not installed. Using heuristic mode.")
            self.is_trained = False
            return self, {"accuracy": "N/A", "note": "Heuristic mode"}

        X_list = []
        y_list = []

        for ticker, df in historical_data.items():
            if len(df) < 60:
                continue
            feat = self._build_features(df)
            # Target: did next day gap exceed threshold?
            next_gap = df["close"].pct_change().shift(-1).abs()
            target = (next_gap > self.threshold).astype(int)

            valid = feat.dropna().index.intersection(target.dropna().index)
            if len(valid) < 30:
                continue

            X_list.append(feat.loc[valid].values)
            y_list.append(target.loc[valid].values)

        if not X_list:
            self.is_trained = False
            return self, {"accuracy": 0, "note": "Insufficient training data"}

        X = np.vstack(X_list)
        y = np.concatenate(y_list)

        # Train/test split
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        self.model = GradientBoostingClassifier(
            n_estimators=TRAIN_CFG.get("n_estimators", 200),
            max_depth=TRAIN_CFG.get("max_depth", 5),
            learning_rate=TRAIN_CFG.get("learning_rate", 0.1),
            min_samples_split=TRAIN_CFG.get("min_samples", 100),
            random_state=42
        )
        self.model.fit(X_train, y_train)
        self.is_trained = True

        accuracy = self.model.score(X_test, y_test)

        return self, {
            "accuracy": round(accuracy, 3),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "features": len(self.feature_cols)
        }

    def predict(self, df: pd.DataFrame, ticker: str = "", segment: str = "high_activity") -> GapPrediction:
        """Predict overnight gap for current bar."""
        feat = self._build_features(df)
        latest = feat.iloc[[-1]].values

        if not self.is_trained or self.model is None:
            # Heuristic fallback
            recent_vol = df["close"].pct_change().rolling(5).std().iloc[-1]
            recent_return = df["close"].pct_change(1).iloc[-1]
            prob = min(0.8, recent_vol * 10)
            direction = "up" if recent_return > 0 else "down" if recent_return < 0 else "neutral"
            magnitude = recent_vol * 2

            t0 = segment in CONFIG.get("t0_rules", {}).get("t0_enabled_segments", [])
            boost = self.t0_boost if t0 else 1.0

            return GapPrediction(
                ticker=ticker,
                gap_direction=direction,
                gap_probability=round(prob, 2),
                expected_magnitude=round(magnitude * boost, 4),
                t0_liquidity_boost=round(boost, 1),
                confidence=0.3,
                features_used=len(self.feature_cols)
            )

        prob = self.model.predict_proba(latest)[0][1]  # probability of gap > threshold
        pred_class = self.model.predict(latest)[0]

        if pred_class == 1 and df["close"].pct_change(1).iloc[-1] > 0:
            direction = "up"
        elif pred_class == 1:
            direction = "down"
        else:
            direction = "neutral"

        magnitude = self.threshold * (1 + prob)

        t0 = segment in CONFIG.get("t0_rules", {}).get("t0_enabled_segments", [])
        boost = self.t0_boost if t0 else 1.0

        return GapPrediction(
            ticker=ticker,
            gap_direction=direction,
            gap_probability=round(prob, 2),
            expected_magnitude=round(magnitude * boost, 4),
            t0_liquidity_boost=round(boost, 1),
            confidence=round(prob, 2),
            features_used=len(self.feature_cols)
        )

    def save(self, path: str):
        """Save trained model."""
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path: str):
        """Load trained model."""
        import pickle
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        self.is_trained = True


def predict_overnight_gap(df: pd.DataFrame, ticker: str, segment: str = "high_activity") -> GapPrediction:
    """Convenience wrapper for single-ticker gap prediction."""
    predictor = GapPredictor()
    return predictor.predict(df, ticker, segment)


def train_gap_model(historical_data: Dict[str, pd.DataFrame], segments: Dict[str, str]) -> Tuple[GapPredictor, Dict]:
    """Train gap model on multiple tickers."""
    predictor = GapPredictor()
    return predictor.train(historical_data, segments)


if __name__ == "__main__":
    print("GapPredictor v4.2 ready: 35 features + XGB/RF + T+0 boost")
