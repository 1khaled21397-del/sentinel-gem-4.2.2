"""
Sentinel-EGX v4.2.2 — Hedge Engine
EGX30 futures beta-adjusted hedge calculator
"""
import json
from typing import Dict, List
import numpy as np
import pandas as pd

class EGXHedgeEngine:
    """Calculate beta-adjusted hedge ratio for EGX30 futures."""

    def __init__(self, config_path: str = "sentinel_config.json"):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.cfg = cfg["hedge_engine"]
        self.futures_beta = self.cfg.get("egx30_futures_beta", 0.85)
        self.lookback = self.cfg.get("beta_lookback_days", 60)

    def calculate_beta(self, stock_returns: pd.Series, index_returns: pd.Series) -> float:
        """Calculate beta = cov(stock, index) / var(index)."""
        aligned = pd.concat([stock_returns, index_returns], axis=1).dropna()
        if len(aligned) < 20:
            return 1.0
        cov = aligned.cov().iloc[0, 1]
        var = aligned.iloc[:, 1].var()
        if var == 0 or np.isnan(var):
            return 1.0
        return float(cov / var) if var != 0 else 1.0

    def compute_hedge(self, portfolio: List[Dict], index_df: pd.DataFrame) -> Dict:
        """
        portfolio: list of {ticker, position_value, stock_df}
        Returns: hedge details
        """
        if index_df.empty or len(index_df) < self.lookback:
            return {"hedge_value": 0, "hedge_ratio": 0, "residual_beta": 1.0, "note": "Insufficient index data"}

        index_returns = index_df['close'].pct_change().tail(self.lookback)
        total_value = sum(p['position_value'] for p in portfolio)
        weighted_beta = 0.0

        for p in portfolio:
            stock_returns = p['stock_df']['close'].pct_change().tail(self.lookback)
            beta = self.calculate_beta(stock_returns, index_returns)
            weight = p['position_value'] / total_value if total_value > 0 else 0
            weighted_beta += beta * weight

        # Hedge to reduce portfolio beta to near-zero
        hedge_ratio = weighted_beta / self.futures_beta if self.futures_beta > 0 else 0
        hedge_value = hedge_ratio * total_value
        residual_beta = weighted_beta - hedge_ratio * self.futures_beta

        return {
            "portfolio_beta": round(float(weighted_beta), 3),
            "futures_beta": self.futures_beta,
            "hedge_ratio": round(float(hedge_ratio), 3),
            "hedge_value": round(float(hedge_value), 2),
            "residual_beta": round(float(residual_beta), 3),
            "portfolio_value": round(float(total_value), 2)
        }


if __name__ == "__main__":
    print("HedgeEngine v4.2 ready: EGX30 futures beta-adjusted hedging")
