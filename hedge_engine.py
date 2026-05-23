"""
Sentinel-EGX v4.2 — Hedge Engine
EGX30 futures beta-adjusted hedge calculator
"""
import json
from typing import Dict, List
import numpy as np
import pandas as pd

class EGXHedgeEngine:
    """Calculate beta-adjusted hedge ratio for EGX30 futures."""

    def __init__(self, config_path: str = "sentinel_config.json"):
        with open(config_path) as f:
            cfg = json.load(f)
        self.cfg = cfg["hedge_engine"]
        self.futures_symbol = self.cfg["egx30_futures_symbol"]
        self.target_delta = self.cfg["target_residual_delta"]
        self.lookback = self.cfg["beta_lookback_days"]

    def calculate_beta(self, stock_returns: pd.Series, index_returns: pd.Series) -> float:
        """Calculate beta = cov(stock, index) / var(index)."""
        aligned = pd.concat([stock_returns, index_returns], axis=1).dropna()
        if len(aligned) < 20:
            return 1.0
        cov = aligned.cov().iloc[0, 1]
        var = aligned.iloc[:, 1].var()
        if var == 0 or np.isnan(var):
            return 1.0
        return float(cov / var)

    def compute_hedge(self, portfolio: List[Dict], index_df: pd.DataFrame) -> Dict:
        """
        portfolio: list of {ticker, position_value, stock_df}
        Returns: hedge_shares, hedge_value, residual_delta
        """
        if index_df.empty or len(index_df) < self.lookback:
            return {"hedge_value": 0, "hedge_ratio": 0, "residual_delta": 1.0, "note": "Insufficient index data"}

        index_returns = index_df['close'].pct_change().tail(self.lookback)
        total_value = sum(p['position_value'] for p in portfolio)
        weighted_beta = 0.0

        for p in portfolio:
            stock_returns = p['stock_df']['close'].pct_change().tail(self.lookback)
            beta = self.calculate_beta(stock_returns, index_returns)
            weight = p['position_value'] / total_value if total_value > 0 else 0
            weighted_beta += beta * weight

        # Target residual delta = 10% of portfolio beta
        hedge_ratio = max(0, weighted_beta - self.target_delta)
        hedge_value = hedge_ratio * total_value
        residual_delta = weighted_beta - hedge_ratio

        return {
            "portfolio_beta": float(weighted_beta),
            "hedge_ratio": float(hedge_ratio),
            "hedge_value": float(hedge_value),
            "residual_delta": float(residual_delta),
            "target_delta": self.target_delta,
            "futures_symbol": self.futures_symbol
        }


if __name__ == "__main__":
    print("HedgeEngine ready.")
