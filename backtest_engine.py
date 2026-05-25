"""
Sentinel-EGX v4.2 — Backtest Engine
Walk-forward simulation for overnight gap strategy
Ported and enhanced from v3.7, aligned with sentinel_config.json v4.2
NEW v4.2.2: DeltaCache integration for historical data loading.
"""
import os, json, time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

class BacktestEngine:
    """Walk-forward backtest for overnight gap and ML forecast strategies."""

    def __init__(self, config_path: str = "sentinel_config.json"):
        with open(config_path) as f:
            cfg = json.load(f)
        self.cfg = cfg["backtest"]
        self.alpha_cfg = cfg.get("alpha_scorer", {})
        self.gap_cfg = cfg.get("gap_predictor", {})
        self.ml_cfg = cfg.get("ml_forecast", {})

        self.walk_splits = self.cfg.get("walk_forward_splits", 5)
        self.train_size = self.cfg.get("train_size", 0.7)
        self.metrics_list = self.cfg.get("metrics", ["accuracy", "precision", "recall", "sharpe", "max_drawdown", "win_rate"])
        self.regime_filter = self.cfg.get("regime_filter", True)

        self.tx_cost = 0.00125  # 0.125% per side
        self.slippage = 0.001   # 0.1% slippage

        self.results = []

    def _split_windows(self, df: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Generate walk-forward train/test windows."""
        windows = []
        total = len(df)
        train_len = int(total * self.train_size)
        test_len = total - train_len
        step = max(1, test_len // self.walk_splits)

        for i in range(self.walk_splits):
            start = i * step
            if start + train_len + test_len > total:
                break
            train = df.iloc[start:start + train_len]
            test = df.iloc[start + train_len:start + train_len + test_len]
            if len(train) >= 50 and len(test) >= 10:
                windows.append((train, test))
        return windows

    def run_overnight_gap(self, df: pd.DataFrame, gap_threshold: float = 0.015,
                          confidence_threshold: float = 0.6) -> Dict:
        """
        Simulate: buy at close, sell at next open if gap > threshold.
        Uses actual gap column for ground truth.
        """
        if 'gap' not in df.columns or len(df) < 100:
            return {"error": "Insufficient data or no gap column"}

        windows = self._split_windows(df)
        all_trades = []

        for train, test in windows:
            train_gaps = train['gap'].dropna()
            avg_gap = train_gaps.mean()
            std_gap = train_gaps.std()

            for _, row in test.iterrows():
                gap = row.get('gap', 0)
                if pd.isna(gap):
                    continue

                predicted_gap = avg_gap + 0.5 * std_gap
                confidence = min(1.0, abs(gap - avg_gap) / (std_gap + 0.001))

                if predicted_gap > gap_threshold and confidence > confidence_threshold:
                    entry = row['close']
                    exit_price = row['open']
                    pnl = (exit_price / entry - 1) - self.tx_cost - self.slippage

                    all_trades.append({
                        "entry_date": row['date'],
                        "entry": entry,
                        "exit": exit_price,
                        "pnl_pct": pnl * 100,
                        "predicted_gap": predicted_gap,
                        "actual_gap": gap,
                        "confidence": confidence,
                        "win": pnl > 0
                    })

        return self._compute_metrics(all_trades, df)

    def run_ml_forecast(self, df: pd.DataFrame, ml_engine, threshold: float = 0.02) -> Dict:
        """Simulate ML 7-day forecast strategy."""
        windows = self._split_windows(df)
        all_trades = []

        for train, test in windows:
            ml_engine.train(train)

            for _, row in test.iterrows():
                sub = df[df['date'] <= row['date']].tail(60)
                if len(sub) < 30:
                    continue

                pred = ml_engine.predict(sub)
                target = pred.get("target_return", 0)
                conf = pred.get("confidence", 0)

                if target > threshold and conf > 0.5:
                    entry = row['close']
                    future_idx = df.index[df['date'] > row['date']]
                    horizon = self.ml_cfg.get("target_horizon_days", 7)
                    if len(future_idx) >= horizon:
                        exit_price = df.loc[future_idx[horizon-1], 'close']
                    else:
                        continue

                    pnl = (exit_price / entry - 1) - self.tx_cost - self.slippage
                    all_trades.append({
                        "entry_date": row['date'],
                        "entry": entry,
                        "exit": exit_price,
                        "pnl_pct": pnl * 100,
                        "predicted_return": target,
                        "confidence": conf,
                        "win": pnl > 0
                    })

        return self._compute_metrics(all_trades, df)

    def _compute_metrics(self, trades: List[Dict], df: pd.DataFrame) -> Dict:
        if not trades:
            return {"trades": 0, "total_return": 0, "win_rate": 0}

        pnls = [t["pnl_pct"] for t in trades]
        wins = sum(1 for p in pnls if p > 0)

        cumulative = 1.0
        for p in pnls:
            cumulative *= (1 + p / 100)

        peak = 1.0
        max_dd = 0.0
        running = 1.0
        for p in pnls:
            running *= (1 + p / 100)
            if running > peak:
                peak = running
            dd = (peak - running) / peak
            if dd > max_dd:
                max_dd = dd

        returns = np.array(pnls) / 100
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0

        if 'close' in df.columns and len(df) > 1:
            bench_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
        else:
            bench_return = 0

        alpha = (cumulative - 1) * 100 - bench_return

        return {
            "trades": len(trades),
            "total_return_pct": (cumulative - 1) * 100,
            "win_rate": wins / len(trades),
            "avg_trade_pnl": np.mean(pnls),
            "max_drawdown_pct": max_dd * 100,
            "sharpe_ratio": sharpe,
            "profit_factor": abs(sum(p for p in pnls if p > 0)) / abs(sum(p for p in pnls if p < 0)) if sum(p for p in pnls if p < 0) != 0 else float('inf'),
            "alpha_vs_benchmark": alpha,
            "benchmark_return_pct": bench_return,
            "trade_log": trades[:50]
        }

    def run_full_backtest(self, data_dict: Dict[str, pd.DataFrame], gap_engine, ml_engine,
                          regime_detector) -> Dict:
        """Run backtest across all tickers with regime filtering."""
        all_results = {}

        for ticker, df in data_dict.items():
            if len(df) < 100:
                continue

            if self.regime_filter:
                try:
                    regime = regime_detector.detect(df)
                    position_mult = regime.get("position_size", 1.0)
                except Exception:
                    position_mult = 1.0
            else:
                position_mult = 1.0

            gap_result = self.run_overnight_gap(df)
            gap_result["regime"] = regime.get("regime", "unknown") if self.regime_filter else "disabled"
            gap_result["position_multiplier"] = position_mult

            all_results[ticker] = {
                "overnight_gap": gap_result,
                "ticker": ticker
            }

        all_pnls = []
        for r in all_results.values():
            if "overnight_gap" in r and "trade_log" in r["overnight_gap"]:
                for t in r["overnight_gap"]["trade_log"]:
                    all_pnls.append(t["pnl_pct"])

        if all_pnls:
            total_trades = len(all_pnls)
            wins = sum(1 for p in all_pnls if p > 0)
            cumulative = np.prod([1 + p/100 for p in all_pnls])

            return {
                "tickers_tested": len(all_results),
                "total_trades": total_trades,
                "aggregate_return_pct": (cumulative - 1) * 100,
                "aggregate_win_rate": wins / total_trades,
                "per_ticker": all_results
            }

        return {"tickers_tested": 0, "error": "No valid trades"}


if __name__ == "__main__":
    print("BacktestEngine v4.2 ready. Run with historical EOD data.")
