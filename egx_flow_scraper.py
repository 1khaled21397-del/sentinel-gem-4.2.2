"""
Sentinel-EGX v4.2.2 — EGX Flow Sentiment Scraper
===============================================
Scrapes EGX market flow data (foreign/institutional/retail ratios).
Aligned with sentinel_config.json v4.2 flow_sentiment specs.
"""

import requests
import json
import numpy as np
from typing import Dict, Optional
from datetime import datetime

with open("sentinel_config.json", "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

FLOW_CFG = CONFIG.get("flow_sentiment", {})

# ── FLOW SCRAPER CONSTANTS (module-level) ─────────────────────────────────────
MAX_CONFIDENCE        = 1.0
RETAIL_FOMO_THRESHOLD = 0.1
# ─────────────────────────────────────────────────────────────────────────────


class EGXFlowScraper:
    """Scrape EGX market flow data for sentiment scoring."""

    def __init__(self):
        self.enabled = FLOW_CFG.get("enabled", True)
        self.source = FLOW_CFG.get("source", "egx.com.eg")
        self.cache_ttl = FLOW_CFG.get("cache_ttl_minutes", 30)
        self.threshold = FLOW_CFG.get("sentiment_threshold", 0.3)
        self.foreign_buy_boost = FLOW_CFG.get("foreign_buy_boost", 0.15)
        self.foreign_sell_penalty = FLOW_CFG.get("foreign_sell_penalty", -0.15)
        self.institutional_ratio_bonus = FLOW_CFG.get("institutional_ratio_bonus", 0.1)
        self.retail_fomo_penalty = FLOW_CFG.get("retail_fomo_penalty", -0.1)

    def fetch_flow(self) -> Optional[Dict]:
        """Fetch EGX flow data. Placeholder for actual scraping logic."""
        if not self.enabled:
            return None
        # In production, this would scrape egx.com.eg or use an API
        # For now, return demo data structure
        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "total_volume": 150000000,
            "foreign_buy": 45000000,
            "foreign_sell": 30000000,
            "institutional_buy": 60000000,
            "institutional_sell": 40000000,
            "retail_buy": 45000000,
            "retail_sell": 80000000,
            "net_foreign": 15000000,
            "net_institutional": 20000000,
            "net_retail": -35000000
        }

    def compute_sentiment(self, flow_data: Dict) -> Dict:
        """Compute sentiment score from flow data."""
        if not flow_data:
            return {"score": 0, "confidence": 0, "meta": {}}

        total = flow_data.get("total_volume", 1)
        foreign_ratio = (flow_data.get("foreign_buy", 0) + flow_data.get("foreign_sell", 0))  / total if total > 0 else 0
        inst_ratio = (flow_data.get("institutional_buy", 0) + flow_data.get("institutional_sell", 0))  / total if total > 0 else 0
        retail_ratio = (flow_data.get("retail_buy", 0) + flow_data.get("retail_sell", 0))  / total if total > 0 else 0

        net_foreign = flow_data.get("net_foreign", 0)
        net_inst = flow_data.get("net_institutional", 0)
        net_retail = flow_data.get("net_retail", 0)

        # Score components
        score = 0.0
        if net_foreign > 0:
            score += self.foreign_buy_boost * (net_foreign / total if total > 0 else 0)
        else:
            score += self.foreign_sell_penalty * abs(net_foreign / total if total > 0 else 0)

        if net_inst > 0:
            score += self.institutional_ratio_bonus * (net_inst / total if total > 0 else 0)

        if net_retail < 0 and abs(net_retail / total if total > 0 else 0) > RETAIL_FOMO_THRESHOLD:
            score += self.retail_fomo_penalty  # retail selling = smart money buying

        confidence = min(MAX_CONFIDENCE, foreign_ratio + inst_ratio)

        return {
            "score": round(np.clip(score, -1, 1), 3),
            "confidence": round(confidence, 2),
            "meta": {
                "foreign_ratio": round(foreign_ratio, 3),
                "institutional_ratio": round(inst_ratio, 3),
                "retail_ratio": round(retail_ratio, 3),
                "net_foreign": net_foreign,
                "net_institutional": net_inst,
                "net_retail": net_retail
            }
        }

    def get_flow_sentiment(self) -> Dict:
        """Fetch and compute flow sentiment in one call."""
        flow = self.fetch_flow()
        return self.compute_sentiment(flow)


def get_flow_sentiment() -> Dict:
    """Convenience wrapper."""
    scraper = EGXFlowScraper()
    return scraper.get_flow_sentiment()


if __name__ == "__main__":
    print("EGXFlowScraper v4.2 ready: Market flow sentiment from EGX data")
    demo = get_flow_sentiment()
    print(f"Flow sentiment: {demo['score']:+.3f} (conf: {demo['confidence']:.0%})")
