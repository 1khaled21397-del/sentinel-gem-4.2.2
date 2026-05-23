"""
Sentinel-EGX v4.2 — EGX.com Institutional Flow Scraper
==========================================================
Scrapes EGX.com.eg investor type data and converts flow ratios into 
sentiment signals for the alpha pipeline.

Integrated with sentinel_config.json v4.2
"""

import os
import requests
from datetime import datetime
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
import json
import time

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    import pandas as pd
    PD_AVAILABLE = True
except ImportError:
    PD_AVAILABLE = False

with open("sentinel_config.json", "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

FLOW_CFG = CONFIG.get("flow_sentiment", {})


@dataclass
class FlowSentiment:
    """Sentiment signal derived from institutional flow data."""
    ticker: str
    timestamp: str
    arab_net_flow: float
    foreign_net_flow: float
    egyptian_net_flow: float
    institutional_ratio: float
    foreign_ratio: float
    sentiment_score: float
    confidence: float
    source: str
    raw_data: Dict


class EGXFlowScraper:
    """Scrape EGX.com.eg investor type data."""

    URL = "https://www.egx.com.eg/en/investorstypepiechart.aspx"

    VIEW_TYPES = {
        "all": {"type": "All", "event": "ctl00$C$rblSecuritiesBonds$0"},
        "securities": {"type": "Securities", "event": "ctl00$C$rblSecuritiesBonds$1"},
        "bonds": {"type": "Bonds", "event": "ctl00$C$rblSecuritiesBonds$2"},
    }

    def __init__(self):
        if not BS4_AVAILABLE:
            raise ImportError("beautifulsoup4 required. pip install beautifulsoup4 lxml")
        self.session = requests.Session()
        self.last_data: Dict[str, pd.DataFrame] = {}

    def _get_viewstate(self) -> str:
        resp = self.session.get(self.URL, timeout=30)
        soup = BeautifulSoup(resp.text, 'lxml')
        viewstate = soup.find('input', {'name': '__VIEWSTATE'})
        return viewstate['value'] if viewstate else ""

    def fetch_flow_data(self, view_type: str = "securities") -> Optional[pd.DataFrame]:
        if view_type not in self.VIEW_TYPES:
            raise ValueError(f"view_type must be one of {list(self.VIEW_TYPES.keys())}")

        viewstate = self._get_viewstate()
        cfg = self.VIEW_TYPES[view_type]

        params = {
            "__EVENTTARGET": cfg["event"],
            "__VIEWSTATE": viewstate,
            "__VIEWSTATEGENERATOR": "F88730C4",
            "ctl00$H$rblSearchType": "1",
            "ctl00$C$rblSecuritiesBonds": cfg["type"]
        }

        resp = self.session.post(self.URL, data=params, timeout=30)
        soup = BeautifulSoup(resp.text, 'lxml')

        tables = {
            'total': 'ctl00_C_Pc_GridView1',
            'institutions': 'ctl00_C_Pc_gvInstByNationality',
            'individuals': 'ctl00_C_Pc_gvIndByNationality'
        }

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        all_data = []

        for table_name, table_id in tables.items():
            table = soup.find('table', id=table_id)
            if not table:
                continue

            rows = table.find_all('tr')[1:]
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 2:
                    category = cols[0].get_text(strip=True)
                    value = cols[1].get_text(strip=True).replace(',', '')
                    try:
                        value = float(value)
                    except ValueError:
                        value = 0.0

                    all_data.append({
                        'table': table_name,
                        'category': category,
                        'value': value,
                        'timestamp': timestamp,
                        'view_type': view_type
                    })

        df = pd.DataFrame(all_data)
        self.last_data[view_type] = df
        return df

    def calculate_sentiment(self, df: pd.DataFrame) -> FlowSentiment:
        """Convert raw flow data into Sentinel-compatible sentiment score."""
        if df.empty:
            return FlowSentiment(
                ticker="EGX", timestamp=datetime.now().isoformat(),
                arab_net_flow=0, foreign_net_flow=0, egyptian_net_flow=0,
                institutional_ratio=0, foreign_ratio=0,
                sentiment_score=0, confidence=0, source="EGX.com",
                raw_data={}
            )

        total_df = df[df['table'] == 'total']

        # Extract values by nationality
        arab_val = total_df[total_df['category'].str.contains('Arab', case=False, na=False)]['value'].sum()
        foreign_val = total_df[total_df['category'].str.contains('Foreign', case=False, na=False)]['value'].sum()
        egyptian_val = total_df[total_df['category'].str.contains('Egyptian', case=False, na=False)]['value'].sum()

        total = arab_val + foreign_val + egyptian_val
        if total == 0:
            total = 1

        foreign_ratio = foreign_val / total
        arab_ratio = arab_val / total
        egyptian_ratio = egyptian_val / total

        # Sentinel scoring logic (config-driven)
        sentiment = (foreign_ratio - 0.33) * 3  # Baseline 33% foreign

        # Apply config bonuses/penalties
        if FLOW_CFG.get("enabled", True):
            if sentiment > FLOW_CFG.get("sentiment_threshold", 0.3):
                sentiment += FLOW_CFG.get("foreign_buy_boost", 0.15)
            elif sentiment < -FLOW_CFG.get("sentiment_threshold", 0.3):
                sentiment += FLOW_CFG.get("foreign_sell_penalty", -0.15)

        sentiment = max(-1.0, min(1.0, sentiment))

        # Confidence based on magnitude
        confidence = min(1.0, total / 1e9)

        return FlowSentiment(
            ticker="EGX30",
            timestamp=df['timestamp'].iloc[-1] if not df.empty else datetime.now().isoformat(),
            arab_net_flow=arab_val,
            foreign_net_flow=foreign_val,
            egyptian_net_flow=egyptian_val,
            institutional_ratio=0.5,
            foreign_ratio=round(foreign_ratio, 3),
            sentiment_score=round(sentiment, 3),
            confidence=round(confidence, 3),
            source="EGX.com",
            raw_data={
                "arab_ratio": round(arab_ratio, 3),
                "egyptian_ratio": round(egyptian_ratio, 3),
                "total_value": total
            }
        )

    def get_market_sentiment(self) -> Dict:
        """Fetch and analyze flow data for all view types."""
        results = {}
        for view_type in ["securities", "all"]:
            try:
                df = self.fetch_flow_data(view_type)
                if df is not None and not df.empty:
                    sentiment = self.calculate_sentiment(df)
                    results[view_type] = {
                        "sentiment_score": sentiment.sentiment_score,
                        "foreign_ratio": sentiment.foreign_ratio,
                        "confidence": sentiment.confidence,
                        "timestamp": sentiment.timestamp
                    }
            except Exception as e:
                print(f"[EGXFlow] Error fetching {view_type}: {e}")
                continue

        return results


def get_egx_flow_sentiment() -> Optional[FlowSentiment]:
    """Convenience function for Sentinel pipeline."""
    try:
        scraper = EGXFlowScraper()
        df = scraper.fetch_flow_data("securities")
        if df is not None:
            return scraper.calculate_sentiment(df)
    except Exception as e:
        print(f"[EGXFlow] Failed: {e}")
    return None


def get_flow_sentiment_for_alpha() -> Tuple[float, float, Dict]:
    """
    Returns (sentiment_score, confidence, metadata) for alpha scoring.
    Used by overnight_alpha.py.
    """
    flow = get_egx_flow_sentiment()
    if flow is None:
        # Fallback: neutral if EGX.com is down
        if FLOW_CFG.get("fallback_if_down", True):
            return 0.0, 0.0, {"note": "EGX.com unavailable, using neutral fallback"}
        return 0.0, 0.0, {"note": "EGX.com unavailable"}

    return flow.sentiment_score, flow.confidence, {
        "foreign_ratio": flow.foreign_ratio,
        "arab_net": flow.arab_net_flow,
        "foreign_net": flow.foreign_net_flow,
        "timestamp": flow.timestamp
    }


if __name__ == "__main__":
    print("EGX Flow Scraper v4.2 — Sentinel Integrated")
    print("Usage:")
    print("  from egx_flow_scraper import get_flow_sentiment_for_alpha")
    print("  score, conf, meta = get_flow_sentiment_for_alpha()")
    print("  # score: -1.0 to +1.0 | conf: 0.0 to 1.0")
