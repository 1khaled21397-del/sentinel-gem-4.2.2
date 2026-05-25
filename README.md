# 🛡️ Sentinel-EGX v4.2.2

**AI-Powered Overnight Alpha & Technical Analysis for the Egyptian Exchange (EGX)**

> EOD data only | 263-ticker universe | T+0/T+1 aware | Delta-Update Caching

---

## 📋 Overview

Sentinel-EGX is a comprehensive quantitative trading intelligence system for the Egyptian Stock Exchange (EGX). It combines:

- **Technical Analysis**: VWAP, Anchored VWAP, CMF, OBV, RSI, Stochastic RSI, MACD, EMA 20/50, SMA 200
- **ML Forecasting**: XGBoost + Random Forest ensemble for 7-day return prediction
- **Gap Prediction**: 35-feature model for overnight gap forecasting
- **Sentiment Analysis**: Triple AI (Claude + Kimi + Gemini) with keyword fallback
- **Hybrid Regime Detection**: Heuristic per-ticker + Claude macro + disagreement handling
- **Auto Skills**: 7 pattern recognition algorithms (breakout, mean reversion, trend following, etc.)
- **Delta-Update Caching**: Intelligent EOD data caching — fetches only missing date ranges

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file or set Streamlit Secrets:

```
EODHD_API_KEY=your_eodhd_key
ANTHROPIC_API_KEY=your_claude_key
KIMI_API_KEY=your_kimi_key
GEMINI_API_KEY=your_gemini_key
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

### Run Streamlit App

```bash
streamlit run sentinel_app.py
```

### Run GitHub Actions Pipeline

The pre-market pipeline runs automatically at 06:30 Cairo time (Sun-Thu):
- Fetches EOD data (delta-update: only missing ranges)
- Runs full analysis pipeline
- Sends Telegram digest with top setups

---

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  EODHD API      │────▶│  DeltaCache     │────▶│  data_engine    │
│  (EOD data)     │     │  (row-level)    │     │  (indicators)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                         │
    ┌──────────────────────────────────────────────────────┘
    ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ technical_      │────▶│  overnight_     │────▶│  sentinel_app   │
│ analysis.py     │     │  alpha.py       │     │  (Streamlit UI) │
└─────────────────┘     └─────────────────┘     └─────────────────┘
    │                           │
    ▼                           ▼
┌─────────────────┐     ┌─────────────────┐
│  gap_predictor  │     │  ml_forecast    │
│  (35 features)  │     │  (XGB + RF)     │
└─────────────────┘     └─────────────────┘
    │                           │
    └──────────┬────────────────┘
               ▼
        ┌─────────────────┐
        │  regime_detector│
        │  (hybrid)       │
        └─────────────────┘
```

---

## 📁 File Structure

| File | Purpose |
|------|---------|
| `sentinel_app.py` | Streamlit UI — Scanner, Single Stock, Watchlists, Diagnostics |
| `data_engine.py` | EODHD fetcher + 9 indicators + **DeltaCache integration** |
| `delta_cache.py` | **NEW v4.2.2** — Row-level SQLite cache with gap detection |
| `technical_analysis.py` | Full TA engine (VWAP, CMF, OBV, RSI, StochRSI, MACD, EMA/SMA) |
| `gap_predictor.py` | 35-feature overnight gap prediction (XGB/RF) |
| `ml_forecast.py` | 7-day return forecast (XGBoost + Random Forest ensemble) |
| `sentiment_scraper.py` | Triple AI sentiment (Claude + Kimi + Gemini) |
| `regime_detector.py` | Hybrid regime detection (heuristic + macro + ensemble) |
| `auto_skills.py` | 7 pattern recognition skills |
| `overnight_alpha.py` | Full pipeline orchestrator |
| `backtest_engine.py` | Walk-forward backtesting |
| `hedge_engine.py` | EGX30 futures beta-adjusted hedging |
| `egx_flow_scraper.py` | EGX market flow sentiment |
| `sentinel_config.json` | Central configuration |
| `deploy.yml` | GitHub Actions pre-market pipeline |
| `devcontainer.json` | GitHub Codespaces configuration |
| `requirements.txt` | Python dependencies |

---

## 🔄 Delta-Update Caching (v4.2.2)

### Problem
Previously, `data_engine.py` stored entire DataFrames as JSON blobs in SQLite. Every cache refresh required re-fetching the full history — expensive for 263 tickers on a $20/mo EODHD plan.

### Solution
**DeltaCache** stores data **per-row** (symbol + date PK) and only fetches missing trading days:

```python
from delta_cache import DeltaCache

dc = DeltaCache()

# Check what's missing
gaps = dc.get_missing_ranges("COMI.EGX", "2024-01-01", "2026-05-25")
# Returns: [("2026-05-20", "2026-05-25")]  # only 6 days to fetch!

# Fetch delta, merge, return full dataset
full_df = dc.merge_fetch("COMI.EGX", new_data)
```

### Benefits
- **~95% fewer API calls** for daily updates
- **Faster startup** (no full re-fetch)
- **Separate DB** (`sentinel_delta_cache.db`) — no migration from legacy cache
- **Backward compatible** — falls back to legacy `DataCache` if unavailable

---

## 📊 Key Features

### T+0 / T+1 Awareness
EGX market segments (per Feb 2024 restructure):
- **T+0**: high_activity, moderate_activity, egx30
- **T+1 only**: low_activity

### Gemini Flash Framework
3-step entry confirmation:
1. **Trend**: EMA50 > SMA200
2. **Volume**: OBV rising + CMF > 0
3. **Timing**: RSI pullback to 40-50 + price near EMA20

### Hybrid Regime Detection
- **Layer 1**: Enhanced heuristic (per-ticker, free, deterministic)
- **Layer 2**: Claude macro analyzer (market-wide, 1 call/day)
- **Layer 3**: Ensemble with disagreement handling + EGX shock detection

---

## ⚙️ Configuration

Edit `sentinel_config.json`:

```json
{
  "cache_ttl_hours": 12,
  "technical_analysis": {
    "ema_periods": [20, 50],
    "sma_periods": [200],
    "rsi_period": 14
  },
  "alpha_scorer": {
    "min_alpha_to_report": 0.55,
    "top_n_setups": 10
  }
}
```

---

## 🧪 Testing

```bash
# Verify all imports
python -c "import data_engine; import delta_cache; import technical_analysis; print('OK')"

# Test DeltaCache
dc = DeltaCache()
dc.get_missing_ranges("COMI.EGX", "2024-01-01", "2026-05-25")

# Test synthetic data
df = data_engine._generate_synthetic_data("TEST.EGX", 100)
assert len(df) == 100
```

---

## 📜 License

MIT License — see LICENSE file

---

**Version**: 4.2.2 | **Date**: 2026-05-25 | **Exchange**: EGX (Egyptian Exchange)
