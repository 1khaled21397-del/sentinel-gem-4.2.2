# Sentinel-EGX v4.2

> **Complete EGX Stock Scanner & Overnight Alpha Pipeline**  
> 263-ticker universe | EOD Data Only | T+0/T+1 Aware | Gemini Flash Framework

---

## ⚠️ DISCLAIMER

**Sentinel is a financial analytics and data visualization platform. All forecasts, signals, and AI-generated content are for informational and educational purposes only. Users must conduct their own research and consult a licensed financial advisor before making investment decisions. Sentinel does not provide investment advice, manage portfolios, or execute trades.**

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    Sentinel-EGX v4.2 Pipeline                    │
├────────────────────────────────────────────────────────────────────────────┤
│  Data Layer                                                      │
│    ├── data_engine.py        → EODHD fetch + 9 indicators       │
│    └── sentinel_cache.db      → SQLite TTL cache (6h)           │
├────────────────────────────────────────────────────────────────────────────┤
│  Analysis Layer                                                  │
│    ├── technical_analysis.py → VWAP, CMF, OBV, RSI, StochRSI  │
│    │                            MACD, EMA20/50, SMA200          │
│    ├── gap_predictor.py      → 35-feature gap prediction        │
│    ├── ml_forecast.py        → XGBoost + Random Forest ensemble │
│    ├── auto_skills.py        → 7 pattern recognition skills     │
│    ├── sentiment_scraper.py  → Keyword-based sentiment scoring  │
│    └── regime_detector.py    → Bull/sideways/bear classifier    │
├────────────────────────────────────────────────────────────────────────────┤
│  Integration Layer                                               │
│    ├── overnight_alpha.py    → Full pipeline orchestrator       │
│    ├── hedge_engine.py       → EGX30 futures beta hedge         │
│    └── backtest_engine.py    → Walk-forward backtester          │
├────────────────────────────────────────────────────────────────────────────┤
│  Presentation Layer                                              │
│    └── sentinel_app.py       → Streamlit dashboard              │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

Create `.env` in project root:

```bash
EODHD_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
KIMI_API_KEY=your_key_here
```

Or add to Streamlit Secrets for cloud deployment.

### 3. Run Streamlit App

```bash
streamlit run sentinel_app.py
```

### 4. Run Pre-Market Pipeline (CLI)

```bash
python overnight_alpha.py
```

---

## Key Features

| Feature | Description | Config Key |
|---------|-------------|------------|
| **263 Tickers** | Complete EGX universe | `tickers` |
| **T+0/T+1 Filter** | Segment-aware liquidity scoring | `t0_rules` |
| **Gemini Flash** | 3-step framework (Trend → Volume → Timing) | `gemini_flash_framework` |
| **Gap Prediction** | 35 features, XGB/RF ensemble | `gap_predictor` |
| **ML Forecast** | 7-day return prediction | `ml_forecast` |
| **7 Auto Skills** | Breakout, mean reversion, trend, volume, gap fill, S/R | `auto_skills` |
| **Walk-Forward Backtest** | 5-split regime-aware testing | `backtest` |
| **Hedge Engine** | EGX30 futures beta-adjusted | `hedge_engine` |
| **Telegram Digest** | Daily pre-market alerts | `telegram` |

---

## T+0/T+1 Market Segments (EGX Feb 2024)

| Segment | T+0 Eligible | Price Limit | Examples |
|---------|-------------|-------------|----------|
| `egx30` | ✅ Yes | ±30% | COMI, HRHO, TMGH |
| `egx70` | ✅ Yes | ±15% | DSCW, MEPA, ATQA |
| `high_activity` | ✅ Yes | ±20% | FWRY, ORWE, EFID |
| `moderate_activity` | ✅ Yes | ±20% | EFIC, CIEB, SVCE |
| `low_activity` | ❌ T+1 Only | ±5% | BTFH, SIDI, FPCM |

---

## Module Reference

### `data_engine.py`
- `fetch_and_build(symbol, exchange="EGX", lookback=400)` → enriched DataFrame
- `get_segment(ticker)` → market segment lookup
- `EGXCalendar(start, end)` → trading day generator

### `technical_analysis.py`
- `analyze_ticker(df, ticker, segment)` → (IndicatorSnapshot, SetupQuality)
- `get_indicator_summary(snapshot)` → UI-ready dict

### `gap_predictor.py`
- `predict_overnight_gap(df, ticker, segment)` → GapPrediction
- `train_gap_model(historical_data, segments)` → (predictor, metrics)

### `ml_forecast.py`
- `MLForecastEngine.train(df)` → fits XGB+RF ensemble
- `MLForecastEngine.predict(df)` → 7-day return forecast

### `auto_skills.py`
- `analyze_skills(df, ticker, segment)` → composite score + triggered skills

### `overnight_alpha.py`
- `run_pipeline(tickers)` → top setups sorted by alpha score

---

## Configuration

All behavior is controlled by `sentinel_config.json`:

```json
{
  "alpha_scorer": {
    "weights": {
      "gap_magnitude": 0.20,
      "gap_confidence": 0.15,
      "ml_7d": 0.15,
      "sentiment": 0.15,
      "auto_skills": 0.10,
      "technical": 0.10,
      "weekly_confluence": 0.10,
      "sr_bonus": 0.05
    }
  }
}
```

---

## Deployment

### GitHub Actions (Pre-Market Pipeline)

See `deploy.yml` for cron-scheduled pipeline at 06:30 Cairo time (Sun-Thu).

### Dev Container / Codespaces

Open in GitHub Codespaces — `.devcontainer/devcontainer.json` pre-configured.

---

## License

MIT License — see full text in repository.

---

## Version History

| Version | Date | Key Changes |
|---------|------|-------------|
| v3.7 | 2024 | 87 tickers, VAMP prediction, basic backtest |
| v4.1 | 2025 | Modular architecture, sentiment scraper |
| **v4.2** | **2026** | **263 tickers, Gemini Flash, T+0/T+1, full indicator suite** |

---

*Built for the Egyptian Exchange (EGX). EOD data only — no intraday feeds available.*
