"""
Sentinel-EGX v4.2.1 — Fixed & Integrated Streamlit App
========================================================
FIXES: Double exchange suffix (data_engine), Clear Cache & Re-fetch button,
       Kimi env var name fix (KIMI_API_KEY), Dual AI sentiment wiring,
       Synthetic calendar (Sun-Thu), Debug logging.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json, os, sqlite3
from pathlib import Path
import tempfile
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── PATH SETUP ──
SCRIPT_DIR = Path(__file__).parent.resolve()
CACHE_DB = SCRIPT_DIR / "sentinel_cache.db"
CONFIG_FILE = SCRIPT_DIR / "sentinel_config.json"
WATCHLIST_FILE = SCRIPT_DIR / "my_watchlists.json"

# ── SAFE CACHE TEST ──
try:
    with open(CACHE_DB, "a"):
        pass
except OSError:
    CACHE_DB = Path(tempfile.gettempdir()) / "sentinel_cache.db"

# ── CONFIG LOAD ──
try:
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        config = json.load(f)
except FileNotFoundError:
    st.error("❌ sentinel_config.json not found. Please upload it.")
    st.stop()

# ── API KEYS (Streamlit Secrets → .env → fallback) ──
EODHD_API_KEY = ""
ANTHROPIC_API_KEY = ""
KIMI_API_KEY = ""
GEMINI_API_KEY = ""

try:
    EODHD_API_KEY = st.secrets.get("EODHD_API_KEY", "").strip()
    ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY", "").strip()
    KIMI_API_KEY = st.secrets.get("KIMI_API_KEY", "").strip()  # FIX v4.2.1: was "sentinel"
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "").strip()
except Exception:
    pass

# Fallback to env vars
if not all([EODHD_API_KEY, ANTHROPIC_API_KEY, KIMI_API_KEY, GEMINI_API_KEY]):
    try:
        from dotenv import load_dotenv
        env_path = SCRIPT_DIR / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
        else:
            load_dotenv()
        EODHD_API_KEY = EODHD_API_KEY or os.getenv("EODHD_API_KEY", "").strip()
        ANTHROPIC_API_KEY = ANTHROPIC_API_KEY or os.getenv("ANTHROPIC_API_KEY", "").strip()
        KIMI_API_KEY = KIMI_API_KEY or os.getenv("KIMI_API_KEY", "").strip()  # FIX v4.2.1
        GEMINI_API_KEY = GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "").strip()
    except ImportError:
        pass

# Export keys to environment for child modules (data_engine, sentiment_scraper)
os.environ["EODHD_API_KEY"] = EODHD_API_KEY
os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY
os.environ["KIMI_API_KEY"] = KIMI_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

# ── VALIDATE KEYS ──
missing = []
if not EODHD_API_KEY: missing.append("EODHD")
if not ANTHROPIC_API_KEY: missing.append("Claude")
if not KIMI_API_KEY: missing.append("Kimi")
if not GEMINI_API_KEY: missing.append("Gemini")

if missing:
    st.sidebar.error(f"🔴 Missing API keys: {', '.join(missing)}")
    st.sidebar.markdown("Add keys in Streamlit Secrets or .env file")

if not EODHD_API_KEY:
    st.error("❌ EODHD_API_KEY is required.")
    st.stop()

# ── PAGE CONFIG ──
st.set_page_config(page_title="Sentinel-EGX v4.2.1", layout="wide")

# ── IMPORT v4.2.1 MODULES (with graceful fallback) ──
MODULES_AVAILABLE = {}

def _clean_val(v):
    """Convert numpy types to native Python for clean display."""
    if hasattr(v, "item"):
        return v.item()
    if isinstance(v, (list, tuple)):
        return [_clean_val(x) for x in v]
    return v


try:
    from data_engine import fetch_and_build, EGXCalendar, get_segment, DataCache
    MODULES_AVAILABLE["data_engine"] = True
except Exception as e:
    MODULES_AVAILABLE["data_engine"] = False
    st.sidebar.warning(f"data_engine not loaded: {e}")

try:
    from technical_analysis import analyze_ticker, get_indicator_summary
    MODULES_AVAILABLE["technical_analysis"] = True
except Exception as e:
    MODULES_AVAILABLE["technical_analysis"] = False
    st.sidebar.warning(f"technical_analysis not loaded: {e}")

try:
    from auto_skills import analyze_skills
    MODULES_AVAILABLE["auto_skills"] = True
except Exception as e:
    MODULES_AVAILABLE["auto_skills"] = False
    st.sidebar.warning(f"auto_skills not loaded: {e}")

try:
    from gap_predictor import predict_overnight_gap
    MODULES_AVAILABLE["gap_predictor"] = True
except Exception as e:
    MODULES_AVAILABLE["gap_predictor"] = False
    st.sidebar.warning(f"gap_predictor not loaded: {e}")

try:
    from ml_forecast import MLForecastEngine
    MODULES_AVAILABLE["ml_forecast"] = True
except Exception as e:
    MODULES_AVAILABLE["ml_forecast"] = False
    st.sidebar.warning(f"ml_forecast not loaded: {e}")

try:
    from sentiment_scraper import batch_sentiment, get_sentiment_for_ticker
    MODULES_AVAILABLE["sentiment"] = True
except Exception as e:
    MODULES_AVAILABLE["sentiment"] = False
    st.sidebar.warning(f"sentiment_scraper not loaded: {e}")

try:
    from overnight_alpha import run_pipeline
    MODULES_AVAILABLE["overnight_alpha"] = True
except Exception as e:
    MODULES_AVAILABLE["overnight_alpha"] = False
    st.sidebar.warning(f"overnight_alpha not loaded: {e}")

# NEW v4.2.2: Hybrid Regime Detector
try:
    from regime_detector_v2 import HybridRegimeEnsemble, HeuristicRegimeDetector, MacroRegimeAnalyzer
    MODULES_AVAILABLE["regime_detector_v2"] = True
except Exception as e:
    MODULES_AVAILABLE["regime_detector_v2"] = False
    st.sidebar.warning(f"regime_detector_v2 not loaded: {e}")
# ── SQLITE CACHE ──
def _get_db():
    conn = sqlite3.connect(str(CACHE_DB))
    conn.row_factory = sqlite3.Row
    return conn

def init_cache():
    conn = _get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS eod_cache (symbol TEXT PRIMARY KEY, data TEXT, fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE IF NOT EXISTS alpha_results (date TEXT, ticker TEXT, data TEXT, PRIMARY KEY (date, ticker));
        CREATE TABLE IF NOT EXISTS api_calls (date TEXT PRIMARY KEY, count INTEGER DEFAULT 0);
    """)
    conn.commit()
    conn.close()

def clear_cache():
    """Clear all cached EOD data."""
    conn = _get_db()
    conn.execute("DELETE FROM eod_cache")
    conn.commit()
    conn.close()
    return True

def get_cache_stats():
    """Return cache statistics."""
    conn = _get_db()
    cursor = conn.execute("SELECT COUNT(*) as cnt, MAX(fetched_at) as latest FROM eod_cache")
    row = cursor.fetchone()
    conn.close()
    return row["cnt"], row["latest"]

init_cache()

# ── WATCHLISTS ──
def load_watchlists() -> dict:
    if os.path.exists(WATCHLIST_FILE):
        with open(WATCHLIST_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_watchlists(wl: dict):
    with open(WATCHLIST_FILE, "w", encoding="utf-8") as f:
        json.dump(wl, f, indent=2, ensure_ascii=False)

# ── TICKER DATA ──
ALL_TICKERS = config.get("tickers", [])
MARKET_SEGMENTS = config.get("market_segments", {})
T0_ENABLED = config.get("t0_rules", {}).get("t0_enabled_segments", [])

if not ALL_TICKERS:
    st.error("❌ No tickers found in sentinel_config.json")
    st.stop()

# Build reverse segment lookup
TICKER_TO_SEGMENT = {}
for seg, tickers in MARKET_SEGMENTS.items():
    for t in tickers:
        TICKER_TO_SEGMENT[t] = seg

def get_ticker_segment(ticker: str) -> str:
    return TICKER_TO_SEGMENT.get(ticker, "moderate_activity")

def is_t0_eligible(ticker: str) -> bool:
    return get_ticker_segment(ticker) in T0_ENABLED

# ── SIDEBAR ──
with st.sidebar:
    st.header("⚙️ Sentinel-EGX v4.2.1")

    st.subheader("🔑 API Status")
    st.write("🟢 EODHD" if EODHD_API_KEY else "🔴 EODHD")
    st.write("🟢 Claude" if ANTHROPIC_API_KEY else "🔴 Claude")
    st.write("🟢 Kimi" if KIMI_API_KEY else "🔴 Kimi")
    st.write("🟢 Gemini" if GEMINI_API_KEY else "🔴 Gemini")

    st.subheader("📦 Modules")
    for mod, ok in MODULES_AVAILABLE.items():
        st.write(f"{'🟢' if ok else '🔴'} {mod}")

    st.subheader("📊 Settings")
    min_alpha = st.slider("Min Alpha Score", 0.0, 1.0, 0.55, 0.05)
    top_n = st.slider("Top N Setups", 1, 20, 10)

    st.subheader("🔥 T+0 Filter")
    t0_only = st.checkbox("T+0 Eligible Only", value=False)

    st.divider()
    st.caption(f"📈 {len(ALL_TICKERS)} tickers configured")
    st.caption(f"⏰ Cache TTL: {config.get('rules', {}).get('cache_ttl_hours', 6)}h")

# ── MAIN TABS ──
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🚀 Overnight Alpha", "🔬 Single Stock", "📊 Scanner", "🗂️ Watchlists", "⚙️ Diagnostics"
])

# ── TAB 1: OVERNIGHT ALPHA ──
with tab1:
    st.header("🚀 Overnight Alpha Pipeline")
    st.caption("Full v4.2.1 pipeline: Data → Sentiment → Gap → ML → Skills → Technical → Alpha")

    col1, col2 = st.columns([3, 1])
    with col1:
        selected_tickers = st.multiselect(
            "Select Tickers to Analyze",
            options=ALL_TICKERS,
            default=ALL_TICKERS[:20] if len(ALL_TICKERS) > 20 else ALL_TICKERS,
            key="alpha_tickers"
        )
    with col2:
        st.write("")
        st.write("")
        run_alpha = st.button("▶️ Run Pipeline", type="primary", disabled=not MODULES_AVAILABLE.get("overnight_alpha"))

    if run_alpha and selected_tickers:
        progress = st.progress(0)
        status = st.empty()

        results = []
        for i, ticker in enumerate(selected_tickers):
            progress.progress((i + 1) / len(selected_tickers))
            status.text(f"Analyzing {ticker}... ({i+1}/{len(selected_tickers)})")

            try:
                if MODULES_AVAILABLE.get("overnight_alpha"):
                    from overnight_alpha import OvernightAlphaPipeline
                    pipeline = OvernightAlphaPipeline(selected_tickers)
                    res = pipeline.run_ticker(ticker)
                    if res and res.alpha >= min_alpha:
                        results.append({
                            "ticker": res.ticker,
                            "alpha": res.alpha,
                            "t0_eligible": res.t0_eligible,
                            "setup": res.setup,
                            "gap": res.gap,
                            "technical": res.technical,
                        })
                else:
                    st.warning("overnight_alpha module not available — using basic mode")
                    break
            except Exception as e:
                st.error(f"Error analyzing {ticker}: {e}")
                continue

        progress.empty()
        status.empty()

        results.sort(key=lambda x: x.get("alpha", 0), reverse=True)
        if t0_only:
            results = [r for r in results if r.get("t0_eligible", False)]
        results = results[:top_n]

        if results:
            st.success(f"Found {len(results)} setups above {min_alpha} alpha threshold")

            for i, r in enumerate(results, 1):
                with st.expander(f"#{i} {r['ticker']} | Alpha: {r['alpha']:.2f} | {r['setup']['type']}", expanded=(i==1)):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Alpha", f"{r['alpha']:.2f}")
                    c2.metric("T+0", "✅ Yes" if r['t0_eligible'] else "⏳ T+1")
                    c3.metric("Gap", f"{r['gap']['direction']} {r['gap']['probability']:.0%}")
                    c4.metric("R/R", f"{r['setup']['rr']:.1f}:1")

                    # NEW v4.2.2: Hybrid Regime Display
                    if 'hybrid_regime' in r and r['hybrid_regime']:
                        hr = r['hybrid_regime']
                        st.markdown("---")
                        hr_c1, hr_c2, hr_c3, hr_c4 = st.columns(4)
                        hr_c1.metric("Regime", hr.get('regime', 'unknown'))
                        hr_c2.metric("Position Size", f"{hr.get('position_size', 1.0):.0%}")
                        hr_c3.metric("Confidence", f"{hr.get('confidence', 0):.0%}")
                        hr_c4.metric("Disagreement", f"{hr.get('disagreement_index', 0):.2f}")

                        if hr.get('conflict_flag'):
                            st.warning(f"⚠️ Conflict: Heuristic={hr.get('heuristic_regime')} vs Macro={hr.get('macro_score'):+.2f}")

                        if hr.get('shock_detected'):
                            st.error(f"🚨 Shock: {hr.get('shock_type')}")

                        st.caption(f"💡 {hr.get('recommendation', '')}")
                        st.markdown("---")
                    # Flow Sentiment
                    if 'flow_sentiment' in r and r['flow_sentiment']:
                        flow = r['flow_sentiment']
                        flow_score = flow.get('score', 0)
                        flow_conf = flow.get('confidence', 0)
                        if flow_conf > 0.2:
                            st.markdown("**🌊 EGX Flow Sentiment:**")
                            fc1, fc2, fc3 = st.columns(3)
                            fc1.metric("Flow Score", f"{flow_score:+.2f}")
                            fc2.metric("Foreign Ratio", f"{flow.get('meta', {}).get('foreign_ratio', 0):.1%}")
                            fc3.metric("Confidence", f"{flow_conf:.0%}")

                    st.markdown("**Setup Rationale:**")
                    for note in r['setup']['rationale']:
                        st.write(f"  {note}")

                    st.markdown(f"**Entry:** {r['setup']['entry']} | **Stop:** {r['setup']['stop']} | **Target:** {r['setup']['targets']}")
                    st.markdown(f"**Best Session:** {r['setup']['best_session']} | **Gemini Score:** {r['technical']['gemini_framework']['composite']:.2f}")
        else:
            st.info("No setups above threshold. Try lowering min alpha or selecting more tickers.")

# ── TAB 2: SINGLE STOCK ──
with tab2:
    st.header("🔬 Single Stock Deep Analysis")

    ticker = st.selectbox("Select Ticker", ALL_TICKERS, key="single_ticker")
    segment = get_ticker_segment(ticker)
    t0 = is_t0_eligible(ticker)

    st.caption(f"Segment: {segment} | T+0: {'✅ Eligible' if t0 else '⏳ T+1 Only'}")

    if st.button("Analyze", type="primary"):
        with st.spinner(f"Analyzing {ticker}..."):
            try:
                if MODULES_AVAILABLE.get("data_engine"):
                    df = fetch_and_build(ticker, "EGX", lookback=400, use_cache=True)
                    is_synthetic = df.attrs.get("synthetic", False)
                    if is_synthetic:
                        st.warning(f"⚠️ Using synthetic data for {ticker} (EODHD unavailable)")
                    else:
                        st.success(f"Loaded {len(df)} bars for {ticker}")

                    # Technical Analysis
                    if MODULES_AVAILABLE.get("technical_analysis"):
                        snap, setup = analyze_ticker(df, ticker, segment)
                        summary = get_indicator_summary(snap)

                        st.subheader("📊 Technical Snapshot")
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Trend", snap.trend_direction)
                        c2.metric("RSI", f"{snap.rsi_14:.1f}")
                        c3.metric("MACD", snap.macd_state)
                        c4.metric("Gemini", f"{snap.gemini_framework_score:.2f}")

                        st.subheader("🎯 Gemini Flash Framework")
                        g = summary["gemini_framework"]
                        cg1, cg2, cg3 = st.columns(3)
                        cg1.metric("Trend Score", f"{g['trend_score']:.2f}")
                        cg2.metric("Volume Score", f"{g['volume_score']:.2f}")
                        cg3.metric("Timing Score", f"{g['timing_score']:.2f}")
                        st.progress(g["composite"], text=f"Composite: {g['composite']:.2f} — {g['signal']}")

                        st.subheader("📈 Setup Quality")
                        st.write(f"Type: **{setup.setup_type}** | Score: {_clean_val(setup.quality_score)}")
                        st.write(f"Entry: {_clean_val(setup.entry_zone)} | Stop: {_clean_val(setup.stop_loss)} | Targets: {_clean_val(setup.targets)}")
                        st.write(f"Best Session: {setup.best_session}")

                        st.subheader("📝 Rationale")
                        for note in setup.rationale:
                            st.write(note)

                    # Auto Skills
                    if MODULES_AVAILABLE.get("auto_skills"):
                        skills = analyze_skills(df, ticker, segment)
                        st.subheader("🎯 Auto Skills")
                        st.write(f"Composite Score: {skills['composite_score']:.2f} | Triggered: {skills['skills_triggered']}/7")
                        for detail in skills.get("triggered_details", []):
                            st.write(f"  {detail['skill']}: {detail['direction']} ({detail['confidence']:.0%}) — Gemini aligned: {detail['gemini_aligned']}")

                    # Gap Predictor
                    if MODULES_AVAILABLE.get("gap_predictor"):
                        gap = predict_overnight_gap(df, ticker, segment)
                        st.subheader("🌙 Gap Prediction")
                        st.write(f"Direction: {gap.gap_direction} | Probability: {gap.gap_probability:.0%}")
                        st.write(f"Expected Magnitude: {gap.expected_magnitude:.2%}")
                        st.write(f"T+0 Boost: {gap.t0_liquidity_boost:.1f}x")

                    # ML Forecast
                    if MODULES_AVAILABLE.get("ml_forecast"):
                        ml = MLForecastEngine()
                        ml.train(df)
                        ml_pred = ml.predict(df)
                        if ml_pred:
                            st.subheader("🧬 ML Forecast (7d)")
                            target_return = ml_pred.get('target_return', 0)
                            st.write(f"Target Return: {target_return:+.2%}")
                            st.write(f"Confidence: {ml_pred.get('confidence', 0):.0%}")
                            st.write(f"XGB: {ml_pred.get('xgb_pred', 0):+.2%} | RF: {ml_pred.get('rf_pred', 0):+.2%}")

                    # Sentiment (Triple AI)
                    if MODULES_AVAILABLE.get("sentiment"):
                        st.subheader("📰 Sentiment Analysis")
                        # Try to get real headlines from a news source, or use placeholder
                        # In production, you'd integrate with a news API
                        demo_headlines = [
                            f"{ticker.split('.')[0]} reports strong quarterly earnings",
                            f"Foreign investors increase holdings in {ticker.split('.')[0]}",
                            f"{ticker.split('.')[0]} announces expansion into new markets"
                        ]
                        try:
                            sent = get_sentiment_for_ticker(ticker, demo_headlines)
                            sc1, sc2, sc3 = st.columns(3)
                            sc1.metric("Score", f"{sent.score:+.2f}")
                            sc2.metric("Confidence", f"{sent.confidence:.0%}")
                            sc3.metric("AI Source", sent.ai_source or "Keyword")
                            st.write(f"Summary: {sent.summary}")
                            if sent.ai_score is not None:
                                st.caption(f"AI Score: {sent.ai_score:+.2f} (conf: {sent.ai_confidence:.0%})")
                        except Exception as e:
                            st.warning(f"Sentiment analysis failed: {e}")

                    # Chart
                    st.subheader("📈 Price Chart")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df["date"], y=df["close"], mode="lines", name="Close"))
                    if "ema_20" in df.columns:
                        fig.add_trace(go.Scatter(x=df["date"], y=df["ema_20"], mode="lines", name="EMA20", line=dict(dash="dot")))
                    if "ema_50" in df.columns:
                        fig.add_trace(go.Scatter(x=df["date"], y=df["ema_50"], mode="lines", name="EMA50", line=dict(dash="dash")))
                    if "sma_200" in df.columns:
                        fig.add_trace(go.Scatter(x=df["date"], y=df["sma_200"], mode="lines", name="SMA200"))
                    if "vwap_20d" in df.columns:
                        fig.add_trace(go.Scatter(x=df["date"], y=df["vwap_20d"], mode="lines", name="VWAP"))
                    fig.update_layout(template="plotly_dark", height=500)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("data_engine module not available")
            except ValueError as e:
                error_msg = str(e)
                if "Insufficient data" in error_msg or "0 bars" in error_msg:
                    st.error(f"📊 {error_msg}")
                    st.info("""
                    **Possible causes:**
                    • Ticker is newly listed (< 50 trading days)
                    • Ticker was delisted or suspended
                    • EODHD API key invalid or quota exceeded
                    • EODHD does not cover this ticker

                    **Try:** Select a different ticker (e.g., COMI.EGX, HRHO.EGX, FWRY.EGX)
                    """)
                elif "Unable to fetch" in error_msg:
                    st.error(f"🔌 {error_msg}")
                    st.info("Check your EODHD_API_KEY in .env or Streamlit Secrets")
                else:
                    st.error(f"Analysis failed: {e}")

                with st.expander("🔧 Debug Details"):
                    import traceback
                    st.code(traceback.format_exc())

            except Exception as e:
                st.error(f"Unexpected error: {e}")
                import traceback
                st.code(traceback.format_exc())

# ── TAB 3: SCANNER ──
with tab3:
    st.header("📊 Market Scanner")
    st.info("Batch scan across selected universe. Uses v4.2.1 Alpha scorer with all layers.")

    scan_universe = st.multiselect("Select Universe", ALL_TICKERS, default=ALL_TICKERS[:30], key="scan_universe")

    if st.button("🔍 Run Scanner", type="primary"):
        if MODULES_AVAILABLE.get("overnight_alpha"):
            with st.spinner("Scanning..."):
                try:
                    results = run_pipeline(scan_universe)
                    if results:
                        st.success(f"Found {len(results)} setups")
                        df_results = pd.DataFrame([
                            {
                                "Ticker": r["ticker"],
                                "Alpha": r["alpha"],
                                "T0": "✅" if r["t0_eligible"] else "⏳",
                                "Setup": r["setup"]["type"],
                                "Gap": r["gap"]["direction"],
                                "R/R": r["setup"]["rr"],
                                "Session": r["setup"]["best_session"],
                                "Flow": r.get("flow_sentiment", {}).get("score", 0),
                            }
                            for r in results[:top_n]
                        ])
                        st.dataframe(df_results, use_container_width=True, hide_index=True)
                    else:
                        st.info("No setups found.")
                except Exception as e:
                    st.error(f"Scanner failed: {e}")
        else:
            st.error("overnight_alpha module required for scanner")

# ── TAB 4: WATCHLISTS ──
with tab4:
    st.header("🗂️ My Watchlists")
    watchlists = load_watchlists()

    col_a, col_b = st.columns([1, 2])
    with col_a:
        new_name = st.text_input("New Watchlist Name")
        if st.button("➕ Create") and new_name.strip():
            if new_name not in watchlists:
                watchlists[new_name] = []
                save_watchlists(watchlists)
                st.rerun()

        wl_names = list(watchlists.keys())
        selected_wl = st.selectbox("Select", wl_names if wl_names else ["(none)"])

        if selected_wl in watchlists and st.button("🗑️ Delete"):
            del watchlists[selected_wl]
            save_watchlists(watchlists)
            st.rerun()

    with col_b:
        if selected_wl in watchlists:
            st.subheader(f"📌 {selected_wl} ({len(watchlists[selected_wl])} tickers)")
            add_ticker = st.selectbox("Add Ticker", [""] + [t for t in ALL_TICKERS if t not in watchlists[selected_wl]])
            if st.button("➕ Add") and add_ticker:
                watchlists[selected_wl].append(add_ticker)
                save_watchlists(watchlists)
                st.rerun()

            for t in list(watchlists[selected_wl]):
                c1, c2 = st.columns([5, 1])
                c1.write(f"`{t}` ({get_ticker_segment(t)})")
                if c2.button("🗑️", key=f"del_{t}"):
                    watchlists[selected_wl].remove(t)
                    save_watchlists(watchlists)
                    st.rerun()

# ── TAB 5: DIAGNOSTICS ──
with tab5:
    st.header("⚙️ System Diagnostics")

    st.subheader("Module Status")
    for mod, ok in MODULES_AVAILABLE.items():
        st.write(f"{'🟢' if ok else '🔴'} {mod}")

    st.subheader("Config Preview")
    with st.expander("View sentinel_config.json"):
        st.json(config)

    st.subheader("Cache Status")
    try:
        cache_count, cache_latest = get_cache_stats()
        st.write(f"Cached tickers: {cache_count}")
        if cache_latest:
            st.write(f"Latest cache update: {cache_latest}")
    except Exception as e:
        st.write(f"Cache error: {e}")

    # NEW v4.2.1: Clear Cache & Re-fetch
    st.subheader("🧹 Cache Management")
    st.caption("Clear cached data to force fresh EODHD fetches. Useful after fixing API keys or when data seems stale.")

    col_cache1, col_cache2 = st.columns([1, 3])
    with col_cache1:
        if st.button("🗑️ Clear Cache", type="secondary"):
            try:
                clear_cache()
                st.success("✅ Cache cleared successfully! Next analysis will fetch fresh data from EODHD.")
                st.balloons()
            except Exception as e:
                st.error(f"Failed to clear cache: {e}")

    with col_cache2:
        st.write("")
        st.write("")
        st.info("💡 After clearing cache, re-run any analysis to fetch fresh EODHD data.")

    st.subheader("T+0 Segment Map")
    seg_df = pd.DataFrame([
        {"Segment": seg, "Tickers": len(tickers), "T+0": seg in T0_ENABLED}
        for seg, tickers in MARKET_SEGMENTS.items()
    ])
    st.dataframe(seg_df, hide_index=True)

    st.subheader("🧠 Gap Model Training")
    # NEW v4.2.2: Regime Analysis
    st.subheader("🧠 Regime Analysis")
    st.caption("Hybrid regime detection: Heuristic per-ticker + Claude macro (1 call/day) + Disagreement handling")

    if MODULES_AVAILABLE.get("regime_detector_v2"):
        col_reg1, col_reg2 = st.columns([1, 2])

        with col_reg1:
            regime_ticker = st.selectbox(
                "Select ticker for regime analysis",
                options=ALL_TICKERS,
                key="regime_ticker"
            )
            regime_segment = TICKER_TO_SEGMENT.get(regime_ticker, "moderate_activity")

            # Optional headlines input for macro analysis
            st.caption("Optional: Add headlines for Claude macro analysis")
            headline_input = st.text_area(
                "Headlines (one per line)",
                value="CBE maintains interest rates\nEGX trading volume rises on foreign buying",
                height=80,
                key="regime_headlines"
            )
            headlines = [h.strip() for h in headline_input.split("\n") if h.strip()]

            run_regime = st.button("🔍 Analyze Regime", type="primary")

        with col_reg2:
            if run_regime:
                with st.spinner(f"Analyzing regime for {regime_ticker}..."):
                    try:
                        from regime_detector_v2 import HybridRegimeEnsemble
                        ensemble = HybridRegimeEnsemble()

                        df_reg = fetch_and_build(regime_ticker, "EGX", lookback=100, use_cache=True)
                        hybrid = ensemble.detect(
                            df_reg,
                            ticker=regime_ticker,
                            segment=regime_segment,
                            headlines=headlines if headlines else None
                        )

                        st.subheader("📊 Hybrid Regime Result")

                        rc1, rc2, rc3, rc4 = st.columns(4)
                        rc1.metric("Final Regime", hybrid.regime)
                        rc2.metric("Position Size", f"{hybrid.position_size:.0%}")
                        rc3.metric("Confidence", f"{hybrid.confidence:.0%}")
                        rc4.metric("Disagreement", f"{hybrid.disagreement_index:.2f}")

                        st.progress(hybrid.confidence, text=f"Confidence: {hybrid.confidence:.0%}")

                        # Heuristic vs Macro breakdown
                        st.subheader("🔍 Layer Breakdown")
                        h = hybrid.heuristic
                        m = hybrid.macro

                        h_c1, h_c2 = st.columns(2)
                        with h_c1:
                            st.markdown("**Heuristic Layer (per-ticker)**")
                            st.write(f"Regime: `{h.regime}`")
                            st.write(f"Slope: {h.slope_pct}% | RSI: {h.rsi}")
                            st.write(f"Volatility: {h.volatility_annual}")
                            st.write(f"Shock: {'🚨 ' + h.shock_type if h.shock_detected else '✅ None'}")
                            st.write(f"T+0 Spike: {'🔥 Yes' if h.t0_volatility_spike else '✅ No'}")

                        with h_c2:
                            st.markdown("**Macro Layer (Claude, market-wide)**")
                            st.write(f"Score: {m.macro_score:+.2f}")
                            st.write(f"Confidence: {m.confidence:.0%}")
                            st.write(f"Risk-off: {'🚨 Yes' if m.risk_off_flag else '✅ No'}")
                            st.write(f"Risk-on: {'🚀 Yes' if m.risk_on_flag else '✅ No'}")
                            st.write(f"Source: {m.source}")
                            if m.key_factors:
                                st.write(f"Factors: {', '.join(m.key_factors[:3])}")

                        # Recommendation
                        st.info(f"💡 **Recommendation:** {hybrid.recommendation}")

                        # Conflict/shock alerts
                        if hybrid.conflict_flag:
                            st.warning("⚠️ **CONFLICT DETECTED** — Heuristic and macro disagree. Reduce position or skip.")

                        if h.shock_detected:
                            st.error(f"🚨 **SHOCK DETECTED:** {h.shock_type} — Do not add new positions.")

                        # Paper trading log
                        st.subheader("📝 Paper Trading Log Entry")
                        log_entry = f"""
Date: {hybrid.timestamp[:10]}
Ticker: {regime_ticker}
Regime: {hybrid.regime}
Heuristic: {h.regime} | Macro: {m.macro_score:+.2f}
Disagreement: {hybrid.disagreement_index} | Conflict: {hybrid.conflict_flag}
Recommendation: {hybrid.recommendation}
Action: 
Outcome: 
"""
                        st.code(log_entry)

                    except Exception as e:
                        st.error(f"Regime analysis failed: {e}")
                        import traceback
                        st.code(traceback.format_exc())
    else:
        st.warning("regime_detector_v2 not available. Please ensure regime_detector_v2.py is in the project directory.")

    st.divider()
    st.caption("Train the overnight gap predictor on historical EOD data. Required before ML gap predictions.")

    col_train1, col_train2 = st.columns([1, 3])
    with col_train1:
        train_tickers = st.multiselect(
            "Tickers to train on",
            options=ALL_TICKERS,
            default=ALL_TICKERS[:10] if len(ALL_TICKERS) > 10 else ALL_TICKERS,
            key="train_tickers"
        )
    with col_train2:
        st.write("")
        st.write("")
        run_training = st.button("🏋️ Train Gap Model", type="primary", disabled=not MODULES_AVAILABLE.get("gap_predictor"))

    if run_training and train_tickers:
        from gap_predictor import train_gap_model
        train_progress = st.progress(0)
        train_status = st.empty()

        historical_data = {}
        segments = {}
        total = len(train_tickers)

        for idx, ticker in enumerate(train_tickers):
            train_status.text(f"Fetching {ticker}... ({idx+1}/{total})")
            try:
                df = fetch_and_build(ticker, "EGX", lookback=400, use_cache=True)
                historical_data[ticker] = df
                segments[ticker] = get_segment(ticker)
            except Exception as e:
                st.warning(f"Skipping {ticker}: {e}")
            train_progress.progress((idx + 1) / total)

        if len(historical_data) >= 3:
            train_status.text("Training model...")
            try:
                predictor, metrics = train_gap_model(historical_data, segments)
                predictor.save("gap_model_v42.pkl")
                train_progress.empty()
                train_status.empty()
                st.success(f"✅ Model trained on {len(historical_data)} tickers | Accuracy: {metrics.get('accuracy', 'N/A')}")
                st.json(metrics)
                st.info("🔄 Refresh the app to use the trained model for gap predictions.")
            except Exception as e:
                train_progress.empty()
                train_status.empty()
                st.error(f"Training failed: {e}")
        else:
            train_progress.empty()
            train_status.empty()
            st.error("Need at least 3 tickers with valid data to train.")

# ── FOOTER ──
st.divider()
st.caption("Sentinel-EGX v4.2.1 | Overnight Alpha + Gemini Flash + T+0/T+1 Aware | EOD Data Only")
