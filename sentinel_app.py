"""
Sentinel-EGX v4.2.1-defensive — Defensive Patch
=================================================
Handles both old and new module versions gracefully.
Adds: module verification, restart button, defensive sentiment access.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json, os, sqlite3, sys
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

# ── API KEYS ──
EODHD_API_KEY = ""
ANTHROPIC_API_KEY = ""
KIMI_API_KEY = ""

try:
    EODHD_API_KEY = st.secrets.get("EODHD_API_KEY", "").strip()
    ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY", "").strip()
    KIMI_API_KEY = st.secrets.get("KIMI_API_KEY", "").strip()
    if not KIMI_API_KEY and "sentinel" in st.secrets:
        sentinel_cfg = st.secrets.get("sentinel", {})
        if isinstance(sentinel_cfg, dict):
            KIMI_API_KEY = sentinel_cfg.get("KIMI_API_KEY", "").strip()
except Exception:
    pass

if not all([EODHD_API_KEY, ANTHROPIC_API_KEY, KIMI_API_KEY]):
    try:
        from dotenv import load_dotenv
        env_path = SCRIPT_DIR / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
        else:
            load_dotenv()
        EODHD_API_KEY = EODHD_API_KEY or os.getenv("EODHD_API_KEY", "").strip()
        ANTHROPIC_API_KEY = ANTHROPIC_API_KEY or os.getenv("ANTHROPIC_API_KEY", "").strip()
        KIMI_API_KEY = KIMI_API_KEY or os.getenv("KIMI_API_KEY", "").strip()
    except ImportError:
        pass

os.environ["EODHD_API_KEY"] = EODHD_API_KEY
os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY
os.environ["KIMI_API_KEY"] = KIMI_API_KEY

# ── VALIDATE KEYS ──
missing = []
if not EODHD_API_KEY: missing.append("EODHD")
if not ANTHROPIC_API_KEY: missing.append("Claude")
if not KIMI_API_KEY: missing.append("Kimi")

if missing:
    st.sidebar.error(f"🔴 Missing API keys: {', '.join(missing)}")

if not EODHD_API_KEY:
    st.error("❌ EODHD_API_KEY is required.")
    st.stop()

st.set_page_config(page_title="Sentinel-EGX v4.2.1", layout="wide")

# ── MODULE VERIFICATION ──
MODULES_AVAILABLE = {}
MODULE_VERSIONS = {}

def _clean_val(v):
    if hasattr(v, "item"):
        return v.item()
    if isinstance(v, (list, tuple)):
        return [_clean_val(x) for x in v]
    return v

def verify_module(module_name, expected_attrs=None):
    """Import module and verify it has expected attributes."""
    try:
        mod = __import__(module_name)
        MODULES_AVAILABLE[module_name] = True

        # Check version marker
        ver = getattr(mod, '__version__', getattr(mod, 'VERSION', 'unknown'))
        MODULE_VERSIONS[module_name] = ver

        # Check expected attributes
        if expected_attrs:
            missing_attrs = [a for a in expected_attrs if not hasattr(mod, a)]
            if missing_attrs:
                st.sidebar.warning(f"⚠️ {module_name} loaded but missing: {', '.join(missing_attrs)}")
                return False
        return True
    except Exception as e:
        MODULES_AVAILABLE[module_name] = False
        st.sidebar.warning(f"🔴 {module_name} not loaded: {e}")
        return False

# Verify data_engine
de_ok = verify_module("data_engine", ["fetch_and_build", "get_segment", "DataCache"])
if de_ok:
    from data_engine import fetch_and_build, get_segment, DataCache
    # Verify URL fix is present
    import data_engine as de_mod
    if hasattr(de_mod, '_build_url'):
        test_url = de_mod._build_url("COMI.EGX", "EGX", "d", 500)
        if "COMI.EGX.EGX" in test_url:
            st.sidebar.error("🔴 data_engine has DOUBLE SUFFIX BUG — replace file and restart!")
        else:
            st.sidebar.success("🟢 data_engine URL fix verified")

# Verify other modules
verify_module("technical_analysis", ["analyze_ticker", "get_indicator_summary"])
if MODULES_AVAILABLE.get("technical_analysis"):
    from technical_analysis import analyze_ticker, get_indicator_summary

verify_module("auto_skills", ["analyze_skills"])
if MODULES_AVAILABLE.get("auto_skills"):
    from auto_skills import analyze_skills

verify_module("gap_predictor", ["predict_overnight_gap"])
if MODULES_AVAILABLE.get("gap_predictor"):
    from gap_predictor import predict_overnight_gap

verify_module("ml_forecast", ["MLForecastEngine"])
if MODULES_AVAILABLE.get("ml_forecast"):
    from ml_forecast import MLForecastEngine

# Verify sentiment_scraper with defensive handling
sentiment_ok = verify_module("sentiment_scraper", ["get_sentiment_for_ticker"])
if sentiment_ok:
    from sentiment_scraper import get_sentiment_for_ticker
    # Check if new version (has ai_source)
    import sentiment_scraper as ss_mod
    from dataclasses import fields
    if hasattr(ss_mod, 'SentimentResult'):
        sent_fields = [f.name for f in fields(ss_mod.SentimentResult)]
        if 'ai_source' in sent_fields:
            st.sidebar.success("🟢 sentiment_scraper v4.2.1+ verified")
        else:
            st.sidebar.warning("⚠️ sentiment_scraper is OLD version — replace file and restart!")
            sentiment_ok = False
    else:
        sentiment_ok = False

MODULES_AVAILABLE["sentiment"] = sentiment_ok

verify_module("overnight_alpha", ["run_pipeline"])
if MODULES_AVAILABLE.get("overnight_alpha"):
    from overnight_alpha import run_pipeline

# ── CACHE ──
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

def clear_eod_cache():
    conn = _get_db()
    conn.execute("DELETE FROM eod_cache")
    conn.commit()
    conn.close()
    return True

def get_cache_stats():
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

    st.subheader("📦 Modules")
    for mod, ok in MODULES_AVAILABLE.items():
        st.write(f"{'🟢' if ok else '🔴'} {mod}")

    st.subheader("📊 Settings")
    min_alpha = st.slider("Min Alpha Score", 0.0, 1.0, 0.55, 0.05)
    top_n = st.slider("Top N Setups", 1, 20, 10)
    t0_only = st.checkbox("T+0 Eligible Only", value=False)

    st.divider()
    st.caption(f"📈 {len(ALL_TICKERS)} tickers")
    st.caption(f"⏰ Cache TTL: {config.get('rules', {}).get('cache_ttl_hours', 6)}h")

# ── TABS ──
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🚀 Overnight Alpha", "🔬 Single Stock", "📊 Scanner", "🗂️ Watchlists", "⚙️ Diagnostics"
])

# ── TAB 1: OVERNIGHT ALPHA ──
with tab1:
    st.header("🚀 Overnight Alpha Pipeline")
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_tickers = st.multiselect(
            "Select Tickers", options=ALL_TICKERS,
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
                            "ticker": res.ticker, "alpha": res.alpha,
                            "t0_eligible": res.t0_eligible, "setup": res.setup,
                            "gap": res.gap, "technical": res.technical,
                        })
            except Exception as e:
                st.error(f"Error analyzing {ticker}: {e}")
        progress.empty()
        status.empty()

        results.sort(key=lambda x: x.get("alpha", 0), reverse=True)
        if t0_only:
            results = [r for r in results if r.get("t0_eligible", False)]
        results = results[:top_n]

        if results:
            st.success(f"Found {len(results)} setups")
            for i, r in enumerate(results, 1):
                with st.expander(f"#{i} {r['ticker']} | Alpha: {r['alpha']:.2f} | {r['setup']['type']}", expanded=(i==1)):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Alpha", f"{r['alpha']:.2f}")
                    c2.metric("T+0", "✅ Yes" if r['t0_eligible'] else "⏳ T+1")
                    c3.metric("Gap", f"{r['gap']['direction']} {r['gap']['probability']:.0%}")
                    c4.metric("R/R", f"{r['setup']['rr']:.1f}:1")
                    st.markdown("**Setup Rationale:**")
                    for note in r['setup']['rationale']:
                        st.write(f"  {note}")
        else:
            st.info("No setups above threshold.")

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
                        st.warning(f"⚠️ Using synthetic data for {ticker} (EODHD unavailable or API quota exceeded).")
                        st.info("""
                        **Most likely cause:** EODHD API quota exhausted

                        Your plan: 20 calls/day (free tier) | Configured for: 5,000 calls/day

                        **Solutions:**
                        1. Wait for daily reset (midnight UTC)
                        2. Upgrade to $19.99/mo plan at eodhd.com
                        3. Check EODHD dashboard for quota status
                        """)
                    else:
                        st.success(f"Loaded {len(df)} real EODHD bars for {ticker}")

                    # Technical
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

                    # Gap
                    if MODULES_AVAILABLE.get("gap_predictor"):
                        gap = predict_overnight_gap(df, ticker, segment)
                        st.subheader("🌙 Gap Prediction")
                        st.write(f"Direction: {gap.gap_direction} | Probability: {gap.gap_probability:.0%}")
                        st.write(f"Expected Magnitude: {gap.expected_magnitude:.2%}")
                        st.write(f"T+0 Boost: {gap.t0_liquidity_boost:.1f}x")

                    # ML
                    if MODULES_AVAILABLE.get("ml_forecast"):
                        ml = MLForecastEngine()
                        ml.train(df)
                        ml_pred = ml.predict(df)
                        if ml_pred:
                            st.subheader("🧬 ML Forecast (7d)")
                            st.write(f"Target Return: {ml_pred.get('target_return', 0):+.2%}")
                            st.write(f"Confidence: {ml_pred.get('confidence', 0):.0%}")

                    # Sentiment — DEFENSIVE: handles old and new sentiment_scraper
                    st.subheader("📰 Sentiment Analysis")
                    demo_headlines = [
                        f"{ticker.split('.')[0]} reports strong quarterly earnings",
                        f"Foreign investors increase holdings in {ticker.split('.')[0]}",
                        f"{ticker.split('.')[0]} announces expansion into new markets"
                    ]
                    try:
                        if sentiment_ok:
                            sent = get_sentiment_for_ticker(ticker, demo_headlines)
                            sc1, sc2, sc3 = st.columns(3)
                            sc1.metric("Score", f"{sent.score:+.2f}")
                            sc2.metric("Confidence", f"{sent.confidence:.0%}")
                            # DEFENSIVE: check if ai_source exists (handles old module)
                            ai_src = getattr(sent, 'ai_source', None)
                            sc3.metric("AI Source", ai_src or "Keyword")
                            st.write(f"Summary: {sent.summary}")
                            ai_score = getattr(sent, 'ai_score', None)
                            ai_conf = getattr(sent, 'ai_confidence', None)
                            if ai_score is not None:
                                st.caption(f"AI Score: {ai_score:+.2f} (conf: {ai_conf:.0%})")
                        else:
                            st.info("Sentiment module unavailable or outdated. Replace sentiment_scraper.py and restart.")
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
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                with st.expander("🔧 Debug Details"):
                    import traceback
                    st.code(traceback.format_exc())

# ── TAB 3: SCANNER ──
with tab3:
    st.header("📊 Market Scanner")
    scan_universe = st.multiselect("Select Universe", ALL_TICKERS, default=ALL_TICKERS[:30], key="scan_universe")
    if st.button("🔍 Run Scanner", type="primary"):
        if MODULES_AVAILABLE.get("overnight_alpha"):
            with st.spinner("Scanning..."):
                try:
                    results = run_pipeline(scan_universe)
                    if results:
                        st.success(f"Found {len(results)} setups")
                        df_results = pd.DataFrame([{
                            "Ticker": r["ticker"], "Alpha": r["alpha"],
                            "T0": "✅" if r["t0_eligible"] else "⏳",
                            "Setup": r["setup"]["type"], "Gap": r["gap"]["direction"],
                            "R/R": r["setup"]["rr"], "Session": r["setup"]["best_session"],
                        } for r in results[:top_n]])
                        st.dataframe(df_results, use_container_width=True, hide_index=True)
                    else:
                        st.info("No setups found.")
                except Exception as e:
                    st.error(f"Scanner failed: {e}")
        else:
            st.error("overnight_alpha module required")

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
        ver = MODULE_VERSIONS.get(mod, '?')
        st.write(f"{'🟢' if ok else '🔴'} {mod} (ver: {ver})")

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

    # Cache Management
    st.subheader("🧹 Cache Management")
    col_cache1, col_cache2 = st.columns([1, 3])
    with col_cache1:
        if st.button("🗑️ Clear EOD Cache", type="secondary"):
            try:
                clear_eod_cache()
                st.success("✅ EOD cache cleared! Next analysis will fetch fresh EODHD data.")
                st.balloons()
            except Exception as e:
                st.error(f"Failed: {e}")
    with col_cache2:
        st.write("")
        st.write("")
        st.info("💡 Clear cache to force fresh EODHD fetches after fixing API keys.")

    # NEW: Restart Server
    st.subheader("🔄 Server Management")
    st.caption("Python caches imported modules in memory. After replacing .py files, you MUST restart the server.")
    if st.button("🔄 Restart Streamlit Server", type="primary"):
        st.warning("Restarting server... Please wait 10-20 seconds and refresh the page.")
        # Trigger Streamlit's auto-reload by touching this file
        Path(__file__).touch()
        # Alternative: use os._exit to force restart
        import time
        time.sleep(1)
        os._exit(0)  # Force process restart

    st.subheader("T+0 Segment Map")
    seg_df = pd.DataFrame([
        {"Segment": seg, "Tickers": len(tickers), "T+0": seg in T0_ENABLED}
        for seg, tickers in MARKET_SEGMENTS.items()
    ])
    st.dataframe(seg_df, hide_index=True)

st.divider()
st.caption("Sentinel-EGX v4.2.1 | Overnight Alpha + Gemini Flash + T+0/T+1 Aware | EOD Data Only")
