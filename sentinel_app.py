"""
Sentinel-EGX v4.2.2 — Streamlit App (Integrated)
==================================================
Adds to v4.2.1:
  • Tab 6 — 🧠 Self-Learning  (sentinel_learning.render_tab)
  • Tab 7 — 📋 Reports        (sentinel_reports.render_tab)
  • Anthropic SDK client for analyst-report PDF/image parsing
  • Daily learning maintenance triggered once per session on startup

FIXES carried forward from v4.2.1:
  Double exchange suffix, Clear Cache button, Kimi env-var name,
  Dual AI sentiment wiring, Synthetic calendar (Sun-Thu), Debug logging.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json, os, sqlite3
from pathlib import Path
import tempfile
import anthropic                          # ← NEW: SDK client for reports module

# --- UI Constants ---
DEFAULT_MIN_ALPHA       = 0.55
ALPHA_STEP              = 0.05
TOP_N_MIN               = 1
TOP_N_MAX               = 20
TOP_N_DEFAULT           = 10
CHART_HEIGHT            = 500
HEADLINES_TEXTAREA_HEIGHT = 80
DATE_SLICE_LENGTH       = 10
# --------------------

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── PATH SETUP ──
SCRIPT_DIR   = Path(__file__).parent.resolve()
CACHE_DB     = SCRIPT_DIR / "sentinel_cache.db"
CONFIG_FILE  = SCRIPT_DIR / "sentinel_config.json"
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
EODHD_API_KEY       = ""
ANTHROPIC_API_KEY   = ""
KIMI_API_KEY        = ""
GEMINI_API_KEY      = ""

try:
    EODHD_API_KEY     = st.secrets.get("EODHD_API_KEY",     "").strip()
    ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY", "").strip()
    KIMI_API_KEY      = st.secrets.get("KIMI_API_KEY",       "").strip()
    GEMINI_API_KEY    = st.secrets.get("GEMINI_API_KEY",     "").strip()
except Exception:
    pass

if not all([EODHD_API_KEY, ANTHROPIC_API_KEY, KIMI_API_KEY, GEMINI_API_KEY]):
    try:
        from dotenv import load_dotenv
        env_path = SCRIPT_DIR / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
        else:
            load_dotenv()
        EODHD_API_KEY     = EODHD_API_KEY     or os.getenv("EODHD_API_KEY",     "").strip()
        ANTHROPIC_API_KEY = ANTHROPIC_API_KEY or os.getenv("ANTHROPIC_API_KEY", "").strip()
        KIMI_API_KEY      = KIMI_API_KEY      or os.getenv("KIMI_API_KEY",       "").strip()
        GEMINI_API_KEY    = GEMINI_API_KEY    or os.getenv("GEMINI_API_KEY",     "").strip()
    except ImportError:
        pass

os.environ["EODHD_API_KEY"]     = EODHD_API_KEY
os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY
os.environ["KIMI_API_KEY"]      = KIMI_API_KEY
os.environ["GEMINI_API_KEY"]    = GEMINI_API_KEY

# ── ANTHROPIC SDK CLIENT (used by sentinel_reports for PDF / image parsing) ──
claude_client: anthropic.Anthropic | None = None
if ANTHROPIC_API_KEY:
    try:
        claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    except Exception:
        claude_client = None

# ── VALIDATE KEYS ──
missing = []
if not EODHD_API_KEY:     missing.append("EODHD")
if not ANTHROPIC_API_KEY: missing.append("Claude")
if not KIMI_API_KEY:      missing.append("Kimi")
if not GEMINI_API_KEY:    missing.append("Gemini")

if missing:
    st.sidebar.error(f"🔴 Missing API keys: {', '.join(missing)}")
    st.sidebar.markdown("Add keys in Streamlit Secrets or .env file")

if not EODHD_API_KEY:
    st.error("❌ EODHD_API_KEY is required.")
    st.stop()

# ── PAGE CONFIG ──
st.set_page_config(page_title="Sentinel-EGX v4.2.2", layout="wide")

# ── IMPORT MODULES ──
MODULES_AVAILABLE = {}

def _clean_val(v):
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

try:
    from regime_detector import HybridRegimeEnsemble, HeuristicRegimeDetector, MacroRegimeAnalyzer
    MODULES_AVAILABLE["regime_detector_v2"] = True
except Exception as e:
    MODULES_AVAILABLE["regime_detector_v2"] = False
    st.sidebar.warning(f"regime_detector_v2 not loaded: {e}")

# ── NEW: Self-Learning Module ──
try:
    import sentinel_learning as learning
    MODULES_AVAILABLE["sentinel_learning"] = True
except Exception as e:
    learning = None
    MODULES_AVAILABLE["sentinel_learning"] = False
    st.sidebar.warning(f"sentinel_learning not loaded: {e}")

# ── NEW: Analyst Reports Module ──
try:
    import sentinel_reports as reports
    MODULES_AVAILABLE["sentinel_reports"] = True
except Exception as e:
    reports = None
    MODULES_AVAILABLE["sentinel_reports"] = False
    st.sidebar.warning(f"sentinel_reports not loaded: {e}")

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

# ── DAILY LEARNING INIT (runs once per Streamlit session) ──
if learning and "learning_init_done" not in st.session_state:
    try:
        _summary = learning.run_daily_if_needed()
        st.session_state["learning_init_done"] = True
        st.session_state["learning_init_summary"] = _summary
    except Exception as _e:
        st.session_state["learning_init_done"] = True
        st.session_state["learning_init_summary"] = {"status": "error", "detail": str(_e)}

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
ALL_TICKERS      = config.get("tickers", [])
MARKET_SEGMENTS  = config.get("market_segments", {})
T0_ENABLED       = config.get("t0_rules", {}).get("t0_enabled_segments", [])

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
    st.header("⚙️ Sentinel-EGX v4.2.2")

    st.subheader("🔑 API Status")
    st.write("🟢 EODHD"  if EODHD_API_KEY     else "🔴 EODHD")
    st.write("🟢 Claude" if ANTHROPIC_API_KEY  else "🔴 Claude")
    st.write("🟢 Kimi"   if KIMI_API_KEY       else "🔴 Kimi")
    st.write("🟢 Gemini" if GEMINI_API_KEY      else "🔴 Gemini")

    st.subheader("📦 Modules")
    for mod, ok in MODULES_AVAILABLE.items():
        st.write(f"{'🟢' if ok else '🔴'} {mod}")

    # Learning daily-init status
    if "learning_init_summary" in st.session_state:
        _ls = st.session_state["learning_init_summary"]
        if _ls.get("status") == "ran":
            st.caption(
                f"🧠 Learning: {_ls.get('outcomes_filled',0)} outcomes, "
                f"{len(_ls.get('adjustments',[]))} adjustments"
            )
        elif _ls.get("status") == "skipped":
            st.caption("🧠 Learning: already ran today")

    st.subheader("📊 Settings")
    min_alpha = st.slider("Min Alpha Score", 0.0, 1.0, DEFAULT_MIN_ALPHA, ALPHA_STEP)
    top_n     = st.slider("Top N Setups", TOP_N_MIN, TOP_N_MAX, TOP_N_DEFAULT)

    st.subheader("🔥 T+0 Filter")
    t0_only = st.checkbox("T+0 Eligible Only", value=False)

    st.divider()
    st.caption(f"📈 {len(ALL_TICKERS)} tickers configured")
    st.caption(f"⏰ Cache TTL: {config.get('rules', {}).get('cache_ttl_hours', 6)}h")

# ── MAIN TABS ──
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🚀 Overnight Alpha",
    "🔬 Single Stock",
    "📊 Scanner",
    "🗂️ Watchlists",
    "⚙️ Diagnostics",
    "🧠 Self-Learning",      # NEW
    "📋 Reports",             # NEW
])

# ══════════════════════════════════════════════════════════════════════
# TAB 1 — OVERNIGHT ALPHA
# ══════════════════════════════════════════════════════════════════════
with tab1:
    st.header("🚀 Overnight Alpha Pipeline")
    st.caption("Full v4.2.2 pipeline: Data → Sentiment → Gap → ML → Skills → Technical → Alpha")

    col1, col2 = st.columns([3, 1])
    with col1:
        selected_tickers = st.multiselect(
            "Select Tickers to Analyze",
            options=ALL_TICKERS,
            default=ALL_TICKERS[:20] if len(ALL_TICKERS) > 20 else ALL_TICKERS,
            key="alpha_tickers",
        )
    with col2:
        st.write("")
        st.write("")
        run_alpha = st.button(
            "▶️ Run Pipeline", type="primary",
            disabled=not MODULES_AVAILABLE.get("overnight_alpha"),
        )

    if run_alpha and selected_tickers:
        progress = st.progress(0)
        status   = st.empty()
        results  = []

        for i, ticker in enumerate(selected_tickers):
            progress.progress((i + 1) / len(selected_tickers))
            status.text(f"Analysing {ticker}… ({i+1}/{len(selected_tickers)})")
            try:
                if MODULES_AVAILABLE.get("overnight_alpha"):
                    from overnight_alpha import OvernightAlphaPipeline
                    pipeline = OvernightAlphaPipeline(selected_tickers)
                    res = pipeline.run_ticker(ticker)
                    if res and res.alpha >= min_alpha:
                        results.append({
                            "ticker":       res.ticker,
                            "alpha":        res.alpha,
                            "t0_eligible":  res.t0_eligible,
                            "setup":        res.setup,
                            "gap":          res.gap,
                            "technical":    res.technical,
                            "learning_regime": res.learning_regime,  # NEW
                        })
                else:
                    st.warning("overnight_alpha module not available")
                    break
            except Exception as e:
                st.error(f"Error analysing {ticker}: {e}")

        progress.empty()
        status.empty()

        results.sort(key=lambda x: x.get("alpha", 0), reverse=True)
        if t0_only:
            results = [r for r in results if r.get("t0_eligible", False)]
        results = results[:top_n]

        if results:
            st.success(f"Found {len(results)} setups above {min_alpha} alpha threshold")
            for i, r in enumerate(results, 1):
                with st.expander(
                    f"#{i} {r['ticker']} | Alpha: {r['alpha']:.2f} | "
                    f"{r['setup'].get('setup_type','?')} | "
                    f"Regime: {r.get('learning_regime','?')}",    # NEW
                    expanded=(i == 1),
                ):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Alpha",   f"{r['alpha']:.2f}")
                    c2.metric("T+0",     "✅ Yes" if r["t0_eligible"] else "⏳ T+1")
                    c3.metric("Gap",     f"{r['gap']['direction']} {r['gap'].get('probability',0):.0%}")
                    c4.metric("R/R",     f"{r['setup'].get('rr',0):.1f}:1")

                    # Learning regime badge (NEW)
                    if r.get("learning_regime") and r["learning_regime"] != "unknown":
                        lr = r["learning_regime"]
                        regime_meta = getattr(learning, "REGIMES", {}).get(lr, {}) if learning else {}
                        st.caption(
                            f"{regime_meta.get('emoji','🔍')} Learning Regime: "
                            f"**{regime_meta.get('label', lr)}** — "
                            f"{regime_meta.get('desc','')}"
                        )

                    st.markdown("**Setup Rationale:**")
                    for note in r["setup"].get("rationale", []):
                        st.write(f"  {note}")
                    st.markdown(
                        f"**Entry:** {_clean_val(r['setup'].get('entry_zone','—'))} | "
                        f"**Stop:** {_clean_val(r['setup'].get('stop_loss','—'))} | "
                        f"**Target:** {_clean_val(r['setup'].get('targets','—'))}"
                    )
                    st.markdown(
                        f"**Best Session:** {r['setup'].get('best_session','—')} | "
                        f"**Gemini Score:** "
                        f"{r['technical'].get('gemini_framework',{}).get('composite',0):.2f}"
                    )
        else:
            st.info("No setups above threshold. Try lowering min alpha or selecting more tickers.")

# ══════════════════════════════════════════════════════════════════════
# TAB 2 — SINGLE STOCK
# ══════════════════════════════════════════════════════════════════════
with tab2:
    st.header("🔬 Single Stock Deep Analysis")

    ticker  = st.selectbox("Select Ticker", ALL_TICKERS, key="single_ticker")
    segment = get_ticker_segment(ticker)
    t0      = is_t0_eligible(ticker)
    st.caption(f"Segment: {segment} | T+0: {'✅ Eligible' if t0 else '⏳ T+1 Only'}")

    # Analyst signal banner (from sentinel_reports, NEW)
    if reports and MODULES_AVAILABLE.get("sentinel_reports"):
        try:
            active_sig = reports.get_active_signal(ticker)
            if active_sig:
                action = active_sig.get("action", "WATCH")
                colour = {"BUY": "🟢", "STRONG_BUY": "🟢🟢",
                          "SELL": "🔴", "STRONG_SELL": "🔴🔴",
                          "HOLD": "🟡", "WATCH": "⚪"}.get(action, "⚪")
                st.info(
                    f"📋 **Analyst Signal:** {colour} {action}  |  "
                    f"Confidence: {active_sig.get('confidence','—')}  |  "
                    f"Entry: {active_sig.get('entry_low','—')}–{active_sig.get('entry_high','—')}  |  "
                    f"Target 1: {active_sig.get('target1','—')}  |  "
                    f"Valid until: {str(active_sig.get('valid_until','—'))[:10]}"
                )
        except Exception:
            pass

    if st.button("Analyse", type="primary"):
        with st.spinner(f"Analysing {ticker}…"):
            try:
                if MODULES_AVAILABLE.get("data_engine"):
                    df          = fetch_and_build(ticker, "EGX", lookback=400, use_cache=True)
                    is_synthetic = df.attrs.get("synthetic", False)
                    if is_synthetic:
                        st.warning(f"⚠️ Using synthetic data for {ticker}")
                    else:
                        st.success(f"Loaded {len(df)} bars for {ticker}")

                    if MODULES_AVAILABLE.get("technical_analysis"):
                        snap, setup = analyze_ticker(df, ticker, segment)
                        summary     = get_indicator_summary(snap)

                        st.subheader("📊 Technical Snapshot")
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Trend",  snap.trend_direction)
                        c2.metric("RSI",    f"{snap.rsi_14:.1f}")
                        c3.metric("MACD",   snap.macd_state)
                        c4.metric("Gemini", f"{snap.gemini_framework_score:.2f}")

                        st.subheader("🎯 Gemini Flash Framework")
                        g = summary["gemini_framework"]
                        cg1, cg2, cg3 = st.columns(3)
                        cg1.metric("Trend Score",  f"{g['trend_score']:.2f}")
                        cg2.metric("Volume Score", f"{g['volume_score']:.2f}")
                        cg3.metric("Timing Score", f"{g['timing_score']:.2f}")
                        st.progress(g["composite"],
                                    text=f"Composite: {g['composite']:.2f} — {g['signal']}")

                        st.subheader("📈 Setup Quality")
                        st.write(
                            f"Type: **{setup.setup_type}** | "
                            f"Score: {_clean_val(setup.quality_score)}"
                        )
                        st.write(
                            f"Entry: {_clean_val(setup.entry_zone)} | "
                            f"Stop: {_clean_val(setup.stop_loss)} | "
                            f"Targets: {_clean_val(setup.targets)}"
                        )
                        st.write(f"Best Session: {setup.best_session}")

                        # Learning regime (NEW)
                        if learning and MODULES_AVAILABLE.get("sentinel_learning"):
                            try:
                                close_val   = float(df["close"].iloc[-1])
                                atr_pct     = snap.atr_14 / close_val * 100 if close_val > 0 else 0
                                obv_norm    = (float(snap.obv_slope) / max(abs(float(snap.obv or 1)), 1)
                                               if snap.obv else 0)
                                skills_full = (analyze_skills(df, ticker, segment)
                                               if MODULES_AVAILABLE.get("auto_skills") else None)
                                skills_list = ([{"id": d["skill"]}
                                                 for d in (skills_full or {}).get("triggered_details", [])]
                                               if skills_full else [])
                                w_conf      = {"aligned": snap.confluence_score > 0}
                                lr, _       = learning.inject_regime_weights(
                                    skills_list, df, w_conf, atr_pct, obv_norm,
                                    config.get("rules", {}),
                                )
                                regime_meta = learning.REGIMES.get(lr, {})
                                st.info(
                                    f"🧠 Learning Regime: **{regime_meta.get('label', lr)}** "
                                    f"{regime_meta.get('emoji','')} — {regime_meta.get('desc','')}"
                                )
                                active_cfg = learning.get_active_config(lr, "classic")
                                with st.expander("⚙️ Adaptive Weights (from Self-Learning)"):
                                    acols = st.columns(4)
                                    acols[0].metric("Trend w", f"{active_cfg.get('vamp_trend_weight',0):.3f}")
                                    acols[1].metric("EMA w",   f"{active_cfg.get('vamp_ema_weight',0):.3f}")
                                    acols[2].metric("Volume w",f"{active_cfg.get('volume_weight',0):.3f}")
                                    acols[3].metric("Weekly w",f"{active_cfg.get('w_weekly',0):.3f}")
                            except Exception:
                                pass

                        st.subheader("📝 Rationale")
                        for note in setup.rationale:
                            st.write(note)

                    if MODULES_AVAILABLE.get("auto_skills"):
                        skills = analyze_skills(df, ticker, segment)
                        st.subheader("🎯 Auto Skills")
                        st.write(
                            f"Composite Score: {skills['composite_score']:.2f} | "
                            f"Triggered: {skills['skills_triggered']}/7"
                        )
                        for detail in skills.get("triggered_details", []):
                            st.write(
                                f"  {detail['skill']}: {detail['direction']} "
                                f"({detail['confidence']:.0%}) — "
                                f"Gemini aligned: {detail['gemini_aligned']}"
                            )

                    if MODULES_AVAILABLE.get("gap_predictor"):
                        gap = predict_overnight_gap(df, ticker, segment)
                        st.subheader("🌙 Gap Prediction")
                        st.write(f"Direction: {gap.gap_direction} | Probability: {gap.gap_probability:.0%}")
                        st.write(f"Expected Magnitude: {gap.expected_magnitude:.2%}")
                        st.write(f"T+0 Boost: {gap.t0_liquidity_boost:.1f}x")

                    if MODULES_AVAILABLE.get("ml_forecast"):
                        ml     = MLForecastEngine()
                        ml.train(df)
                        ml_pred = ml.predict(df)
                        if ml_pred:
                            st.subheader("🧬 ML Forecast (7d)")
                            st.write(f"Target Return: {ml_pred.get('target_return',0):+.2%}")
                            st.write(f"Confidence: {ml_pred.get('confidence',0):.0%}")
                            st.write(
                                f"XGB: {ml_pred.get('xgb_pred',0):+.2%} | "
                                f"RF: {ml_pred.get('rf_pred',0):+.2%}"
                            )

                    if MODULES_AVAILABLE.get("sentiment"):
                        st.subheader("📰 Sentiment Analysis")
                        demo_headlines = [
                            f"{ticker.split('.')[0]} reports strong quarterly earnings",
                            f"Foreign investors increase holdings in {ticker.split('.')[0]}",
                        ]
                        try:
                            sent  = get_sentiment_for_ticker(ticker, demo_headlines)
                            sc1, sc2, sc3 = st.columns(3)
                            sc1.metric("Score",      f"{sent.score:+.2f}")
                            sc2.metric("Confidence", f"{sent.confidence:.0%}")
                            sc3.metric("AI Source",  sent.ai_source or "Keyword")
                            st.write(f"Summary: {sent.summary}")
                        except Exception as e:
                            st.warning(f"Sentiment failed: {e}")

                    # Chart
                    st.subheader("📈 Price Chart")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df["date"], y=df["close"], mode="lines", name="Close"))
                    for col_name, label, dash in [
                        ("ema_20","EMA20","dot"), ("ema_50","EMA50","dash"), ("sma_200","SMA200","solid")
                    ]:
                        if col_name in df.columns:
                            fig.add_trace(go.Scatter(x=df["date"], y=df[col_name],
                                                     mode="lines", name=label,
                                                     line=dict(dash=dash)))
                    if "vwap_20d" in df.columns:
                        fig.add_trace(go.Scatter(x=df["date"], y=df["vwap_20d"],
                                                 mode="lines", name="VWAP"))
                    fig.update_layout(template="plotly_dark", height=CHART_HEIGHT)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("data_engine module not available")

            except ValueError as e:
                error_msg = str(e)
                if "Insufficient data" in error_msg or "0 bars" in error_msg:
                    st.error(f"📊 {error_msg}")
                    st.info("Possible causes: new listing, delisted, invalid API key, EODHD quota.")
                elif "Unable to fetch" in error_msg:
                    st.error(f"🔌 {error_msg}")
                    st.info("Check your EODHD_API_KEY.")
                else:
                    st.error(f"Analysis failed: {e}")
                with st.expander("🔧 Debug"):
                    import traceback
                    st.code(traceback.format_exc())
            except Exception as e:
                st.error(f"Unexpected error: {e}")
                import traceback
                st.code(traceback.format_exc())

# ══════════════════════════════════════════════════════════════════════
# TAB 3 — SCANNER
# ══════════════════════════════════════════════════════════════════════
with tab3:
    st.header("📊 Market Scanner")
    st.info("Batch scan across selected universe. Uses v4.2.2 Alpha scorer with all layers.")

    scan_universe = st.multiselect("Select Universe", ALL_TICKERS,
                                   default=ALL_TICKERS[:30], key="scan_universe")

    if st.button("🔍 Run Scanner", type="primary"):
        if MODULES_AVAILABLE.get("overnight_alpha"):
            with st.spinner("Scanning…"):
                try:
                    results = run_pipeline(scan_universe)
                    if results:
                        st.success(f"Found {len(results)} setups")
                        df_results = pd.DataFrame([
                            {
                                "Ticker":  r["ticker"],
                                "Alpha":   r["alpha"],
                                "T0":      "✅" if r["t0_eligible"] else "⏳",
                                "Setup":   r["setup"].get("setup_type","?"),
                                "Gap":     r["gap"].get("direction","?"),
                                "R/R":     r["setup"].get("rr", 0),
                                "Session": r["setup"].get("best_session","?"),
                                "Regime":  r.get("learning_regime","?"),   # NEW
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

# ══════════════════════════════════════════════════════════════════════
# TAB 4 — WATCHLISTS
# ══════════════════════════════════════════════════════════════════════
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

        wl_names    = list(watchlists.keys())
        selected_wl = st.selectbox("Select", wl_names if wl_names else ["(none)"])

        if selected_wl in watchlists and st.button("🗑️ Delete"):
            del watchlists[selected_wl]
            save_watchlists(watchlists)
            st.rerun()

    with col_b:
        if selected_wl in watchlists:
            st.subheader(f"📌 {selected_wl} ({len(watchlists[selected_wl])} tickers)")
            add_ticker = st.selectbox(
                "Add Ticker",
                [""] + [t for t in ALL_TICKERS if t not in watchlists[selected_wl]],
            )
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

# ══════════════════════════════════════════════════════════════════════
# TAB 5 — DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════
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

    st.subheader("🧹 Cache Management")
    st.caption("Clear cached data to force fresh EODHD fetches.")
    col_cache1, col_cache2 = st.columns([1, 3])
    with col_cache1:
        if st.button("🗑️ Clear Cache", type="secondary"):
            try:
                clear_cache()
                st.success("✅ Cache cleared! Next analysis will fetch fresh EODHD data.")
                st.balloons()
            except Exception as e:
                st.error(f"Failed to clear cache: {e}")
    with col_cache2:
        st.write("")
        st.write("")
        st.info("💡 After clearing, re-run any analysis to fetch fresh data.")

    st.subheader("T+0 Segment Map")
    seg_df = pd.DataFrame([
        {"Segment": seg, "Tickers": len(tickers), "T+0": seg in T0_ENABLED}
        for seg, tickers in MARKET_SEGMENTS.items()
    ])
    st.dataframe(seg_df, hide_index=True)

    st.subheader("🧠 Regime Analysis")
    st.caption("Hybrid regime detection: Heuristic per-ticker + Claude macro + Disagreement handling")

    if MODULES_AVAILABLE.get("regime_detector_v2"):
        col_reg1, col_reg2 = st.columns([1, 2])
        with col_reg1:
            regime_ticker  = st.selectbox("Ticker for regime", ALL_TICKERS, key="regime_ticker")
            regime_segment = TICKER_TO_SEGMENT.get(regime_ticker, "moderate_activity")
            headline_input = st.text_area(
                "Headlines (one per line, optional)",
                value="CBE maintains interest rates\nEGX volume rises on foreign buying",
                height=HEADLINES_TEXTAREA_HEIGHT, key="regime_headlines",
            )
            headlines   = [h.strip() for h in headline_input.split("\n") if h.strip()]
            run_regime  = st.button("🔍 Analyse Regime", type="primary")

        with col_reg2:
            if run_regime:
                with st.spinner(f"Analysing regime for {regime_ticker}…"):
                    try:
                        ensemble = HybridRegimeEnsemble()
                        df_reg   = fetch_and_build(regime_ticker, "EGX", lookback=100, use_cache=True)
                        hybrid   = ensemble.detect(df_reg, ticker=regime_ticker,
                                                    segment=regime_segment, headlines=headlines or None)
                        st.subheader("📊 Hybrid Regime Result")
                        rc1, rc2, rc3, rc4 = st.columns(4)
                        rc1.metric("Final Regime",  hybrid.regime)
                        rc2.metric("Position Size", f"{hybrid.position_size:.0%}")
                        rc3.metric("Confidence",    f"{hybrid.confidence:.0%}")
                        rc4.metric("Disagreement",  f"{hybrid.disagreement_index:.2f}")
                        st.progress(hybrid.confidence, text=f"Confidence: {hybrid.confidence:.0%}")
                        h, m = hybrid.heuristic, hybrid.macro
                        h_c1, h_c2 = st.columns(2)
                        with h_c1:
                            st.markdown("**Heuristic Layer**")
                            st.write(f"Regime: `{h.regime}`")
                            st.write(f"Slope: {h.slope_pct}% | RSI: {h.rsi}")
                            st.write(f"Shock: {'🚨 ' + h.shock_type if h.shock_detected else '✅ None'}")
                        with h_c2:
                            st.markdown("**Macro Layer (Claude)**")
                            st.write(f"Score: {m.macro_score:+.2f} | Conf: {m.confidence:.0%}")
                            st.write(f"Risk-off: {'🚨 Yes' if m.risk_off_flag else '✅ No'}")
                            st.write(f"Source: {m.source}")
                        st.info(f"💡 {hybrid.recommendation}")
                        if hybrid.conflict_flag:
                            st.warning("⚠️ CONFLICT — Heuristic and macro disagree. Reduce position.")
                        if h.shock_detected:
                            st.error(f"🚨 SHOCK: {h.shock_type}")
                    except Exception as e:
                        st.error(f"Regime analysis failed: {e}")
                        import traceback; st.code(traceback.format_exc())
    else:
        st.warning("regime_detector_v2 not available.")

    st.divider()
    st.caption("Train the overnight gap predictor on historical EOD data.")
    col_train1, col_train2 = st.columns([1, 3])
    with col_train1:
        train_tickers = st.multiselect(
            "Tickers to train on", options=ALL_TICKERS,
            default=ALL_TICKERS[:10] if len(ALL_TICKERS) > 10 else ALL_TICKERS,
            key="train_tickers",
        )
    with col_train2:
        st.write(""); st.write("")
        run_training = st.button("🏋️ Train Gap Model", type="primary",
                                 disabled=not MODULES_AVAILABLE.get("gap_predictor"))

    if run_training and train_tickers:
        from gap_predictor import train_gap_model
        train_progress = st.progress(0)
        train_status   = st.empty()
        historical_data, segments = {}, {}
        for idx, t in enumerate(train_tickers):
            train_status.text(f"Fetching {t}… ({idx+1}/{len(train_tickers)})")
            try:
                df_t = fetch_and_build(t, "EGX", lookback=400, use_cache=True)
                historical_data[t] = df_t
                segments[t]        = get_segment(t)
            except Exception as e:
                st.warning(f"Skipping {t}: {e}")
            train_progress.progress((idx + 1) / len(train_tickers))
        if len(historical_data) >= 3:
            train_status.text("Training model…")
            try:
                predictor, metrics = train_gap_model(historical_data, segments)
                predictor.save("gap_model_v42.pkl")
                train_progress.empty(); train_status.empty()
                st.success(f"✅ Trained on {len(historical_data)} tickers | "
                           f"Accuracy: {metrics.get('accuracy','N/A')}")
                st.json(metrics)
            except Exception as e:
                train_progress.empty(); train_status.empty()
                st.error(f"Training failed: {e}")
        else:
            train_progress.empty(); train_status.empty()
            st.error("Need ≥3 tickers with valid data to train.")

# ══════════════════════════════════════════════════════════════════════
# TAB 6 — SELF-LEARNING  (NEW)
# ══════════════════════════════════════════════════════════════════════
with tab6:
    if learning and MODULES_AVAILABLE.get("sentinel_learning"):
        learning.render_tab()
    else:
        st.header("🧠 Adaptive Self-Learning Engine")
        st.error(
            "sentinel_learning module not available. "
            "Ensure sentinel_learning.py is present in the project directory."
        )
        st.code("pip install -r requirements.txt  # then restart the app")

# ══════════════════════════════════════════════════════════════════════
# TAB 7 — ANALYST REPORTS  (NEW)
# ══════════════════════════════════════════════════════════════════════
with tab7:
    if reports and MODULES_AVAILABLE.get("sentinel_reports"):
        if claude_client is None:
            st.warning(
                "⚠️ Anthropic SDK client not initialised — "
                "PDF/image parsing will be unavailable. "
                "Set ANTHROPIC_API_KEY and ensure `anthropic` is installed."
            )
        reports.render_tab(claude_client)
    else:
        st.header("📋 Analyst Reports & Signals")
        st.error(
            "sentinel_reports module not available. "
            "Ensure sentinel_reports.py is present in the project directory."
        )
        st.code("pip install anthropic pillow openpyxl  # then restart the app")

# ── FOOTER ──
st.divider()
st.caption(
    "Sentinel-EGX v4.2.2 | "
    "Overnight Alpha + Gemini Flash + T+0/T+1 + "
    "Adaptive Self-Learning + Analyst Reports | EOD Data Only"
)
