"""
Sentinel-EGX v4.2.2 — Adaptive Self-Learning Engine
=====================================================
Logs every forecast → fetches actual outcomes daily → diagnoses errors per Regime →
auto-adjusts weights → Rollback on performance regression.

v4.2.2 adaptations vs. original v3.8:
  • fetch_pending_outcomes(): uses data_engine.fetch_eod() instead of EODHD SDK client
  • inject_regime_weights(): resolves both v3.7 column names (EMA20, ADX, BB_width)
    and v4.2.2 names (ema_20, ema_50, atr_14 — no ADX or BB_width in default pipeline)
  • run_daily_if_needed(): api_client parameter kept for backward compat but ignored

Files created:
    sentinel_learning.db         — all learning data
    sentinel_regime_configs.json — per-regime adaptive weights

Integration (overnight_alpha.py):
  1. import sentinel_learning as _learning
  2. After detect_skills(): regime, adaptive_rules = _learning.inject_regime_weights(...)
  3. In result: result["regime"] = regime
  4. Before return: _learning.log_forecast(result)

Integration (sentinel_app.py):
  5. if "learning_init" not in st.session_state:
         _learning.run_daily_if_needed()
         st.session_state["learning_init"] = True
  6. Tab: learning.render_tab()
"""

import json
import sqlite3
import warnings
from datetime import datetime, timedelta
from math import erfc, sqrt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SCRIPT_DIR  = Path(__file__).parent.resolve()
LEARNING_DB = SCRIPT_DIR / "sentinel_learning.db"
REGIME_CFG  = SCRIPT_DIR / "sentinel_regime_configs.json"
LAST_RUN_F  = SCRIPT_DIR / ".learning_last_run"

# ===========================================================================
# CONSTANTS
# ===========================================================================

STAGES: List[Tuple] = [
    (0,   0.000, 1.00, "Collecting  (0-9)"),
    (10,  0.010, 0.15, "Learning    (10-19)"),
    (20,  0.020, 0.10, "Adjusting   (20-29)"),
    (30,  0.030, 0.05, "Active      (30-49)"),
    (50,  0.050, 0.05, "Optimized   (50+)"),
]

ROLLBACK_WINDOW = 10
ROLLBACK_RATIO  = 1.20

WEIGHT_BOUNDS = {
    "vamp_trend_weight":  (0.15, 0.70),
    "vamp_ema_weight":    (0.20, 0.75),
    "volume_weight":      (0.05, 0.25),
    "w_weekly":           (0.05, 0.30),
    "rsi_overbought":     (65.0, 90.0),
    "rsi_oversold":       (10.0, 35.0),
    "stop_loss_atr_mult": (1.0,  5.0),
}

REGIMES = {
    "trending": {"label": "Trending",  "emoji": "📈", "color": "#10b981",
                 "desc": "ADX>25 + Weekly Aligned + EMA20>EMA50"},
    "ranging":  {"label": "Ranging",   "emoji": "↔️", "color": "#f59e0b",
                 "desc": "ADX<20 + BB Squeeze"},
    "volatile": {"label": "Volatile",  "emoji": "🔥", "color": "#ef4444",
                 "desc": "ATR > 5% of price"},
    "obv_led":  {"label": "OBV-Led",   "emoji": "💥", "color": "#8b5cf6",
                 "desc": "OBV Corr > 0.6 + Volume Spike"},
    "reversal": {"label": "Reversal",  "emoji": "🌊", "color": "#06b6d4",
                 "desc": "Supertrend Flip or MFI Divergence"},
    "unknown":  {"label": "Unknown",   "emoji": "❓", "color": "#64748b",
                 "desc": "No dominant regime"},
}

DEFAULT_CONFIGS: Dict[str, Dict[str, Dict]] = {
    "trending": {
        "classic": {"vamp_trend_weight":0.40,"vamp_ema_weight":0.45,"volume_weight":0.125,
                    "w_weekly":0.15,"rsi_overbought":75,"rsi_oversold":25,"stop_loss_atr_mult":2.0},
        "ml":      {"vamp_trend_weight":0.35,"vamp_ema_weight":0.40,"volume_weight":0.10,
                    "w_weekly":0.10,"rsi_overbought":75,"rsi_oversold":25,"stop_loss_atr_mult":2.0},
    },
    "ranging": {
        "classic": {"vamp_trend_weight":0.25,"vamp_ema_weight":0.60,"volume_weight":0.10,
                    "w_weekly":0.10,"rsi_overbought":70,"rsi_oversold":30,"stop_loss_atr_mult":1.5},
        "ml":      {"vamp_trend_weight":0.20,"vamp_ema_weight":0.55,"volume_weight":0.10,
                    "w_weekly":0.10,"rsi_overbought":70,"rsi_oversold":30,"stop_loss_atr_mult":1.5},
    },
    "volatile": {
        "classic": {"vamp_trend_weight":0.35,"vamp_ema_weight":0.50,"volume_weight":0.125,
                    "w_weekly":0.15,"rsi_overbought":80,"rsi_oversold":20,"stop_loss_atr_mult":3.0},
        "ml":      {"vamp_trend_weight":0.30,"vamp_ema_weight":0.45,"volume_weight":0.125,
                    "w_weekly":0.15,"rsi_overbought":80,"rsi_oversold":20,"stop_loss_atr_mult":3.0},
    },
    "obv_led": {
        "classic": {"vamp_trend_weight":0.30,"vamp_ema_weight":0.40,"volume_weight":0.20,
                    "w_weekly":0.15,"rsi_overbought":75,"rsi_oversold":25,"stop_loss_atr_mult":2.0},
        "ml":      {"vamp_trend_weight":0.25,"vamp_ema_weight":0.35,"volume_weight":0.20,
                    "w_weekly":0.15,"rsi_overbought":75,"rsi_oversold":25,"stop_loss_atr_mult":2.0},
    },
    "reversal": {
        "classic": {"vamp_trend_weight":0.25,"vamp_ema_weight":0.45,"volume_weight":0.125,
                    "w_weekly":0.25,"rsi_overbought":70,"rsi_oversold":30,"stop_loss_atr_mult":2.5},
        "ml":      {"vamp_trend_weight":0.20,"vamp_ema_weight":0.40,"volume_weight":0.125,
                    "w_weekly":0.25,"rsi_overbought":70,"rsi_oversold":30,"stop_loss_atr_mult":2.5},
    },
    "unknown": {
        "classic": {"vamp_trend_weight":0.38,"vamp_ema_weight":0.48,"volume_weight":0.125,
                    "w_weekly":0.15,"rsi_overbought":75,"rsi_oversold":25,"stop_loss_atr_mult":2.0},
        "ml":      {"vamp_trend_weight":0.35,"vamp_ema_weight":0.45,"volume_weight":0.125,
                    "w_weekly":0.15,"rsi_overbought":75,"rsi_oversold":25,"stop_loss_atr_mult":2.0},
    },
}

# ===========================================================================
# 1. DATABASE
# ===========================================================================

def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(LEARNING_DB))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = _db()
    conn.executescript("""
    PRAGMA journal_mode=WAL;

    CREATE TABLE IF NOT EXISTS forecast_log (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol           TEXT NOT NULL,
        forecast_date    TEXT NOT NULL,
        horizon_days     INTEGER NOT NULL,
        model_type       TEXT NOT NULL,
        regime           TEXT NOT NULL,
        entry_price      REAL NOT NULL,
        target_price     REAL NOT NULL,
        predicted_growth REAL NOT NULL,
        rsi              REAL,
        adx              REAL,
        atr_pct          REAL,
        obv_signal       REAL,
        weekly_aligned   INTEGER,
        active_skills    TEXT,
        weights_snap     TEXT,
        components_snap  TEXT,
        outcome_date     TEXT,
        outcome_price    REAL,
        actual_growth    REAL,
        error_pct        REAL,
        direction_ok     INTEGER,
        diagnosed        INTEGER DEFAULT 0,
        diagnosis_json   TEXT,
        created_at       TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS regime_versions (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        regime      TEXT NOT NULL,
        model_type  TEXT NOT NULL,
        version     INTEGER NOT NULL DEFAULT 1,
        config_json TEXT NOT NULL,
        is_active   INTEGER DEFAULT 1,
        created_at  TEXT DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(regime, model_type, version)
    );

    CREATE TABLE IF NOT EXISTS config_changes (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        regime            TEXT NOT NULL,
        model_type        TEXT NOT NULL,
        from_version      INTEGER NOT NULL,
        to_version        INTEGER NOT NULL,
        old_config        TEXT NOT NULL,
        new_config        TEXT NOT NULL,
        samples_used      INTEGER,
        culprit_component TEXT,
        change_reason     TEXT,
        rmse_before       REAL,
        dir_acc_before    REAL,
        rmse_after        REAL,
        applied_at        TEXT DEFAULT CURRENT_TIMESTAMP,
        rolled_back       INTEGER DEFAULT 0,
        rollback_reason   TEXT,
        rollback_at       TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_fl_regime  ON forecast_log(regime, model_type);
    CREATE INDEX IF NOT EXISTS idx_fl_outcome ON forecast_log(outcome_price);
    CREATE INDEX IF NOT EXISTS idx_fl_sym     ON forecast_log(symbol, forecast_date);
    """)
    conn.commit()
    conn.close()

init_db()

# ===========================================================================
# 2. REGIME CLASSIFIER
# ===========================================================================

def classify_regime(result: Dict) -> str:
    """
    Classifies market regime from a result dict.
    Priority: Volatile > OBV-Led > Reversal > Trending > Ranging > Unknown
    """
    ind     = result.get("indicators", {})
    skills  = {s.get("id") or s.get("skill", "") for s in result.get("active_skills", [])}
    atr_pct = float(result.get("atr_pct") or 0)
    adx     = float(ind.get("adx") or 0)
    obv     = float(result.get("obv_signal") or 0)
    ema20   = float(ind.get("ema20") or 0)
    ema50   = float(ind.get("ema50") or 0)
    bb_w    = float(ind.get("bb_width") or 0.10)
    aligned = bool((result.get("weekly_confluence") or {}).get("aligned", False))

    if atr_pct > 5.0:
        return "volatile"
    if obv > 0.6 and "volume_spike" in skills:
        return "obv_led"
    if "supertrend_flip" in skills or "mfi_divergence" in skills:
        return "reversal"
    if adx > 25 and aligned and ema20 > ema50:
        return "trending"
    if adx < 20 and bb_w < 0.08:
        return "ranging"
    # v4.2.2 fallback: use EMA alignment when ADX is unavailable
    if aligned and ema20 > ema50 and ema50 > 0:
        return "trending"
    return "unknown"

# ===========================================================================
# 3. CONFIG MANAGER
# ===========================================================================

def _load_regime_cfgs() -> Dict:
    if REGIME_CFG.exists():
        try:
            return json.loads(REGIME_CFG.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

def _save_regime_cfgs(cfgs: Dict):
    REGIME_CFG.write_text(json.dumps(cfgs, indent=2, ensure_ascii=False), encoding="utf-8")

def get_active_config(regime: str, model_type: str) -> Dict:
    """Returns current config = defaults + any learned adjustments."""
    default  = DEFAULT_CONFIGS.get(regime, DEFAULT_CONFIGS["unknown"]).get(model_type, {})
    override = _load_regime_cfgs().get(regime, {}).get(model_type, {})
    return {**default, **override}

def inject_regime_weights(active_skills, df, w_conf, atr_pct, obv_signal, base_rules) -> Tuple[str, Dict]:
    """
    [INTEGRATION POINT — call in overnight_alpha.run_ticker() after detect_skills()]
    Classifies regime and returns RULES updated with regime-specific adaptive weights.

    v4.2.2 adaptation: resolves both v3.7 column names (EMA20, ADX, BB_width)
    and v4.2.2 names (ema_20, ema_50, atr_14). Missing columns fall back to 0.
    active_skills may be a list of {"id":...} (v3.7) or {"skill":...} (v4.2.2).
    """
    def _col(df: pd.DataFrame, *names, default: float = 0.0) -> float:
        for n in names:
            if n in df.columns and not df[n].isna().all():
                return float(df[n].iloc[-1])
        return default

    # Compute ATR% from dataframe if not supplied
    atr_computed = atr_pct or 0.0
    close_val = _col(df, "close", default=1.0)
    if close_val > 0:
        raw_atr = _col(df, "ATR", "atr_14", default=0.0)
        if raw_atr > 0:
            atr_computed = raw_atr / close_val * 100

    tmp = {
        "indicators": {
            "adx":      _col(df, "ADX", "adx_proxy", default=0.0),
            "ema20":    _col(df, "EMA20", "ema_20", default=0.0),
            "ema50":    _col(df, "EMA50", "ema_50", default=0.0),
            "bb_width": _col(df, "BB_width", "bb_width", default=0.10),
        },
        # normalise skill dicts: accept {"id": x} or {"skill": x}
        "active_skills": [
            {"id": s.get("id") or s.get("skill", "")} for s in (active_skills or [])
        ],
        "atr_pct": atr_computed,
        "obv_signal": obv_signal,
        "weekly_confluence": w_conf,
    }
    regime     = classify_regime(tmp)
    model_type = "classic"
    cfg        = get_active_config(regime, model_type)

    new_rules = dict(base_rules)
    new_rules["vamp_trend_weight"]  = cfg.get("vamp_trend_weight",  base_rules.get("vamp_trend_weight",  0.45))
    new_rules["vamp_ema_weight"]    = cfg.get("vamp_ema_weight",    base_rules.get("vamp_ema_weight",    0.55))
    new_rules["volume_weight"]      = cfg.get("volume_weight",      base_rules.get("volume_weight",      0.125))
    new_rules["_w_weekly"]          = cfg.get("w_weekly", 0.15)
    new_rules["rsi_overbought"]     = cfg.get("rsi_overbought",     base_rules.get("rsi_overbought",     75))
    new_rules["rsi_oversold"]       = cfg.get("rsi_oversold",       base_rules.get("rsi_oversold",       25))
    new_rules["stop_loss_atr_mult"] = cfg.get("stop_loss_atr_mult", base_rules.get("stop_loss_atr_mult", 2.0))
    return regime, new_rules

# ===========================================================================
# 4. FORECAST LOGGER
# ===========================================================================

def log_forecast(result: Dict):
    """
    [INTEGRATION POINT — call at end of overnight_alpha.run_ticker()]
    Logs every prediction. Fails silently.

    Expected keys in result (v4.2.2 mapping):
        Symbol           → ticker
        days             → prediction horizon
        current          → entry price (close)
        target           → take_profit
        growth           → predicted growth %
        regime           → from inject_regime_weights()
        forecast_mode    → "ml" or "classic"
        active_skills    → list of {"id": skill_name}
        weekly_confluence→ {"aligned": bool}
        rsi              → snap.rsi_14
        atr_pct          → snap.atr_14 / close * 100
        obv_signal       → normalised OBV slope
        indicators       → {"adx": 0, "ema20": ..., "ema50": ...}
        w_trend/w_ema/w_vol/w_weekly → Gemini component scores
        trend_component/ema20_component/volume_component/weekly_component → price levels
    """
    try:
        regime = result.get("regime") or classify_regime(result)
        model  = "ml" if result.get("forecast_mode") == "ml" else "classic"
        skills = [s.get("id") or s.get("skill", "") for s in result.get("active_skills", [])]
        wk     = result.get("weekly_confluence") or {}
        ind    = result.get("indicators") or {}
        weights_snap = json.dumps({
            "w_t": result.get("w_trend", 0),
            "w_e": result.get("w_ema", 0),
            "w_v": result.get("w_vol", 0),
            "w_w": result.get("w_weekly", 0),
        })
        components_snap = json.dumps({
            "trend_c":  result.get("trend_component", 0),
            "ema_c":    result.get("ema20_component", 0),
            "vol_c":    result.get("volume_component", 0),
            "weekly_c": result.get("weekly_component", 0),
        })
        conn = _db()
        conn.execute("""
            INSERT INTO forecast_log (
                symbol, forecast_date, horizon_days, model_type, regime,
                entry_price, target_price, predicted_growth,
                rsi, adx, atr_pct, obv_signal, weekly_aligned,
                active_skills, weights_snap, components_snap
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            result.get("Symbol", ""),
            datetime.now().strftime("%Y-%m-%d"),
            result.get("days", 7),
            model, regime,
            result.get("current", 0), result.get("target", 0), result.get("growth", 0),
            result.get("rsi", 0), ind.get("adx", 0), result.get("atr_pct", 0),
            result.get("obv_signal", 0),
            1 if wk.get("aligned") else 0,
            json.dumps(skills), weights_snap, components_snap,
        ))
        conn.commit()
        conn.close()
    except Exception:
        pass

# ===========================================================================
# 5. OUTCOME FETCHER — v4.2.2: uses data_engine.fetch_eod() (no SDK client)
# ===========================================================================

def fetch_pending_outcomes(api_client=None) -> int:
    """
    Fetches actual closing prices for forecasts whose horizon has elapsed.
    api_client is accepted for backward compatibility but ignored in v4.2.2;
    data_engine.fetch_eod() is used instead.
    """
    try:
        from data_engine import fetch_eod
    except ImportError:
        return 0

    today  = datetime.now().date()
    conn   = _db()
    pending = conn.execute("""
        SELECT id, symbol, forecast_date, horizon_days, entry_price, predicted_growth
        FROM forecast_log
        WHERE outcome_price IS NULL
          AND date(forecast_date, '+' || horizon_days || ' days') <= ?
        LIMIT 50
    """, (today.isoformat(),)).fetchall()
    conn.close()

    filled = 0
    for row in pending:
        try:
            df = fetch_eod(row["symbol"], "EGX", "d", days=30,
                           use_synthetic_fallback=False, use_delta_cache=True)
            if df is None or df.empty:
                continue

            target_dt  = datetime.strptime(row["forecast_date"], "%Y-%m-%d") + timedelta(days=row["horizon_days"])
            target_str = target_dt.strftime("%Y-%m-%d")
            df["_ds"]  = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            future     = df[df["_ds"] >= target_str]
            if future.empty:
                continue

            actual = float(future["close"].iloc[0])
            if actual <= 0 or np.isnan(actual):
                continue

            ep            = float(row["entry_price"])
            actual_growth = (actual - ep) / ep * 100
            error_pct     = float(row["predicted_growth"]) - actual_growth
            direction_ok  = 1 if (float(row["predicted_growth"]) > 0) == (actual_growth > 0) else 0

            conn2 = _db()
            conn2.execute("""
                UPDATE forecast_log
                SET outcome_date=?, outcome_price=?, actual_growth=?,
                    error_pct=?, direction_ok=?
                WHERE id=?
            """, (future["_ds"].iloc[0], actual, actual_growth, error_pct, direction_ok, row["id"]))
            conn2.commit()
            conn2.close()
            filled += 1
        except Exception:
            pass
    return filled

# ===========================================================================
# 6. STATISTICS HELPERS
# ===========================================================================

def _ttest(data: np.ndarray) -> Tuple[float, float]:
    n = len(data)
    if n < 5:
        return 0.0, 1.0
    mean = float(np.mean(data))
    std  = float(np.std(data, ddof=1))
    if std < 1e-10:
        return (99.0, 0.001) if abs(mean) > 1e-6 else (0.0, 1.0)
    t = mean / (std / sqrt(n))
    p = float(min(erfc(abs(t) / sqrt(2)), 1.0))
    return round(t, 3), round(p, 4)

def _rmse(arr: np.ndarray) -> float:
    return float(np.sqrt(np.mean(arr ** 2))) if len(arr) > 0 else float("inf")

def _get_stage(n: int) -> Dict:
    result = {"min_samples":0,"step":0.0,"p_thresh":1.0,
              "label":STAGES[0][3],"idx":0,"confidence":0.0,"next_at":10}
    for i, (mn, step, p, lbl) in enumerate(STAGES):
        if n >= mn:
            nxt = STAGES[i+1][0] if i < len(STAGES)-1 else None
            result = {"min_samples":mn,"step":step,"p_thresh":p,"label":lbl,
                      "idx":i,"confidence":i/(len(STAGES)-1),"next_at":nxt}
    return result

# ===========================================================================
# 7. DIAGNOSIS ENGINE
# ===========================================================================

def diagnose_regime(regime: str, model_type: str, last_n: int = 60) -> Optional[Dict]:
    """Analyses forecast errors for a regime and identifies which VAMP component causes drift."""
    conn = _db()
    rows = conn.execute("""
        SELECT error_pct, actual_growth, predicted_growth, direction_ok,
               rsi, adx, weekly_aligned, weights_snap, components_snap, entry_price
        FROM forecast_log
        WHERE regime=? AND model_type=? AND outcome_price IS NOT NULL
        ORDER BY created_at DESC LIMIT ?
    """, (regime, model_type, last_n)).fetchall()
    conn.close()

    if len(rows) < 5:
        return None

    errors     = np.array([r["error_pct"]    for r in rows], dtype=float)
    directions = np.array([r["direction_ok"] for r in rows], dtype=float)
    n          = len(errors)

    contribs = {"trend": [], "ema": [], "volume": [], "weekly": []}
    for row in rows:
        try:
            w  = json.loads(row["weights_snap"])
            c  = json.loads(row["components_snap"])
            ep = float(row["entry_price"])
            ag = row["actual_growth"]
            if ep > 0 and ag is not None:
                ap = ep * (1 + float(ag) / 100)
                contribs["trend"].append(  w["w_t"] * (c["trend_c"]  - ap) / ap)
                contribs["ema"].append(    w["w_e"] * (c["ema_c"]    - ap) / ap)
                contribs["volume"].append( w["w_v"] * (c["vol_c"]    - ap) / ap)
                contribs["weekly"].append( w["w_w"] * (c["weekly_c"] - ap) / ap)
        except Exception:
            pass

    weight_key_map = {"trend":"vamp_trend_weight","ema":"vamp_ema_weight",
                      "volume":"volume_weight","weekly":"w_weekly"}
    component_analysis = {}
    weight_adjustments = {}

    for name, arr_list in contribs.items():
        if not arr_list:
            continue
        arr = np.array(arr_list)
        mean_c        = float(np.mean(arr))
        t_stat, p_val = _ttest(arr)
        significant   = p_val < 0.20
        component_analysis[name] = {
            "mean_contribution_pct": round(mean_c * 100, 3),
            "t_stat": round(t_stat, 2), "p_value": round(p_val, 4),
            "significant": significant,
            "verdict": "Overshooting -> reduce" if mean_c > 0 else "Undershooting -> increase",
        }
        if abs(mean_c) > 0.001:
            weight_adjustments[weight_key_map[name]] = {
                "direction": -1 if mean_c > 0 else 1,
                "magnitude": abs(mean_c), "p_value": p_val, "component": name,
            }

    rsi_adjustments = {}
    rsi_vals = np.array([float(r["rsi"]) for r in rows if r["rsi"] is not None])
    if len(rsi_vals) >= 8:
        ob_mask = rsi_vals > 70
        if ob_mask.sum() >= 4:
            ob_acc = float(np.mean(directions[:len(ob_mask)][ob_mask[:len(directions)]]))
            if ob_acc < 0.45:
                rsi_adjustments["rsi_overbought"] = {"direction":1,"reason":f"Acc={ob_acc:.0%} at RSI>70"}
        os_mask = rsi_vals < 30
        if os_mask.sum() >= 4:
            os_acc = float(np.mean(directions[:len(os_mask)][os_mask[:len(directions)]]))
            if os_acc < 0.45:
                rsi_adjustments["rsi_oversold"] = {"direction":-1,"reason":f"Acc={os_acc:.0%} at RSI<30"}

    culprit  = None
    max_mag  = 0.0
    for name, info in component_analysis.items():
        if info["significant"] and abs(info["mean_contribution_pct"]) > max_mag:
            max_mag = abs(info["mean_contribution_pct"])
            culprit = name

    return {
        "regime": regime, "model_type": model_type, "n_samples": n,
        "rmse": round(_rmse(errors), 3),
        "direction_accuracy": round(float(np.mean(directions)), 3),
        "bias_pct": round(float(np.mean(errors)), 3),
        "culprit_component": culprit,
        "component_analysis": component_analysis,
        "weight_adjustments": weight_adjustments,
        "rsi_adjustments": rsi_adjustments,
    }

# ===========================================================================
# 8. ADAPTIVE LEARNER
# ===========================================================================

def _clamp(val: float, key: str) -> float:
    lo, hi = WEIGHT_BOUNDS[key]
    return round(float(np.clip(val, lo, hi)), 4)

def _adjust_pair(ct: float, ce: float, direction: int, step: float) -> Tuple[float, float]:
    new_t = _clamp(ct + direction * step, "vamp_trend_weight")
    delta = new_t - ct
    new_e = _clamp(ce - delta, "vamp_ema_weight")
    return new_t, new_e

def apply_adjustments(regime: str, model_type: str, diagnosis: Dict) -> Optional[Dict]:
    """Applies weight adjustments based on diagnosis with adaptive step sizes and hard bounds."""
    n        = diagnosis["n_samples"]
    stage    = _get_stage(n)
    step     = stage["step"]
    p_thresh = stage["p_thresh"]

    if step == 0:
        return None

    current_cfg = get_active_config(regime, model_type)
    new_cfg     = dict(current_cfg)
    changes     = []

    t_adj = diagnosis["weight_adjustments"].get("vamp_trend_weight")
    e_adj = diagnosis["weight_adjustments"].get("vamp_ema_weight")
    primary = None
    if t_adj and e_adj:
        primary = t_adj if t_adj["p_value"] <= e_adj["p_value"] else e_adj
    elif t_adj:
        primary = t_adj
    elif e_adj:
        primary = e_adj

    if primary and primary["p_value"] < p_thresh:
        direction = primary["direction"] if primary["component"] == "trend" else -primary["direction"]
        new_t, new_e = _adjust_pair(current_cfg["vamp_trend_weight"],
                                     current_cfg["vamp_ema_weight"], direction, step)
        if new_t != current_cfg["vamp_trend_weight"]:
            changes.append(f"trend {current_cfg['vamp_trend_weight']:.3f}->{new_t:.3f}")
            changes.append(f"ema {current_cfg['vamp_ema_weight']:.3f}->{new_e:.3f}")
            new_cfg["vamp_trend_weight"] = new_t
            new_cfg["vamp_ema_weight"]   = new_e

    v_adj = diagnosis["weight_adjustments"].get("volume_weight")
    if v_adj and v_adj["p_value"] < p_thresh:
        new_v = _clamp(current_cfg["volume_weight"] + v_adj["direction"] * step * 0.5, "volume_weight")
        if new_v != current_cfg["volume_weight"]:
            changes.append(f"volume_weight {current_cfg['volume_weight']:.3f}->{new_v:.3f}")
            new_cfg["volume_weight"] = new_v

    w_adj = diagnosis["weight_adjustments"].get("w_weekly")
    if w_adj and w_adj["p_value"] < p_thresh:
        new_w = _clamp(current_cfg["w_weekly"] + w_adj["direction"] * step * 0.5, "w_weekly")
        if new_w != current_cfg["w_weekly"]:
            changes.append(f"w_weekly {current_cfg['w_weekly']:.3f}->{new_w:.3f}")
            new_cfg["w_weekly"] = new_w

    for rk, ri in diagnosis.get("rsi_adjustments", {}).items():
        new_r = _clamp(current_cfg[rk] + ri["direction"] * 1.0, rk)
        if new_r != current_cfg[rk]:
            changes.append(f"{rk} {current_cfg[rk]:.0f}->{new_r:.0f}")
            new_cfg[rk] = new_r

    if not changes:
        return None

    loaded = _load_regime_cfgs()
    loaded.setdefault(regime, {})[model_type] = new_cfg
    _save_regime_cfgs(loaded)

    conn = _db()
    row      = conn.execute("SELECT MAX(version) as v FROM regime_versions WHERE regime=? AND model_type=?",
                             (regime, model_type)).fetchone()
    from_ver = row["v"] or 0
    to_ver   = from_ver + 1
    conn.execute("UPDATE regime_versions SET is_active=0 WHERE regime=? AND model_type=?", (regime, model_type))
    conn.execute("INSERT INTO regime_versions (regime,model_type,version,config_json,is_active) VALUES (?,?,?,?,1)",
                 (regime, model_type, to_ver, json.dumps(new_cfg)))
    conn.execute("""
        INSERT INTO config_changes (regime,model_type,from_version,to_version,
            old_config,new_config,samples_used,culprit_component,
            change_reason,rmse_before,dir_acc_before)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, (regime, model_type, from_ver, to_ver,
          json.dumps(current_cfg), json.dumps(new_cfg),
          n, diagnosis.get("culprit_component"),
          " | ".join(changes),
          diagnosis["rmse"], diagnosis["direction_accuracy"]))
    conn.commit()
    conn.close()

    return {"regime":regime,"model_type":model_type,"from_version":from_ver,
            "to_version":to_ver,"changes":changes,"old_config":current_cfg,
            "new_config":new_cfg,"stage":stage["label"],"n_samples":n,
            "applied_at":datetime.now().isoformat()}

# ===========================================================================
# 9. ROLLBACK GUARD
# ===========================================================================

def check_and_rollback() -> List[Dict]:
    """Rolls back any weight change that degraded RMSE by > ROLLBACK_RATIO over ROLLBACK_WINDOW forecasts."""
    conn      = _db()
    rollbacks = []
    pending   = conn.execute("""
        SELECT id, regime, model_type, from_version, old_config, rmse_before, applied_at
        FROM config_changes WHERE rolled_back=0 AND rmse_before IS NOT NULL
    """).fetchall()

    for chg in pending:
        recent = conn.execute("""
            SELECT error_pct, direction_ok FROM forecast_log
            WHERE regime=? AND model_type=? AND outcome_price IS NOT NULL AND created_at > ?
            ORDER BY created_at DESC LIMIT ?
        """, (chg["regime"], chg["model_type"], chg["applied_at"], ROLLBACK_WINDOW)).fetchall()

        if len(recent) < ROLLBACK_WINDOW:
            continue

        new_rmse = _rmse(np.array([r["error_pct"] for r in recent]))
        old_rmse = float(chg["rmse_before"])
        conn.execute("UPDATE config_changes SET rmse_after=? WHERE id=?", (new_rmse, chg["id"]))

        if new_rmse > old_rmse * ROLLBACK_RATIO:
            old_cfg = json.loads(chg["old_config"])
            loaded  = _load_regime_cfgs()
            loaded.setdefault(chg["regime"], {})[chg["model_type"]] = old_cfg
            _save_regime_cfgs(loaded)
            reason = (f"RMSE degraded {old_rmse:.2f}->{new_rmse:.2f} "
                      f"({(new_rmse/old_rmse-1)*100:.1f}% worse)")
            conn.execute("""
                UPDATE config_changes SET rolled_back=1, rollback_reason=?, rollback_at=?
                WHERE id=?
            """, (reason, datetime.now().isoformat(), chg["id"]))
            conn.execute("UPDATE regime_versions SET is_active=0 WHERE regime=? AND model_type=?",
                         (chg["regime"], chg["model_type"]))
            conn.execute("UPDATE regime_versions SET is_active=1 WHERE regime=? AND model_type=? AND version=?",
                         (chg["regime"], chg["model_type"], chg["from_version"]))
            rollbacks.append({"regime":chg["regime"],"model_type":chg["model_type"],
                               "reason":reason,"old_rmse":old_rmse,"new_rmse":new_rmse})

    conn.commit()
    conn.close()
    return rollbacks

# ===========================================================================
# 10. DAILY MAINTENANCE JOB
# ===========================================================================

def run_daily_if_needed(api_client=None, force: bool = False) -> Dict:
    """
    [INTEGRATION POINT — call on app startup]
    Runs once per day: fetch outcomes → rollback check → diagnose → adjust.
    api_client is accepted for backward compatibility but ignored in v4.2.2.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    if not force and LAST_RUN_F.exists() and LAST_RUN_F.read_text().strip() == today:
        return {"status": "skipped", "reason": "Already ran today", "date": today}

    summary = {"status":"ran","date":today,
               "outcomes_filled":0,"rollbacks":[],"adjustments":[],"errors":[]}
    try:
        summary["outcomes_filled"] = fetch_pending_outcomes()
    except Exception as e:
        summary["errors"].append(f"Outcome fetch: {e}")

    try:
        summary["rollbacks"] = check_and_rollback()
    except Exception as e:
        summary["errors"].append(f"Rollback: {e}")

    for regime in REGIMES:
        for model in ("classic", "ml"):
            try:
                diag = diagnose_regime(regime, model)
                if diag and diag["n_samples"] >= 10:
                    change = apply_adjustments(regime, model, diag)
                    if change:
                        summary["adjustments"].append(change)
            except Exception as e:
                summary["errors"].append(f"{regime}/{model}: {e}")

    try:
        LAST_RUN_F.write_text(today)
    except Exception:
        pass
    return summary

# ===========================================================================
# 11. DB QUERY HELPERS
# ===========================================================================

def get_learning_stats() -> pd.DataFrame:
    conn = _db()
    cur = conn.execute("""
        SELECT regime, model_type,
               COUNT(*) AS total,
               SUM(outcome_price IS NOT NULL) AS completed,
               ROUND(100.0*AVG(direction_ok),1) AS dir_acc_pct,
               ROUND(AVG(CASE WHEN outcome_price IS NOT NULL THEN ABS(error_pct) END),2) AS mae,
               ROUND(AVG(CASE WHEN outcome_price IS NOT NULL THEN error_pct END),2) AS bias
        FROM forecast_log GROUP BY regime, model_type ORDER BY dir_acc_pct DESC
    """)
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    conn.close()
    return pd.DataFrame([dict(zip(cols, r)) for r in rows])

def get_recent_forecasts(limit: int = 25) -> pd.DataFrame:
    conn = _db()
    cur = conn.execute("""
        SELECT symbol, forecast_date, horizon_days, model_type, regime,
               ROUND(entry_price,2) entry, ROUND(target_price,2) target,
               ROUND(predicted_growth,2) pred_pct,
               ROUND(actual_growth,2) actual_pct,
               ROUND(error_pct,2) error_pct,
               CASE direction_ok WHEN 1 THEN 'ok' ELSE 'wrong' END direction
        FROM forecast_log WHERE outcome_price IS NOT NULL
        ORDER BY created_at DESC LIMIT ?
    """, (limit,))
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    conn.close()
    return pd.DataFrame([dict(zip(cols, r)) for r in rows])

def get_change_history(limit: int = 20) -> pd.DataFrame:
    conn = _db()
    cur = conn.execute("""
        SELECT regime, model_type, to_version version, culprit_component culprit,
               change_reason changes, samples_used samples,
               ROUND(rmse_before,3) rmse_before, ROUND(rmse_after,3) rmse_after,
               CASE rolled_back WHEN 1 THEN 'Rolled Back' ELSE 'Active' END status,
               SUBSTR(applied_at,1,10) date
        FROM config_changes ORDER BY applied_at DESC LIMIT ?
    """, (limit,))
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    conn.close()
    return pd.DataFrame([dict(zip(cols, r)) for r in rows])

def reset_regime(regime: str, model_type: str):
    loaded = _load_regime_cfgs()
    if regime in loaded and model_type in loaded[regime]:
        del loaded[regime][model_type]
        if not loaded[regime]:
            del loaded[regime]
        _save_regime_cfgs(loaded)

# ===========================================================================
# 12. STREAMLIT TAB
# ===========================================================================

def render_tab(api_client=None):
    """[INTEGRATION POINT] Render self-learning dashboard in a Streamlit tab."""
    import streamlit as st
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    st.header("🧠 Adaptive Self-Learning Engine")
    st.caption(
        "Logs every forecast → diagnoses errors per Market Regime → "
        "auto-adjusts weights → Rollback on performance regression."
    )

    col_btn, col_info = st.columns([3, 1])
    with col_btn:
        if st.button("🔄 Run Daily Maintenance Now", type="primary"):
            if LAST_RUN_F.exists():
                LAST_RUN_F.unlink()
            with st.spinner("Analysing errors and adjusting weights…"):
                summary = run_daily_if_needed(force=True)
            st.success(
                f"✅ {summary['outcomes_filled']} outcomes | "
                f"{len(summary['adjustments'])} adjustments | "
                f"{len(summary['rollbacks'])} rollbacks | "
                f"{len(summary['errors'])} errors"
            )
            for a in summary.get("adjustments", []):
                st.info(f"✏️ {a['regime']}/{a['model_type']} v{a['to_version']}: {' | '.join(a['changes'])}")
            for r in summary.get("rollbacks", []):
                st.warning(f"🔄 Rollback {r['regime']}/{r['model_type']}: {r['reason']}")

    with col_info:
        last = LAST_RUN_F.read_text().strip() if LAST_RUN_F.exists() else "Not yet run"
        st.info(f"Last run:\n{last}")

    st.divider()

    conn = _db()
    total    = conn.execute("SELECT COUNT(*) FROM forecast_log").fetchone()[0]
    pending  = conn.execute("SELECT COUNT(*) FROM forecast_log WHERE outcome_price IS NULL").fetchone()[0]
    dir_avg  = conn.execute("SELECT AVG(direction_ok) FROM forecast_log WHERE outcome_price IS NOT NULL").fetchone()[0]
    n_active = conn.execute("SELECT COUNT(*) FROM config_changes WHERE rolled_back=0").fetchone()[0]
    n_rback  = conn.execute("SELECT COUNT(*) FROM config_changes WHERE rolled_back=1").fetchone()[0]
    conn.close()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("📊 Forecasts Logged", total)
    k2.metric("⏳ Pending Outcomes", pending)
    k3.metric("🎯 Direction Accuracy", f"{(dir_avg or 0)*100:.1f}%" if dir_avg else "N/A")
    k4.metric("✏️ Active Changes", n_active, f"-{n_rback} rolled back")

    st.divider()
    st.subheader("📊 Performance per Regime")
    stats_df = get_learning_stats()
    if not stats_df.empty:
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=("Direction Accuracy %", "MAE %"),
                            horizontal_spacing=0.1)
        for mi, (model, opacity) in enumerate([("classic", 0.95), ("ml", 0.60)]):
            mdf = stats_df[stats_df["model_type"] == model]
            if mdf.empty:
                continue
            clrs  = [REGIMES.get(r, {}).get("color", "#64748b") for r in mdf["regime"]]
            xlbls = [f"{REGIMES.get(r,{}).get('emoji','?')} {r}" for r in mdf["regime"]]
            fig.add_trace(go.Bar(name=model, x=xlbls, y=mdf["dir_acc_pct"].fillna(0), opacity=opacity,
                                 marker_color=clrs, showlegend=True,
                                 text=mdf["dir_acc_pct"].fillna(0).round(1), textposition="auto"), row=1, col=1)
            fig.add_trace(go.Bar(name=f"MAE {model}", x=xlbls, y=mdf["mae"].fillna(0), opacity=opacity,
                                 marker_color=clrs, showlegend=False,
                                 text=mdf["mae"].fillna(0).round(2), textposition="auto"), row=1, col=2)
        fig.add_hline(y=50, line_dash="dash", line_color="#94a3b8",
                      annotation_text="50% (random)", row=1, col=1)
        fig.update_layout(template="plotly_dark", height=360, barmode="group",
                          paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                          font=dict(color="#e2e8f0"), margin=dict(t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)  # noqa: deprecated but safe
        with st.expander("📋 Performance Table"):
            st.dataframe(stats_df, hide_index=True, width="stretch")
    else:
        st.info("No data yet. Run forecasts and wait for horizon expiry.")

    st.divider()
    st.subheader("⚙️ Current Weights per Regime")
    loaded_cfgs = _load_regime_cfgs()
    for regime, rinfo in REGIMES.items():
        modified = bool(loaded_cfgs.get(regime))
        suffix   = " ✏️ (Modified)" if modified else " (Defaults)"
        with st.expander(f"{rinfo['emoji']} {rinfo['label']}{suffix}  —  {rinfo['desc']}"):
            cols = st.columns(2)
            for ci, model in enumerate(("classic", "ml")):
                with cols[ci]:
                    st.markdown(f"**{model.upper()}**")
                    default = DEFAULT_CONFIGS.get(regime, {}).get(model, {})
                    active  = get_active_config(regime, model)
                    changed_keys = {k for k,v in active.items()
                                    if abs(float(v) - float(default.get(k,v))) > 0.001}
                    conn = _db()
                    sn = conn.execute(
                        "SELECT COUNT(*) FROM forecast_log WHERE regime=? AND model_type=? AND outcome_price IS NOT NULL",
                        (regime, model)).fetchone()[0]
                    conn.close()
                    stg = _get_stage(sn)
                    st.caption(f"{stg['label']} — {sn} samples")
                    if stg["next_at"]:
                        st.progress(stg["confidence"], text=f"Next stage at {stg['next_at']} samples")
                    rows_data = []
                    for k in ["vamp_trend_weight","vamp_ema_weight","volume_weight","w_weekly",
                               "rsi_overbought","rsi_oversold","stop_loss_atr_mult"]:
                        rows_data.append({"Parameter": k, "Default": default.get(k,"—"),
                                          "Active": active.get(k,"—"),
                                          "Changed": "✏️" if k in changed_keys else ""})
                    st.dataframe(pd.DataFrame(rows_data),
                                 hide_index=True, height=270, width="stretch")
                    if st.button(f"↩️ Reset {model}", key=f"rst_{regime}_{model}"):
                        reset_regime(regime, model)
                        st.success("Reset to defaults!")
                        st.rerun()

    st.divider()
    st.subheader("📈 Change History")
    chg_df = get_change_history(20)
    if not chg_df.empty:
        st.dataframe(chg_df, hide_index=True, width="stretch")
    else:
        st.info("No changes yet — adjustments begin after 10 completed forecasts per regime.")

    st.subheader("🔍 Recent Completed Forecasts")
    recent_df = get_recent_forecasts(25)
    if not recent_df.empty:
        st.dataframe(recent_df, hide_index=True, width="stretch")
    else:
        st.info("No completed forecasts yet.")

    st.divider()
    st.subheader("🩺 Detailed Diagnosis")
    sel_regime = st.selectbox("Select Regime", list(REGIMES.keys()),
                               format_func=lambda r: f"{REGIMES[r]['emoji']} {REGIMES[r]['label']}")
    sel_model  = st.radio("Model", ["classic","ml"], horizontal=True, key="diag_mdl")
    diag = diagnose_regime(sel_regime, sel_model)
    if diag:
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Samples", diag["n_samples"])
        d2.metric("RMSE", f"{diag['rmse']:.2f}%")
        d3.metric("Direction Acc", f"{diag['direction_accuracy']*100:.1f}%")
        d4.metric("Bias", f"{diag['bias_pct']:+.2f}%",
                  "Over-forecasting" if diag["bias_pct"] > 0 else "Under-forecasting")
        if diag["culprit_component"]:
            st.error(f"⚠️ Primary error driver: **{diag['culprit_component'].upper()} component**")
        comp_rows = []
        for nm, info in diag["component_analysis"].items():
            comp_rows.append({"Component": nm,
                               "Mean Contribution%": info["mean_contribution_pct"],
                               "Verdict": info["verdict"],
                               "t-stat": info["t_stat"],
                               "p-value": info["p_value"],
                               "Significant": "✅" if info["significant"] else "—"})
        st.dataframe(pd.DataFrame(comp_rows), hide_index=True, width="stretch")
        if diag["weight_adjustments"]:
            st.markdown("**Proposed adjustments (applied automatically per stage):**")
            for wk, wadj in diag["weight_adjustments"].items():
                arrow = "⬇️ Reduce" if wadj["direction"] == -1 else "⬆️ Increase"
                st.write(f"- **{wk}**: {arrow}  (p={wadj['p_value']:.3f}, magnitude={wadj['magnitude']*100:.2f}%)")
    else:
        st.info(f"Requires 5 completed forecasts for {sel_regime}/{sel_model}.")

    st.divider()
    st.caption(
        f"🧠 Learning Engine v4.2.2  |  DB: {LEARNING_DB.name}  |  "
        f"Regime Configs: {REGIME_CFG.name}  |  "
        f"Rollback window: {ROLLBACK_WINDOW} forecasts, {int((ROLLBACK_RATIO-1)*100)}% threshold"
    )
