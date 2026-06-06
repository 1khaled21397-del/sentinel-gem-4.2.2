"""
Sentinel-EGX v3.8 — Reports & Analyst Signals
==============================================
يستقبل ملفات التقارير (PDF/صورة/Excel) يدوياً،
يحللها بـ Claude API، يستخرج التوصيات،
ويستخدمها كـ signal إضافي في VAMP.

Integration (sentinel_app.py):
    import sentinel_reports as reports

    # Tab 6:
    with tab6:
        reports.render_tab(claude)

    # في get_egx_prediction() قبل return result:
    result["analyst_signal"] = reports.get_active_signal(result["Symbol"])
"""

import json
import sqlite3
import base64
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

warnings.filterwarnings("ignore")

SCRIPT_DIR   = Path(__file__).parent.resolve()
REPORTS_DB   = SCRIPT_DIR / "sentinel_reports.db"

# عمر الـ signal قبل ما يعتبر قديم (أيام)
SIGNAL_TTL_DAYS = 14

# ===========================================================================
# 1. DATABASE
# ===========================================================================

def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(REPORTS_DB))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = _db()
    conn.executescript("""
    PRAGMA journal_mode=WAL;

    CREATE TABLE IF NOT EXISTS reports (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        filename     TEXT NOT NULL,
        file_type    TEXT NOT NULL,
        source       TEXT DEFAULT 'manual',
        raw_text     TEXT,
        uploaded_at  TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS analyst_signals (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        report_id    INTEGER REFERENCES reports(id),
        symbol       TEXT NOT NULL,
        action       TEXT,         -- BUY / SELL / HOLD / WATCH
        entry_low    REAL,
        entry_high   REAL,
        target1      REAL,
        target2      REAL,
        target3      REAL,
        stop_loss    REAL,
        timeframe    TEXT,
        confidence   TEXT,         -- HIGH / MEDIUM / LOW
        notes        TEXT,
        valid_until  TEXT,
        created_at   TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE INDEX IF NOT EXISTS idx_sig_symbol ON analyst_signals(symbol);
    CREATE INDEX IF NOT EXISTS idx_sig_valid  ON analyst_signals(valid_until);
    """)
    conn.commit()
    conn.close()

init_db()

# ===========================================================================
# 2. FILE PARSER — Claude API
# ===========================================================================

_EXTRACT_PROMPT = """أنت محلل مالي متخصص في سوق EGX.
اقرأ هذا التقرير واستخرج كل التوصيات الموجودة فيه.

أرجع JSON فقط، بدون أي نص إضافي، بهذا الشكل:
{
  "signals": [
    {
      "symbol": "COMI",
      "action": "BUY",
      "entry_low": 75.0,
      "entry_high": 78.0,
      "target1": 85.0,
      "target2": 92.0,
      "target3": null,
      "stop_loss": 71.0,
      "timeframe": "2-4 أسابيع",
      "confidence": "HIGH",
      "notes": "اختراق مقاومة مع حجم مرتفع"
    }
  ],
  "report_summary": "ملخص التقرير في جملة واحدة"
}

قواعد:
- symbol بدون .EGX
- action: BUY أو SELL أو HOLD أو WATCH فقط
- الأرقام كـ float أو null لو مش موجودة
- لو مفيش توصيات واضحة أرجع signals: []
"""

def parse_report(file_bytes: bytes, file_name: str, claude_client) -> Optional[Dict]:
    """
    يبعت الملف لـ Claude ويستخرج التوصيات المنظّمة.
    يدعم: PDF، صور (PNG/JPG)، Excel/CSV، نصوص.
    """
    if not claude_client:
        return None

    ext = Path(file_name).suffix.lower()

    try:
        # ── PDF أو صورة → Claude Vision ──────────────────────────────────
        if ext in (".pdf", ".png", ".jpg", ".jpeg", ".webp"):
            media_map = {
                ".pdf":  "application/pdf",
                ".png":  "image/png",
                ".jpg":  "image/jpeg",
                ".jpeg": "image/jpeg",
                ".webp": "image/webp",
            }
            media_type = media_map[ext]
            b64 = base64.standard_b64encode(file_bytes).decode("utf-8")

            if ext == ".pdf":
                content_block = {
                    "type": "document",
                    "source": {"type": "base64", "media_type": media_type, "data": b64},
                }
            else:
                content_block = {
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": b64},
                }

            resp = claude_client.messages.create(
                model="claude-opus-4-5",
                max_tokens=2000,
                messages=[{
                    "role": "user",
                    "content": [content_block, {"type": "text", "text": _EXTRACT_PROMPT}]
                }]
            )
            raw_text = resp.content[0].text.strip()

        # ── Excel / CSV → pandas → نص ─────────────────────────────────────
        elif ext in (".xlsx", ".xls", ".csv"):
            import io
            if ext == ".csv":
                df = pd.read_csv(io.BytesIO(file_bytes))
            else:
                df = pd.read_excel(io.BytesIO(file_bytes))
            text_content = df.to_string(index=False, max_rows=200)

            resp = claude_client.messages.create(
                model="claude-opus-4-5",
                max_tokens=2000,
                messages=[{
                    "role": "user",
                    "content": f"هذا جدول بيانات من تقرير تحليل فني:\n\n{text_content}\n\n{_EXTRACT_PROMPT}"
                }]
            )
            raw_text = resp.content[0].text.strip()

        # ── نص عادي ───────────────────────────────────────────────────────
        else:
            text_content = file_bytes.decode("utf-8", errors="ignore")[:8000]
            resp = claude_client.messages.create(
                model="claude-opus-4-5",
                max_tokens=2000,
                messages=[{
                    "role": "user",
                    "content": f"هذا تقرير تحليل فني:\n\n{text_content}\n\n{_EXTRACT_PROMPT}"
                }]
            )
            raw_text = resp.content[0].text.strip()

        # ── Parse JSON ─────────────────────────────────────────────────────
        # نظّف لو فيه markdown code fences
        clean = raw_text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean)

    except Exception as e:
        return {"signals": [], "report_summary": f"خطأ في التحليل: {e}", "error": str(e)}

# ===========================================================================
# 3. STORAGE
# ===========================================================================

def store_report_signals(filename: str, file_type: str,
                          parsed: Dict, file_bytes: bytes) -> int:
    """يحفظ التقرير والـ signals في DB. Returns report_id."""
    conn   = _db()
    valid  = (datetime.now() + timedelta(days=SIGNAL_TTL_DAYS)).strftime("%Y-%m-%d")

    cur = conn.execute("""
        INSERT INTO reports (filename, file_type, raw_text)
        VALUES (?,?,?)
    """, (filename, file_type, parsed.get("report_summary", "")))
    report_id = cur.lastrowid

    for sig in parsed.get("signals", []):
        symbol = sig.get("symbol", "").strip().upper()
        if not symbol:
            continue
        # أضف .EGX لو مش موجودة
        if not symbol.endswith(".EGX"):
            symbol_full = symbol + ".EGX"
        else:
            symbol_full = symbol

        conn.execute("""
            INSERT INTO analyst_signals (
                report_id, symbol, action,
                entry_low, entry_high,
                target1, target2, target3,
                stop_loss, timeframe, confidence,
                notes, valid_until
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            report_id, symbol_full,
            sig.get("action", "WATCH"),
            sig.get("entry_low"), sig.get("entry_high"),
            sig.get("target1"), sig.get("target2"), sig.get("target3"),
            sig.get("stop_loss"),
            sig.get("timeframe", ""),
            sig.get("confidence", "MEDIUM"),
            sig.get("notes", ""),
            valid,
        ))

    conn.commit()
    conn.close()
    return report_id

# ===========================================================================
# 4. SIGNAL RETRIEVAL — للاستخدام في get_egx_prediction()
# ===========================================================================

def get_active_signal(symbol: str) -> Optional[Dict]:
    """
    [INTEGRATION POINT]
    يرجع أحدث analyst signal نشط للسهم ده.
    أضف في get_egx_prediction() قبل return result:
        result["analyst_signal"] = reports.get_active_signal(result["Symbol"])
    """
    today = datetime.now().strftime("%Y-%m-%d")
    conn  = _db()
    row   = conn.execute("""
        SELECT * FROM analyst_signals
        WHERE symbol=? AND valid_until >= ?
        ORDER BY created_at DESC LIMIT 1
    """, (symbol, today)).fetchone()
    conn.close()

    if not row:
        return None
    return {
        "action":     row["action"],
        "entry_low":  row["entry_low"],
        "entry_high": row["entry_high"],
        "target1":    row["target1"],
        "target2":    row["target2"],
        "target3":    row["target3"],
        "stop_loss":  row["stop_loss"],
        "timeframe":  row["timeframe"],
        "confidence": row["confidence"],
        "notes":      row["notes"],
        "valid_until":row["valid_until"],
        "created_at": row["created_at"],
    }

def get_all_active_signals() -> pd.DataFrame:
    today = datetime.now().strftime("%Y-%m-%d")
    conn  = _db()
    cur = conn.execute("""
        SELECT s.symbol, s.action, s.entry_low, s.entry_high,
               s.target1, s.target2, s.stop_loss,
               s.confidence, s.timeframe, s.notes,
               s.valid_until, s.created_at,
               r.filename
        FROM analyst_signals s
        JOIN reports r ON s.report_id = r.id
        WHERE s.valid_until >= ?
        ORDER BY s.created_at DESC
    """, (today,))
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    conn.close()
    return pd.DataFrame([dict(zip(cols, r)) for r in rows])

def get_reports_history(limit: int = 20) -> pd.DataFrame:
    conn = _db()
    cur = conn.execute("""
        SELECT r.id, r.filename, r.file_type,
               COUNT(s.id) as signals_count,
               r.uploaded_at
        FROM reports r
        LEFT JOIN analyst_signals s ON s.report_id = r.id
        GROUP BY r.id ORDER BY r.uploaded_at DESC LIMIT ?
    """, (limit,))
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    conn.close()
    return pd.DataFrame([dict(zip(cols, r)) for r in rows])

# ===========================================================================
# 5. VAMP CONFIDENCE ADJUSTMENT
# ===========================================================================

ACTION_MULTIPLIER = {
    "BUY":   1.05,   # +5% على الـ target
    "SELL": -1.05,   # عكس الإشارة
    "HOLD":  1.00,
    "WATCH": 1.00,
}

CONFIDENCE_BOOST = {"HIGH": 0.08, "MEDIUM": 0.04, "LOW": 0.01}

def apply_analyst_signal(vamp_target: float, vamp_growth: float,
                          signal: Optional[Dict]) -> Dict:
    """
    يدمج الـ analyst signal مع نتيجة VAMP.
    Returns dict بالـ adjusted values.
    """
    if not signal:
        return {"adjusted_target": vamp_target, "adjusted_growth": vamp_growth,
                "signal_applied": False}

    action     = signal.get("action", "WATCH")
    confidence = signal.get("confidence", "MEDIUM")
    boost      = CONFIDENCE_BOOST.get(confidence, 0.04)

    if action == "BUY":
        # رفع الـ target بنسبة الـ boost
        adjusted_target = vamp_target * (1 + boost)
    elif action == "SELL":
        # خفض الـ target
        adjusted_target = vamp_target * (1 - boost)
    else:
        adjusted_target = vamp_target

    current = vamp_target / (1 + vamp_growth / 100) if vamp_growth != -100 else vamp_target
    adjusted_growth = (adjusted_target - current) / current * 100 if current > 0 else vamp_growth

    return {
        "adjusted_target": round(adjusted_target, 2),
        "adjusted_growth": round(adjusted_growth, 2),
        "signal_applied":  True,
        "signal_action":   action,
        "signal_boost":    boost,
    }

# ===========================================================================
# 6. STREAMLIT TAB
# ===========================================================================

def render_tab(claude_client=None):
    """
    [INTEGRATION POINT]
    tab1,...,tab6 = st.tabs([..., "📋 Reports"])
    with tab6:
        reports.render_tab(claude)
    """
    import streamlit as st

    st.header("📋 Reports & Analyst Signals")
    st.caption("حمّل تقارير التحليل الفني → Claude يستخرج التوصيات → تظهر كـ signal جنب VAMP")

    if not claude_client:
        st.error("❌ Claude API key مطلوب لتحليل التقارير.")
        return

    # ── Upload ────────────────────────────────────────────────────────────
    st.subheader("📤 رفع تقرير جديد")
    uploaded = st.file_uploader(
        "اختار الملف (PDF / صورة / Excel / CSV)",
        type=["pdf", "png", "jpg", "jpeg", "xlsx", "xls", "csv", "txt"],
        key="report_uploader",
    )

    if uploaded:
        file_bytes = uploaded.read()
        file_ext   = Path(uploaded.name).suffix.lower()
        size_kb    = len(file_bytes) / 1024

        st.info(f"📄 {uploaded.name}  —  {size_kb:.1f} KB")

        if st.button("🔍 تحليل التقرير", type="primary"):
            with st.spinner("Claude بيقرأ التقرير ويستخرج التوصيات …"):
                parsed = parse_report(file_bytes, uploaded.name, claude_client)

            if not parsed:
                st.error("فشل التحليل — تأكد من Claude API key")
                return

            if parsed.get("error"):
                st.warning(f"⚠️ {parsed['error']}")

            signals = parsed.get("signals", [])
            summary = parsed.get("report_summary", "")

            if summary:
                st.success(f"📝 {summary}")

            if not signals:
                st.warning("لم يتم استخراج توصيات من هذا التقرير")
                store_report_signals(uploaded.name, file_ext, parsed, file_bytes)
                return

            # ── Preview ───────────────────────────────────────────────────
            st.subheader(f"✅ استُخرج {len(signals)} توصية")
            preview_rows = []
            for s in signals:
                action = s.get("action","?")
                color  = {"BUY":"🟢","SELL":"🔴","HOLD":"🟡","WATCH":"⚪"}.get(action,"⚪")
                preview_rows.append({
                    "": color,
                    "Symbol": s.get("symbol",""),
                    "Action": action,
                    "Entry": f"{s.get('entry_low','')} – {s.get('entry_high','')}",
                    "T1": s.get("target1",""),
                    "T2": s.get("target2",""),
                    "Stop": s.get("stop_loss",""),
                    "Confidence": s.get("confidence",""),
                    "Notes": (s.get("notes","") or "")[:60],
                })
            st.dataframe(pd.DataFrame(preview_rows),
                         hide_index=True, width="stretch")

            if st.button("💾 حفظ التوصيات", type="secondary"):
                report_id = store_report_signals(uploaded.name, file_ext, parsed, file_bytes)
                st.success(f"✅ تم الحفظ — {len(signals)} توصية (صالحة {SIGNAL_TTL_DAYS} يوم)")
                st.rerun()

    st.divider()

    # ── Active Signals ────────────────────────────────────────────────────
    st.subheader("🎯 التوصيات النشطة")
    active_df = get_all_active_signals()

    if not active_df.empty:
        # Color-code by action
        def color_action(val):
            colors = {"BUY":"background-color:#064e3b;color:#10b981",
                      "SELL":"background-color:#4c0519;color:#f43f5e",
                      "HOLD":"background-color:#1c1917;color:#f59e0b",
                      "WATCH":"background-color:#1e293b;color:#94a3b8"}
            return colors.get(val,"")

        styled = active_df.style.map(color_action, subset=["action"])
        st.dataframe(styled, hide_index=True, width="stretch")

        # Conflict check مع VAMP
        st.caption(f"⏳ التوصيات صالحة لـ {SIGNAL_TTL_DAYS} يوم من تاريخ الرفع")
    else:
        st.info("لا يوجد توصيات نشطة حالياً — ارفع تقريراً جديداً.")

    st.divider()

    # ── Reports History ───────────────────────────────────────────────────
    st.subheader("📁 سجل التقارير")
    hist_df = get_reports_history(20)
    if not hist_df.empty:
        st.dataframe(hist_df, hide_index=True, width="stretch")
    else:
        st.info("لا يوجد تقارير محفوظة بعد.")

    st.divider()
    st.caption(
        f"DB: {REPORTS_DB.name}  |  "
        f"Signal TTL: {SIGNAL_TTL_DAYS} days  |  "
        "Powered by Claude API (vision + text)"
    )
