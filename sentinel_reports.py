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
import os
import re
import sqlite3
import base64
import requests
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

warnings.filterwarnings("ignore")

SCRIPT_DIR      = Path(__file__).parent.resolve()
REPORTS_DB      = SCRIPT_DIR / "sentinel_reports.db"
SIGNAL_TTL_DAYS = 14

# ── Gemini API key (primary engine for report analysis) ──────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
if not GEMINI_API_KEY:
    try:
        import streamlit as st
        GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "").strip()
    except Exception:
        pass
# ─────────────────────────────────────────────────────────────────────────────

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

_EXTRACT_PROMPT = """أنت محلل مالي متخصص في تقارير مباشر (Mubasher) للبورصة المصرية EGX.
اقرأ هذا التقرير بعناية واستخرج كل التوصيات ومستويات التحليل الفني لكل سهم.

أرجع JSON فقط، بدون أي نص إضافي أو Markdown، بهذا الشكل بالضبط:
{
  "report_type": "technical_analysis",
  "report_date": "2025-06-01",
  "market_overview": "نظرة عامة على السوق في جملة واحدة أو null",
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
      "s1": 73.0,
      "s2": 70.0,
      "s3": null,
      "r1": 82.0,
      "r2": 89.0,
      "r3": null,
      "timeframe": "2-4 أسابيع",
      "confidence": "HIGH",
      "pattern": "اختراق مقاومة مع حجم مرتفع",
      "rsi": 58.0,
      "volume_signal": "مرتفع",
      "notes": "أي ملاحظات إضافية"
    }
  ],
  "report_summary": "ملخص التقرير في جملة واحدة"
}

قواعد صارمة:
- symbol: رمز السهم بالأحرف الإنجليزية بدون .EGX (مثل COMI وليس COMI.EGX)
- action: BUY أو SELL أو HOLD أو WATCH فقط — لا قيم أخرى
- report_type: technical_analysis | morning_call | daily_summary | insider_trading | stock_info | egx_daily
- report_date: تاريخ التقرير بصيغة YYYY-MM-DD أو null
- جميع الأرقام الفنية كـ float أو null إذا لم تُذكر صراحةً في التقرير
- s1/s2/s3: مستويات الدعم الأول والثاني والثالث
- r1/r2/r3: مستويات المقاومة الأول والثاني والثالث
- target1/target2/target3: الأهداف السعرية المستهدفة بالترتيب
- entry_low/entry_high: منطقة الدخول الموصى بها (سعر الشراء)
- stop_loss: مستوى وقف الخسارة
- pattern: النمط الفني أو الإشارة (مثل: اختراق، ارتداد، ضغط، تقاطع)
- confidence: HIGH إذا كانت التوصية واضحة وقوية، MEDIUM إذا كانت محتملة، LOW إذا كانت مشروطة
- إذا احتوى التقرير على أسهم متعددة، استخرج توصية لكل سهم في مصفوفة signals
- إذا لم تجد توصيات واضحة، أرجع signals كـ []
"""

# ===========================================================================
# 2. FILE PARSER — Gemini primary / Claude fallback
# ===========================================================================

_GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.0-flash:generateContent?key={key}"
)
_GEMINI_UPLOAD_URL = (
    "https://generativelanguage.googleapis.com/upload/v1beta/files?key={key}"
)
_INLINE_SIZE_LIMIT = 15 * 1024 * 1024   # 15 MB raw → ~20 MB base64


def _upload_to_gemini_files(file_bytes: bytes, mime_type: str) -> Optional[str]:
    """
    Upload large file (>15 MB) via Gemini Files API.
    Returns the file URI to reference in the generate request.
    """
    boundary = "sentinel_gemini_boundary"
    metadata  = json.dumps({"file": {"displayName": "mubasher_report"}}).encode()

    body = (
        f"--{boundary}\r\n"
        f"Content-Type: application/json; charset=utf-8\r\n\r\n"
    ).encode() + metadata + (
        f"\r\n--{boundary}\r\n"
        f"Content-Type: {mime_type}\r\n\r\n"
    ).encode() + file_bytes + f"\r\n--{boundary}--".encode()

    try:
        resp = requests.post(
            _GEMINI_UPLOAD_URL.format(key=GEMINI_API_KEY),
            headers={
                "Content-Type": f"multipart/related; boundary={boundary}",
                "Content-Length": str(len(body)),
            },
            data=body,
            timeout=180,
        )
        resp.raise_for_status()
        return resp.json().get("file", {}).get("uri")
    except Exception as e:
        print(f"[Reports] Gemini file upload error: {e}")
        return None


def _call_gemini(parts: list) -> Optional[str]:
    """Send parts list to Gemini 1.5 Flash and return raw text response."""
    try:
        resp = requests.post(
            _GEMINI_URL.format(key=GEMINI_API_KEY),
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{"parts": parts}],
                "generationConfig": {"temperature": 0.1, "maxOutputTokens": 2048},
            },
            timeout=90,
        )
        if not resp.ok:
            # Return structured error so caller can surface it to the UI
            return f"__GEMINI_ERROR__{resp.status_code}::{resp.text[:400]}"
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"__GEMINI_ERROR__network::{str(e)}"


def _parse_with_gemini(file_bytes: bytes, ext: str) -> Optional[str]:
    """
    Route file to Gemini:
      • PDF / image  → inline base64 (≤15 MB) or Files API (>15 MB)
      • Excel / CSV  → pandas text → Gemini text
      • Plain text   → Gemini text
    """
    mime_map = {
        ".pdf":  "application/pdf",
        ".png":  "image/png",
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }
    mime_type = mime_map.get(ext)

    if mime_type:
        size = len(file_bytes)
        if size <= _INLINE_SIZE_LIMIT:
            # Inline base64 — fast path
            b64 = base64.standard_b64encode(file_bytes).decode("utf-8")
            parts = [
                {"inline_data": {"mime_type": mime_type, "data": b64}},
                {"text": _EXTRACT_PROMPT},
            ]
        else:
            # Large file — upload via Files API first
            file_uri = _upload_to_gemini_files(file_bytes, mime_type)
            if not file_uri:
                return None
            parts = [
                {"file_data": {"mime_type": mime_type, "file_uri": file_uri}},
                {"text": _EXTRACT_PROMPT},
            ]
        return _call_gemini(parts)

    # Excel / CSV → extract text, send as prompt
    if ext in (".xlsx", ".xls", ".csv"):
        import io
        try:
            df = pd.read_csv(io.BytesIO(file_bytes)) if ext == ".csv" \
                 else pd.read_excel(io.BytesIO(file_bytes))
            text_content = df.to_string(index=False, max_rows=300)
        except Exception:
            text_content = file_bytes.decode("utf-8", errors="ignore")[:10000]
        parts = [{"text": f"هذا جدول بيانات من تقرير تحليل فني:\n\n{text_content}\n\n{_EXTRACT_PROMPT}"}]
        return _call_gemini(parts)

    # Plain text fallback
    text_content = file_bytes.decode("utf-8", errors="ignore")[:10000]
    parts = [{"text": f"هذا تقرير تحليل فني:\n\n{text_content}\n\n{_EXTRACT_PROMPT}"}]
    return _call_gemini(parts)


def parse_report(file_bytes: bytes, file_name: str,
                 claude_client=None) -> Optional[Dict]:
    """
    Extract structured signals from a report file.
    Primary engine  : Gemini 1.5 Flash (free, no credits needed)
    Fallback engine : Claude SDK (used only if Gemini fails and claude_client provided)
    Supports: PDF (text-based or image-based), PNG/JPG, Excel, CSV, TXT
    Large PDFs (>15 MB) are uploaded via Gemini Files API automatically.
    """
    ext      = Path(file_name).suffix.lower()
    raw_text = None

    # ── Primary: Gemini ────────────────────────────────────────────────────
    if GEMINI_API_KEY:
        raw_text = _parse_with_gemini(file_bytes, ext)

    # ── Fallback: Claude SDK ───────────────────────────────────────────────
    if not raw_text and claude_client:
        try:
            mime_map = {".pdf": "application/pdf", ".png": "image/png",
                        ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
            if ext in mime_map:
                b64  = base64.standard_b64encode(file_bytes).decode("utf-8")
                block = ({"type": "document",
                           "source": {"type": "base64", "media_type": mime_map[ext], "data": b64}}
                          if ext == ".pdf" else
                          {"type": "image",
                           "source": {"type": "base64", "media_type": mime_map[ext], "data": b64}})
                resp = claude_client.messages.create(
                    model="claude-sonnet-4-6", max_tokens=2000,
                    messages=[{"role": "user",
                                "content": [block, {"type": "text", "text": _EXTRACT_PROMPT}]}])
            else:
                text_content = file_bytes.decode("utf-8", errors="ignore")[:8000]
                resp = claude_client.messages.create(
                    model="claude-sonnet-4-6", max_tokens=2000,
                    messages=[{"role": "user",
                                "content": f"هذا تقرير:\n\n{text_content}\n\n{_EXTRACT_PROMPT}"}])
            raw_text = resp.content[0].text.strip()
        except Exception as e:
            print(f"[Reports] Claude fallback error: {e}")

    if not raw_text:
        return {"signals": [], "report_summary": "فشل التحليل — لم يُستلم رد من Gemini أو Claude",
                "error": "no_response"}

    # Surface any Gemini HTTP/network error so the UI can display it
    if isinstance(raw_text, str) and raw_text.startswith("__GEMINI_ERROR__"):
        err_detail = raw_text.replace("__GEMINI_ERROR__", "")
        return {"signals": [], "report_summary": f"Gemini API Error: {err_detail}",
                "error": "gemini_api_error", "error_detail": err_detail}

    try:
        clean = re.sub(r"```(?:json)?|```", "", raw_text).strip()
        return json.loads(clean)
    except Exception as e:
        return {"signals": [], "report_summary": f"خطأ في تحليل JSON: {e}",
                "error": str(e), "raw": raw_text[:500]}

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
    Renders the Reports & Analyst Signals tab.
    Primary engine: Gemini 1.5 Flash (free).
    Fallback: Claude SDK (if available and Gemini fails).
    """
    import streamlit as st

    st.header("📋 Reports & Analyst Signals")
    st.caption("حمّل تقارير مباشر → Gemini يستخرج التوصيات ومستويات الدعم/المقاومة → تظهر كـ signal في Single Stock")

    if not GEMINI_API_KEY:
        st.error("❌ GEMINI_API_KEY غير موجود — أضفه في Streamlit Secrets باسم GEMINI_API_KEY")
        return

    engine_label = "🤖 Gemini 1.5 Flash"
    if claude_client:
        engine_label += " + Claude fallback"
    st.caption(f"محرك التحليل: {engine_label} | الحد الأقصى للملف: 15 MB (inline) أو أكبر via Files API")

    # ── Gemini connectivity test ──────────────────────────────────────────
    with st.expander("🔧 Gemini API Diagnostic", expanded=False):
        if st.button("▶️ Test Gemini Connection", key="test_gemini_btn"):
            with st.spinner("Testing Gemini API key…"):
                test_result = _call_gemini([{"text": "Reply with the single word: OK"}])
            if test_result and not test_result.startswith("__GEMINI_ERROR__"):
                st.success(f"✅ Gemini working — response: {test_result.strip()[:80]}")
            elif test_result:
                err = test_result.replace("__GEMINI_ERROR__", "")
                st.error(f"❌ Gemini failed:\n```\n{err}\n```")
            else:
                st.error("❌ No response — possible network block")

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
            with st.spinner("Gemini بيقرأ التقرير ويستخرج التوصيات ومستويات الدعم/المقاومة …"):
                parsed = parse_report(file_bytes, uploaded.name, claude_client)

            if not parsed:
                st.error("فشل التحليل — تأكد من Gemini API key")
                return

            if parsed.get("error") == "gemini_api_error":
                st.error(f"❌ Gemini API Error:\n```\n{parsed.get('error_detail','')}\n```")
                st.info("الحل: تحقق من المفتاح في Streamlit Secrets أو جرّب مفتاح جديد من aistudio.google.com")
                return

            if parsed.get("error"):
                st.warning(f"⚠️ {parsed.get('report_summary','')}")

            signals = parsed.get("signals", [])
            summary = parsed.get("report_summary", "")

            if summary:
                st.success(f"📝 {summary}")

            if not signals:
                st.warning("لم يتم استخراج توصيات من هذا التقرير")
                store_report_signals(uploaded.name, file_ext, parsed, file_bytes)
                return

            # ── Report metadata ───────────────────────────────────────────
            meta_cols = st.columns(3)
            meta_cols[0].caption(f"📋 نوع التقرير: {parsed.get('report_type','—')}")
            meta_cols[1].caption(f"📅 التاريخ: {parsed.get('report_date','—')}")
            if parsed.get("market_overview"):
                st.info(f"📊 نظرة السوق: {parsed['market_overview']}")

            # ── Preview ───────────────────────────────────────────────────
            st.subheader(f"✅ استُخرج {len(signals)} توصية")
            preview_rows = []
            for s in signals:
                action = s.get("action", "?")
                color  = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡", "WATCH": "⚪"}.get(action, "⚪")
                preview_rows.append({
                    "":         color,
                    "Symbol":   s.get("symbol", ""),
                    "Action":   action,
                    "Entry":    f"{s.get('entry_low','')} – {s.get('entry_high','')}",
                    "T1":       s.get("target1", ""),
                    "T2":       s.get("target2", ""),
                    "T3":       s.get("target3", ""),
                    "Stop":     s.get("stop_loss", ""),
                    "S1":       s.get("s1", ""),
                    "S2":       s.get("s2", ""),
                    "R1":       s.get("r1", ""),
                    "R2":       s.get("r2", ""),
                    "Pattern":  (s.get("pattern") or "")[:30],
                    "Conf":     s.get("confidence", ""),
                    "Notes":    (s.get("notes") or "")[:40],
                })
            st.dataframe(pd.DataFrame(preview_rows),
                         hide_index=True, use_container_width=True)

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
        st.dataframe(styled, hide_index=True, use_container_width=True)

        # Conflict check مع VAMP
        st.caption(f"⏳ التوصيات صالحة لـ {SIGNAL_TTL_DAYS} يوم من تاريخ الرفع")
    else:
        st.info("لا يوجد توصيات نشطة حالياً — ارفع تقريراً جديداً.")

    st.divider()

    # ── Reports History ───────────────────────────────────────────────────
    st.subheader("📁 سجل التقارير")
    try:
        hist_df = get_reports_history(20)
        if not hist_df.empty:
            st.dataframe(hist_df, hide_index=True, use_container_width=True)
        else:
            st.info("لا يوجد تقارير محفوظة بعد.")
    except Exception as _hist_err:
        st.warning(f"⚠️ Could not load reports history: {_hist_err}")

    st.divider()
    st.caption(
        f"DB: {REPORTS_DB.name}  |  "
        f"Signal TTL: {SIGNAL_TTL_DAYS} days  |  "
        "Powered by Claude API (vision + text)"
    )
