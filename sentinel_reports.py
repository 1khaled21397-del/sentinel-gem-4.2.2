"""
Sentinel-EGX v4.3.0 — Reports & Analyst Signals
================================================
Engine  : OpenRouter vision (KIMI_API_KEY) — primary
Fallback: Gemini 2.0 Flash direct API
PDF     : pymupdf → page images → batched vision calls (5 pages/batch)
Drive   : public folder URL → list + auto-download + skip already-analyzed
Cache   : SQLite drive_cache table — never re-processes same file
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
PAGES_PER_BATCH = 5       # images per OpenRouter request
PDF_DPI         = 100     # lower = fewer tokens, faster

# ── API KEYS ──────────────────────────────────────────────────────────────────
def _get_secret(key: str) -> str:
    val = os.getenv(key, "").strip()
    if not val:
        try:
            import streamlit as st
            val = st.secrets.get(key, "").strip()
        except Exception:
            pass
    return val

OPENROUTER_KEY = _get_secret("KIMI_API_KEY")     # OpenRouter stored as KIMI
GEMINI_KEY     = _get_secret("GEMINI_API_KEY")    # Gemini fallback

OPENROUTER_MODEL = "google/gemini-2.0-flash-exp:free"
OPENROUTER_URL   = "https://openrouter.ai/api/v1/chat/completions"
GEMINI_URL       = ("https://generativelanguage.googleapis.com/v1beta/models/"
                    "gemini-2.0-flash:generateContent?key={key}")

# ── DATABASE ──────────────────────────────────────────────────────────────────
def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(REPORTS_DB))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = _db()
    conn.executescript("""
    PRAGMA journal_mode=WAL;

    CREATE TABLE IF NOT EXISTS reports (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        filename        TEXT NOT NULL,
        file_type       TEXT NOT NULL,
        drive_file_id   TEXT,
        source          TEXT DEFAULT 'manual',
        raw_text        TEXT,
        report_type     TEXT,
        report_date     TEXT,
        market_overview TEXT,
        uploaded_at     TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS analyst_signals (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        report_id     INTEGER REFERENCES reports(id),
        symbol        TEXT NOT NULL,
        action        TEXT,
        entry_low     REAL, entry_high REAL,
        target1       REAL, target2 REAL, target3 REAL,
        stop_loss     REAL,
        s1 REAL, s2 REAL, s3 REAL,
        r1 REAL, r2 REAL, r3 REAL,
        timeframe     TEXT,
        confidence    TEXT,
        pattern       TEXT,
        rsi           REAL,
        volume_signal TEXT,
        notes         TEXT,
        valid_until   TEXT,
        created_at    TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS drive_cache (
        file_id     TEXT PRIMARY KEY,
        filename    TEXT NOT NULL,
        report_id   INTEGER REFERENCES reports(id),
        analyzed_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE INDEX IF NOT EXISTS idx_sig_symbol ON analyst_signals(symbol);
    CREATE INDEX IF NOT EXISTS idx_sig_valid  ON analyst_signals(valid_until);
    """)
    conn.commit()
    conn.close()

init_db()

# ── GOOGLE DRIVE ──────────────────────────────────────────────────────────────
def _extract_folder_id(url: str) -> Optional[str]:
    for pattern in [r'/drive/folders/([A-Za-z0-9_-]{25,})',
                    r'id=([A-Za-z0-9_-]{25,})',
                    r'^([A-Za-z0-9_-]{25,})$']:
        m = re.search(pattern, url.strip())
        if m:
            return m.group(1)
    return None

def _list_drive_folder(folder_id: str) -> List[Dict]:
    """List PDF files in a public Google Drive folder — no API key needed."""
    try:
        resp = requests.get(
            f"https://drive.google.com/drive/folders/{folder_id}",
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
            timeout=15,
        )
        # Drive embeds file data as JSON — extract file IDs and PDF names
        matches = re.findall(
            r'\["([A-Za-z0-9_-]{25,})",null,"([^"]+\.pdf)"',
            resp.text, re.IGNORECASE
        )
        seen, files = set(), []
        for fid, fname in matches:
            if fid not in seen:
                seen.add(fid)
                files.append({"id": fid, "name": fname})
        return sorted(files, key=lambda x: x["name"], reverse=True)
    except Exception as e:
        print(f"[Reports] Drive listing error: {e}")
        return []

def _download_drive_file(file_id: str) -> Optional[bytes]:
    """Download a public Google Drive file using gdown."""
    try:
        import gdown, tempfile
        url    = f"https://drive.google.com/uc?id={file_id}"
        fpath  = tempfile.mktemp(suffix=".pdf")
        result = gdown.download(url, fpath, quiet=True, fuzzy=True, use_cookies=False)
        if result and Path(result).exists():
            data = Path(result).read_bytes()
            Path(result).unlink(missing_ok=True)
            return data
        return None
    except Exception as e:
        print(f"[Reports] Drive download error: {e}")
        return None

def is_drive_analyzed(file_id: str) -> bool:
    conn = _db()
    row  = conn.execute("SELECT 1 FROM drive_cache WHERE file_id=?", (file_id,)).fetchone()
    conn.close()
    return row is not None

def _mark_drive_analyzed(file_id: str, filename: str, report_id: int):
    conn = _db()
    conn.execute("""
        INSERT OR REPLACE INTO drive_cache (file_id, filename, report_id, analyzed_at)
        VALUES (?,?,?,?)
    """, (file_id, filename, report_id, datetime.now().isoformat()))
    conn.commit()
    conn.close()

# ── PDF → IMAGES ──────────────────────────────────────────────────────────────
def _pdf_to_images(pdf_bytes: bytes) -> List[str]:
    """Convert every PDF page to base64 PNG using pymupdf."""
    try:
        import fitz
        doc  = fitz.open(stream=pdf_bytes, filetype="pdf")
        mat  = fitz.Matrix(PDF_DPI / 72, PDF_DPI / 72)
        imgs = []
        for page in doc:
            pix = page.get_pixmap(matrix=mat)
            imgs.append(base64.b64encode(pix.tobytes("png")).decode())
        doc.close()
        return imgs
    except Exception as e:
        print(f"[Reports] PDF→images error: {e}")
        return []

# ── EXTRACTION PROMPT ─────────────────────────────────────────────────────────
_PROMPT = """You are a senior financial analyst specializing in Mubasher institutional research for the Egyptian Exchange (EGX).
Analyze ALL pages and extract every stock recommendation and technical level.
The report may contain Arabic and English text — extract from both.

Return ONLY valid JSON, no markdown:
{
  "report_type": "morning_call",
  "report_date": "2026-06-08",
  "market_overview": "One sentence market summary or null",
  "signals": [
    {
      "symbol": "COMI",
      "action": "BUY",
      "entry_low": 75.0, "entry_high": 78.0,
      "target1": 85.0, "target2": 92.0, "target3": null,
      "stop_loss": 71.0,
      "s1": 73.0, "s2": 70.0, "s3": null,
      "r1": 82.0, "r2": 89.0, "r3": null,
      "timeframe": "2-4 weeks",
      "confidence": "HIGH",
      "pattern": "Resistance breakout high volume",
      "rsi": 58.0,
      "volume_signal": "High",
      "notes": "Additional context"
    }
  ],
  "report_summary": "One sentence summary"
}

RULES:
- symbol: ticker WITHOUT .EGX suffix
- action: BUY | SELL | HOLD | WATCH only
- report_type: morning_call | technical_analysis | daily_summary | insider_trading | stock_info | egx_daily
- All price fields: float or null if not mentioned
- s1/s2/s3 = support levels, r1/r2/r3 = resistance levels
- confidence: HIGH=clear strong call, MEDIUM=conditional, LOW=speculative
- Extract ALL stocks mentioned across all pages
- signals: [] if no clear recommendations
"""

# ── OPENROUTER VISION ─────────────────────────────────────────────────────────
def _call_openrouter(images_b64: List[str]) -> str:
    """Send image batch to OpenRouter free vision model."""
    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b}"}}
        for b in images_b64
    ]
    content.append({"type": "text", "text": _PROMPT})
    try:
        resp = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://sentinel-egx.streamlit.app",
            },
            json={
                "model": OPENROUTER_MODEL,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 3000,
            },
            timeout=120,
        )
        if not resp.ok:
            return f"__ERROR__{resp.status_code}::{resp.text[:300]}"
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"__ERROR__network::{e}"

def _call_gemini_vision(images_b64: List[str]) -> Optional[str]:
    """Gemini fallback for vision."""
    if not GEMINI_KEY:
        return None
    parts = [
        {"inline_data": {"mime_type": "image/png", "data": b}}
        for b in images_b64
    ]
    parts.append({"text": _PROMPT})
    try:
        resp = requests.post(
            GEMINI_URL.format(key=GEMINI_KEY),
            headers={"Content-Type": "application/json"},
            json={"contents": [{"parts": parts}],
                  "generationConfig": {"temperature": 0.1, "maxOutputTokens": 3000}},
            timeout=90,
        )
        if not resp.ok:
            return None
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return None

def _parse_json(raw: str) -> Optional[Dict]:
    if not raw:
        return None
    try:
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        return json.loads(clean)
    except Exception:
        return None

def _merge_batches(results: List[Dict]) -> Dict:
    """Merge signals from multiple page-batch calls."""
    merged = {"report_type": "unknown", "report_date": None,
              "market_overview": None, "signals": [], "report_summary": ""}
    seen = set()
    for r in results:
        if not r:
            continue
        if r.get("report_type", "unknown") != "unknown":
            merged["report_type"] = r["report_type"]
        merged["report_date"]     = merged["report_date"]     or r.get("report_date")
        merged["market_overview"] = merged["market_overview"] or r.get("market_overview")
        merged["report_summary"]  = r.get("report_summary")  or merged["report_summary"]
        for sig in r.get("signals", []):
            sym = sig.get("symbol", "").upper()
            if sym and sym not in seen:
                seen.add(sym)
                merged["signals"].append(sig)
    return merged

# ── MAIN PARSE ENTRY POINT ────────────────────────────────────────────────────
def parse_report(file_bytes: bytes, file_name: str,
                 claude_client=None) -> Dict:
    """
    Parse a report file → structured signals.
    PDF/image → pymupdf pages → OpenRouter batches → merge.
    Falls back to Gemini vision if OpenRouter fails.
    """
    ext = Path(file_name).suffix.lower()

    # Convert to images
    if ext == ".pdf":
        images = _pdf_to_images(file_bytes)
        if not images:
            return {"signals": [], "error": "pdf_failed",
                    "report_summary": "pymupdf could not open PDF"}
    elif ext in (".png", ".jpg", ".jpeg", ".webp"):
        images = [base64.b64encode(file_bytes).decode()]
    else:
        return {"signals": [], "error": "unsupported",
                "report_summary": f"Unsupported format: {ext}"}

    batch_results = []
    for i in range(0, len(images), PAGES_PER_BATCH):
        batch = images[i:i + PAGES_PER_BATCH]

        # Primary: OpenRouter
        raw = _call_openrouter(batch) if OPENROUTER_KEY else None

        if raw and raw.startswith("__ERROR__"):
            err = raw.replace("__ERROR__", "")
            # Fallback to Gemini on error
            raw = _call_gemini_vision(batch)
            if not raw:
                return {"signals": [], "error": "openrouter_error",
                        "error_detail": err,
                        "report_summary": f"OpenRouter Error: {err}"}

        parsed = _parse_json(raw)
        if parsed:
            batch_results.append(parsed)

    if not batch_results:
        return {"signals": [], "error": "no_response",
                "report_summary": "No response from vision engine"}

    return _merge_batches(batch_results)

# ── STORAGE ───────────────────────────────────────────────────────────────────
def store_report_signals(filename: str, file_type: str,
                          parsed: Dict, drive_file_id: str = None) -> int:
    conn  = _db()
    valid = (datetime.now() + timedelta(days=SIGNAL_TTL_DAYS)).strftime("%Y-%m-%d")

    cur = conn.execute("""
        INSERT INTO reports
            (filename, file_type, drive_file_id, raw_text, report_type, report_date, market_overview)
        VALUES (?,?,?,?,?,?,?)
    """, (filename, file_type, drive_file_id,
          parsed.get("report_summary", ""),
          parsed.get("report_type", ""),
          parsed.get("report_date", ""),
          parsed.get("market_overview", "")))
    rid = cur.lastrowid

    for sig in parsed.get("signals", []):
        sym = sig.get("symbol", "").strip().upper()
        if not sym:
            continue
        sym_full = sym if sym.endswith(".EGX") else sym + ".EGX"
        conn.execute("""
            INSERT INTO analyst_signals
                (report_id, symbol, action, entry_low, entry_high,
                 target1, target2, target3, stop_loss,
                 s1, s2, s3, r1, r2, r3,
                 timeframe, confidence, pattern,
                 rsi, volume_signal, notes, valid_until)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (rid, sym_full, sig.get("action", "WATCH"),
              sig.get("entry_low"), sig.get("entry_high"),
              sig.get("target1"), sig.get("target2"), sig.get("target3"),
              sig.get("stop_loss"),
              sig.get("s1"), sig.get("s2"), sig.get("s3"),
              sig.get("r1"), sig.get("r2"), sig.get("r3"),
              sig.get("timeframe", ""),
              sig.get("confidence", "MEDIUM"),
              sig.get("pattern", ""),
              sig.get("rsi"), sig.get("volume_signal", ""),
              sig.get("notes", ""), valid))

    conn.commit()
    conn.close()

    if drive_file_id:
        _mark_drive_analyzed(drive_file_id, filename, rid)

    return rid

# ── SIGNAL RETRIEVAL ──────────────────────────────────────────────────────────
def get_active_signal(symbol: str) -> Optional[Dict]:
    today = datetime.now().strftime("%Y-%m-%d")
    conn  = _db()
    row   = conn.execute("""
        SELECT s.*, r.report_type, r.market_overview, r.report_date, r.filename
        FROM analyst_signals s
        JOIN reports r ON s.report_id = r.id
        WHERE s.symbol=? AND s.valid_until >= ?
        ORDER BY s.created_at DESC LIMIT 1
    """, (symbol, today)).fetchone()
    conn.close()
    return dict(row) if row else None

def get_all_active_signals() -> pd.DataFrame:
    today = datetime.now().strftime("%Y-%m-%d")
    conn  = _db()
    cur   = conn.execute("""
        SELECT s.symbol, s.action, s.entry_low, s.entry_high,
               s.target1, s.target2, s.stop_loss,
               s.s1, s.r1, s.confidence, s.pattern,
               s.valid_until, s.created_at, r.filename
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
    cur  = conn.execute("""
        SELECT r.filename, r.report_type, r.report_date,
               COUNT(s.id) as signals, r.uploaded_at
        FROM reports r
        LEFT JOIN analyst_signals s ON s.report_id = r.id
        GROUP BY r.id ORDER BY r.uploaded_at DESC LIMIT ?
    """, (limit,))
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    conn.close()
    return pd.DataFrame([dict(zip(cols, r)) for r in rows])

# ── ALPHA RECOMMENDATION ──────────────────────────────────────────────────────
def get_alpha_recommendation(symbol: str, current_alpha: float,
                              current_target: float,
                              current_price: float) -> Optional[Dict]:
    """
    Returns specific number adjustments to alpha score and price target
    based on the latest Mubasher signal for this ticker.
    """
    sig = get_active_signal(symbol)
    if not sig:
        return None

    action     = sig.get("action", "WATCH")
    confidence = sig.get("confidence", "LOW")
    target1    = sig.get("target1")

    conf_mult  = {"HIGH": 0.12, "MEDIUM": 0.07, "LOW": 0.03}.get(confidence, 0.03)
    direction  = {"BUY": 1.0, "SELL": -1.0, "HOLD": 0.3}.get(action)
    if direction is None:
        return None

    alpha_delta = conf_mult * direction
    new_alpha   = round(min(1.0, max(0.0, current_alpha + alpha_delta)), 3)

    new_target = current_target
    if target1 and current_price > 0:
        mb_ret     = (target1 - current_price) / current_price
        vamp_ret   = (current_target - current_price) / current_price
        blended    = vamp_ret * 0.6 + mb_ret * 0.4
        new_target = round(current_price * (1 + blended), 2)

    stop = sig.get("stop_loss")
    rr   = None
    if stop and current_price > 0 and new_target > current_price:
        risk   = current_price - stop
        reward = new_target - current_price
        rr     = round(reward / risk, 1) if risk > 0 else None

    return {
        "current_alpha":  round(current_alpha, 3),
        "new_alpha":      new_alpha,
        "alpha_delta":    f"{alpha_delta:+.3f}",
        "current_target": round(current_target, 2),
        "new_target":     new_target,
        "target_delta":   f"{new_target - current_target:+.2f}",
        "new_rr":         rr,
        "action":         action,
        "confidence":     confidence,
        "pattern":        sig.get("pattern", ""),
        "valid_until":    sig.get("valid_until", ""),
        "source_file":    sig.get("filename", ""),
    }

# ── STREAMLIT UI ──────────────────────────────────────────────────────────────
def render_tab(claude_client=None):
    import streamlit as st

    st.header("📋 Reports & Analyst Signals")
    st.caption(
        "Mubasher reports → OpenRouter vision → signals + S/R levels → "
        "Single Stock panel with specific alpha recommendations"
    )

    if not OPENROUTER_KEY:
        st.error("❌ KIMI_API_KEY (OpenRouter) not found in Streamlit Secrets")
        return

    engine = f"🤖 OpenRouter ({OPENROUTER_MODEL})"
    if GEMINI_KEY:
        engine += " + Gemini fallback"
    st.caption(engine)

    # ── Upload mode ───────────────────────────────────────────────────────
    mode = st.radio(
        "Source",
        ["📁 Google Drive Folder", "📤 Manual Upload"],
        horizontal=True, key="rpt_upload_mode",
    )

    if mode == "📁 Google Drive Folder":
        _ui_drive(claude_client)
    else:
        _ui_manual(claude_client)

    st.divider()

    # ── Active signals table ──────────────────────────────────────────────
    st.subheader("🎯 Active Signals")
    df_active = get_all_active_signals()
    if df_active.empty:
        st.info("No active signals — upload a report to start.")
    else:
        def _ca(val):
            return {"BUY": "background-color:#064e3b;color:#10b981",
                    "SELL": "background-color:#4c0519;color:#f43f5e",
                    "HOLD": "background-color:#1c1917;color:#f59e0b",
                    "WATCH": "background-color:#1e293b;color:#94a3b8"}.get(val, "")
        st.dataframe(
            df_active.style.map(_ca, subset=["action"]),
            hide_index=True, use_container_width=True,
        )
        st.caption(f"⏳ Signals valid for {SIGNAL_TTL_DAYS} days from upload")

    st.divider()

    # ── Reports history ───────────────────────────────────────────────────
    st.subheader("📁 Reports History")
    df_hist = get_reports_history()
    if not df_hist.empty:
        st.dataframe(df_hist, hide_index=True, use_container_width=True)
    else:
        st.info("No reports yet.")

    st.caption(
        f"DB: {REPORTS_DB.name} | TTL: {SIGNAL_TTL_DAYS} days | "
        f"Engine: OpenRouter {OPENROUTER_MODEL}"
    )


def _ui_drive(claude_client):
    import streamlit as st

    folder_url = st.text_input(
        "Google Drive folder URL",
        placeholder="https://drive.google.com/drive/folders/...",
        key="drive_folder_url",
    )
    if not folder_url.strip():
        return

    folder_id = _extract_folder_id(folder_url)
    if not folder_id:
        st.error("❌ Could not extract folder ID — paste the full Drive folder URL")
        return

    with st.spinner("Loading folder…"):
        files = _list_drive_folder(folder_id)

    if not files:
        st.warning("No PDFs found — make sure folder is shared as 'Anyone with link'")
        return

    tagged  = [{**f, "done": is_drive_analyzed(f["id"])} for f in files]
    new_f   = [f for f in tagged if not f["done"]]
    done_f  = [f for f in tagged if f["done"]]

    st.success(
        f"📂 {len(files)} PDFs found — "
        f"🆕 {len(new_f)} new | ✅ {len(done_f)} already analyzed"
    )

    if not new_f:
        st.info("✅ All files already analyzed — upload new reports to continue.")
        return

    selected = st.multiselect(
        "Select reports to analyze",
        options=[f["name"] for f in new_f],
        default=[f["name"] for f in new_f],
        key="drive_selected",
    )

    if not selected:
        return

    if st.button("🔍 Analyze Selected", type="primary", key="drive_analyze_btn"):
        to_proc = [f for f in new_f if f["name"] in selected]
        bar     = st.progress(0)

        for idx, f in enumerate(to_proc):
            st.caption(f"⏳ {f['name']} ({idx+1}/{len(to_proc)})")

            pdf_bytes = _download_drive_file(f["id"])
            if not pdf_bytes:
                st.warning(f"⚠️ Could not download: {f['name']}")
                bar.progress((idx + 1) / len(to_proc))
                continue

            parsed = parse_report(pdf_bytes, f["name"], claude_client)

            if parsed.get("error") == "openrouter_error":
                st.error(f"❌ {f['name']}: {parsed.get('error_detail','')}")
                bar.progress((idx + 1) / len(to_proc))
                continue

            sigs = parsed.get("signals", [])
            store_report_signals(f["name"], ".pdf", parsed, drive_file_id=f["id"])

            st.success(f"✅ {f['name']}: {len(sigs)} signals")
            if parsed.get("market_overview"):
                st.caption(f"📊 {parsed['market_overview']}")

            bar.progress((idx + 1) / len(to_proc))

        bar.empty()
        st.rerun()


def _ui_manual(claude_client):
    import streamlit as st

    uploaded = st.file_uploader(
        "Upload file (PDF / image / Excel / CSV)",
        type=["pdf", "png", "jpg", "jpeg", "xlsx", "xls", "csv", "txt"],
        key="manual_upload",
    )
    if not uploaded:
        return

    file_bytes = uploaded.read()
    st.info(f"📄 {uploaded.name}  —  {len(file_bytes)/1024:.1f} KB")

    if st.button("🔍 Analyze Report", type="primary", key="manual_analyze_btn"):
        with st.spinner("Analyzing pages…"):
            parsed = parse_report(file_bytes, uploaded.name, claude_client)

        if parsed.get("error") == "openrouter_error":
            st.error(f"❌ OpenRouter Error: {parsed.get('error_detail','')}")
            return

        sigs = parsed.get("signals", [])
        if parsed.get("report_summary"):
            st.success(f"📝 {parsed['report_summary']}")
        if parsed.get("market_overview"):
            st.info(f"📊 Market: {parsed['market_overview']}")

        if not sigs:
            st.warning("No recommendations extracted")
            store_report_signals(uploaded.name, Path(uploaded.name).suffix.lower(), parsed)
            return

        # Preview table
        st.subheader(f"✅ {len(sigs)} signals extracted")
        rows = []
        for s in sigs:
            act   = s.get("action", "?")
            color = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡", "WATCH": "⚪"}.get(act, "⚪")
            rows.append({
                "":       color,
                "Symbol": s.get("symbol", ""),
                "Action": act,
                "Entry":  f"{s.get('entry_low','')} – {s.get('entry_high','')}",
                "T1":     s.get("target1", ""),
                "T2":     s.get("target2", ""),
                "Stop":   s.get("stop_loss", ""),
                "S1/R1":  f"{s.get('s1','')} / {s.get('r1','')}",
                "Pattern":(s.get("pattern") or "")[:25],
                "Conf":   s.get("confidence", ""),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        if st.button("💾 Save Signals", type="secondary", key="manual_save_btn"):
            store_report_signals(
                uploaded.name,
                Path(uploaded.name).suffix.lower(),
                parsed,
            )
            st.success(f"✅ {len(sigs)} signals saved (valid {SIGNAL_TTL_DAYS} days)")
            st.rerun()
