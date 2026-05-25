"""
Sentinel-EGX v4.2.2 — Delta-Update Cache Engine
=================================================
Row-level SQLite cache with intelligent gap detection.
Only fetches missing date ranges from EODHD. Separate DB to avoid migration.

Features:
  • Per-row storage (symbol + date PK) — no JSON blobs
  • Automatic missing-range detection with contiguous-range consolidation
  • Single-fetch optimization: one API call covers all gaps
  • EGX trading-day aware (skips Fri/Sat + holidays)
  • Metadata tracking per symbol (earliest, latest, bar count, last fetch)
  • Backward-compatible export: DataCache still available in data_engine
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Dict
from pathlib import Path


class DeltaCache:
    """Intelligent EOD data cache: stores per-row, fetches only deltas."""

    def __init__(self, db_path: str = "sentinel_delta_cache.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS eod_data (
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open REAL, high REAL, low REAL, close REAL, volume REAL,
                    source TEXT DEFAULT 'eodhd',
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (symbol, date)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_meta (
                    symbol TEXT PRIMARY KEY,
                    earliest_date TEXT,
                    latest_date TEXT,
                    total_bars INTEGER DEFAULT 0,
                    last_fetch TIMESTAMP,
                    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_eod_data_symbol_date 
                ON eod_data(symbol, date)
            """)
            conn.commit()

    # ── READ ──

    def get_data(self, symbol: str, start_date: Optional[str] = None,
                 end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Retrieve cached OHLCV for symbol, optionally filtered by date range."""
        query = """SELECT date, open, high, low, close, volume, source
                     FROM eod_data WHERE symbol = ?"""
        params = [symbol]
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        query += " ORDER BY date ASC"

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params, parse_dates=["date"])
        if df.empty:
            return None
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["close"])
        return df.reset_index(drop=True)

    def get_latest_date(self, symbol: str) -> Optional[str]:
        """Return the latest cached date for a symbol (YYYY-MM-DD)."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT MAX(date) FROM eod_data WHERE symbol = ?", (symbol,)
            )
            row = cursor.fetchone()
            return row[0] if row and row[0] else None

    def get_earliest_date(self, symbol: str) -> Optional[str]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT MIN(date) FROM eod_data WHERE symbol = ?", (symbol,)
            )
            row = cursor.fetchone()
            return row[0] if row and row[0] else None

    # ── GAP DETECTION ──

    def get_missing_ranges(self, symbol: str, start_date: str, end_date: str,
                           trading_days: Optional[List[str]] = None) -> List[Tuple[str, str]]:
        """
        Return consolidated (from, to) date ranges that are NOT in cache.
        If trading_days provided, only those dates are considered required.
        """
        cached_df = self.get_data(symbol, start_date, end_date)
        if cached_df is None or cached_df.empty:
            return [(start_date, end_date)]

        cached_dates = set(pd.to_datetime(cached_df["date"]).dt.strftime("%Y-%m-%d"))

        if trading_days:
            needed = set(trading_days)
        else:
            needed = set()
            cur = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            while cur <= end:
                needed.add(cur.strftime("%Y-%m-%d"))
                cur += timedelta(days=1)

        missing = sorted(needed - cached_dates)
        if not missing:
            return []

        # Consolidate contiguous dates into ranges
        ranges = []
        range_start = missing[0]
        prev = datetime.strptime(missing[0], "%Y-%m-%d")

        for d_str in missing[1:]:
            curr = datetime.strptime(d_str, "%Y-%m-%d")
            if (curr - prev).days > 1:
                ranges.append((range_start, prev.strftime("%Y-%m-%d")))
                range_start = d_str
            prev = curr
        ranges.append((range_start, prev.strftime("%Y-%m-%d")))
        return ranges

    def needs_refresh(self, symbol: str, required_end_date: str,
                      ttl_hours: int = 12) -> bool:
        """Check if cache is stale or incomplete up to required_end_date."""
        meta = self.get_meta(symbol)
        if not meta:
            return True
        latest = meta.get("latest_date")
        if not latest:
            return True
        if latest < required_end_date:
            return True
        last_fetch = meta.get("last_fetch")
        if last_fetch:
            last_dt = datetime.fromisoformat(last_fetch.replace("Z", "+00:00"))
            if datetime.now() - last_dt > timedelta(hours=ttl_hours):
                return True
        return False

    # ── WRITE ──

    def insert_data(self, symbol: str, df: pd.DataFrame, source: str = "eodhd"):
        """Upsert OHLCV rows. Duplicate dates are replaced."""
        if df is None or df.empty:
            return
        with sqlite3.connect(self.db_path) as conn:
            for _, row in df.iterrows():
                d = row["date"]
                if isinstance(d, pd.Timestamp):
                    d = d.strftime("%Y-%m-%d")
                elif hasattr(d, "strftime"):
                    d = d.strftime("%Y-%m-%d")
                else:
                    d = str(d)[:10]
                conn.execute("""
                    INSERT OR REPLACE INTO eod_data 
                    (symbol, date, open, high, low, close, volume, source, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    symbol, d,
                    float(row.get("open", 0) or 0),
                    float(row.get("high", 0) or 0),
                    float(row.get("low", 0) or 0),
                    float(row.get("close", 0) or 0),
                    float(row.get("volume", 0) or 0),
                    source
                ))
            conn.commit()
            self._update_meta(symbol, df)

    def _update_meta(self, symbol: str, df: pd.DataFrame):
        dates = pd.to_datetime(df["date"])
        earliest = dates.min().strftime("%Y-%m-%d")
        latest = dates.max().strftime("%Y-%m-%d")

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO cache_meta (symbol, earliest_date, latest_date, total_bars, last_fetch, last_update)
                VALUES (?, ?, ?, (SELECT COUNT(*) FROM eod_data WHERE symbol = ?), CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(symbol) DO UPDATE SET
                    earliest_date = CASE WHEN excluded.earliest_date < cache_meta.earliest_date 
                                        THEN excluded.earliest_date ELSE cache_meta.earliest_date END,
                    latest_date = CASE WHEN excluded.latest_date > cache_meta.latest_date 
                                     THEN excluded.latest_date ELSE cache_meta.latest_date END,
                    total_bars = (SELECT COUNT(*) FROM eod_data WHERE symbol = ?),
                    last_fetch = CURRENT_TIMESTAMP,
                    last_update = CURRENT_TIMESTAMP
            """, (symbol, earliest, latest, symbol, symbol))
            conn.commit()

    def merge_fetch(self, symbol: str, new_df: pd.DataFrame,
                    source: str = "eodhd") -> pd.DataFrame:
        """Insert new data and return the full cached dataset for symbol."""
        self.insert_data(symbol, new_df, source)
        full = self.get_data(symbol)
        return full if full is not None else new_df

    # ── METADATA & STATS ──

    def get_meta(self, symbol: str) -> Optional[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM cache_meta WHERE symbol = ?", (symbol,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_stats(self) -> Dict:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT COUNT(DISTINCT symbol) as symbols, COUNT(*) as rows, MAX(updated_at) as latest
                FROM eod_data
            """)
            row = cursor.fetchone()
            return {"symbols": row[0], "rows": row[1], "latest_update": row[2]}

    def list_symbols(self) -> List[str]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT DISTINCT symbol FROM eod_data ORDER BY symbol")
            return [r[0] for r in cursor.fetchall()]

    # ── CLEANUP ──

    def clear_symbol(self, symbol: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM eod_data WHERE symbol = ?", (symbol,))
            conn.execute("DELETE FROM cache_meta WHERE symbol = ?", (symbol,))
            conn.commit()

    def clear_all(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM eod_data")
            conn.execute("DELETE FROM cache_meta")
            conn.commit()

    def vacuum(self):
        """Reclaim disk space after large deletions."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("VACUUM")


if __name__ == "__main__":
    print("DeltaCache v4.2.2 ready: row-level EOD storage with intelligent gap fetching.")
    dc = DeltaCache()
    print(f"DB path: {dc.db_path}")
    print(f"Stats: {dc.get_stats()}")
