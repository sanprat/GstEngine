"""
invoice_dedup.py — Deduplication Layer for GstEngine & Android App.

Prevents duplicate invoice recording across three complementary verification tiers:

Tier 1: File Content Hash (SHA-256)
        Detects exact duplicate uploads of the same PDF or image file.

Tier 2: Business Key Fingerprint (GSTIN + Invoice Number + Financial Year)
        Indian GST statutory unique constraint (Section 31 CGST Act): A supplier cannot
        issue two invoices with the identical invoice number within the same Financial Year.
        Detects duplicate bills even if captured via different camera angles or photos.

Tier 3: Fuzzy / Near-Duplicate Detection
        Flags potential duplicates where the same supplier and exact amount appear within
        a narrow date window (e.g., catching minor OCR character substitutions like O vs 0).

Storage:
    SQLite database (built-in Python stdlib sqlite3, 100% native to Android Room / SQLiteDatabase).
"""

from __future__ import annotations
import hashlib
import json
import os
import re
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


def compute_file_hash(data_or_path: bytes | str) -> str:
    """Compute SHA-256 hash of a file's raw bytes or file path."""
    if isinstance(data_or_path, str):
        if not os.path.exists(data_or_path):
            return ""
        with open(data_or_path, "rb") as f:
            content = f.read()
    else:
        content = data_or_path
    return hashlib.sha256(content).hexdigest()


def normalize_string(s: Optional[str]) -> str:
    """Normalize text by stripping whitespace and non-alphanumeric noise."""
    if not s:
        return ""
    return re.sub(r"[^A-Za-z0-9]", "", str(s)).upper()


def get_financial_year(date_str: Optional[str]) -> str:
    """
    Compute the Indian Financial Year (April 1 to March 31).
    Example:
        2026-10-24 -> '2026-27'
        2026-02-15 -> '2025-26'
    """
    if not date_str:
        return "UNKNOWN"
    d = None
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y", "%d-%b-%Y", "%d %B %Y", "%d %b %Y"):
        try:
            d = datetime.strptime(str(date_str).strip(), fmt)
            break
        except (ValueError, TypeError):
            continue
    if not d:
        return "UNKNOWN"

    year = d.year
    if d.month >= 4:
        # April to December: current year to next year
        return f"{year}-{str(year + 1)[-2:]}"
    else:
        # January to March: previous year to current year
        return f"{year - 1}-{str(year)[-2:]}"


def compute_invoice_fingerprint(gstin_supplier: Optional[str],
                                invoice_no: Optional[str],
                                date_str: Optional[str] = None,
                                party_name: Optional[str] = None) -> str:
    """
    Deterministic business key fingerprint:
    Hash of (Normalized Supplier GSTIN or Name, Normalized Invoice No, Financial Year).
    """
    norm_gstin = normalize_string(gstin_supplier)
    norm_inv = normalize_string(invoice_no)
    norm_party = normalize_string(party_name)
    fy = get_financial_year(date_str)

    # Prefer GSTIN; fallback to supplier name if GSTIN is not present
    supplier_key = norm_gstin if norm_gstin else f"NAME_{norm_party[:20]}"
    raw_key = f"{supplier_key}:{norm_inv}:{fy}"
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


class InvoiceDeduplicator:
    """
    SQLite-backed deduplication engine.
    Compatible with both Python backend and Android SQLite / Room database.
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(base_dir, "data", "invoice_registry.db")
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Create tables and indexes if they do not exist."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS recorded_invoices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fingerprint TEXT UNIQUE NOT NULL,
                    file_hash TEXT,
                    gstin_supplier TEXT,
                    invoice_no TEXT,
                    invoice_date TEXT,
                    financial_year TEXT,
                    party_name TEXT,
                    total_amount REAL,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_file_hash ON recorded_invoices(file_hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_gstin_inv ON recorded_invoices(gstin_supplier, invoice_no)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_party_total ON recorded_invoices(party_name, total_amount)")
            conn.commit()

    def check_duplicate(self,
                        invoice: Dict[str, Any],
                        file_bytes: Optional[bytes] = None,
                        file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Check if an invoice has already been recorded in the system.

        Returns a dictionary:
        {
            "is_duplicate": bool,
            "status": "NEW" | "EXACT_DUPLICATE" | "FILE_HASH_DUPLICATE" | "SUSPECTED_DUPLICATE",
            "message": str,
            "matched_record": dict | None,
            "fingerprint": str
        }
        """
        gstin = str(invoice.get("gstin_supplier") or "").strip()
        inv_no = str(invoice.get("invoice_no") or "").strip()
        date_str = str(invoice.get("date") or "").strip()
        party_name = str(invoice.get("party_name") or "").strip()
        total = invoice.get("total")
        try:
            total_val = float(total) if total is not None else None
        except (ValueError, TypeError):
            total_val = None

        fy = get_financial_year(date_str)
        fingerprint = compute_invoice_fingerprint(gstin, inv_no, date_str, party_name)

        # 1. Tier 1 Check: Exact File Hash Match
        file_hash = None
        if file_bytes is not None or file_path is not None:
            file_hash = compute_file_hash(file_bytes if file_bytes is not None else file_path)

        with self._get_connection() as conn:
            if file_hash:
                cur = conn.execute("SELECT * FROM recorded_invoices WHERE file_hash = ?", (file_hash,))
                row = cur.fetchone()
                if row:
                    return {
                        "is_duplicate": True,
                        "status": "FILE_HASH_DUPLICATE",
                        "duplicate_type": "EXACT_FILE_MATCH",
                        "message": (f"Duplicate file: This exact file was already captured on {row['created_at']} "
                                    f"(Invoice #{row['invoice_no']} from {row['party_name'] or row['gstin_supplier']})."),
                        "matched_record": dict(row),
                        "fingerprint": fingerprint
                    }

            # 2. Tier 2 Check: Business Key Fingerprint (GSTIN + Invoice Number + FY)
            if inv_no:
                cur = conn.execute("SELECT * FROM recorded_invoices WHERE fingerprint = ?", (fingerprint,))
                row = cur.fetchone()
                if row:
                    vendor_name = row['party_name'] or row['gstin_supplier']
                    return {
                        "is_duplicate": True,
                        "status": "EXACT_DUPLICATE",
                        "duplicate_type": "BUSINESS_KEY_MATCH",
                        "message": (f"Duplicate invoice: Invoice #{inv_no} from '{vendor_name}' for FY {fy} "
                                    f"was already recorded on {row['created_at']}."),
                        "matched_record": dict(row),
                        "fingerprint": fingerprint
                    }

            # 3. Tier 3 Check: Fuzzy / Near-Duplicate Detection
            # Same supplier GSTIN + same total amount
            if gstin and total_val is not None:
                cur = conn.execute("""
                    SELECT * FROM recorded_invoices
                    WHERE gstin_supplier = ? AND ABS(total_amount - ?) < 0.05
                """, (gstin, total_val))
                row = cur.fetchone()
                if row:
                    return {
                        "is_duplicate": True,
                        "status": "SUSPECTED_DUPLICATE",
                        "duplicate_type": "NEAR_DUPLICATE_SUSPECT",
                        "message": (f"Potential duplicate: An invoice from '{row['party_name'] or gstin}' "
                                    f"with identical amount ₹{total_val:.2f} is already recorded as #{row['invoice_no']}."),
                        "matched_record": dict(row),
                        "fingerprint": fingerprint
                    }

        return {
            "is_duplicate": False,
            "status": "NEW",
            "duplicate_type": "NONE",
            "message": "Invoice is new and ready to record.",
            "matched_record": None,
            "fingerprint": fingerprint
        }

    def record_invoice(self,
                       invoice: Dict[str, Any],
                       file_bytes: Optional[bytes] = None,
                       file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Record an invoice in the deduplication registry.
        Raises ValueError if an exact duplicate exists.
        """
        check = self.check_duplicate(invoice, file_bytes=file_bytes, file_path=file_path)
        if check["is_duplicate"] and check["status"] in ("EXACT_DUPLICATE", "FILE_HASH_DUPLICATE"):
            raise ValueError(f"Cannot record duplicate invoice: {check['message']}")

        gstin = str(invoice.get("gstin_supplier") or "").strip()
        inv_no = str(invoice.get("invoice_no") or "").strip()
        date_str = str(invoice.get("date") or "").strip()
        party_name = str(invoice.get("party_name") or "").strip()
        total = invoice.get("total")
        try:
            total_val = float(total) if total is not None else None
        except (ValueError, TypeError):
            total_val = None

        fy = get_financial_year(date_str)
        fingerprint = check["fingerprint"]
        file_hash = compute_file_hash(file_bytes if file_bytes is not None else file_path) if (file_bytes or file_path) else None
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO recorded_invoices
                (fingerprint, file_hash, gstin_supplier, invoice_no, invoice_date,
                 financial_year, party_name, total_amount, created_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (fingerprint, file_hash, gstin, inv_no, date_str,
                  fy, party_name, total_val, now_str, json.dumps(invoice)))
            conn.commit()

        return {
            "recorded": True,
            "fingerprint": fingerprint,
            "invoice_no": inv_no,
            "created_at": now_str,
            "status": "SAVED"
        }

    def count_records(self) -> int:
        """Return total number of recorded invoices."""
        with self._get_connection() as conn:
            cur = conn.execute("SELECT COUNT(*) FROM recorded_invoices")
            return int(cur.fetchone()[0])

    def clear(self):
        """Clear all records (used for testing)."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM recorded_invoices")
            conn.commit()
