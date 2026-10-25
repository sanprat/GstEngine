"""
Unit tests for invoice_dedup.py — Deduplication Layer for GstEngine.
Tests all three deduplication tiers:
  1. Exact File Hash match (SHA-256)
  2. Business Key Fingerprint match (Supplier GSTIN + Invoice Number + FY)
  3. Near-Duplicate / Suspected Duplicate match (Supplier + Total Amount)
Also tests Indian Financial Year calculation and SQLite persistence.
"""

import os
import tempfile
import unittest
from invoice_dedup import (
    InvoiceDeduplicator,
    compute_file_hash,
    compute_invoice_fingerprint,
    get_financial_year,
    normalize_string,
)


class TestInvoiceDeduplication(unittest.TestCase):

    def setUp(self):
        # Use an isolated temporary SQLite database for each test
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_registry.db")
        self.dedup = InvoiceDeduplicator(db_path=self.db_path)

        # Sample invoice fixture
        self.sample_invoice = {
            "gstin_supplier": "27ABCTY1234E1ZL",
            "invoice_no": "INV/2026/001",
            "date": "24-10-2026",
            "party_name": "Sharma Electronics",
            "total": 33512.00,
        }
        self.sample_file_bytes = b"%PDF-1.4 test invoice dummy content 12345"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_normalize_string(self):
        self.assertEqual(normalize_string(" 27abcty1234e1zl "), "27ABCTY1234E1ZL")
        self.assertEqual(normalize_string("INV / 2026 - 001!"), "INV2026001")
        self.assertEqual(normalize_string(None), "")
        self.assertEqual(normalize_string(""), "")

    def test_get_financial_year(self):
        # Indian FY runs April 1 to March 31
        self.assertEqual(get_financial_year("2026-10-24"), "2026-27")
        self.assertEqual(get_financial_year("2026-04-01"), "2026-27")
        self.assertEqual(get_financial_year("2026-03-31"), "2025-26")
        self.assertEqual(get_financial_year("2026-01-15"), "2025-26")
        self.assertEqual(get_financial_year("15-02-2026"), "2025-26")
        self.assertEqual(get_financial_year("05/11/2025"), "2025-26")
        self.assertEqual(get_financial_year("invalid-date"), "UNKNOWN")
        self.assertEqual(get_financial_year(None), "UNKNOWN")

    def test_compute_file_hash(self):
        h1 = compute_file_hash(b"content A")
        h2 = compute_file_hash(b"content A")
        h3 = compute_file_hash(b"content B")
        self.assertEqual(h1, h2)
        self.assertNotEqual(h1, h3)
        self.assertEqual(len(h1), 64)

    def test_compute_invoice_fingerprint_invariance(self):
        fp1 = compute_invoice_fingerprint("27ABCTY1234E1ZL", "INV/2026/001", "2026-10-24")
        fp2 = compute_invoice_fingerprint(" 27abcty1234e1zl ", "inv-2026-001", "24-10-2026")
        self.assertEqual(fp1, fp2)

    def test_tier1_file_hash_duplicate(self):
        # Initially new
        check = self.dedup.check_duplicate(self.sample_invoice, file_bytes=self.sample_file_bytes)
        self.assertFalse(check["is_duplicate"])
        self.assertEqual(check["status"], "NEW")

        # Record invoice
        self.dedup.record_invoice(self.sample_invoice, file_bytes=self.sample_file_bytes)
        self.assertEqual(self.dedup.count_records(), 1)

        # Checking with same file bytes (even if metadata is altered or OCR failed)
        altered_invoice = {
            "gstin_supplier": "UNKNOWN",
            "invoice_no": "CORRUPTED",
            "date": "2026-10-24",
            "total": 9999.00
        }
        check2 = self.dedup.check_duplicate(altered_invoice, file_bytes=self.sample_file_bytes)
        self.assertTrue(check2["is_duplicate"])
        self.assertEqual(check2["status"], "FILE_HASH_DUPLICATE")
        self.assertEqual(check2["duplicate_type"], "EXACT_FILE_MATCH")
        self.assertIn("Duplicate file", check2["message"])

        # Recording exact duplicate file must raise ValueError
        with self.assertRaises(ValueError):
            self.dedup.record_invoice(altered_invoice, file_bytes=self.sample_file_bytes)

    def test_tier2_business_key_duplicate(self):
        # Record original
        self.dedup.record_invoice(self.sample_invoice, file_bytes=self.sample_file_bytes)

        # Different scan/photo file bytes of the SAME bill
        different_scan_bytes = b"%PDF-1.4 completely different scan bits or camera photo bytes"
        duplicate_bill = {
            "gstin_supplier": "27ABCTY1234E1ZL",
            "invoice_no": "INV/2026/001",
            "date": "2026-10-24",
            "party_name": "Sharma Electronics",
            "total": 33512.00,
        }

        check = self.dedup.check_duplicate(duplicate_bill, file_bytes=different_scan_bytes)
        self.assertTrue(check["is_duplicate"])
        self.assertEqual(check["status"], "EXACT_DUPLICATE")
        self.assertEqual(check["duplicate_type"], "BUSINESS_KEY_MATCH")
        self.assertIn("Duplicate invoice: Invoice #INV/2026/001", check["message"])

        # Recording must raise ValueError
        with self.assertRaises(ValueError):
            self.dedup.record_invoice(duplicate_bill, file_bytes=different_scan_bytes)

    def test_financial_year_distinction(self):
        # Section 31 CGST Act: Invoice numbers reset every Financial Year
        # FY 2025-26
        inv_fy25 = {
            "gstin_supplier": "27ABCTY1234E1ZL",
            "invoice_no": "001",
            "date": "2025-05-10",  # FY 2025-26
            "total": 500.0,
        }
        self.dedup.record_invoice(inv_fy25, file_bytes=b"file_2025")

        # FY 2026-27 with identical invoice_no from same vendor is statutory valid
        inv_fy26 = {
            "gstin_supplier": "27ABCTY1234E1ZL",
            "invoice_no": "001",
            "date": "2026-05-10",  # FY 2026-27
            "total": 750.0,
        }
        check = self.dedup.check_duplicate(inv_fy26, file_bytes=b"file_2026")
        self.assertFalse(check["is_duplicate"])
        self.assertEqual(check["status"], "NEW")

        # Should record without error
        res = self.dedup.record_invoice(inv_fy26, file_bytes=b"file_2026")
        self.assertTrue(res["recorded"])
        self.assertEqual(self.dedup.count_records(), 2)

    def test_different_suppliers_same_invoice_no(self):
        # Two different suppliers can legitimately have invoice "INV-001"
        inv_vendor_a = {
            "gstin_supplier": "27ABCTY1234E1ZL",
            "invoice_no": "INV-001",
            "date": "2026-06-15",
            "total": 1000.0,
        }
        inv_vendor_b = {
            "gstin_supplier": "29XYZAB5678K1ZY",
            "invoice_no": "INV-001",
            "date": "2026-06-15",
            "total": 1000.0,
        }
        self.dedup.record_invoice(inv_vendor_a, file_bytes=b"a")
        check_b = self.dedup.check_duplicate(inv_vendor_b, file_bytes=b"b")
        self.assertFalse(check_b["is_duplicate"])
        self.assertEqual(check_b["status"], "NEW")

    def test_tier3_near_duplicate_detection(self):
        # Same supplier + exact same total amount, but slight OCR typo in invoice number
        # e.g. OCR read 'INV-001' as '1NV-001'
        self.dedup.record_invoice(self.sample_invoice, file_bytes=self.sample_file_bytes)

        suspected_invoice = {
            "gstin_supplier": "27ABCTY1234E1ZL",
            "invoice_no": "1NV/2026/001",  # Slight OCR misread
            "date": "24-10-2026",
            "party_name": "Sharma Electronics",
            "total": 33512.00,  # Exact amount match
        }
        different_photo_bytes = b"different photo bytes"

        check = self.dedup.check_duplicate(suspected_invoice, file_bytes=different_photo_bytes)
        self.assertTrue(check["is_duplicate"])
        self.assertEqual(check["status"], "SUSPECTED_DUPLICATE")
        self.assertEqual(check["duplicate_type"], "NEAR_DUPLICATE_SUSPECT")
        self.assertIn("Potential duplicate", check["message"])

    def test_sqlite_persistence_across_instances(self):
        # Record using one instance
        self.dedup.record_invoice(self.sample_invoice, file_bytes=self.sample_file_bytes)
        self.assertEqual(self.dedup.count_records(), 1)

        # Open database with a second fresh instance
        dedup2 = InvoiceDeduplicator(db_path=self.db_path)
        self.assertEqual(dedup2.count_records(), 1)
        check = dedup2.check_duplicate(self.sample_invoice)
        self.assertTrue(check["is_duplicate"])
        self.assertEqual(check["status"], "EXACT_DUPLICATE")


if __name__ == "__main__":
    unittest.main()
