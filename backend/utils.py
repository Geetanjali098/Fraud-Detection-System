"""
utils.py
Utility helpers: CSV parsing, input validation, response formatting.
"""

import io
import csv
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"transaction_id", "amount"}


# ──────────────────────────────────────────────
# CSV parsing
# ──────────────────────────────────────────────

def parse_csv_bytes(content: bytes) -> List[Dict[str, Any]]:
    """
    Parse raw CSV bytes into a list of record dicts.
    Raises ValueError with a human-readable message on failure.
    """
    try:
        text   = content.decode("utf-8")
        reader = csv.DictReader(io.StringIO(text))
        rows   = list(reader)
    except Exception as exc:
        raise ValueError(f"Could not read CSV file: {exc}") from exc

    if not rows:
        raise ValueError("CSV file is empty.")

    cols = set(rows[0].keys())
    missing = REQUIRED_COLUMNS - cols
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}. "
            f"Found: {cols}"
        )

    records: List[Dict[str, Any]] = []
    for i, row in enumerate(rows, start=2):          # row 1 = header
        tid = row.get("transaction_id", "").strip()
        if not tid:
            raise ValueError(f"Row {i}: 'transaction_id' is empty.")
        try:
            amount = float(row["amount"])
        except (ValueError, TypeError):
            raise ValueError(
                f"Row {i}: 'amount' must be a number, got '{row['amount']}'."
            )
        if amount < 0:
            raise ValueError(f"Row {i}: 'amount' cannot be negative ({amount}).")
        records.append({"transaction_id": tid, "amount": amount})

    logger.debug("Parsed %d records from CSV.", len(records))
    return records


# ──────────────────────────────────────────────
# Single-transaction validation
# ──────────────────────────────────────────────

def validate_single(transaction_id: str, amount: float) -> None:
    """Raise ValueError if single-record inputs are invalid."""
    if not transaction_id or not transaction_id.strip():
        raise ValueError("'transaction_id' must not be empty.")
    if amount < 0:
        raise ValueError(f"'amount' cannot be negative ({amount}).")


# ──────────────────────────────────────────────
# Response summary helper
# ──────────────────────────────────────────────

def summarise(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return high-level summary stats to include in API response."""
    total   = len(results)
    flagged = sum(r["anomaly_flag"] for r in results)
    by_level: Dict[str, int] = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    for r in results:
        by_level[r["risk_level"]] += 1
    avg_risk = round(sum(r["risk_score"] for r in results) / total, 2) if total else 0.0
    return {
        "total_transactions": total,
        "flagged_transactions": flagged,
        "flag_rate_pct": round(flagged / total * 100, 2) if total else 0.0,
        "avg_risk_score": avg_risk,
        "by_risk_level": by_level,
    }
