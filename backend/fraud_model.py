
"""
fraud_model.py
Core ML logic: Z-score anomaly detection + Isolation Forest
sklearn is lazy-loaded inside functions to keep startup time fast.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
import logging

# ── NO sklearn import at top level ──
# Importing sklearn here causes 10+ second startup → Leapcell timeout
# Instead we import inside run_isolation_forest() on first call

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Z-Score  (numpy only — loads instantly)
# ──────────────────────────────────────────────

def compute_z_scores(amounts: np.ndarray) -> np.ndarray:
    """Return per-transaction Z-scores (absolute value)."""
    mean = np.mean(amounts)
    std  = np.std(amounts)
    if std == 0:
        return np.zeros_like(amounts, dtype=float)
    return np.abs((amounts - mean) / std)


# ──────────────────────────────────────────────
# Isolation Forest  (sklearn lazy-loaded)
# ──────────────────────────────────────────────

def run_isolation_forest(
    amounts: np.ndarray,
    contamination: float = 0.1,
    random_state: int = 42,
):
    """
    Fit Isolation Forest and return:
      - labels    : 1 = normal, -1 = anomaly
      - raw_scores: decision_function scores (higher = more normal)

    sklearn is imported HERE (not at module level) so the app
    starts in ~1 second instead of 10+ seconds.
    """
    from sklearn.ensemble import IsolationForest  # ← lazy import

    X = amounts.reshape(-1, 1)
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100,
    )
    model.fit(X)
    labels     = model.predict(X)
    raw_scores = model.decision_function(X)
    logger.debug("Isolation Forest fitted. Anomalies: %d", (labels == -1).sum())
    return labels, raw_scores


# ──────────────────────────────────────────────
# Risk Score (0–100)
# ──────────────────────────────────────────────

def compute_risk_scores(
    z_scores: np.ndarray,
    if_labels: np.ndarray,
    if_raw: np.ndarray,
    z_weight: float = 0.5,
    if_weight: float = 0.5,
) -> np.ndarray:
    """
    Combine Z-score component and Isolation Forest component
    into a single 0–100 risk score.
    """
    # Z-score component (0–1)
    z_clipped   = np.clip(z_scores, 0, 10)
    z_component = z_clipped / 10.0

    # IF component (0–1)  — lower raw_score = more anomalous
    raw_min = if_raw.min()
    raw_max = if_raw.max()
    if raw_max - raw_min == 0:
        if_component = np.where(if_labels == -1, 1.0, 0.0)
    else:
        if_component = 1.0 - (if_raw - raw_min) / (raw_max - raw_min)

    combined = z_weight * z_component + if_weight * if_component
    risk     = np.clip(combined * 100, 0, 100)
    return risk.round(2)


# ──────────────────────────────────────────────
# Risk Level Label
# ──────────────────────────────────────────────

def risk_level(score: float) -> str:
    if score <= 30:
        return "LOW"
    elif score <= 70:
        return "MEDIUM"
    return "HIGH"


# ──────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────

def analyze_transactions(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Accept a list of dicts with keys: transaction_id, amount
    Return enriched list with fraud-detection fields.
    """
    df      = pd.DataFrame(records)
    amounts = df["amount"].to_numpy(dtype=float)

    z_scores            = compute_z_scores(amounts)
    if_labels, if_raw   = run_isolation_forest(amounts)
    risk_scores         = compute_risk_scores(z_scores, if_labels, if_raw)

    results = []
    for i, row in df.iterrows():
        score = float(risk_scores[i])
        results.append({
            "transaction_id": str(row["transaction_id"]),
            "amount":         round(float(row["amount"]), 2),
            "z_score":        round(float(z_scores[i]), 4),
            "anomaly_flag":   bool(if_labels[i] == -1),
            "risk_score":     score,
            "risk_level":     risk_level(score),
        })

    logger.info(
        "Analyzed %d transactions. Flagged: %d",
        len(results),
        sum(r["anomaly_flag"] for r in results),
    )
    return results
