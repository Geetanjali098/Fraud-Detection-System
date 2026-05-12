"""
app.py
Streamlit frontend – Fraud Detection System
"""

import io
import time
import logging

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

API_BASE = "http://localhost:8000"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RISK_COLORS = {"LOW": "#22c55e", "MEDIUM": "#f59e0b", "HIGH": "#ef4444"}
RISK_BG     = {"LOW": "#dcfce7", "MEDIUM": "#fef9c3", "HIGH": "#fee2e2"}

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Top header bar */
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    border: 1px solid #334155;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "";
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 70% 50%, rgba(99,102,241,0.15) 0%, transparent 60%);
}
.hero h1 { color: #f8fafc; font-size: 2rem; font-weight: 700; margin: 0; }
.hero p  { color: #94a3b8; margin: 0.25rem 0 0; font-size: 0.95rem; }

/* Metric cards */
.metric-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    text-align: center;
}
.metric-card .val { font-size: 2rem; font-weight: 700; color: #f1f5f9; }
.metric-card .lbl { font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.06em; }

/* Risk badge */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.05em;
}
.badge-LOW    { background:#dcfce7; color:#166534; }
.badge-MEDIUM { background:#fef9c3; color:#854d0e; }
.badge-HIGH   { background:#fee2e2; color:#991b1b; }

/* Section title */
.section-title {
    font-size: 1rem; font-weight: 600; color: #e2e8f0;
    border-left: 3px solid #6366f1; padding-left: 0.6rem;
    margin-bottom: 0.75rem;
}

/* Scrollable table */
.table-scroll { overflow-x: auto; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def call_api_json(records: list[dict]) -> dict:
    resp = requests.post(f"{API_BASE}/predict", json={"transactions": records}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def call_api_csv(csv_bytes: bytes, filename: str) -> dict:
    resp = requests.post(
        f"{API_BASE}/predict/csv",
        files={"file": (filename, csv_bytes, "text/csv")},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def results_to_df(results: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    col_order = ["transaction_id", "amount", "z_score", "risk_score", "risk_level", "anomaly_flag"]
    return df[col_order]


def badge_html(level: str) -> str:
    return f'<span class="badge badge-{level}">{level}</span>'


def color_row(row):
    """Apply background tint to flagged rows."""
    if row["anomaly_flag"]:
        return [f"background-color: #fff1f2"] * len(row)
    return [""] * len(row)


def render_styled_table(df: pd.DataFrame):
    """Render DataFrame with risk-level badges via HTML."""
    display = df.copy()
    display["risk_level"] = display["risk_level"].apply(badge_html)
    display["anomaly_flag"] = display["anomaly_flag"].apply(
        lambda x: "🚨 Yes" if x else "✅ No"
    )
    display = display.rename(columns={
        "transaction_id": "Transaction ID",
        "amount": "Amount ($)",
        "z_score": "Z-Score",
        "risk_score": "Risk Score",
        "risk_level": "Risk Level",
        "anomaly_flag": "Flagged",
    })
    st.markdown(display.to_html(escape=False, index=False), unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Charts
# ──────────────────────────────────────────────

def chart_risk_distribution(df: pd.DataFrame):
    counts = df["risk_level"].value_counts().reindex(["LOW", "MEDIUM", "HIGH"], fill_value=0)
    fig = go.Figure(go.Bar(
        x=counts.index,
        y=counts.values,
        marker_color=[RISK_COLORS[k] for k in counts.index],
        text=counts.values,
        textposition="outside",
    ))
    fig.update_layout(
        title="Anomaly Distribution by Risk Level",
        xaxis_title="Risk Level",
        yaxis_title="Count",
        plot_bgcolor="#0f172a",
        paper_bgcolor="#1e293b",
        font=dict(color="#e2e8f0"),
        margin=dict(t=50, b=30, l=40, r=20),
        height=320,
    )
    return fig


def chart_amount_vs_risk(df: pd.DataFrame):
    color_map = {"LOW": "#22c55e", "MEDIUM": "#f59e0b", "HIGH": "#ef4444"}
    fig = px.scatter(
        df,
        x="amount",
        y="risk_score",
        color="risk_level",
        color_discrete_map=color_map,
        hover_data=["transaction_id", "z_score"],
        title="Amount vs Risk Score",
        labels={"amount": "Transaction Amount ($)", "risk_score": "Risk Score (0–100)"},
    )
    fig.update_traces(marker=dict(size=9, opacity=0.85, line=dict(width=1, color="white")))
    fig.update_layout(
        plot_bgcolor="#0f172a",
        paper_bgcolor="#1e293b",
        font=dict(color="#e2e8f0"),
        margin=dict(t=50, b=30, l=40, r=20),
        height=320,
        legend_title_text="Risk Level",
    )
    return fig


def chart_z_score_histogram(df: pd.DataFrame):
    fig = px.histogram(
        df, x="z_score", nbins=20,
        color="risk_level",
        color_discrete_map={"LOW": "#22c55e", "MEDIUM": "#f59e0b", "HIGH": "#ef4444"},
        title="Z-Score Distribution",
        labels={"z_score": "Z-Score", "count": "Transactions"},
    )
    fig.update_layout(
        plot_bgcolor="#0f172a",
        paper_bgcolor="#1e293b",
        font=dict(color="#e2e8f0"),
        margin=dict(t=50, b=30, l=40, r=20),
        height=320,
        barmode="stack",
    )
    return fig


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    api_url = st.text_input("API Base URL", value=API_BASE)
    API_BASE = api_url.rstrip("/")

    st.markdown("---")
    st.markdown("**Risk Level Thresholds**")
    st.markdown("🟢 **LOW**: 0 – 30")
    st.markdown("🟡 **MEDIUM**: 31 – 70")
    st.markdown("🔴 **HIGH**: 71 – 100")

    st.markdown("---")
    st.markdown("**About**")
    st.caption(
        "This tool uses Z-score analysis and "
        "Isolation Forest ML to detect anomalous transactions."
    )

    # Health check
    if st.button("🔌 Check API Health"):
        try:
            r = requests.get(f"{API_BASE}/health", timeout=5)
            if r.status_code == 200:
                st.success("API is online ✅")
            else:
                st.error(f"API returned {r.status_code}")
        except Exception as e:
            st.error(f"Cannot reach API: {e}")


# ──────────────────────────────────────────────
# Hero header
# ──────────────────────────────────────────────

st.markdown("""
<div class="hero">
  <h1>🛡️ Fraud Detection System</h1>
  <p>AI-powered transaction anomaly detection using Z-score analysis &amp; Isolation Forest ML</p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Input section
# ──────────────────────────────────────────────

tab_csv, tab_manual = st.tabs(["📂 Upload CSV", "✏️ Manual Entry"])

results_data = None

# ── Tab 1: CSV Upload ──────────────────────────
with tab_csv:
    st.markdown('<div class="section-title">Upload Transaction File</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Choose a CSV file with `transaction_id` and `amount` columns",
        type=["csv"],
        help="CSV must contain columns: transaction_id, amount",
    )

    if uploaded:
        try:
            preview_df = pd.read_csv(uploaded)
            st.caption(f"Preview — {len(preview_df)} rows")
            st.dataframe(preview_df.head(5), use_container_width=True)
            uploaded.seek(0)
        except Exception as e:
            st.warning(f"Could not preview: {e}")

    analyze_csv = st.button("🔍 Analyze CSV", type="primary", key="btn_csv",
                             disabled=uploaded is None)
    if analyze_csv and uploaded:
        with st.spinner("Running fraud detection …"):
            try:
                uploaded.seek(0)
                raw = uploaded.read()
                resp = call_api_csv(raw, uploaded.name)
                results_data = resp
            except requests.exceptions.ConnectionError:
                st.error("❌ Could not connect to the API. Make sure the backend is running.")
            except requests.exceptions.HTTPError as e:
                st.error(f"❌ API Error: {e.response.json().get('detail', str(e))}")
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")


# ── Tab 2: Manual Entry ────────────────────────
with tab_manual:
    st.markdown('<div class="section-title">Enter Transactions Manually</div>', unsafe_allow_html=True)

    if "manual_rows" not in st.session_state:
        st.session_state.manual_rows = [
            {"transaction_id": "TXN001", "amount": 150.0},
            {"transaction_id": "TXN002", "amount": 9500.0},
            {"transaction_id": "TXN003", "amount": 200.0},
        ]

    cols_add = st.columns([1, 1, 2])
    with cols_add[0]:
        new_id = st.text_input("Transaction ID", placeholder="TXN_NEW")
    with cols_add[1]:
        new_amount = st.number_input("Amount ($)", min_value=0.01, value=100.0, step=0.01)
    with cols_add[2]:
        st.write("")
        st.write("")
        if st.button("➕ Add Row"):
            if new_id.strip():
                st.session_state.manual_rows.append(
                    {"transaction_id": new_id.strip(), "amount": new_amount}
                )
            else:
                st.warning("Please enter a Transaction ID.")

    if st.session_state.manual_rows:
        manual_df = pd.DataFrame(st.session_state.manual_rows)
        st.dataframe(manual_df, use_container_width=True)

        col_btn1, col_btn2, _ = st.columns([1, 1, 4])
        analyze_manual = col_btn1.button("🔍 Analyze", type="primary", key="btn_manual")
        if col_btn2.button("🗑️ Clear All", key="btn_clear"):
            st.session_state.manual_rows = []
            st.rerun()

        if analyze_manual:
            with st.spinner("Running fraud detection …"):
                try:
                    records = st.session_state.manual_rows
                    resp = call_api_json(records)
                    results_data = resp
                except requests.exceptions.ConnectionError:
                    st.error("❌ Could not connect to the API. Make sure the backend is running.")
                except requests.exceptions.HTTPError as e:
                    st.error(f"❌ API Error: {e.response.json().get('detail', str(e))}")
                except Exception as e:
                    st.error(f"❌ Unexpected error: {e}")


# ──────────────────────────────────────────────
# Results section
# ──────────────────────────────────────────────

if results_data:
    st.markdown("---")
    st.markdown("## 📊 Analysis Results")

    df     = results_to_df(results_data["results"])
    summ   = results_data["summary"]
    flagged_df = df[df["anomaly_flag"] == True]

    # ── Summary metrics ──────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    metrics = [
        (c1, summ["total_transactions"],    "Total Transactions"),
        (c2, summ["flagged_transactions"],  "Flagged"),
        (c3, f'{summ["flag_rate_pct"]}%',   "Flag Rate"),
        (c4, summ["avg_risk_score"],        "Avg Risk Score"),
        (c5, summ["by_risk_level"]["HIGH"], "High Risk"),
    ]
    for col, val, lbl in metrics:
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="val">{val}</div>'
                f'<div class="lbl">{lbl}</div></div>',
                unsafe_allow_html=True,
            )

    st.write("")

    # ── Charts row ────────────────────────────
    ch1, ch2, ch3 = st.columns(3)
    with ch1:
        st.plotly_chart(chart_risk_distribution(df), use_container_width=True)
    with ch2:
        st.plotly_chart(chart_amount_vs_risk(df), use_container_width=True)
    with ch3:
        st.plotly_chart(chart_z_score_histogram(df), use_container_width=True)

    # ── Full results table ────────────────────
    st.markdown('<div class="section-title">All Transactions</div>', unsafe_allow_html=True)
    render_styled_table(df)

    # ── Flagged only ──────────────────────────
    if not flagged_df.empty:
        st.write("")
        st.markdown(
            f'<div class="section-title">🚨 Flagged Transactions ({len(flagged_df)})</div>',
            unsafe_allow_html=True,
        )
        render_styled_table(flagged_df)

    # ── Download ──────────────────────────────
    st.write("")
    csv_out = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Results as CSV",
        data=csv_out,
        file_name="fraud_detection_results.csv",
        mime="text/csv",
    )
