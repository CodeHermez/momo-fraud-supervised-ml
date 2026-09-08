"""Compose several transactions by hand and score them together."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import schema, scorer, ui  # noqa: E402
from momo_fraud import constants as C  # noqa: E402

st.set_page_config(page_title="Build a table", layout="wide")
st.title("Build a table of transactions")
st.markdown(
    "Edit the grid directly **+** adds a row, backspace deletes a selected "
    "one then score them together."
)

if not scorer.artifacts_present():
    st.error("No model bundle in `artifacts/`.")
    st.stop()

fraud_scorer = scorer.get_scorer()
thresholds = scorer.get_thresholds()

SEED_ROWS = pd.DataFrame([p.transaction for p in schema.SAMPLE_PROFILES[:3]])

if "grid" not in st.session_state:
    st.session_state.grid = SEED_ROWS.copy()

left, right = st.columns([1, 5])
if left.button("Reset grid"):
    st.session_state.grid = SEED_ROWS.copy()
    st.rerun()
if right.button("Add a blank row"):
    st.session_state.grid = pd.concat(
        [st.session_state.grid, pd.DataFrame([schema.blank_row()])],
        ignore_index=True,
    )
    st.rerun()

edited = st.data_editor(
    st.session_state.grid,
    num_rows="dynamic",
    width="stretch",
    key="grid_editor",
    column_config={
        "step": st.column_config.NumberColumn(
            "Step", min_value=1, max_value=C.MAX_STEP, step=1,
            help=schema.FIELD_HELP["step"],
        ),
        "type": st.column_config.SelectboxColumn(
            "Type", options=C.TRANSACTION_TYPES, required=True,
            help=schema.FIELD_HELP["type"],
        ),
        "amount": st.column_config.NumberColumn(
            "Amount", min_value=0.0, format="%.2f",
        ),
        "oldbalanceOrg": st.column_config.NumberColumn(
            "Sender before", min_value=0.0, format="%.2f",
            help="Collected but unused by this model.",
        ),
        "newbalanceOrig": st.column_config.NumberColumn(
            "Sender after", min_value=0.0, format="%.2f",
            help="Collected but unused by this model.",
        ),
        "oldbalanceDest": st.column_config.NumberColumn(
            "Recipient before", min_value=0.0, format="%.2f",
        ),
        "newbalanceDest": st.column_config.NumberColumn(
            "Recipient after", min_value=0.0, format="%.2f",
        ),
    },
)

if st.button("Score all rows", type="primary"):
    clean, problems = schema.validate(edited)

    if problems:
        st.warning(f"{len(problems)} problem(s) found. Those rows were skipped.")
        st.dataframe(
            pd.DataFrame([p.__dict__ for p in problems]),
            hide_index=True, width="stretch",
        )

    if clean.empty:
        st.error("Nothing left to score.")
    else:
        scored = fraud_scorer.score_batch(clean)
        combined = clean.join(scored)

        st.divider()
        st.markdown("##### Results")

        a, b, c = st.columns(3)
        a.metric("Rows scored", f"{len(combined):,}")
        a_flagged = int(combined["flagged"].sum())
        b.metric("Flagged", f"{a_flagged:,}")
        c.metric(
            "Highest risk",
            f"{combined['risk_score'].max():.2f} / 100",
        )

        st.dataframe(
            combined.sort_values("risk_score", ascending=False),
            width="stretch",
            column_config={
                "risk_score": st.column_config.ProgressColumn(
                    "Risk", min_value=0.0, max_value=100.0, format="%.2f",
                ),
                "probability": st.column_config.NumberColumn(
                    "Probability", format="%.6f",
                ),
            },
        )

        st.bar_chart(ui.band_distribution(combined), color="#b3261e")

        st.download_button(
            "Download scored table (CSV)",
            combined.to_csv(index=False).encode("utf-8"),
            file_name="scored_transactions.csv",
            mime="text/csv",
        )
