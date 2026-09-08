"""Upload a CSV in the PaySim schema and score every row."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import schema, scorer, ui  # noqa: E402

st.set_page_config(page_title="Batch upload", layout="wide")
st.title("Score a file of transactions")

if not scorer.artifacts_present():
    st.error("No model bundle in `artifacts/`.")
    st.stop()

fraud_scorer = scorer.get_scorer()
thresholds = scorer.get_thresholds()

#: Above this, scoring is still fine but the browser struggles to render the
#: table. A demo should not be lost to a spinning tab.
LARGE_UPLOAD = 200_000

TEMPLATE = Path(__file__).resolve().parents[1] / "assets" / "sample_transactions.csv"

intro, download = st.columns([3, 1])
with intro:
    st.markdown(
        "The file needs these seven columns: "
        + ", ".join(f"`{c}`" for c in schema.REQUIRED_FIELDS)
        + ". Extra columns are ignored, so a raw PaySim export works unchanged."
    )
with download:
    if TEMPLATE.exists():
        st.download_button(
            "Download a sample CSV",
            TEMPLATE.read_bytes(),
            file_name="sample_transactions.csv",
            mime="text/csv",
            width="stretch",
        )

uploaded = st.file_uploader("Choose a CSV", type=["csv"])

if uploaded is None:
    st.info("Waiting for a file. The sample above is a valid one to start with.")
    st.stop()

try:
    raw = pd.read_csv(uploaded)
except Exception as error:  # noqa: BLE001 -- surface any parse failure as UI
    st.error(f"Could not read that file as CSV: {error}")
    st.stop()

st.success(f"Read **{len(raw):,}** rows and **{len(raw.columns)}** columns.")

if len(raw) > LARGE_UPLOAD:
    st.warning(
        f"That is a large file. Scoring all {len(raw):,} rows may take a while "
        f"and the table below will be truncated for display."
    )

extra = schema.describe_extra_columns(raw)
if extra:
    with st.expander(f"{len(extra)} column(s) present but not used as input"):
        st.dataframe(
            pd.DataFrame(extra, columns=["column", "why it is ignored"]),
            hide_index=True, width="stretch",
        )

clean, problems = schema.validate(raw)

if problems:
    st.warning(f"{len(problems)} problem(s) found. Affected rows were skipped.")
    with st.expander("See every problem", expanded=len(clean) == 0):
        st.dataframe(
            pd.DataFrame([p.__dict__ for p in problems]),
            hide_index=True, width="stretch",
        )

if clean.empty:
    st.error("No valid rows to score.")
    st.stop()

with st.spinner(f"Scoring {len(clean):,} transactions ..."):
    scored = fraud_scorer.score_batch(clean)

combined = clean.join(scored)

st.divider()
st.markdown("##### Summary")

a, b, c, d = st.columns(4)
flagged = int(combined["flagged"].sum())
a.metric("Rows scored", f"{len(combined):,}")
b.metric("Flagged for review", f"{flagged:,}")
c.metric("Alert rate", f"{flagged / len(combined):.2%}")
d.metric("Highest risk", f"{combined['risk_score'].max():.2f} / 100")

chart, table = st.columns([1, 2])
with chart:
    st.caption("Transactions per risk band")
    st.bar_chart(ui.band_distribution(combined), color="#b3261e")
with table:
    st.caption("Highest risk first")
    st.dataframe(
        combined.sort_values("risk_score", ascending=False).head(500),
        width="stretch",
        column_config={
            "risk_score": st.column_config.ProgressColumn(
                "Risk", min_value=0.0, max_value=100.0, format="%.2f",
            ),
            "probability": st.column_config.NumberColumn("Probability", format="%.6f"),
        },
    )
    if len(combined) > 500:
        st.caption(f"Showing the top 500 of {len(combined):,}. The download has everything.")

st.download_button(
    "Download all scored rows (CSV)",
    combined.to_csv(index=False).encode("utf-8"),
    file_name="scored_transactions.csv",
    mime="text/csv",
    type="primary",
)

# If the upload carried labels, grade the model against them. This is what
# makes it possible to point the demo at a real slice of PaySim.
if "isFraud" in raw.columns:
    st.divider()
    st.markdown("##### The file carried labels, so here is the scorecard")

    truth = pd.to_numeric(raw.loc[combined.index, "isFraud"], errors="coerce")
    usable = truth.notna()

    if not usable.any():
        st.info("`isFraud` was present but held no usable values.")
    else:
        counts = ui.confusion_counts(truth[usable], combined.loc[usable, "flagged"])
        left, right = st.columns([2, 3])
        with left:
            ui.render_confusion(counts)
        with right:
            st.caption(
                "Precision is low by design — the *Threshold and risk* page "
                "shows the trade."
            )
            st.markdown(
                f"- **{counts['TP']:,}** frauds caught\n"
                f"- **{counts['FN']:,}** frauds missed\n"
                f"- **{counts['FP']:,}** legitimate transactions flagged\n"
                f"- **{counts['TN']:,}** legitimate transactions passed through"
            )
