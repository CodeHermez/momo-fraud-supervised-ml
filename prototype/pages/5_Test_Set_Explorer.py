"""Walk the confusion matrix and open any transaction inside it."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import evidence, schema, scorer, ui  # noqa: E402

st.set_page_config(page_title="Test set explorer", layout="wide")
st.title("Where the model is right, and where it is wrong")

if not evidence.demo_sample_available():
    st.error(
        "The bundled test-set sample is missing. Build it with:\n\n"
        "```\npython prototype/scripts/build_demo_sample.py\n```\n\n"
        "It needs the raw PaySim CSV in `data/raw/`."
    )
    st.stop()

if not scorer.artifacts_present():
    st.error("No model bundle in `artifacts/`.")
    st.stop()

fraud_scorer = scorer.get_scorer()
thresholds = scorer.get_thresholds()
card = scorer.get_model_card()
shipped_threshold = thresholds["decision_threshold"]

sample = evidence.load_demo_sample_scored()

st.markdown(
    f"A **{len(sample):,}-transaction** slice of the held-out split: every one of "
    f"its **{int(sample['isFraud'].sum()):,}** frauds, plus a random sample of "
    "legitimate traffic."
)

threshold = st.slider(
    "Alert threshold",
    min_value=0.0001, max_value=0.05,
    value=float(shipped_threshold),
    step=0.0001, format="%.4f",
    help="Drag to see transactions move between quadrants.",
)
if abs(threshold - shipped_threshold) < 1e-9:
    st.caption("This is the shipped threshold.")

flagged = sample["probability"] >= threshold
counts = ui.confusion_counts(sample["isFraud"], flagged)

st.divider()

matrix_column, note_column = st.columns([2, 3], gap="large")
with matrix_column:
    st.markdown("##### Confusion matrix")
    ui.render_confusion(counts)
with note_column:
    st.markdown("##### Reading this honestly")
    st.markdown(
        f"""
        The **{counts['FP']:,} false positives** are the price the severity ratio
        deliberately pays, not a defect.

        The **{counts['FN']:,} missed frauds** are the interesting column: mostly
        modest cash-outs into accounts that already held a balance transactions
        that look like ordinary withdrawals.

        **This sample is enriched**, so precision here reads far better than in
        production. The *Threshold and risk* page has the honest figure, over the
        whole split.
        """
    )

with st.expander("Why these scores differ slightly from the recorded experiment"):
    drift = (sample["probability"] - sample["experiment_score"]).abs()
    agreement = (
        (sample["probability"] >= shipped_threshold)
        == (sample["experiment_score"] >= shipped_threshold)
    ).mean()
    st.markdown(
        f"""
        The sample carries two scores per row.

        - `experiment_score` came from the **train-only** model, which every
          metric and figure in the study was computed from.
        - `probability` comes from **`artifacts/`**, a refit on train *and*
          validation ({card['data']['trained_on_rows']:,} rows) with the
          hyperparameters and threshold frozen from the experiment.

        Median absolute difference **{np.median(drift):.2e}**; they agree on the
        flag decision for **{agreement:.2%}** of these rows. This page uses the
        shipped model throughout.
        """
    )

st.divider()

QUADRANTS = {
    "True positives": (sample["isFraud"] == 1) & flagged,
    "False positives": (sample["isFraud"] == 0) & flagged,
    "False negatives": (sample["isFraud"] == 1) & ~flagged,
    "True negatives": (sample["isFraud"] == 0) & ~flagged,
}

BLURBS = {
    "True positives": "Fraud the model caught.",
    "False positives": "Legitimate transactions the model stopped. The cost of catching the rest.",
    "False negatives": "Fraud that got through. The expensive column.",
    "True negatives": "Ordinary traffic, correctly left alone.",
}

DISPLAY = ["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
           "oldbalanceDest", "newbalanceDest", "probability", "risk_score", "band"]

tabs = st.tabs([f"{name} ({int(QUADRANTS[name].sum()):,})" for name in QUADRANTS])

for tab, (name, mask) in zip(tabs, QUADRANTS.items()):
    with tab:
        st.caption(BLURBS[name])
        subset = sample.loc[mask, DISPLAY]

        if subset.empty:
            st.info("Nothing in this quadrant at the current threshold.")
            continue

        st.markdown(f"##### Composition of these {len(subset):,} transactions")
        left, right = st.columns([1, 1])
        with left:
            st.dataframe(
                subset["type"].value_counts().rename("transactions").to_frame(),
                width="stretch",
            )
        with right:
            st.dataframe(
                subset["amount"].describe().to_frame("amount").style.format("{:,.2f}"),
                width="stretch",
            )

        st.markdown("##### Pick one to inspect")
        ascending = name in {"False negatives", "True negatives"}
        ordered = subset.sort_values("probability", ascending=ascending)
        shown = ordered.head(300)

        event = st.dataframe(
            shown,
            width="stretch",
            on_select="rerun",
            selection_mode="single-row",
            key=f"table_{name}",
            column_config={
                "probability": st.column_config.NumberColumn("Probability", format="%.6f"),
                "risk_score": st.column_config.ProgressColumn(
                    "Risk", min_value=0.0, max_value=100.0, format="%.2f",
                ),
                "amount": st.column_config.NumberColumn(format="%.2f"),
            },
        )
        if len(ordered) > 300:
            st.caption(
                f"Showing 300 of {len(ordered):,}, "
                + ("lowest scores first." if ascending else "highest scores first.")
            )

        rows = event.selection.rows if event and event.selection else []
        if not rows:
            st.info("Select a row above to score it live and see the explanation.")
            continue

        record = shown.iloc[rows[0]]
        transaction = {field: record[field] for field in schema.REQUIRED_FIELDS}
        transaction["step"] = int(transaction["step"])
        transaction["type"] = str(transaction["type"])

        result = fraud_scorer.score(transaction, explain=True)

        st.divider()
        verdict, factors = st.columns([2, 3], gap="large")
        with verdict:
            st.markdown("##### Verdict")
            ui.render_verdict(result, thresholds["severity_ratio_R"])
            truth = "fraud" if name in {"True positives", "False negatives"} else "legitimate"
            st.caption(f"Ground truth: this transaction was **{truth}**.")
        with factors:
            st.markdown("##### What drove it")
            ui.render_factors(result)
