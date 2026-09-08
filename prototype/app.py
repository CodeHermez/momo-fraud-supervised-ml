"""Entry point for the fraud-detection prototype.

Run from the repo root:

    streamlit run prototype/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import evidence, scorer  # noqa: E402

st.set_page_config(
    page_title="Mobile Money Fraud Detection",
    layout="wide",
)

st.title("Mobile money fraud detection")
st.markdown(
    "A risk-based, cost-sensitive fraud classifier for mobile money transactions, "
    "trained on the PaySim simulation of 6.36 million transfers."
)

if not scorer.artifacts_present():
    st.error(
        "No model bundle found in `artifacts/`. Run "
        "`notebooks/08_model_export.ipynb` to produce it, then reload this page."
    )
    st.stop()

card = scorer.get_model_card()
thresholds = scorer.get_thresholds()
performance = card["held_out_performance"]

st.subheader("The shipped model")

a, b, c, d = st.columns(4)
a.metric("PR-AUC", f"{performance['pr_auc']:.4f}",
         help="Precision-recall area under curve on the held-out test split. "
              "The right measure at 0.129% prevalence.")
b.metric("Recall", f"{performance['recall']:.2%}",
         help="Share of all fraud the model catches at the frozen threshold.")
c.metric("Alert rate", f"{performance['alert_rate']:.2%}",
         help="Share of all traffic that gets flagged for review.")
d.metric("NER", f"{performance['ner']:.4f}",
         help="Normalised expected risk. Flagging nothing scores 1.0, so this "
              "model removes about 87% of the risk of doing nothing.")

st.caption(
    f"{card['learner'].upper()} on the `{card['data']['feature_set']}` feature set, "
    f"trained on {card['data']['trained_on_rows']:,} transactions at "
    f"{card['data']['prevalence']:.3%} fraud prevalence. "
    f"Decision threshold {thresholds['decision_threshold']:.6f}, "
    f"severity ratio R = {thresholds['severity_ratio_R']:,.1f}."
)

st.divider()

st.subheader("What you can do here")

left, right = st.columns(2)

with left:
    st.markdown(
        """
        **Score transactions**

        - **Score a transaction** — type one in, or load a real one from the
          held-out split.
        - **Build a table** — compose several rows in a grid and score them
          together.
        - **Batch upload** — score a CSV, with a confusion matrix if it carries
          labels.
        """
    )

with right:
    st.markdown(
        """
        **See the evidence**

        - **Threshold and risk** — move the alert line across 954,394 held-out
          transactions.
        - **Test set explorer** — walk the confusion matrix and open any
          transaction in it.
        - **Experiment results** — the 31 figures and 41 tables behind the model.
        - **Model card** — what it was trained on, and what it cannot do.
        """
    )

st.divider()

with st.expander("Read this before drawing conclusions", expanded=False):
    st.warning(
        "**PaySim is a simulator, not real mobile money traffic.** A three-clause "
        "rule recovers almost all of its fraud, which is why this model excludes "
        "the origin-balance columns. The *Experiment results* page has the evidence."
    )
    for limitation in card["limitations"]:
        st.markdown(f"- {limitation}")

if not evidence.demo_sample_available():
    st.info(
        "The bundled test-set sample is missing, so the *Test set explorer* will "
        "be unavailable. Build it with "
        "`python prototype/scripts/build_demo_sample.py` (needs the raw PaySim CSV)."
    )
