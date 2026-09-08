"""The model card, rendered: what it is, how it scored, what it cannot do."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import scorer  # noqa: E402
from momo_fraud import constants as C  # noqa: E402

st.set_page_config(page_title="Model card", layout="wide")
st.title("Model card")

if not scorer.artifacts_present():
    st.error("No model bundle in `artifacts/`.")
    st.stop()

card = scorer.get_model_card()
thresholds = scorer.get_thresholds()
data = card["data"]
performance = card["held_out_performance"]

st.caption(f"Frozen {card['created_utc']}. Task: {card['task']}.")

# --- Limitations first ------------------------------------------------------
# Deliberately above the metrics. A card that leads with 0.985 ROC-AUC and
# buries the caveats is doing the opposite of what a card is for.

st.header("Read this first")
for limitation in card["limitations"]:
    st.markdown(f"- {limitation}")

st.divider()

st.header("What it is")
a, b, c = st.columns(3)
a.metric("Learner", card["learner"].upper())
b.metric("Feature set", data["feature_set"])
c.metric("Features used", len(thresholds["feature_names"]))

st.markdown(f"**Cost sensitivity:** {card['cost_sensitivity']}")

st.markdown("**Excluded features and why:**")
st.markdown(
    ", ".join(f"`{c}`" for c in data["features_excluded"])
    + " — these encode PaySim's fraud-generation rule almost perfectly "
      "(see the *Experiment results* page)."
)

with st.expander("The 14 features the model actually reads"):
    st.dataframe(
        pd.DataFrame({"feature": thresholds["feature_names"]}),
        hide_index=True, width="stretch",
    )

st.divider()

st.header("What it was trained on")
a, b, c, d = st.columns(4)
a.metric("Dataset", data["dataset"])
b.metric("Transactions", f"{data['n_transactions']:,}")
c.metric("Trained on", f"{data['trained_on_rows']:,}")
d.metric("Fraud prevalence", f"{data['prevalence']:.4%}")
st.caption(f"Source file SHA-256: `{data['csv_sha256']}`")

st.divider()

st.header("How it decides")
decision = card["decision"]
a, b, c = st.columns(3)
a.metric("Severity ratio R", f"{decision['severity_ratio_R']:,.1f}")
b.metric("Decision threshold", f"{decision['threshold']:.6f}")
c.metric("Bayes threshold", f"{thresholds['bayes_threshold']:.6f}")
st.markdown(f"*{decision['interpretation']}.* Threshold {decision['selected_on']}.")

st.markdown("##### Risk bands")
bands = pd.DataFrame([
    {"band": "Low", "risk score above": 0.0, "action": C.BAND_ACTIONS["Low"]},
    *[
        {"band": band, "risk score above": cutoff, "action": C.BAND_ACTIONS[band]}
        for band, cutoff in thresholds["band_cutoffs"].items()
    ],
])
st.dataframe(
    bands, hide_index=True, width="stretch",
    column_config={"risk score above": st.column_config.NumberColumn(format="%.4f")},
)

st.divider()

st.header("How it scored on held-out data")

a, b, c, d = st.columns(4)
a.metric("PR-AUC", f"{performance['pr_auc']:.4f}")
b.metric("Recall", f"{performance['recall']:.4f}")
c.metric("Precision", f"{performance['precision']:.4f}")
d.metric("NER", f"{performance['ner']:.4f}")

a, b, c, d = st.columns(4)
a.metric("F1", f"{performance['f1']:.4f}")
b.metric("MCC", f"{performance['mcc']:.4f}")
c.metric("Brier", f"{performance['brier']:.6f}")
d.metric("Alert rate", f"{performance['alert_rate']:.4%}")

st.caption(
    f"ROC-AUC is {performance['roc_auc']:.4f}, reported only for comparability "
    "with published work. At this prevalence it should not be used to rank models."
)

st.markdown("##### Fraud caught, against what a team can actually review")
budgets = pd.DataFrame([
    {"alerts per day": int(k), "fraud caught": v}
    for k, v in card["recall_at_alerts_per_day"].items()
])
top_k = pd.DataFrame([
    {"top-K alerts on the split": int(k), "fraud caught": v}
    for k, v in card["recall_at_top_k_on_test_split"].items()
])
left, right = st.columns(2)
left.dataframe(
    budgets, hide_index=True, width="stretch",
    column_config={"fraud caught": st.column_config.NumberColumn(format="%.2%")},
)
right.dataframe(
    top_k, hide_index=True, width="stretch",
    column_config={"fraud caught": st.column_config.NumberColumn(format="%.2%")},
)
st.caption(card["budget_note"])

with st.expander("Raw model_card.json"):
    st.json(card)
with st.expander("Raw thresholds.json"):
    st.json(thresholds)
