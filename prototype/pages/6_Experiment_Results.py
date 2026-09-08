"""The evidence behind the model: 31 figures and 41 tables, in study order."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import evidence, scorer  # noqa: E402

st.set_page_config(page_title="Experiment results", layout="wide")
st.title("The evidence behind the model")

card = scorer.get_model_card()

st.markdown(
    "Nine notebooks and 28 model fits. Nothing here is recomputed — these are "
    "the figures and tables the experiments wrote."
)

# --- The finding that shaped every other decision ---------------------------

st.divider()
st.header("The result that changed the study")

st.markdown(
    "This three-clause rule — no training, no parameters, no model — recovers "
    "almost every fraud in PaySim:"
)
st.code(
    "type in (TRANSFER, CASH_OUT)\n"
    "    AND amount == oldbalanceOrg      # the whole balance moved\n"
    "    AND newbalanceOrig == 0          # the account was left empty",
    language="text",
)

try:
    rules = evidence.load_result("02_rule_recoverability.csv")
    st.dataframe(
        rules, hide_index=True, width="stretch",
        column_config={
            "n_flagged": st.column_config.NumberColumn("Flagged", format="%d"),
            "precision": st.column_config.NumberColumn(format="%.4f"),
            "recall": st.column_config.NumberColumn(format="%.4f"),
        },
    )
except Exception:  # noqa: BLE001
    st.info("`results/02_rule_recoverability.csv` is not available.")

if evidence.demo_sample_available():
    sample = evidence.load_demo_sample()
    rule = (
        sample["type"].isin(["TRANSFER", "CASH_OUT"])
        & np.isclose(sample["amount"], sample["oldbalanceOrg"])
        & np.isclose(sample["newbalanceOrig"], 0.0)
    )
    truth = sample["isFraud"].astype(bool)
    tp = int((rule & truth).sum())
    fp = int((rule & ~truth).sum())
    fn = int((~rule & truth).sum())

    st.markdown("##### Run the rule live, on the bundled test-set sample")
    a, b, c = st.columns(3)
    a.metric("Precision", f"{tp / (tp + fp):.4f}" if tp + fp else "n/a")
    b.metric("Recall", f"{tp / (tp + fn):.4f}" if tp + fn else "n/a")
    c.metric("Model's PR-AUC, for contrast",
             f"{card['held_out_performance']['pr_auc']:.4f}")

st.warning(
    "**This is the simulator, not fraud.** PaySim does not update the sender's "
    "balance on fraudulent transactions, so the origin-balance columns encode the "
    "label almost perfectly. The shipped model therefore drops "
    + ", ".join(f"`{c}`" for c in card["data"]["features_excluded"])
    + f", giving up the easy 0.9999 for an honest "
    f"{card['held_out_performance']['pr_auc']:.4f} PR-AUC on a harder task."
)

# --- The gallery ------------------------------------------------------------

st.divider()
st.header("Every experiment, in order")

for prefix, title, blurb in evidence.CHAPTERS:
    figures = evidence.figures_for(prefix)
    tables = evidence.results_for(prefix)
    if not figures and not tables:
        continue

    with st.expander(f"**{prefix} — {title}**", expanded=False):
        st.markdown(blurb)

        if figures:
            st.markdown("##### Figures")
            for row_start in range(0, len(figures), 2):
                columns = st.columns(2)
                for column, path in zip(columns, figures[row_start:row_start + 2]):
                    with column:
                        st.image(str(path), caption=evidence.prettify(path))

        if tables:
            st.markdown("##### Tables")
            for path in tables:
                st.markdown(f"**{evidence.prettify(path)}** &nbsp; `{path.name}`")
                try:
                    frame = pd.read_csv(path)
                    st.dataframe(frame.head(50), hide_index=True, width="stretch")
                    if len(frame) > 50:
                        st.caption(f"First 50 of {len(frame):,} rows.")
                except Exception as error:  # noqa: BLE001
                    st.caption(f"Could not read: {error}")

# --- Honest closing ---------------------------------------------------------

st.divider()
st.header("What was left on the table")

st.markdown(
    "Notebook 10 went looking for headroom. Two configurations beat the shipped "
    "model on normalised expected risk, where **lower is better**:"
)

try:
    summary = evidence.load_result("10_summary.csv")
    st.dataframe(
        summary, hide_index=True, width="stretch",
        column_config={
            "ner": st.column_config.NumberColumn("NER", format="%.4f"),
            "pr_auc": st.column_config.NumberColumn("PR-AUC", format="%.4f"),
            "beats_baseline": st.column_config.CheckboxColumn("Beats baseline"),
        },
    )
except Exception:  # noqa: BLE001
    st.info("`results/10_summary.csv` is not available.")

st.info(
    "**Neither was promoted, and this prototype serves the older, weaker "
    "model.** Promoting one would mean refitting, re-deriving the threshold on "
    "the validation split, and re-exporting the bundle — and the stack has no "
    "tree, so it would lose the SHAP explanations this prototype relies on. "
    "Reporting the gain without taking it is the honest position."
)
