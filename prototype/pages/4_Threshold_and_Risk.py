"""Move the alert line and watch what it costs, across the full test split."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import evidence, scorer  # noqa: E402
from momo_fraud import constants as C  # noqa: E402
from momo_fraud import risk as R  # noqa: E402

st.set_page_config(page_title="Threshold and risk", layout="wide")
st.title("Where should the alert line sit?")

if not evidence.test_predictions_available():
    st.error(
        "The held-out prediction file is missing from `predictions/`. "
        "Run `notebooks/03_baselines.ipynb` to regenerate it."
    )
    st.stop()

thresholds = scorer.get_thresholds()
card = scorer.get_model_card()
shipped_threshold = thresholds["decision_threshold"]
shipped_r = thresholds["severity_ratio_R"]

preds = evidence.load_test_predictions()
y_true = preds["y_true"].to_numpy()
scores = preds["score"].to_numpy()
recorded_severity = preds["severity"].to_numpy()

st.markdown(
    f"Computed live over the **{len(preds):,}** held-out transactions, of which "
    f"**{int(y_true.sum()):,}** were fraud."
)

st.divider()

controls, explanation = st.columns([2, 3], gap="large")

with controls:
    st.markdown("##### Controls")

    severity_ratio = st.select_slider(
        "Severity ratio R",
        options=[float(v) for v in C.R_SWEEP],
        value=float(shipped_r),
        format_func=lambda v: f"{v:,.0f}",
        help="How many false alarms one missed fraud is worth.",
    )

    weight_by_size = st.checkbox(
        "Price a missed fraud by how much it moved",
        value=False,
        help="Off matches the shipped configuration: every missed fraud counts "
             "the same. On, a large fraud costs more to miss than a small one.",
    )
    severity = recorded_severity if weight_by_size else None

    use_optimal = st.checkbox(
        "Put the line at the risk-optimal point for this R", value=False
    )

    if use_optimal:
        optimal = R.optimal_threshold(y_true, scores, severity_ratio, severity)
        threshold = optimal.threshold
        st.caption(f"Risk-optimal threshold for R = {severity_ratio:,.0f}: `{threshold:.8f}`")
    else:
        threshold = st.slider(
            "Decision threshold",
            min_value=0.0, max_value=0.05,
            value=float(shipped_threshold),
            step=0.00001, format="%.5f",
        )

with explanation:
    st.markdown("##### What R means")
    st.markdown(
        f"""
        **Missing one fraud is treated as costly as raising R false alarms.**
        Raise R and the model interrupts more honest customers to catch more
        fraud; lower it and the reverse.

        The shipped **R = {shipped_r:,.1f}** is the ratio of legitimate to
        fraudulent transactions in the data. Everything else on this page
        follows from that one number.
        """
    )

st.divider()

current = R.evaluate_threshold(y_true, scores, threshold, severity_ratio, severity)
# The reference point is always the shipped configuration -- unweighted, at the
# frozen threshold -- so the deltas mean the same thing whatever the controls say.
shipped = R.evaluate_threshold(y_true, scores, shipped_threshold, shipped_r, None)

st.markdown("##### At this operating point")

a, b, c, d, e = st.columns(5)
a.metric("Recall", f"{current.recall:.2%}",
         delta=f"{current.recall - shipped.recall:+.2%}",
         help="Share of all fraud caught.")
b.metric("Precision", f"{current.precision:.2%}",
         delta=f"{current.precision - shipped.precision:+.2%}",
         help="Share of alerts that were really fraud.")
c.metric("NER", f"{current.ner:.4f}",
         delta=f"{current.ner - shipped.ner:+.4f}", delta_color="inverse",
         help="Normalised expected risk. Lower is better; 1.0 means flagging nothing.")
d.metric("Alerts raised", f"{current.n_flagged:,}",
         delta=f"{current.n_flagged - shipped.n_flagged:+,}", delta_color="off")
e.metric("Alert rate", f"{current.n_flagged / len(preds):.2%}")

st.caption(
    "Deltas compare against the shipped operating point "
    f"(R = {shipped_r:,.1f}, threshold {shipped_threshold:.6f}), which reproduces "
    f"the model card exactly: precision {card['held_out_performance']['precision']:.4f}, "
    f"recall {card['held_out_performance']['recall']:.4f}, "
    f"NER {card['held_out_performance']['ner']:.4f}."
)

st.divider()

curve_column, budget_column = st.columns([3, 2], gap="large")

with curve_column:
    st.markdown("##### Risk against threshold")

    grid, ners = R.ner_curve(y_true, scores, severity_ratio, severity, n_points=300)
    curve = pd.DataFrame({"threshold": grid, "NER": ners}).set_index("threshold")

    st.line_chart(curve, color="#b3261e", height=320)
    st.caption(
        "NER as the alert line moves; the dip is the risk-optimal cut. Flagging "
        "nothing scores 1.0, so anything below that is genuine skill."
    )

    def _row(label: str, at: float) -> dict:
        outcome = R.evaluate_threshold(y_true, scores, at, severity_ratio, severity)
        return {
            "rule": label,
            "threshold": outcome.threshold,
            "NER": outcome.ner,
            "recall": outcome.recall,
            "precision": outcome.precision,
        }

    youden = R.youden_threshold(y_true, scores)
    optimal_here = R.optimal_threshold(y_true, scores, severity_ratio, severity)

    st.dataframe(
        pd.DataFrame([
            _row("Risk-optimal (minimises NER)", optimal_here.threshold),
            _row("Youden / ROC-optimal", youden),
            _row("Naive 0.5 cut-off", 0.5),
        ]),
        hide_index=True, width="stretch",
        column_config={
            "threshold": st.column_config.NumberColumn(format="%.6f"),
            "NER": st.column_config.NumberColumn(format="%.4f"),
            "recall": st.column_config.NumberColumn(format="%.4f"),
            "precision": st.column_config.NumberColumn(format="%.4f"),
        },
    )
    if severity is None and abs(severity_ratio - shipped_r) < 1e-6:
        st.caption(
            "The first two thresholds are **identical** — the framework's central "
            "claim holding: at R = N_legit / N_fraud, minimising expected risk and "
            "maximising Youden's J select the same cut-off."
        )
    else:
        st.caption(
            "The first two rows coincide only at R = N_legit / N_fraud with "
            "unweighted costs."
        )

    st.caption(
        f"The shipped threshold ({shipped_threshold:.6f}) differs from the optimum "
        "measured here because it was chosen on the **validation** split, then "
        "frozen. Re-picking it here would be tuning on held-out data."
    )

with budget_column:
    st.markdown("##### If analysts can only review so much")

    days = C.MAX_STEP / C.STEPS_PER_DAY * C.SPLIT_TEST
    budgets = pd.DataFrame([
        {
            "alerts per day": budget,
            "recall": R.recall_at_budget(y_true, scores, int(budget * days)),
        }
        for budget in C.REVIEW_BUDGETS
    ])
    st.dataframe(
        budgets, hide_index=True, width="stretch",
        column_config={
            "alerts per day": st.column_config.NumberColumn(format="%d"),
            "recall": st.column_config.NumberColumn("Fraud caught", format="%.2%"),
        },
    )
    st.caption(
        f"Budgets scaled to the {days:.1f} days the test split covers. Reviewing "
        "only the top slice by score is what a real fraud team does."
    )

    st.markdown("##### Risk bands")
    cutoffs = R.band_cutoffs(severity_ratio)
    bands = pd.DataFrame(
        [{"band": band, "risk score above": cutoff,
          "action": C.BAND_ACTIONS[band]}
         for band, cutoff in cutoffs.items()]
    )
    bands = pd.concat([
        pd.DataFrame([{"band": "Low", "risk score above": 0.0, "action": "allow"}]),
        bands,
    ], ignore_index=True)
    st.dataframe(
        bands, hide_index=True, width="stretch",
        column_config={"risk score above": st.column_config.NumberColumn(format="%.4f")},
    )
    st.caption("Cutoffs derive from R, so they move with it rather than being hand-picked.")
