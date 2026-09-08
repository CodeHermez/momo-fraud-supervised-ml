"""Score one transaction: type it in, or load a real one."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import schema, scorer, ui  # noqa: E402
from momo_fraud import constants as C  # noqa: E402

st.set_page_config(page_title="Score a transaction", layout="wide")
st.title("Score a transaction")

if not scorer.artifacts_present():
    st.error("No model bundle in `artifacts/`.")
    st.stop()

fraud_scorer = scorer.get_scorer()
thresholds = scorer.get_thresholds()

if "transaction" not in st.session_state:
    st.session_state.transaction = schema.SAMPLE_PROFILES[0].transaction.copy()
    st.session_state.profile_note = None

st.markdown("##### Load a real transaction from the held-out test split")
st.caption(
    "Five rows the model was evaluated on, with their real outcomes: a clear "
    "pass, a near miss, a caught fraud, a false alarm and a missed fraud."
)

columns = st.columns(len(schema.SAMPLE_PROFILES))
for column, profile in zip(columns, schema.SAMPLE_PROFILES):
    if column.button(profile.name, width="stretch", help=profile.blurb):
        st.session_state.transaction = profile.transaction.copy()
        st.session_state.profile_note = profile

st.divider()

form_column, result_column = st.columns([2, 3], gap="large")

with form_column:
    st.markdown("##### Transaction details")
    current = st.session_state.transaction

    # Key must not collide with the ``transaction`` session-state entry below:
    # Streamlit refuses to let a widget key be assigned through session_state.
    with st.form("transaction_form"):
        kind = st.selectbox(
            "Type", C.TRANSACTION_TYPES,
            index=C.TRANSACTION_TYPES.index(current["type"]),
            help=schema.FIELD_HELP["type"],
        )
        step = st.number_input(
            "Step (hour of simulation)", min_value=1, max_value=C.MAX_STEP,
            value=int(current["step"]), step=1, help=schema.FIELD_HELP["step"],
        )
        st.caption(
            f"Step {step} is hour **{step % C.STEPS_PER_DAY}** of day "
            f"**{step // C.STEPS_PER_DAY}**."
        )
        amount = st.number_input(
            "Amount", min_value=0.0, value=float(current["amount"]),
            step=100.0, format="%.2f", help=schema.FIELD_HELP["amount"],
        )

        st.markdown("**Recipient balances**")
        old_dest = st.number_input(
            "Recipient balance before", min_value=0.0,
            value=float(current["oldbalanceDest"]), step=100.0, format="%.2f",
        )
        new_dest = st.number_input(
            "Recipient balance after", min_value=0.0,
            value=float(current["newbalanceDest"]), step=100.0, format="%.2f",
        )

        st.markdown("**Sender balances**")
        st.caption("Required by the feature pipeline, but this model never reads them.")
        old_orig = st.number_input(
            "Sender balance before", min_value=0.0,
            value=float(current["oldbalanceOrg"]), step=100.0, format="%.2f",
        )
        new_orig = st.number_input(
            "Sender balance after", min_value=0.0,
            value=float(current["newbalanceOrig"]), step=100.0, format="%.2f",
        )

        submitted = st.form_submit_button("Score it", type="primary", width="stretch")

    if submitted:
        st.session_state.transaction = {
            "step": int(step), "type": kind, "amount": float(amount),
            "oldbalanceOrg": float(old_orig), "newbalanceOrig": float(new_orig),
            "oldbalanceDest": float(old_dest), "newbalanceDest": float(new_dest),
        }
        st.session_state.profile_note = None

    with st.expander("Why are the sender balances ignored?"):
        st.markdown(
            "`amount == oldbalanceOrg` and `newbalanceOrig == 0` identify almost "
            "every fraud in PaySim outright, so a model given those columns "
            "learns the simulator rather than fraud. This one drops them; the "
            "*Experiment results* page shows the measurement."
        )

with result_column:
    st.markdown("##### Verdict")

    transaction = st.session_state.transaction
    result = fraud_scorer.score(transaction, explain=True)

    ui.render_verdict(result, thresholds["severity_ratio_R"])

    note = st.session_state.get("profile_note")
    if note is not None:
        truth = "was fraud" if note.actual_fraud else "was legitimate"
        correct = note.actual_fraud == result["flagged"]
        message = (
            f"**Ground truth: this transaction {truth}.** "
            f"The model {'got this right' if correct else 'got this wrong'}."
        )
        (st.success if correct else st.error)(message)
        st.caption(note.blurb)

    st.divider()
    st.markdown("##### What drove the score")
    ui.render_factors(result)
