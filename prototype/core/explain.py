"""Turning SHAP contributions into sentences a manager can act on.

``FraudScorer._explain`` already ranks the features that moved a score. What it
returns is ``destZeroAfter: +4.277``, which is precise and useless to anyone
outside the project. This module supplies the missing half: a phrase for each
feature, and a short narration naming the band, the action and the two factors
that mattered most.

Anything unmapped falls back to its raw feature name. A missing phrase should
degrade the sentence, not break the page.
"""

from __future__ import annotations

#: One readable phrase per model feature. Written to complete the sentence
#: "... because <phrase>", so they read as observations, not column names.
FEATURE_PHRASES = {
    "amount": "the amount moved",
    "log_amount": "the size of the amount",
    "oldbalanceDest": "the recipient's balance beforehand",
    "newbalanceDest": "the recipient's balance afterwards",
    "hour": "the hour of day",
    "day": "which day of the month it fell on",
    "destZeroBefore": "the recipient's account was empty beforehand",
    "destZeroAfter": "the recipient's account was left empty",
    "highRiskFlag": "the amount is unusually large for this network",
    "type_CASH_IN": "this was a cash-in",
    "type_CASH_OUT": "this was a cash-out",
    "type_DEBIT": "this was a debit",
    "type_PAYMENT": "this was a payment",
    "type_TRANSFER": "this was a transfer",
}

#: What each band means in plain terms, for the badge caption.
BAND_MEANING = {
    "Low": "Nothing unusual. Let it through.",
    "Medium": "Slightly out of pattern. Worth logging, not worth stopping.",
    "High": "Enough signal to ask the customer to confirm who they are.",
    "Critical": "Strong match for a known fraud pattern. Hold it.",
}

ACTION_WORDS = {
    "allow": "allow it",
    "monitor": "let it through but keep watching the account",
    "step_up_auth": "ask the customer for extra authentication",
    "block": "block it and send it for review",
}

BAND_COLOURS = {
    "Low": "#1a7f37",
    "Medium": "#9a6700",
    "High": "#bc4c00",
    "Critical": "#b3261e",
}


def phrase_for(feature: str) -> str:
    """A readable phrase for a model feature, or the feature name itself."""
    return FEATURE_PHRASES.get(feature, feature)


def narrate(result: dict) -> str:
    """Two or three sentences explaining one scored transaction.

    Reads only what ``FraudScorer.score`` returns, so it works whether or not a
    SHAP explainer was loaded.
    """
    band = result.get("band", "Low")
    action = result.get("action", "allow")
    risk = result.get("risk_score", 0.0)
    factors = result.get("top_factors") or []

    opening = (
        f"This transaction scores **{risk:.2f} out of 100**, which puts it in the "
        f"**{band}** band. The recommended action is to {ACTION_WORDS.get(action, action)}."
    )

    if not factors:
        return opening

    pushed_up = [f for f in factors if f["contribution"] > 0][:2]
    pushed_down = [f for f in factors if f["contribution"] < 0][:1]

    parts = []
    if pushed_up:
        reasons = " and ".join(phrase_for(f["feature"]) for f in pushed_up)
        parts.append(f"What raised the score: {reasons}.")
    if pushed_down:
        eased = " and ".join(phrase_for(f["feature"]) for f in pushed_down)
        parts.append(f"Pulling the other way: {eased}.")

    return opening + "\n\n" + " ".join(parts)


def threshold_sentence(result: dict, severity_ratio: float) -> str:
    """Why the alert line sits at 0.19% rather than the 50% people expect."""
    probability = result.get("probability", 0.0)
    threshold = result.get("decision_threshold", 0.0)
    cleared = result.get("flagged", False)

    verdict = (
        f"{probability:.4%} is **above** the {threshold:.4%} alert line, so this is flagged."
        if cleared
        else f"{probability:.4%} is **below** the {threshold:.4%} alert line, so this is not flagged."
    )
    return (
        f"{verdict}\n\n"
        f"The line is low by design: missing one fraud is priced as high as "
        f"**{severity_ratio:,.0f}** false alarms."
    )
