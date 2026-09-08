"""Shared rendering. The verdict panel appears on two pages and must not drift.

If the explanation an analyst sees when they type a transaction differs from the
one they see when they open a false positive, the demo has two stories. So both
pages call the same function.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from . import explain


def band_badge(band: str) -> str:
    """The band as a coloured pill."""
    colour = explain.BAND_COLOURS.get(band, "#57606a")
    return (
        f'<span style="background:{colour};color:#fff;padding:0.25rem 0.75rem;'
        f'border-radius:1rem;font-weight:600;font-size:0.95rem;">{band}</span>'
    )


def render_verdict(result: dict, severity_ratio: float) -> None:
    """The three headline numbers, the band, and what to do about it."""
    left, middle, right = st.columns(3)
    left.metric("Risk score", f"{result['risk_score']:.2f} / 100")
    middle.metric("Fraud probability", f"{result['probability']:.4%}")
    right.metric("Alert", "FLAGGED" if result["flagged"] else "not flagged")

    st.markdown(
        f"{band_badge(result['band'])} &nbsp; **{result['action']}** &mdash; "
        f"{explain.BAND_MEANING.get(result['band'], '')}",
        unsafe_allow_html=True,
    )

    st.markdown(explain.narrate(result))

    with st.expander("Why the alert line sits where it does"):
        st.markdown(explain.threshold_sentence(result, severity_ratio))


def render_factors(result: dict) -> None:
    """The SHAP contributions, largest first, as a diverging bar chart."""
    factors = result.get("top_factors") or []
    if not factors:
        st.info(
            "No SHAP explainer is loaded, so per-factor contributions are "
            "unavailable. Scoring is unaffected."
        )
        return

    frame = pd.DataFrame(factors)
    frame["reads as"] = frame["feature"].map(explain.phrase_for)

    st.caption(
        "How far each feature pushed this transaction's score, in log-odds. "
        "Positive pushes towards fraud."
    )
    st.bar_chart(
        frame.set_index("reads as")["contribution"],
        horizontal=True,
        color="#b3261e",
    )
    st.dataframe(
        frame[["feature", "reads as", "contribution", "direction"]],
        hide_index=True,
        width="stretch",
    )


def band_distribution(scored: pd.DataFrame) -> pd.DataFrame:
    """Counts per band, always in severity order even when a band is empty."""
    order = ["Low", "Medium", "High", "Critical"]
    counts = scored["band"].value_counts()
    return pd.DataFrame(
        {"transactions": [int(counts.get(b, 0)) for b in order]},
        index=pd.Index(order, name="band"),
    )


def confusion_counts(y_true: pd.Series, flagged: pd.Series) -> dict[str, int]:
    """TP / FP / FN / TN as plain integers."""
    truth = y_true.astype(bool).to_numpy()
    alert = flagged.astype(bool).to_numpy()
    return {
        "TP": int((truth & alert).sum()),
        "FP": int((~truth & alert).sum()),
        "FN": int((truth & ~alert).sum()),
        "TN": int((~truth & ~alert).sum()),
    }


def render_confusion(counts: dict[str, int]) -> None:
    """The 2x2, laid out as a table rather than a heatmap so it reads at a glance."""
    matrix = pd.DataFrame(
        [
            [counts["TN"], counts["FP"]],
            [counts["FN"], counts["TP"]],
        ],
        index=pd.Index(["Actually legitimate", "Actually fraud"], name=""),
        columns=["Not flagged", "Flagged"],
    )
    st.dataframe(matrix.style.format("{:,}"), width="stretch")

    tp, fp, fn = counts["TP"], counts["FP"], counts["FN"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    a, b, c = st.columns(3)
    a.metric("Precision", f"{precision:.2%}", help="Of everything flagged, how much was fraud.")
    b.metric("Recall", f"{recall:.2%}", help="Of all fraud, how much was caught.")
    c.metric("F1", f"{f1:.4f}")
