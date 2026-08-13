"""Exploratory and leakage-audit computations.

These live here rather than inline in the notebooks so they can be tested. The
leakage audit in particular decides which features are admissible, and a bug in
it would quietly invalidate every result downstream -- that is not code to leave
untested in a cell.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

from . import constants as C


# --- Descriptive -------------------------------------------------------------


def fraud_rate_by_type(df: pd.DataFrame) -> pd.DataFrame:
    """Volume and fraud rate per transaction type.

    The headline finding is categorical rather than numeric: fraud appears in
    TRANSFER and CASH_OUT and nowhere else. That is a property of the
    simulator's fraud agents, and it means three of the five transaction types
    carry no positive class at all.
    """
    grouped = df.groupby("type", observed=True)[C.TARGET].agg(["count", "sum"])
    grouped.columns = ["n_transactions", "n_fraud"]

    grouped["fraud_rate"] = grouped["n_fraud"] / grouped["n_transactions"]
    grouped["share_of_volume"] = grouped["n_transactions"] / len(df)
    grouped["share_of_fraud"] = grouped["n_fraud"] / max(int(df[C.TARGET].sum()), 1)

    return grouped.sort_values("n_transactions", ascending=False).reset_index()


def hourly_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Volume and fraud rate by hour of simulated day.

    Legitimate volume collapses overnight while fraud continues at a roughly
    constant rate, so the *rate* spikes in the small hours even though the
    absolute count does not. That asymmetry is why hour-of-day carries signal,
    and it is visible only when both series are plotted -- as two panels, never
    as one chart with two y-axes.
    """
    hours = df["step"] % C.STEPS_PER_DAY
    grouped = df.groupby(hours, observed=True)[C.TARGET].agg(["count", "sum"])
    grouped.columns = ["n_transactions", "n_fraud"]
    grouped["fraud_rate"] = grouped["n_fraud"] / grouped["n_transactions"]

    grouped.index.name = "hour"
    return grouped.reset_index()


def amount_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Amount distribution by class, for comparison with Lokanan's Table 3."""
    quantiles = [0.25, 0.5, 0.75, 0.95, 0.99]
    rows = []
    for label, mask in (("legitimate", df[C.TARGET] == 0), ("fraud", df[C.TARGET] == 1)):
        amounts = df.loc[mask, "amount"]
        row = {"class": label, "n": len(amounts), "mean": amounts.mean(), "std": amounts.std()}
        row.update({f"p{int(q * 100)}": amounts.quantile(q) for q in quantiles})
        row["max"] = amounts.max()
        rows.append(row)
    return pd.DataFrame(rows)


# --- Leakage audit -----------------------------------------------------------


def flagged_fraud_evidence(df: pd.DataFrame) -> dict:
    """Evidence for dropping ``isFlaggedFraud``.

    Lokanan dropped this column for having "no variation". The stronger reason
    is that it is an *output* of the simulator's own rule evaluated against the
    label, not an input a live system would hold at scoring time. Its precision
    against ``isFraud`` demonstrates that directly.
    """
    flagged = df["isFlaggedFraud"] == 1
    fraud = df[C.TARGET] == 1

    n_flagged = int(flagged.sum())
    return {
        "n_flagged": n_flagged,
        "n_flagged_and_fraud": int((flagged & fraud).sum()),
        "precision_against_label": float((flagged & fraud).sum() / n_flagged) if n_flagged else float("nan"),
        "recall_against_label": float((flagged & fraud).sum() / fraud.sum()),
        "share_of_dataset": n_flagged / len(df),
    }


def drain_signature(df: pd.DataFrame, *, tolerance: float = 1.0) -> pd.DataFrame:
    """How often ``amount`` equals the origin's opening balance, by class.

    PaySim's fraud agents empty the account they take over, so for fraud the
    transferred amount matches ``oldbalanceOrg`` almost exactly. Legitimate
    transactions have no such constraint. This is the single strongest signal in
    the dataset and it is a property of the *simulator*, not of mobile money --
    which is why it is reported as a threat to validity rather than a finding.
    """
    rows = []
    for label, mask in (("legitimate", df[C.TARGET] == 0), ("fraud", df[C.TARGET] == 1)):
        subset = df.loc[mask]
        gap = (subset["amount"] - subset["oldbalanceOrg"]).abs()
        rows.append({
            "class": label,
            "n": len(subset),
            "fraction_amount_equals_balance": float((gap <= tolerance).mean()),
            "median_absolute_gap": float(gap.median()),
        })
    return pd.DataFrame(rows)


def zero_balance_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Rate of zero balances on each side, by class.

    PaySim does not credit the destination account on fraudulent transfers, so
    both destination balances sit at zero far more often for fraud. Another
    bookkeeping artifact rather than a behavioural one.
    """
    checks = {
        "origin_zero_before": df["oldbalanceOrg"] == 0,
        "origin_zero_after": df["newbalanceOrig"] == 0,
        "dest_zero_before": df["oldbalanceDest"] == 0,
        "dest_zero_after": df["newbalanceDest"] == 0,
        "dest_zero_both": (df["oldbalanceDest"] == 0) & (df["newbalanceDest"] == 0),
    }
    fraud = df[C.TARGET] == 1

    return pd.DataFrame([
        {
            "condition": name,
            "rate_legitimate": float(mask[~fraud].mean()),
            "rate_fraud": float(mask[fraud].mean()),
        }
        for name, mask in checks.items()
    ])


def rule_recoverability(df: pd.DataFrame, *, tolerance: float = 1.0) -> pd.DataFrame:
    """How well hand-written boolean rules reproduce the label.

    PaySim's fraud agents follow a deterministic script: take over an account,
    transfer the full balance out, cash it out. That script is expressible as a
    few clauses over the raw columns, and if those clauses recover the label
    then the dataset does not pose a learning problem at all -- it poses a
    lookup.

    This matters more than any single feature's AUC. A model that scores 0.99
    on this data may simply have rediscovered the generator, and a
    cost-sensitive comparison run on top of a solved task measures nothing.
    Report these numbers alongside any model result on PaySim.
    """
    y = df[C.TARGET].to_numpy().astype(bool)

    drained = (df["amount"] - df["oldbalanceOrg"]).abs().to_numpy() <= tolerance
    risky_type = df["type"].isin(C.FRAUD_BEARING_TYPES).to_numpy()
    emptied = (df["newbalanceOrig"] == 0).to_numpy()

    candidates = {
        "type in (TRANSFER, CASH_OUT)": risky_type,
        "amount == oldbalanceOrg": drained,
        "risky type AND drained": risky_type & drained,
        "risky type AND drained AND emptied": risky_type & drained & emptied,
    }

    rows = []
    for name, predicted in candidates.items():
        n_flagged = int(predicted.sum())
        n_hit = int((predicted & y).sum())
        rows.append({
            "rule": name,
            "n_flagged": n_flagged,
            "precision": n_hit / n_flagged if n_flagged else 0.0,
            "recall": n_hit / int(y.sum()),
        })
    return pd.DataFrame(rows)


def single_feature_auc(
    X: pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    sample_size: int = 200_000,
    seed: int = C.RANDOM_SEED,
) -> pd.DataFrame:
    """AUC achievable from each feature alone, via a depth-1 stump.

    A feature that separates the classes almost perfectly on its own is a
    simulator artifact, not a discovery. A stump rather than the raw value so
    non-monotone features are not unfairly scored -- the raw AUC of a feature
    whose signal is "equals zero" would look like noise.

    **Scored out of sample.** Fitting and evaluating a stump on the same rows
    inflates the AUC: at 0.1% prevalence a split can capture a handful of
    positives by chance, and pure noise then scores around 0.6 rather than 0.5.
    That would flag innocent features as artifacts. Cross-validated predictions
    cost three fits per feature and remove the bias.

    Sampled by default: 6.3M rows is slow and the ranking is stable well below
    that. Sampling is stratified so the fraud count is preserved exactly.
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    y = np.asarray(y)
    rng = np.random.default_rng(seed)

    if sample_size and len(X) > sample_size:
        pos_idx = np.flatnonzero(y == 1)
        neg_idx = np.flatnonzero(y == 0)
        n_neg = max(sample_size - len(pos_idx), 1)
        keep = np.concatenate([
            pos_idx,
            rng.choice(neg_idx, size=min(n_neg, len(neg_idx)), replace=False),
        ])
        X, y = X.iloc[keep], y[keep]

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

    rows = []
    for column in X.columns:
        values = X[[column]].to_numpy(dtype=float)
        # A constant feature has no split to make; score it at chance rather
        # than letting the stump raise.
        if np.ptp(values) == 0:
            rows.append({"feature": column, "auc": 0.5})
            continue

        oof = cross_val_predict(
            DecisionTreeClassifier(max_depth=1, random_state=seed),
            values, y, cv=cv, method="predict_proba",
        )[:, 1]
        rows.append({"feature": column, "auc": float(roc_auc_score(y, oof))})

    return pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)


def flag_suspicious_features(
    auc_table: pd.DataFrame,
    *,
    threshold: float = 0.90,
) -> list[str]:
    """Features whose solo AUC is high enough to warrant the ablation.

    The threshold is a reporting convention, not a statistical test: anything
    above it gets named in the write-up as a candidate simulator artifact and
    is included in the with/without comparison.
    """
    return auc_table.loc[auc_table["auc"] >= threshold, "feature"].tolist()
