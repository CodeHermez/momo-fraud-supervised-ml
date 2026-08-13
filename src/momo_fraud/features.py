"""Feature engineering, with a single-row inference path built in from the start.

Two properties are load-bearing:

1. **Every fitted quantity is fit on training data only.** The high-risk amount
   cutoff and the severity quantile grid are the only state here, and both come
   from ``fit``. Deriving either from the full dataset would leak the test
   distribution into the decision rule.

2. **Scoring one raw transaction runs the same code as scoring six million.**
   ``transform_one`` wraps a dict into a one-row frame and calls ``transform``.
   The prototype therefore cannot drift from the notebooks -- retrofitting a
   separate inference path later is where that drift normally creeps in.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import constants as C
from .risk import SeverityScaler

#: Features derived from the simulator's balance bookkeeping. Powerful, but
#: powerful partly *because* PaySim does not update mule-account balances --
#: notebook 02 quantifies this. Every headline result is produced with and
#: without them so the report can show conclusions survive their removal.
ARTIFACT_FEATURES = ["errorBalanceOrig", "errorBalanceDest"]

#: Origin-side balance state. PaySim's fraud agents drain the account exactly,
#: so ``amount == oldbalanceOrg`` and ``newbalanceOrig == 0`` together identify
#: fraud almost perfectly. Dropping only the *derived* error features is not
#: enough to remove that: a tree reconstructs the same rule from these columns.
ORIGIN_BALANCE_FEATURES = [
    "oldbalanceOrg", "newbalanceOrig", "errorBalanceOrig",
    "origZeroBefore", "origZeroAfter",
]

#: Destination-side balance state. The simulator never credits the receiving
#: account on a fraudulent transfer, so these carry the same kind of artifact.
DEST_BALANCE_FEATURES = [
    "oldbalanceDest", "newbalanceDest", "errorBalanceDest",
    "destZeroBefore", "destZeroAfter",
]

#: Graduated feature sets, from "everything" to "what a real-time screen would
#: plausibly hold". Each level removes more of the simulator's bookkeeping; the
#: point is to find one where the task is a genuine learning problem rather than
#: a boolean rule waiting to be recovered.
#:
#: The levels are **cumulative and strictly nested** -- each removes a superset
#: of the one before. Without that, the ablation is not a ladder and results at
#: different levels cannot be read as a progression.
FEATURE_SETS: dict[str, list[str]] = {
    "full": [],
    "no_error_features": sorted(set(ARTIFACT_FEATURES)),
    "no_origin_balance": sorted(set(ARTIFACT_FEATURES) | set(ORIGIN_BALANCE_FEATURES)),
    "transaction_only": sorted(
        set(ARTIFACT_FEATURES) | set(ORIGIN_BALANCE_FEATURES) | set(DEST_BALANCE_FEATURES)
    ),
}


#: The columns Lokanan (2023) actually modelled: the raw PaySim numerics plus
#: one-hot transaction type, with nameOrig/nameDest and isFlaggedFraud dropped.
#: No engineered differences.
#:
#: This distinction is load-bearing for the replication. ``errorBalanceOrig``
#: hands a tree the simulator's fraud rule directly and drives precision to
#: ~0.999; without it a depth-limited tree cannot easily express
#: "amount == oldbalanceOrg" as an axis-aligned split, and lands near the 0.96
#: he reports. Replicating him on *our* feature set measures our pipeline, not
#: his.
LOKANAN_RAW_FEATURES = [
    "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
    "type_CASH_IN", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER",
    "day", "hour",  # stand in for his `step`, which they jointly determine
]


def columns_for(feature_set: str, all_columns: list[str]) -> list[str]:
    """Columns retained by a named feature set."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"unknown feature set {feature_set!r}; expected one of {list(FEATURE_SETS)}"
        )
    excluded = set(FEATURE_SETS[feature_set])
    return [c for c in all_columns if c not in excluded]


@dataclass
class FeatureBuilder:
    """Builds the model matrix. Fit on train, apply everywhere.

    Args:
        high_risk_percentile: Percentile of the training ``amount``
            distribution above which ``highRiskFlag`` fires (proposal 4.1,
            which specifies an empirically-determined cutoff).
        include_artifacts: Set False for the ablation run that drops the
            balance-error features.
    """

    high_risk_percentile: float = 99.0
    include_artifacts: bool = True

    high_risk_cutoff: float | None = None
    severity: SeverityScaler = field(default_factory=SeverityScaler)
    feature_names_: list[str] = field(default_factory=list)

    # --- fitting ------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> FeatureBuilder:
        """Learn the training-only quantities. Never call this on val or test."""
        self.high_risk_cutoff = float(np.percentile(df["amount"], self.high_risk_percentile))
        self.severity.fit(df["amount"])
        self.feature_names_ = list(self._build(df).columns)
        return self

    # --- transformation -----------------------------------------------------

    def _build(self, df: pd.DataFrame) -> pd.DataFrame:
        """The single feature definition, shared by batch and single-row paths."""
        out = pd.DataFrame(index=df.index)

        out["amount"] = df["amount"].astype("float32")
        out["log_amount"] = np.log1p(df["amount"]).astype("float32")

        out["oldbalanceOrg"] = df["oldbalanceOrg"].astype("float32")
        out["newbalanceOrig"] = df["newbalanceOrig"].astype("float32")
        out["oldbalanceDest"] = df["oldbalanceDest"].astype("float32")
        out["newbalanceDest"] = df["newbalanceDest"].astype("float32")

        if self.include_artifacts:
            # Proposal 4.1 "Error Balance Origin": the gap between the expected
            # post-transaction balance and the recorded one.
            #
            # Derived from the already-cast float32 columns rather than from
            # ``df`` directly. A single transaction arrives as a dict, which
            # pandas widens to float64, so computing from the raw frame would
            # round differently in the batch and single-row paths -- train/serve
            # skew of ~1e-4 on a feature whose fraud signature is "exactly 0".
            out["errorBalanceOrig"] = (
                out["oldbalanceOrg"] - out["amount"] - out["newbalanceOrig"]
            ).astype("float32")
            out["errorBalanceDest"] = (
                out["oldbalanceDest"] + out["amount"] - out["newbalanceDest"]
            ).astype("float32")

        # Time-of-day matters because legitimate volume collapses overnight
        # while fraud does not -- see notebook 01.
        out["hour"] = (df["step"] % C.STEPS_PER_DAY).astype("int16")
        out["day"] = (df["step"] // C.STEPS_PER_DAY).astype("int16")

        out["origZeroBefore"] = (df["oldbalanceOrg"] == 0).astype("int8")
        out["origZeroAfter"] = (df["newbalanceOrig"] == 0).astype("int8")
        out["destZeroBefore"] = (df["oldbalanceDest"] == 0).astype("int8")
        out["destZeroAfter"] = (df["newbalanceDest"] == 0).astype("int8")

        cutoff = self.high_risk_cutoff
        if cutoff is None:
            raise RuntimeError("FeatureBuilder must be fit before transform.")
        out["highRiskFlag"] = (df["amount"] > cutoff).astype("int8")

        # Built from the fixed type list rather than get_dummies, so a single
        # transaction produces exactly the same columns as the full dataset.
        type_values = df["type"].astype(str)
        for t in C.TRANSACTION_TYPES:
            out[f"type_{t}"] = (type_values == t).astype("int8")

        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build the model matrix for a batch of transactions."""
        out = self._build(df)
        if self.feature_names_ and list(out.columns) != self.feature_names_:
            out = out[self.feature_names_]
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def transform_one(self, transaction: dict) -> pd.DataFrame:
        """Score a single raw transaction. The prototype's entry point.

        Accepts the raw PaySim field names, so the prototype passes through
        whatever the transaction switch emits without reshaping it.
        """
        missing = {"step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
                   "oldbalanceDest", "newbalanceDest"} - transaction.keys()
        if missing:
            raise ValueError(f"transaction is missing required fields: {sorted(missing)}")
        return self.transform(pd.DataFrame([transaction]))

    # --- severity -----------------------------------------------------------

    def severity_of(self, df: pd.DataFrame) -> np.ndarray:
        """Unitless severity in [0, 1] -- the percentile rank of the amount."""
        return self.severity.transform(df["amount"])

    # --- persistence --------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "high_risk_percentile": self.high_risk_percentile,
            "high_risk_cutoff": self.high_risk_cutoff,
            "include_artifacts": self.include_artifacts,
            "severity": self.severity.to_dict(),
            "feature_names": self.feature_names_,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> FeatureBuilder:
        builder = cls(
            high_risk_percentile=payload["high_risk_percentile"],
            include_artifacts=payload["include_artifacts"],
            high_risk_cutoff=payload["high_risk_cutoff"],
            severity=SeverityScaler.from_dict(payload["severity"]),
        )
        builder.feature_names_ = payload["feature_names"]
        return builder
