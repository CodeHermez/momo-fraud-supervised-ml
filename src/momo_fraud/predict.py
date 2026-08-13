"""The inference API the prototype loads.

Scoring one raw transaction runs the *same* feature code as scoring six million
-- ``FeatureBuilder.transform_one`` wraps a dict into a one-row frame and calls
the batch path. That is deliberate: a separate inference path is where train/serve
skew creeps in, and this project has already caught one such bug (float32 versus
float64 balance arithmetic silently disagreeing by 1e-4 on a feature whose fraud
signature is "exactly zero").

The bundle on disk is self-describing: model, fitted preprocessor, frozen
thresholds and band cutoffs, and a model card recording what it was trained on.
Nothing here refits or re-derives anything.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import constants as C
from . import risk as R
from .features import FeatureBuilder

MODEL_FILE = "model.json"
PREPROCESSOR_FILE = "preprocessor.json"
THRESHOLDS_FILE = "thresholds.json"
MODEL_CARD_FILE = "model_card.json"


@dataclass
class FraudScorer:
    """Scores transactions and explains why.

    Attributes:
        model: The fitted estimator.
        builder: The feature pipeline, with its train-only fitted quantities.
        thresholds: Frozen decision threshold, severity ratio and band cutoffs.
        feature_names: Column order the model expects.
    """

    model: object
    builder: FeatureBuilder
    thresholds: dict
    feature_names: list[str]
    explainer: object = None

    # --- persistence --------------------------------------------------------

    @classmethod
    def load(cls, directory: str | Path, *, with_explainer: bool = True) -> FraudScorer:
        """Load a frozen bundle. The prototype's single entry point."""
        directory = Path(directory)
        from xgboost import XGBClassifier

        model = XGBClassifier()
        model.load_model(directory / MODEL_FILE)

        builder = FeatureBuilder.from_dict(
            json.loads((directory / PREPROCESSOR_FILE).read_text())
        )
        thresholds = json.loads((directory / THRESHOLDS_FILE).read_text())

        explainer = None
        if with_explainer:
            try:
                import shap

                explainer = shap.TreeExplainer(model)
            except Exception:
                # Explanations are a nicety; scoring must not depend on them.
                explainer = None

        return cls(model=model, builder=builder, thresholds=thresholds,
                   feature_names=thresholds["feature_names"], explainer=explainer)

    def save(self, directory: str | Path, *, model_card: dict | None = None) -> Path:
        """Freeze everything the prototype needs into one directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        self.model.save_model(directory / MODEL_FILE)
        (directory / PREPROCESSOR_FILE).write_text(
            json.dumps(self.builder.to_dict()), encoding="utf-8"
        )
        (directory / THRESHOLDS_FILE).write_text(
            json.dumps(self.thresholds, indent=2), encoding="utf-8"
        )
        if model_card is not None:
            (directory / MODEL_CARD_FILE).write_text(
                json.dumps(model_card, indent=2), encoding="utf-8"
            )
        return directory

    # --- scoring ------------------------------------------------------------

    def _probabilities(self, features: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(features[self.feature_names])[:, 1]

    def score(self, transaction: dict, *, explain: bool = True) -> dict:
        """Score one raw transaction.

        Args:
            transaction: Raw fields as the transaction switch emits them --
                ``step``, ``type``, ``amount``, ``oldbalanceOrg``,
                ``newbalanceOrig``, ``oldbalanceDest``, ``newbalanceDest``.

        Returns:
            The 0-100 risk score, its band and recommended action, whether it
            clears the frozen threshold, and the features that moved the score.
        """
        features = self.builder.transform_one(transaction)
        probability = float(self._probabilities(features)[0])

        risk = float(R.risk_score(probability))
        cutoffs = self.thresholds["band_cutoffs"]
        band = str(R.assign_bands([risk], cutoffs)[0])

        result = {
            # Deliberately unrounded. The decision threshold at R = 774 sits at
            # 0.129 on this 0-100 scale, so rounding here would both discard
            # meaningful resolution near the cut and break the identity
            # risk_score == probability * 100. Format for display at the edge.
            "risk_score": risk,
            "probability": probability,
            "band": band,
            "action": C.BAND_ACTIONS[band],
            "flagged": probability >= self.thresholds["decision_threshold"],
            "decision_threshold": self.thresholds["decision_threshold"],
        }
        if explain:
            result["top_factors"] = self._explain(features)
        return result

    def score_batch(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """Score many transactions at once, for backfills and monitoring."""
        features = self.builder.transform(transactions)
        probabilities = self._probabilities(features)
        risk = R.risk_score(probabilities)

        cutoffs = self.thresholds["band_cutoffs"]
        bands = R.assign_bands(risk, cutoffs)

        return pd.DataFrame({
            "risk_score": risk,
            "probability": probabilities,
            "band": bands,
            "action": [C.BAND_ACTIONS[b] for b in bands],
            "flagged": probabilities >= self.thresholds["decision_threshold"],
        }, index=transactions.index)

    def _explain(self, features: pd.DataFrame, *, top_n: int = 5) -> list[dict]:
        """The features that moved this transaction's score, largest first.

        Returns an empty list rather than raising when no explainer is loaded --
        an analyst losing the explanation is a degraded experience; a scoring
        endpoint that throws is an outage.
        """
        if self.explainer is None:
            return []

        values = self.explainer.shap_values(features[self.feature_names])[0]
        ranked = sorted(zip(self.feature_names, values),
                        key=lambda pair: abs(pair[1]), reverse=True)

        return [
            {"feature": name,
             "contribution": round(float(value), 5),
             "direction": "increases risk" if value > 0 else "decreases risk"}
            for name, value in ranked[:top_n]
        ]


def build_thresholds(
    *,
    decision_threshold: float,
    r: float,
    feature_names: list[str],
    rung: str,
) -> dict:
    """Assemble the frozen decision configuration.

    Band cutoffs derive from ``r`` rather than being hand-picked, so the bands
    move consistently with the threshold if the severity ratio is ever revised.
    """
    return {
        "decision_threshold": float(decision_threshold),
        "severity_ratio_R": float(r),
        "bayes_threshold": float(R.bayes_threshold(r)),
        "band_cutoffs": {k: float(v) for k, v in R.band_cutoffs(r).items()},
        "band_actions": dict(C.BAND_ACTIONS),
        "feature_names": list(feature_names),
        "feature_set": rung,
    }
