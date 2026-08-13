"""The risk framework: severity ratio, NER, thresholds and review budgets.

Evaluation here is deliberately **unitless**. A cost matrix and a risk score are
the same object -- flagging when ``P(fraud) * C_FN > C_FP`` is identical to
flagging when a risk score clears a threshold -- so the whole framework rests on
one interpretable knob:

    R  =  "missing one fraud is as bad as R false alarms"

Following Johnson & Khoshgoftaar (2022) the false-positive cost is fixed at
``C(1,0) = 1`` and only ``C(0,1) = R`` varies. All three cost-sensitive levels in
the proposal's section 3.3 taxonomy then fall out of that single parameter:

    data level       undersample to  N'_neg = N_neg / R
    algorithm level  scale_pos_weight = R
    decision level   threshold lambda = 1 / (1 + R)

**Decision rule.** A transaction is flagged when ``score >= threshold``. The
``>=`` (rather than ``>``) matters only on exact ties, but fixing it makes the
threshold optimiser and the metrics agree by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_curve

from . import constants as C

ArrayLike = np.ndarray | list


# --- The ratio and its derived quantities -----------------------------------


def bayes_threshold(r: float) -> float:
    """Bayes minimum-risk threshold ``lambda = 1/(1+R)``.

    For a calibrated model this is the cost-minimising decision boundary. At
    the recommended ``R = N_neg/N_pos`` it equals the class prior -- for PaySim,
    0.00129.
    """
    return 1.0 / (1.0 + r)


def undersample_size(n_neg: int, r: float) -> int:
    """Majority-class size after RUS, ``N'_neg = N_neg / R``.

    Johnson & Khoshgoftaar Eq. 1 -- the data-level expression of the same
    cost matrix that produces ``bayes_threshold``.
    """
    return max(1, int(round(n_neg / r)))


def risk_score(proba: ArrayLike) -> np.ndarray:
    """Map calibrated probabilities onto the 0-100 score the prototype shows."""
    return np.asarray(proba, dtype=float) * 100.0


# --- Severity ---------------------------------------------------------------


@dataclass
class SeverityScaler:
    """Maps transaction amount to a unitless severity in [0, 1].

    Severity is the percentile rank of ``amount`` within the *training*
    distribution, so it carries "how much is at stake" without ever naming a
    currency. Fitting stores a quantile grid rather than the full training
    vector, which keeps the exported artifact small enough to ship with the
    prototype while reproducing ranks to within one grid step.
    """

    quantiles: np.ndarray | None = None
    n_grid: int = 1001

    def fit(self, amounts: ArrayLike) -> SeverityScaler:
        probs = np.linspace(0.0, 1.0, self.n_grid)
        self.quantiles = np.quantile(np.asarray(amounts, dtype=float), probs)
        return self

    def transform(self, amounts: ArrayLike) -> np.ndarray:
        if self.quantiles is None:
            raise RuntimeError("SeverityScaler must be fit before transform.")
        idx = np.searchsorted(self.quantiles, np.asarray(amounts, dtype=float), side="right")
        return np.clip(idx / (len(self.quantiles) - 1), 0.0, 1.0)

    def fit_transform(self, amounts: ArrayLike) -> np.ndarray:
        return self.fit(amounts).transform(amounts)

    def to_dict(self) -> dict:
        return {"quantiles": self.quantiles.tolist(), "n_grid": self.n_grid}

    @classmethod
    def from_dict(cls, payload: dict) -> SeverityScaler:
        return cls(quantiles=np.asarray(payload["quantiles"], dtype=float),
                   n_grid=payload["n_grid"])


def per_transaction_thresholds(severity: ArrayLike, r: float) -> np.ndarray:
    """Per-transaction Bayes threshold ``lambda_i = 1/(1 + R * s_i)``.

    Larger transactions carry more severity and therefore face a *lower* bar.
    Compared against a single global threshold in notebook 06.
    """
    sev = np.asarray(severity, dtype=float)
    return 1.0 / (1.0 + r * sev)


def _fn_costs(y_true: np.ndarray, r: float, severity: np.ndarray | None) -> np.ndarray:
    """Per-sample false-negative cost: ``R`` flat, or ``R * s_i`` if severity given."""
    if severity is None:
        return np.full(len(y_true), float(r))
    return float(r) * np.asarray(severity, dtype=float)


# --- Normalized Expected Risk ------------------------------------------------


def total_risk(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    r: float,
    severity: ArrayLike | None = None,
) -> float:
    """The proposal's ``TotalCost = (FP * C_FP) + (FN * C_FN)``, unitless.

    ``C_FP = 1`` throughout; ``C_FN = R``, or ``R * s_i`` when per-instance
    severity is supplied.
    """
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    cfn = _fn_costs(y_true, r, None if severity is None else np.asarray(severity))

    false_positives = float(np.sum(~y_true & y_pred))
    false_negative_cost = float(np.sum(cfn[y_true & ~y_pred]))
    return false_positives + false_negative_cost


def normalized_expected_risk(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    r: float,
    severity: ArrayLike | None = None,
) -> float:
    """Total risk over the flag-nothing baseline. Unitless, lower is better.

    Flagging nothing scores exactly 1.0 by construction, so NER reads directly
    as "fraction of the do-nothing risk that remains".

    A useful anchor: at the recommended ``R = N_neg/N_pos``, flagging
    *everything* also scores exactly 1.0. That ratio is precisely where the two
    trivial strategies break even, so any NER below 1.0 is genuine skill rather
    than an artifact of the cost setting.
    """
    y_true = np.asarray(y_true).astype(bool)
    cfn = _fn_costs(y_true, r, None if severity is None else np.asarray(severity))

    baseline = float(np.sum(cfn[y_true]))  # every fraud missed, nothing flagged
    if baseline == 0:
        raise ValueError("No positive samples: NER is undefined.")
    return total_risk(y_true, y_pred, r, severity) / baseline


# --- Threshold selection -----------------------------------------------------


@dataclass
class ThresholdResult:
    """A chosen operating point and what it costs."""

    threshold: float
    ner: float
    n_flagged: int
    precision: float
    recall: float
    rule: str


def _sorted_scan(y_true: np.ndarray, scores: np.ndarray, cfn: np.ndarray):
    """Cumulative statistics over every distinct cut point, highest score first.

    Returns the candidate thresholds and, for each, the risk of flagging
    everything at or above it. O(n log n) via one sort, so the sweep is cheap
    even on the full 6.3M-row test set.
    """
    order = np.argsort(-scores, kind="mergesort")
    s_sorted = scores[order]
    y_sorted = y_true[order].astype(bool)
    cfn_sorted = cfn[order]

    tp = np.cumsum(y_sorted)
    k = np.arange(1, len(scores) + 1)
    fp = k - tp

    total_fn_cost = float(np.sum(cfn[y_true.astype(bool)]))
    caught_fn_cost = np.cumsum(np.where(y_sorted, cfn_sorted, 0.0))
    remaining_fn_cost = total_fn_cost - caught_fn_cost

    risk = fp.astype(float) + remaining_fn_cost

    # Only cut between distinct scores -- a threshold cannot split a tie group.
    is_boundary = np.empty(len(scores), dtype=bool)
    is_boundary[:-1] = s_sorted[:-1] != s_sorted[1:]
    is_boundary[-1] = True

    return s_sorted, y_sorted, tp, k, risk, is_boundary, total_fn_cost


def optimal_threshold(
    y_true: ArrayLike,
    scores: ArrayLike,
    r: float,
    severity: ArrayLike | None = None,
) -> ThresholdResult:
    """Threshold minimising NER, found by cumulative scan.

    Follows Aleksandrova & Armianova (2022): sort by score, accumulate, take the
    running-total argmin -- rather than sweeping an arbitrary grid, which can
    miss the optimum between grid points.

    Flagging nothing is included as a candidate, so a model with no useful
    signal correctly returns NER 1.0 instead of being forced to flag something.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    cfn = _fn_costs(y_true, r, None if severity is None else np.asarray(severity))

    s_sorted, y_sorted, tp, k, risk, is_boundary, total_fn = _sorted_scan(y_true, scores, cfn)

    valid = np.flatnonzero(is_boundary)
    best = valid[np.argmin(risk[valid])]

    # Compare against flagging nothing (risk = total_fn, NER = 1.0).
    if risk[best] >= total_fn:
        return ThresholdResult(
            threshold=float(np.nextafter(s_sorted[0], np.inf)),
            ner=1.0,
            n_flagged=0,
            precision=0.0,
            recall=0.0,
            rule="ner_optimal",
        )

    n_flagged = int(k[best])
    n_tp = int(tp[best])
    n_pos = int(y_true.sum())
    return ThresholdResult(
        threshold=float(s_sorted[best]),
        ner=float(risk[best] / total_fn),
        n_flagged=n_flagged,
        precision=n_tp / n_flagged if n_flagged else 0.0,
        recall=n_tp / n_pos if n_pos else 0.0,
        rule="ner_optimal",
    )


def youden_threshold(y_true: ArrayLike, scores: ArrayLike) -> float:
    """ROC-optimal threshold (max TPR - FPR).

    **At R = N_neg/N_pos this is identical to the NER-optimal threshold**, and
    not by coincidence. Minimising class-constant risk means minimising

        FP + R*FN  =  FPR*N_neg + R*(1 - TPR)*N_pos

    and substituting R = N_neg/N_pos collapses that to N_neg*(FPR + 1 - TPR),
    whose minimiser is exactly the maximiser of TPR - FPR. The two criteria are
    the same criterion at that one ratio, for every model.

    This is worth stating plainly because it dissolves an apparent
    disagreement in the literature. Aleksandrova & Armianova found ROC-optimal
    and consequence-optimal thresholds producing opposite outcomes; their cost
    ratio sat far from their class ratio, which is precisely where the two
    diverge. Reported as a contrast between "statistical" and "business"
    thresholds, the effect is really a contrast between two *cost ratios*.

    The comparison only carries information when ``R != N_neg/N_pos`` or when
    per-instance severity is in play -- see ``optimal_threshold(severity=...)``.
    """
    fpr, tpr, thresholds = roc_curve(np.asarray(y_true), np.asarray(scores))
    return float(thresholds[np.argmax(tpr - fpr)])


def evaluate_threshold(
    y_true: ArrayLike,
    scores: ArrayLike,
    threshold: float | ArrayLike,
    r: float,
    severity: ArrayLike | None = None,
    rule: str = "fixed",
) -> ThresholdResult:
    """Score a given threshold. Accepts a scalar or a per-transaction vector."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    y_pred = scores >= np.asarray(threshold, dtype=float)

    n_flagged = int(y_pred.sum())
    n_tp = int(np.sum(y_pred & y_true.astype(bool)))
    n_pos = int(y_true.sum())

    return ThresholdResult(
        threshold=float(np.mean(threshold)),
        ner=normalized_expected_risk(y_true, y_pred, r, severity),
        n_flagged=n_flagged,
        precision=n_tp / n_flagged if n_flagged else 0.0,
        recall=n_tp / n_pos if n_pos else 0.0,
        rule=rule,
    )


def ner_curve(
    y_true: ArrayLike,
    scores: ArrayLike,
    r: float,
    severity: ArrayLike | None = None,
    n_points: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """NER as a function of decision threshold -- the proposal's section 4.5 plot.

    Thresholds are sampled on the score *quantiles* rather than uniformly on
    [0, 1]: fraud scores concentrate near zero under extreme imbalance, and a
    uniform grid would spend nearly every point in an empty region.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    cfn = _fn_costs(y_true, r, None if severity is None else np.asarray(severity))

    grid = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, n_points)))
    baseline = float(np.sum(cfn[y_true.astype(bool)]))

    ners = np.array([
        total_risk(y_true, scores >= t, r, severity) / baseline for t in grid
    ])
    return grid, ners


# --- Operational metrics -----------------------------------------------------


def recall_at_budget(y_true: ArrayLike, scores: ArrayLike, budget: int) -> float:
    """Fraction of fraud caught when analysts review the top ``budget`` scores.

    A capacity constraint rather than a monetary one, and closer to how a fraud
    operations team actually works than any fixed threshold.
    """
    y_true = np.asarray(y_true).astype(bool)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(y_true.sum())
    if n_pos == 0:
        raise ValueError("No positive samples: recall@budget is undefined.")

    budget = min(int(budget), len(scores))
    top = np.argpartition(-scores, budget - 1)[:budget] if budget else np.array([], dtype=int)
    return float(y_true[top].sum()) / n_pos


def precision_at_budget(y_true: ArrayLike, scores: ArrayLike, budget: int) -> float:
    """Fraction of the reviewed queue that is genuinely fraud."""
    y_true = np.asarray(y_true).astype(bool)
    scores = np.asarray(scores, dtype=float)

    budget = min(int(budget), len(scores))
    if budget == 0:
        return 0.0
    top = np.argpartition(-scores, budget - 1)[:budget]
    return float(y_true[top].sum()) / budget


def budget_curve(
    y_true: ArrayLike,
    scores: ArrayLike,
    budgets: list[int] | None = None,
) -> dict[int, dict[str, float]]:
    """Recall and precision across a range of analyst review capacities."""
    budgets = budgets or C.REVIEW_BUDGETS
    return {
        b: {
            "recall": recall_at_budget(y_true, scores, b),
            "precision": precision_at_budget(y_true, scores, b),
        }
        for b in budgets
    }


# --- Operational bands -------------------------------------------------------


def band_cutoffs(r: float) -> dict[str, float]:
    """Risk-score cutoffs for the Fraud Management System bands.

    Anchored on the Bayes threshold rather than round numbers: ``Critical``
    begins an order of magnitude above lambda, ``High`` at lambda itself (the
    point where flagging becomes risk-reducing), and ``Medium`` an order of
    magnitude below. Cutoffs are on the 0-100 risk scale.
    """
    lam = bayes_threshold(r) * 100.0
    return {
        "Medium": lam / 10.0,
        "High": lam,
        "Critical": min(lam * 10.0, 100.0),
    }


def assign_bands(scores_0_100: ArrayLike, cutoffs: dict[str, float]) -> np.ndarray:
    """Label each transaction Low / Medium / High / Critical."""
    s = np.asarray(scores_0_100, dtype=float)
    bands = np.full(len(s), "Low", dtype=object)
    bands[s >= cutoffs["Medium"]] = "Medium"
    bands[s >= cutoffs["High"]] = "High"
    bands[s >= cutoffs["Critical"]] = "Critical"
    return bands
