"""Metrics, confidence intervals, and the prediction cache.

The cache is the reason the risk analysis is cheap. Fitting writes out the
*probability vector* for every learner x config x split x seed; notebook 06 then
sweeps nine values of R, three threshold rules and four review budgets over
those cached vectors without refitting anything. The expensive grid runs once.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from . import constants as C
from . import risk as R

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"
RESULTS_DIR = PROJECT_ROOT / "results"


# --- Metrics ----------------------------------------------------------------


def threshold_free_metrics(y_true, scores) -> dict[str, float]:
    """Metrics that do not depend on where the cut is made.

    PR-AUC leads because it is the one that matters at 0.129% prevalence.
    ROC-AUC is reported for comparability with the prior work but should not
    be used to rank models here: it is near-insensitive to false positives when
    negatives outnumber positives 774 to 1, which is exactly how Lokanan's
    models scored 0.92+ while being operationally unusable.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "brier": float(brier_score_loss(y_true, scores)),
    }


def threshold_metrics(y_true, scores, threshold) -> dict[str, float]:
    """Confusion-matrix metrics at a given operating point."""
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(scores, dtype=float) >= np.asarray(threshold, dtype=float)).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(np.mean(threshold)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "n_flagged": int(y_pred.sum()),
        "alert_rate": float(y_pred.mean()),
    }


def risk_metrics(y_true, scores, threshold, r: float, severity=None) -> dict[str, float]:
    """The unitless risk metrics: NER plus recall and precision at budget.

    ``ner`` is always the **class-constant** measure defined in the study design
    (C_FN = R for every fraud). When per-instance severity is supplied, the
    example-dependent variant is reported alongside as
    ``ner_severity_weighted`` rather than replacing it.

    Reporting one under the other's name is how the same model ends up with two
    different "NER" values in two notebooks -- which is exactly what happened
    here before this was split out.
    """
    y_pred = np.asarray(scores, dtype=float) >= np.asarray(threshold, dtype=float)
    out = {
        "R": float(r),
        "ner": R.normalized_expected_risk(y_true, y_pred, r),
        "total_risk": R.total_risk(y_true, y_pred, r),
    }
    if severity is not None:
        out["ner_severity_weighted"] = R.normalized_expected_risk(
            y_true, y_pred, r, severity)
        out["total_risk_severity_weighted"] = R.total_risk(y_true, y_pred, r, severity)
    for budget, values in R.budget_curve(y_true, scores).items():
        out[f"recall_at_{budget}"] = values["recall"]
        out[f"precision_at_{budget}"] = values["precision"]
    return out


def full_report(
    y_true,
    scores,
    *,
    threshold: float,
    r: float = C.R_DEFAULT,
    severity=None,
    **context,
) -> dict:
    """One flat row combining every metric family, ready for a results table."""
    return {
        **context,
        **threshold_free_metrics(y_true, scores),
        **threshold_metrics(y_true, scores, threshold),
        **risk_metrics(y_true, scores, threshold, r, severity),
    }


def select_threshold(
    y_true,
    scores,
    rule: str,
    r: float = C.R_DEFAULT,
    severity=None,
) -> R.ThresholdResult:
    """Apply a threshold rule. **Call this on validation scores only.**

    The chosen value is then frozen and applied once to test. Selecting a
    threshold on the test set is the mistake this whole study is built to
    avoid making.
    """
    if rule == "fixed":
        return R.evaluate_threshold(y_true, scores, 0.5, r, severity, rule="fixed")
    if rule == "youden":
        t = R.youden_threshold(y_true, scores)
        return R.evaluate_threshold(y_true, scores, t, r, severity, rule="youden")
    if rule == "ner_optimal":
        return R.optimal_threshold(y_true, scores, r, severity)
    raise ValueError(f"unknown threshold rule {rule!r}")


# --- Uncertainty ------------------------------------------------------------


def bootstrap_ci(
    y_true,
    scores,
    metric_fn,
    *,
    n_boot: int = 200,
    alpha: float = 0.05,
    seed: int = C.RANDOM_SEED,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for a threshold-free metric.

    Resampling is stratified within each class, so every resample keeps the
    original fraud count. Unstratified bootstrap at 0.129% prevalence would
    occasionally draw a sample with a badly distorted positive count and
    inflate the interval for reasons unrelated to model variance.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    rng = np.random.default_rng(seed)

    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)

    estimates = []
    for _ in range(n_boot):
        sample = np.concatenate([
            rng.choice(pos_idx, size=len(pos_idx), replace=True),
            rng.choice(neg_idx, size=len(neg_idx), replace=True),
        ])
        estimates.append(metric_fn(y_true[sample], scores[sample]))

    point = metric_fn(y_true, scores)
    lo, hi = np.quantile(estimates, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)


def paired_bootstrap_delta(
    y_true,
    scores_a,
    scores_b,
    metric_fn,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = C.RANDOM_SEED,
) -> dict[str, float]:
    """Paired bootstrap of ``metric(A) - metric(B)`` on shared resamples.

    Both score vectors are indexed by the **same** resampled rows in every
    replicate, so the observation-level noise the two models share cancels out of
    the difference. That is the whole point: A and B were scored on the identical
    test set, and treating their intervals as independent throws that pairing
    away.

    This replaces the earlier procedure, which declared a difference when two
    *marginal* CIs failed to overlap. Non-overlap does imply a difference, but
    overlap does **not** imply its absence -- the marginal test is strictly
    weaker than the paired one and cannot support a "no difference" verdict in
    either direction. Anything reported as indistinguishable under the old rule
    has to be re-decided here.

    Resampling is stratified within class, matching ``bootstrap_ci``: at 0.129%
    prevalence an unstratified draw occasionally lands a badly distorted positive
    count and widens the interval for reasons unrelated to either model.

    Returns the point delta, its percentile CI, and ``p_two_sided`` -- the
    proportion of replicates falling on the opposite side of zero from the point
    estimate, doubled. Read that as a bootstrap achieved significance level, not
    as a p-value from a parametric test.
    """
    y_true = np.asarray(y_true)
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    rng = np.random.default_rng(seed)

    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)

    deltas = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = np.concatenate([
            rng.choice(pos_idx, size=len(pos_idx), replace=True),
            rng.choice(neg_idx, size=len(neg_idx), replace=True),
        ])
        y_s = y_true[sample]
        deltas[i] = metric_fn(y_s, scores_a[sample]) - metric_fn(y_s, scores_b[sample])

    point = float(metric_fn(y_true, scores_a) - metric_fn(y_true, scores_b))
    lo, hi = np.quantile(deltas, [alpha / 2, 1 - alpha / 2])

    n_cross = int((deltas >= 0).sum() if point < 0 else (deltas <= 0).sum())
    return {
        "delta": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_boot": int(n_boot),
        "n_crossing_zero": n_cross,
        "prop_crossing_zero": n_cross / n_boot,
        "p_two_sided": float(min(1.0, 2.0 * n_cross / n_boot)),
        "excludes_zero": bool(lo > 0 or hi < 0),
    }


def _weighted_average_precision(y_sorted: np.ndarray, w_sorted: np.ndarray) -> float:
    """Average precision from per-row multiplicities, on a pre-sorted vector.

    A bootstrap resample is a multiset of the original rows, so it is fully
    described by an integer multiplicity per row. Working from multiplicities
    lets the O(n log n) sort be hoisted out of the replicate loop and done once
    per model, which is the difference between minutes and hours on a 954k-row
    test set.
    """
    tp = np.cumsum(y_sorted * w_sorted, dtype=np.float64)
    fp = np.cumsum((1 - y_sorted) * w_sorted, dtype=np.float64)
    n_pos = tp[-1]
    if n_pos == 0:
        return float("nan")

    denominator = tp + fp
    precision = np.divide(tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
    # Each positive copy contributes one recall increment of 1/n_pos.
    return float(np.sum(precision * y_sorted * w_sorted) / n_pos)


def paired_bootstrap_pr_auc(
    y_true,
    scores_a,
    scores_b,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = C.RANDOM_SEED,
) -> dict[str, float]:
    """``paired_bootstrap_delta`` for PR-AUC, fast enough to actually run at 2000.

    Same procedure and same guarantees as the generic version -- shared
    stratified resamples, paired difference -- but each score vector is sorted
    once up front and every replicate is an O(n) weighted pass over that fixed
    order. Agreement with the generic path is asserted in
    ``tests/test_evaluate_repairs.py``.

    The point estimate is computed the same way, so it differs from
    ``sklearn.metrics.average_precision_score`` at around 1e-8 through tie
    handling. That is six orders of magnitude below the smallest delta this
    study reports, and it is identical for both models in the pair, so it
    cancels out of the difference entirely.
    """
    y_true = np.asarray(y_true).astype(np.int64)
    rng = np.random.default_rng(seed)

    prepared = []
    for scores in (scores_a, scores_b):
        order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
        prepared.append((order, y_true[order]))

    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)
    n = len(y_true)

    deltas = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        counts = np.bincount(
            np.concatenate([
                rng.choice(pos_idx, size=len(pos_idx), replace=True),
                rng.choice(neg_idx, size=len(neg_idx), replace=True),
            ]),
            minlength=n,
        )
        a, b = (_weighted_average_precision(y_s, counts[order])
                for order, y_s in prepared)
        deltas[i] = a - b

    ones = np.ones(n, dtype=np.int64)
    point = float(_weighted_average_precision(prepared[0][1], ones[prepared[0][0]])
                  - _weighted_average_precision(prepared[1][1], ones[prepared[1][0]]))
    lo, hi = np.quantile(deltas, [alpha / 2, 1 - alpha / 2])

    n_cross = int((deltas >= 0).sum() if point < 0 else (deltas <= 0).sum())
    return {
        "delta": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_boot": int(n_boot),
        "n_crossing_zero": n_cross,
        "prop_crossing_zero": n_cross / n_boot,
        "p_two_sided": float(min(1.0, 2.0 * n_cross / n_boot)),
        "excludes_zero": bool(lo > 0 or hi < 0),
    }


def paired_test(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Wilcoxon signed-rank on paired per-fold scores.

    Used for H1: whether configs C and E differ from the baseline A, and
    whether D falls below it. Paired because both configs see identical folds,
    which removes fold difficulty as a source of variance.
    """
    from scipy.stats import wilcoxon

    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diff = a - b
    if np.allclose(diff, 0):
        return {"statistic": 0.0, "p_value": 1.0, "median_difference": 0.0}

    stat, p = wilcoxon(a, b)
    return {
        "statistic": float(stat),
        "p_value": float(p),
        "median_difference": float(np.median(diff)),
    }


# --- Persistence ------------------------------------------------------------


def prediction_path(learner: str, config: str, split: str, seed: int, *, tag: str = "") -> Path:
    stem = f"{learner}__{config}__{split}__seed{seed}"
    if tag:
        stem += f"__{tag}"
    return PREDICTIONS_DIR / f"{stem}.parquet"


def save_predictions(
    y_true, scores, *, learner: str, config: str, split: str, seed: int,
    severity=None, tag: str = "",
) -> Path:
    """Cache one probability vector so notebook 06 never has to refit."""
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    frame = {"y_true": np.asarray(y_true).astype("int8"),
             "score": np.asarray(scores, dtype="float32")}
    if severity is not None:
        frame["severity"] = np.asarray(severity, dtype="float32")

    path = prediction_path(learner, config, split, seed, tag=tag)
    pd.DataFrame(frame).to_parquet(path, engine="pyarrow", compression="snappy", index=False)
    return path


def load_predictions(learner: str, config: str, split: str, seed: int, *, tag: str = "") -> pd.DataFrame:
    path = prediction_path(learner, config, split, seed, tag=tag)
    if not path.exists():
        raise FileNotFoundError(
            f"No cached predictions at {path}. Run the fitting notebook (03-05) first."
        )
    return pd.read_parquet(path, engine="pyarrow")


def save_results(rows: list[dict], name: str) -> Path:
    """Write a results table as CSV, for direct import into the report."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def save_json(payload: dict, name: str, *, directory: Path | None = None) -> Path:
    directory = directory or RESULTS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.json"

    def encode(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if hasattr(obj, "__dataclass_fields__"):
            return asdict(obj)
        raise TypeError(f"unserialisable: {type(obj)}")

    path.write_text(json.dumps(payload, indent=2, default=encode), encoding="utf-8")
    return path
