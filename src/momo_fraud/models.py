"""Learner and configuration factory -- the experiment grid.

Four learners (Mienye & Sun's quartet) crossed with seven configurations that
express the proposal's section 3.3 taxonomy through a single severity ratio.

Model *selection* uses PR-AUC, and thresholds are tuned afterwards. That order
is deliberate and inverts the usual habit: Johnson & Khoshgoftaar showed that
tuning against a threshold-dependent metric bakes an arbitrary operating point
into hyperparameter choice, and that thresholding after selection outperforms
weighting during it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from . import constants as C


@dataclass(frozen=True)
class ExperimentConfig:
    """One cell of the grid: how the ratio R enters, and how the cut is made."""

    key: str
    description: str
    level: str  # "none" | "data" | "algorithm" | "decision" | "algorithm+decision"
    class_weighted: bool = False
    instance_weighted: bool = False
    threshold_rule: str = "fixed"  # "fixed" | "youden" | "ner_optimal"
    resample: str | None = None  # None | "rus"


CONFIG_SPECS: dict[str, ExperimentConfig] = {
    "A": ExperimentConfig("A", "baseline, t=0.5", "none"),
    "B": ExperimentConfig("B", "threshold from ROC (Youden)", "decision",
                          threshold_rule="youden"),
    "C": ExperimentConfig("C", "threshold minimising NER", "decision",
                          threshold_rule="ner_optimal"),
    "D": ExperimentConfig("D", "scale_pos_weight = R, t=0.5", "algorithm",
                          class_weighted=True),
    "E": ExperimentConfig("E", "weighted + NER threshold", "algorithm+decision",
                          class_weighted=True, threshold_rule="ner_optimal"),
    "F": ExperimentConfig("F", "per-instance sample_weight ~ severity", "algorithm",
                          instance_weighted=True, threshold_rule="ner_optimal"),
    "G": ExperimentConfig("G", "RUS at N'_neg, natural test set", "data",
                          resample="rus", threshold_rule="ner_optimal"),
}


def make_learner(
    name: str,
    *,
    scale_pos_weight: float | None = None,
    seed: int = C.RANDOM_SEED,
    n_jobs: int = -1,
):
    """Build an unfitted estimator.

    Hyperparameters follow the comparison papers so the numbers stay
    commensurable: RF depth 16 and XGBoost depth 4-6 are Johnson &
    Khoshgoftaar's settings for severely imbalanced big data; the XGBoost
    learning rate and estimator count follow Thar & Wai on this same dataset.

    Args:
        scale_pos_weight: Minority-class weight. Pass ``R`` for the
            algorithm-level configs; leave ``None`` for unweighted.
    """
    weighted = scale_pos_weight is not None and scale_pos_weight != 1.0

    if name == "logistic_regression":
        # Scaling is required, not cosmetic: raw PaySim amounts reach ~9.2e7
        # while the binary flags are 0/1, and lbfgs will not converge sensibly
        # across seven orders of magnitude.
        # No n_jobs: it has been a no-op for LogisticRegression since
        # scikit-learn 1.8 and passing it emits a FutureWarning.
        return Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                solver="lbfgs",
                class_weight={0: 1.0, 1: scale_pos_weight} if weighted else None,
                random_state=seed,
            )),
        ])

    if name == "decision_tree":
        return DecisionTreeClassifier(
            criterion="gini",
            max_depth=8,
            min_samples_leaf=50,
            class_weight={0: 1.0, 1: scale_pos_weight} if weighted else None,
            random_state=seed,
        )

    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=16,
            min_samples_leaf=5,
            max_samples=0.3,  # keeps 4.4M-row fits affordable on 8 cores
            class_weight={0: 1.0, 1: scale_pos_weight} if weighted else None,
            random_state=seed,
            n_jobs=n_jobs,
        )

    if name == "xgboost":
        from xgboost import XGBClassifier  # imported lazily; heavy module

        return XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            scale_pos_weight=scale_pos_weight if weighted else 1.0,
            eval_metric="aucpr",
            random_state=seed,
            n_jobs=n_jobs,
        )

    raise ValueError(f"unknown learner {name!r}; expected one of {C.LEARNERS}")


#: Learners beyond the core quartet, added to check whether a different
#: gradient-boosting implementation beats XGBoost on this rung -- kept out of
#: ``LEARNERS``/``make_learner`` because several existing tests and notebooks
#: assume that constant is exactly the four-learner quartet.
EXTRA_LEARNERS = ["lightgbm"]


def make_learner_extended(
    name: str,
    *,
    scale_pos_weight: float | None = None,
    seed: int = C.RANDOM_SEED,
    n_jobs: int = -1,
):
    """``make_learner`` plus the learners in ``EXTRA_LEARNERS``.

    Hyperparameters for the extras mirror XGBoost's current fixed settings
    (depth 6, 300 estimators, lr 0.1, subsample/colsample 0.8) so the first
    comparison is "same budget, different implementation" -- isolating that
    question from whatever a separate tuning pass might find.
    """
    if name in C.LEARNERS:
        return make_learner(name, scale_pos_weight=scale_pos_weight, seed=seed, n_jobs=n_jobs)

    weighted = scale_pos_weight is not None and scale_pos_weight != 1.0

    if name == "lightgbm":
        from lightgbm import LGBMClassifier  # imported lazily; heavy module

        return LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            # LightGBM silently ignores `subsample` unless `subsample_freq` is
            # also set -- unlike XGBoost, where the equivalent parameter is
            # always active. Without this, "matched hyperparameters" would
            # silently not match.
            subsample_freq=1,
            colsample_bytree=0.8,
            class_weight={0: 1.0, 1: scale_pos_weight} if weighted else None,
            random_state=seed,
            n_jobs=n_jobs,
            verbose=-1,
        )

    raise ValueError(f"unknown learner {name!r}; expected one of {C.LEARNERS + EXTRA_LEARNERS}")


def sample_weights(
    y: np.ndarray,
    config: ExperimentConfig,
    r: float,
    severity: np.ndarray | None = None,
) -> np.ndarray | None:
    """Per-sample training weights for the instance-weighted arm (config F).

    Config F is the novel arm: Aleksandrova & Armianova applied case-based
    severity only when choosing the *threshold* and named applying it during
    *training* as future work. Here a fraud's weight scales with how much is at
    stake, so the loss surface itself learns that large transfers matter more.

    **Three objectives, and they are not the same one.** The audit flagged this
    as a comparability issue, so it is stated here rather than left to be
    inferred:

    ===============  =====================================================
    training         ``C_FN = R * s_i``  (this function)
    threshold        ``C_FN = R``        (experiment.evaluate_configs)
    evaluation       ``C_FN = R``        (evaluate.risk_metrics -> ``ner``)
    ===============  =====================================================

    Because ``s_i`` is a percentile rank, ``E[s_i] ~ 0.5``, so config F carries
    roughly **half** config D's effective positive weight while being scored on
    D's objective. F is therefore a *sensitivity-analysis training variant under
    the class-constant evaluation objective* -- not a self-consistent
    example-dependent cost-sensitive experiment. A negative F result is evidence
    about that variant, and is **not** evidence that severity weighting fails on
    its own terms; testing that requires training, thresholding and scoring all
    under ``R * s_i``, which is a separate experiment this study has not run.

    ``evaluate.risk_metrics`` does report ``ner_severity_weighted`` alongside,
    which scores under the training objective -- but the threshold feeding it was
    still chosen under class-constant costs, so it is not the missing experiment
    either. See ``docs/config_f_cost_objectives.md``.
    """
    if not config.instance_weighted:
        return None
    if severity is None:
        raise ValueError("config F requires per-instance severity.")

    weights = np.ones(len(y), dtype=float)
    positives = y.astype(bool)
    weights[positives] = r * np.asarray(severity)[positives]
    return weights


def resample_indices(
    y: np.ndarray,
    config: ExperimentConfig,
    r: float,
    seed: int = C.RANDOM_SEED,
) -> np.ndarray | None:
    """Row indices after random undersampling, or None if the config resamples nothing.

    The majority target size is ``N_neg / R`` -- Johnson & Khoshgoftaar Eq. 1 --
    so undersampling is driven by the same ratio as the weights and the
    threshold rather than by a hand-picked sampling proportion.
    """
    if config.resample != "rus":
        return None

    from .risk import undersample_size

    rng = np.random.default_rng(seed)
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)

    target = min(undersample_size(len(neg_idx), r), len(neg_idx))
    kept_neg = rng.choice(neg_idx, size=target, replace=False)

    return np.sort(np.concatenate([pos_idx, kept_neg]))


def scale_pos_weight_for(config: ExperimentConfig, r: float) -> float | None:
    """The weight this config passes to the learner, if any."""
    return r if config.class_weighted else None
