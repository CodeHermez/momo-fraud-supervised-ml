"""Fitting, scoring and caching the experiment grid.

**The grid is cheaper than it looks.** Seven configurations do not mean seven
fits: thresholding is a post-hoc operation on a probability vector, so configs
A, B and C share one unweighted model and differ only in where the cut is made,
and D and E share one weighted model. What actually gets trained is four
*variants* per learner:

    unweighted         -> configs A (t=0.5), B (Youden), C (NER-optimal)
    weighted           -> configs D (t=0.5), E (NER-optimal)
    instance_weighted  -> config F
    rus                -> config G

That is 16 fits per seed rather than 28, and it is the reason notebook 06 can
sweep nine values of R across three threshold rules for free.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import constants as C
from . import evaluate as E
from . import models as M
from . import risk as R

#: Which configurations each trained model serves.
FIT_VARIANTS: dict[str, list[str]] = {
    "unweighted": ["A", "B", "C"],
    "weighted": ["D", "E"],
    "instance_weighted": ["F"],
    "rus": ["G"],
}

#: The threshold rule each configuration applies to its variant's scores.
CONFIG_RULES: dict[str, str] = {
    "A": "fixed", "B": "youden", "C": "ner_optimal",
    "D": "fixed", "E": "ner_optimal",
    "F": "ner_optimal", "G": "ner_optimal",
}


@dataclass
class FitResult:
    """One trained model plus the scores it produced."""

    learner: str
    variant: str
    seed: int
    fit_seconds: float
    n_train: int
    val_scores: np.ndarray
    test_scores: np.ndarray
    estimator: object = field(repr=False, default=None)


def fit_one(
    learner: str,
    variant: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    *,
    r: float = C.R_DEFAULT,
    severity_train: np.ndarray | None = None,
    seed: int = C.RANDOM_SEED,
    keep_estimator: bool = False,
) -> FitResult:
    """Train one variant and score validation and test.

    Both score vectors are returned together because the discipline this study
    is built around requires them: the threshold is chosen on validation, frozen,
    and only then applied to test. Producing them in one place makes it hard to
    accidentally select on test.
    """
    config = M.CONFIG_SPECS[FIT_VARIANTS[variant][0]]
    spw = r if variant == "weighted" else None

    rows = np.arange(len(y_train))
    if variant == "rus":
        rows = M.resample_indices(y_train, M.CONFIG_SPECS["G"], r, seed=seed)

    X_fit, y_fit = X_train.iloc[rows], y_train[rows]

    weights = None
    if variant == "instance_weighted":
        if severity_train is None:
            raise ValueError("instance_weighted requires per-instance severity.")
        weights = M.sample_weights(y_fit, M.CONFIG_SPECS["F"], r, severity_train[rows])

    estimator = M.make_learner(learner, scale_pos_weight=spw, seed=seed)

    start = time.perf_counter()
    if weights is not None:
        # Pipelines need the step name prefixed to route a fit parameter.
        if learner == "logistic_regression":
            estimator.fit(X_fit, y_fit, clf__sample_weight=weights)
        else:
            estimator.fit(X_fit, y_fit, sample_weight=weights)
    else:
        estimator.fit(X_fit, y_fit)
    elapsed = time.perf_counter() - start

    return FitResult(
        learner=learner,
        variant=variant,
        seed=seed,
        fit_seconds=elapsed,
        n_train=len(y_fit),
        val_scores=estimator.predict_proba(X_val)[:, 1].astype("float32"),
        test_scores=estimator.predict_proba(X_test)[:, 1].astype("float32"),
        estimator=estimator if keep_estimator else None,
    )


def evaluate_configs(
    fit: FitResult,
    y_val: np.ndarray,
    y_test: np.ndarray,
    *,
    r: float = C.R_DEFAULT,
    severity_val: np.ndarray | None = None,
    severity_test: np.ndarray | None = None,
    feature_set: str = "full",
) -> list[dict]:
    """Turn one fitted variant into a row per configuration it serves.

    The threshold for every rule is selected on **validation** and then applied
    unchanged to test. Both sets of numbers are reported: a large gap between
    them is itself diagnostic.
    """
    rows = []
    for config_key in FIT_VARIANTS[fit.variant]:
        rule = CONFIG_RULES[config_key]

        # Thresholds are selected under **class-constant** costs for every
        # config. Severity belongs to config F, which applies it during
        # *training*; letting it into threshold selection here would silently
        # make every config severity-aware and make config C something other
        # than what the design says it is.
        #
        # A consequence worth expecting rather than debugging: at
        # R = N_neg/N_pos the NER-optimal and Youden thresholds are the same
        # number, so configs B and C coincide at the default R. That is a
        # theorem, not a defect -- see risk.youden_threshold.
        chosen = E.select_threshold(y_val, fit.val_scores, rule, r)
        threshold = chosen.threshold

        for split_name, y_true, scores, severity in (
            ("val", y_val, fit.val_scores, severity_val),
            ("test", y_test, fit.test_scores, severity_test),
        ):
            rows.append(E.full_report(
                y_true, scores,
                threshold=threshold, r=r, severity=severity,
                learner=fit.learner, config=config_key, variant=fit.variant,
                level=M.CONFIG_SPECS[config_key].level,
                threshold_rule=rule, split=split_name, seed=fit.seed,
                feature_set=feature_set, fit_seconds=fit.fit_seconds,
                n_train=fit.n_train,
            ))
    return rows


def run_grid(
    X_train, y_train, X_val, y_val, X_test, y_test,
    *,
    learners: list[str] | None = None,
    variants: list[str] | None = None,
    seeds: list[int] | None = None,
    r: float = C.R_DEFAULT,
    severity: dict[str, np.ndarray] | None = None,
    feature_set: str = "full",
    cache: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Fit every requested cell, cache the scores, and return the results table.

    Caching the probability vectors is what makes notebook 06 cheap: the risk
    sweep re-reads them rather than refitting anything.
    """
    learners = learners or C.LEARNERS
    variants = variants or list(FIT_VARIANTS)
    seeds = seeds or [C.RANDOM_SEED]
    severity = severity or {}

    rows: list[dict] = []
    total = len(learners) * len(variants) * len(seeds)
    done = 0

    for seed in seeds:
        for learner in learners:
            for variant in variants:
                done += 1
                if verbose:
                    print(f"[{done}/{total}] {learner} / {variant} / seed {seed} …",
                          end="", flush=True)

                fit = fit_one(
                    learner, variant, X_train, y_train, X_val, X_test,
                    r=r, severity_train=severity.get("train"), seed=seed,
                )

                if cache:
                    tag = feature_set if feature_set != "full" else ""
                    E.save_predictions(y_val, fit.val_scores, learner=learner,
                                       config=variant, split="val", seed=seed,
                                       severity=severity.get("val"), tag=tag)
                    E.save_predictions(y_test, fit.test_scores, learner=learner,
                                       config=variant, split="test", seed=seed,
                                       severity=severity.get("test"), tag=tag)

                rows.extend(evaluate_configs(
                    fit, y_val, y_test, r=r,
                    severity_val=severity.get("val"),
                    severity_test=severity.get("test"),
                    feature_set=feature_set,
                ))

                if verbose:
                    pr = rows[-1]["pr_auc"]
                    print(f" {fit.fit_seconds:6.1f}s  test PR-AUC {pr:.4f}", flush=True)

    return pd.DataFrame(rows)


#: Above this, a PR-AUC at 0.129% prevalence is more likely leakage than skill.
SUSPICIOUSLY_PERFECT = 0.99


def headroom_by_rung(results: pd.DataFrame) -> pd.Series:
    """Best achievable NER per feature set -- how much risk is left to remove.

    This is the number that decides where a cost-sensitive comparison can
    meaningfully run. NER is the fraction of do-nothing risk remaining after the
    best possible threshold, so a rung scoring near zero is already solved: every
    configuration lands in the same sliver and any difference between them is
    noise rather than effect.
    """
    return (results.query("split == 'test'")
            .groupby("feature_set")["ner"].min()
            .rename("best_ner"))


def choose_rung(
    headroom: pd.Series,
    order: list[str],
    *,
    floor: float = 0.05,
) -> tuple[str, str]:
    """Pick the least-ablated rung that still leaves room to improve.

    Least-ablated rather than most-ablated on purpose: every feature removed is
    information a real system might legitimately have, so the honest choice is
    the mildest ablation that still poses a learning problem -- not the harshest
    one available.

    Returns the rung and the rationale, both of which get recorded so the choice
    is auditable rather than assumed.
    """
    ordered = headroom.reindex(order).dropna()
    viable = ordered[ordered >= floor]

    if viable.empty:
        return ordered.idxmax(), "no rung clears the floor; taking the most headroom"
    return viable.index[0], f"least-ablated rung with best NER >= {floor}"


#: Rungs whose feature set is comparable to the published work. Thar & Wai
#: modelled the raw balance columns without engineering the difference features,
#: so their numbers line up with ``no_error_features``. Below that we have
#: deliberately removed information they had, and the comparison stops being
#: meaningful -- scoring lower there is the intended consequence of the
#: ablation, not evidence of a bug.
COMPARABLE_RUNGS = frozenset({"full", "no_error_features"})


def benchmark_check(
    results: pd.DataFrame,
    *,
    comparable: bool = True,
    shortfall_tolerance: float = 0.08,
    ceiling: float = SUSPICIOUSLY_PERFECT,
) -> pd.DataFrame:
    """Compare baseline PR-AUC against Thar & Wai (2025) on this same dataset.

    A pipeline check, not a finding -- and deliberately **one-sided**.

    Falling well *below* a published number on identical data points at a bug:
    a broken split, a mis-specified metric, a feature that failed to build.
    Landing *above* it does not. This study engineers balance-error features
    that a solo stump already drives to AUC 0.90, so outperforming a paper that
    modelled raw columns is the expected outcome, not a red flag.

    What does warrant suspicion at the top end is near-perfection: a PR-AUC
    above ``ceiling`` at 0.129% prevalence usually means something leaked.

    A useful corollary: the *ablated* feature set, which drops the balance-error
    features, should land closer to the published values than the full set.
    That convergence is a second, sharper check on the pipeline.

    Args:
        comparable: Whether this result set was produced from a feature set the
            published work could plausibly have used. Pass False for the deeper
            ablation rungs -- there, scoring below the benchmark is the point of
            the experiment, and reporting it as a suspected bug would be noise
            dressed up as a warning. See ``COMPARABLE_RUNGS``.
    """
    baseline = results.query("config == 'A' and split == 'test'")
    observed = baseline.groupby("learner")["pr_auc"].mean()

    rows = []
    for learner, expected in C.THAR_WAI_PR_AUC.items():
        if learner not in observed.index:
            continue
        got = float(observed[learner])

        if not comparable:
            verdict, passes = "not comparable (features deliberately removed)", True
        else:
            shortfall = got < expected - shortfall_tolerance
            too_perfect = got > ceiling

            if shortfall:
                verdict = "BELOW - investigate for a pipeline bug"
            elif too_perfect:
                verdict = "SUSPICIOUSLY HIGH - investigate for leakage"
            elif got > expected:
                verdict = "above published (expected: richer features)"
            else:
                verdict = "in line with published"
            passes = not (shortfall or too_perfect)

        rows.append({
            "learner": learner,
            "observed_pr_auc": got,
            "thar_wai_pr_auc": expected,
            "difference": got - expected,
            "comparable": comparable,
            "verdict": verdict,
            "passes": passes,
        })
    return pd.DataFrame(rows)
