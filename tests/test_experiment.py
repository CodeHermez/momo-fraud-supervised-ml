"""Tests for the experiment runner.

The properties that matter: thresholds are chosen on validation and never on
test, each fitted variant serves the configurations it should, and the shared-fit
optimisation does not change any answer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from momo_fraud import constants as C
from momo_fraud import experiment as X
from momo_fraud import models as M
from momo_fraud.features import FeatureBuilder
from momo_fraud.splits import stratified_split


@pytest.fixture
def prepared(paysim_like):
    """Feature matrices and labels for a small train/val/test split."""
    y = paysim_like[C.TARGET].to_numpy()
    split = stratified_split(y)

    builder = FeatureBuilder().fit(paysim_like.iloc[split.train])
    X_all = builder.transform(paysim_like)
    severity_all = builder.severity_of(paysim_like)

    return {
        "X_train": X_all.iloc[split.train], "y_train": y[split.train],
        "X_val": X_all.iloc[split.val], "y_val": y[split.val],
        "X_test": X_all.iloc[split.test], "y_test": y[split.test],
        "severity": {
            "train": severity_all[split.train],
            "val": severity_all[split.val],
            "test": severity_all[split.test],
        },
    }


# --- Grid structure ----------------------------------------------------------


def test_every_config_is_served_by_exactly_one_variant():
    """No configuration may be orphaned or produced twice."""
    served = [c for configs in X.FIT_VARIANTS.values() for c in configs]

    assert sorted(served) == sorted(M.CONFIG_SPECS)
    assert len(served) == len(set(served))


def test_every_config_has_a_threshold_rule():
    assert set(X.CONFIG_RULES) == set(M.CONFIG_SPECS)


def test_variants_are_fewer_than_configs():
    """The shared-fit optimisation is the point: 4 fits, not 7."""
    assert len(X.FIT_VARIANTS) < len(M.CONFIG_SPECS)


# --- Fitting -----------------------------------------------------------------


def test_fit_produces_scores_for_both_splits(prepared):
    fit = X.fit_one("decision_tree", "unweighted",
                    prepared["X_train"], prepared["y_train"],
                    prepared["X_val"], prepared["X_test"])

    assert len(fit.val_scores) == len(prepared["y_val"])
    assert len(fit.test_scores) == len(prepared["y_test"])
    assert np.all((fit.val_scores >= 0) & (fit.val_scores <= 1))
    assert fit.fit_seconds > 0


def test_rus_variant_shrinks_the_training_set(prepared):
    """Undersampling must actually undersample, at N_neg/R."""
    full = X.fit_one("decision_tree", "unweighted",
                     prepared["X_train"], prepared["y_train"],
                     prepared["X_val"], prepared["X_test"])
    rus = X.fit_one("decision_tree", "rus",
                    prepared["X_train"], prepared["y_train"],
                    prepared["X_val"], prepared["X_test"], r=50.0)

    assert rus.n_train < full.n_train


def test_instance_weighted_requires_severity(prepared):
    with pytest.raises(ValueError, match="requires per-instance severity"):
        X.fit_one("decision_tree", "instance_weighted",
                  prepared["X_train"], prepared["y_train"],
                  prepared["X_val"], prepared["X_test"])


def test_instance_weighting_routes_through_a_pipeline(prepared):
    """Logistic regression is wrapped in a Pipeline; weights must still reach it."""
    fit = X.fit_one("logistic_regression", "instance_weighted",
                    prepared["X_train"], prepared["y_train"],
                    prepared["X_val"], prepared["X_test"],
                    severity_train=prepared["severity"]["train"])

    assert len(fit.val_scores) == len(prepared["y_val"])


# --- Evaluation --------------------------------------------------------------


def test_one_variant_yields_all_its_configs(prepared):
    fit = X.fit_one("decision_tree", "unweighted",
                    prepared["X_train"], prepared["y_train"],
                    prepared["X_val"], prepared["X_test"])
    rows = X.evaluate_configs(fit, prepared["y_val"], prepared["y_test"])

    configs = {r["config"] for r in rows}
    assert configs == {"A", "B", "C"}
    assert {r["split"] for r in rows} == {"val", "test"}


def test_baseline_config_uses_the_default_threshold(prepared):
    fit = X.fit_one("decision_tree", "unweighted",
                    prepared["X_train"], prepared["y_train"],
                    prepared["X_val"], prepared["X_test"])
    rows = X.evaluate_configs(fit, prepared["y_val"], prepared["y_test"])

    config_a = next(r for r in rows if r["config"] == "A")
    assert config_a["threshold"] == pytest.approx(0.5)


def test_threshold_is_selected_on_validation_not_test(prepared):
    """The central discipline: val and test must share one frozen threshold."""
    fit = X.fit_one("xgboost", "unweighted",
                    prepared["X_train"], prepared["y_train"],
                    prepared["X_val"], prepared["X_test"])
    rows = X.evaluate_configs(fit, prepared["y_val"], prepared["y_test"])

    for config in ("A", "B", "C"):
        pair = [r for r in rows if r["config"] == config]
        assert pair[0]["threshold"] == pytest.approx(pair[1]["threshold"])


def test_ner_is_always_class_constant(prepared):
    """`ner` must mean the same thing everywhere, severity supplied or not.

    The severity-weighted measure is reported alongside under its own name.
    Letting it occupy the `ner` key is how one model acquires two different
    "NER" values in two notebooks.
    """
    fit = X.fit_one("decision_tree", "unweighted",
                    prepared["X_train"], prepared["y_train"],
                    prepared["X_val"], prepared["X_test"])

    plain = X.evaluate_configs(fit, prepared["y_val"], prepared["y_test"])
    weighted = X.evaluate_configs(
        fit, prepared["y_val"], prepared["y_test"],
        severity_val=prepared["severity"]["val"],
        severity_test=prepared["severity"]["test"],
    )

    assert [r["ner"] for r in plain] == [r["ner"] for r in weighted]
    assert "ner_severity_weighted" not in plain[0]
    assert "ner_severity_weighted" in weighted[0]


def test_threshold_selection_ignores_severity(prepared):
    """Severity must not leak into threshold choice for configs that disown it.

    Config F applies severity during *training*. If it also reached threshold
    selection, every config would quietly become severity-weighted and config C
    would stop being the class-constant rule the design calls for.
    """
    fit = X.fit_one("xgboost", "unweighted",
                    prepared["X_train"], prepared["y_train"],
                    prepared["X_val"], prepared["X_test"])

    without = X.evaluate_configs(fit, prepared["y_val"], prepared["y_test"])
    with_severity = X.evaluate_configs(
        fit, prepared["y_val"], prepared["y_test"],
        severity_val=prepared["severity"]["val"],
        severity_test=prepared["severity"]["test"],
    )

    thresholds_without = {r["config"]: r["threshold"] for r in without}
    thresholds_with = {r["config"]: r["threshold"] for r in with_severity}
    assert thresholds_without == thresholds_with


def test_ner_optimal_beats_default_threshold_on_validation(prepared):
    """Choosing the cut for risk must not be worse than leaving it at 0.5."""
    fit = X.fit_one("xgboost", "unweighted",
                    prepared["X_train"], prepared["y_train"],
                    prepared["X_val"], prepared["X_test"])
    rows = X.evaluate_configs(fit, prepared["y_val"], prepared["y_test"])

    val = {r["config"]: r["ner"] for r in rows if r["split"] == "val"}
    assert val["C"] <= val["A"] + 1e-9


def test_threshold_free_metrics_are_identical_across_configs(prepared):
    """A, B and C share a model, so PR-AUC cannot differ between them."""
    fit = X.fit_one("decision_tree", "unweighted",
                    prepared["X_train"], prepared["y_train"],
                    prepared["X_val"], prepared["X_test"])
    rows = X.evaluate_configs(fit, prepared["y_val"], prepared["y_test"])

    test_rows = [r for r in rows if r["split"] == "test"]
    assert len({round(r["pr_auc"], 12) for r in test_rows}) == 1


# --- Full grid ---------------------------------------------------------------


def test_run_grid_returns_a_row_per_config_and_split(prepared):
    results = X.run_grid(
        prepared["X_train"], prepared["y_train"],
        prepared["X_val"], prepared["y_val"],
        prepared["X_test"], prepared["y_test"],
        learners=["decision_tree"], seeds=[0],
        severity=prepared["severity"], cache=False, verbose=False,
    )

    assert set(results["config"]) == set(M.CONFIG_SPECS)
    assert len(results) == len(M.CONFIG_SPECS) * 2  # val + test


# --- Rung selection ----------------------------------------------------------

RUNG_ORDER = ["full", "no_error_features", "no_origin_balance", "transaction_only"]


def _headroom(values: dict[str, float]) -> pd.Series:
    return pd.Series(values, name="best_ner")


def test_solved_rungs_are_skipped():
    """A rung with no risk left cannot host the experiment."""
    chosen, why = X.choose_rung(
        _headroom({"full": 0.002, "no_error_features": 0.01,
                   "no_origin_balance": 0.31, "transaction_only": 0.62}),
        RUNG_ORDER,
    )
    assert chosen == "no_origin_balance"
    assert "least-ablated" in why


def test_the_mildest_viable_rung_wins():
    """Not the harshest ablation -- the mildest one that still poses a problem."""
    chosen, _ = X.choose_rung(
        _headroom({"full": 0.40, "no_error_features": 0.55,
                   "no_origin_balance": 0.70, "transaction_only": 0.90}),
        RUNG_ORDER,
    )
    assert chosen == "full"


def test_fallback_when_nothing_clears_the_floor():
    """If every rung is solved, take the most headroom and say so."""
    chosen, why = X.choose_rung(
        _headroom({"full": 0.001, "no_error_features": 0.004,
                   "no_origin_balance": 0.02, "transaction_only": 0.03}),
        RUNG_ORDER,
    )
    assert chosen == "transaction_only"
    assert "no rung clears" in why


def test_headroom_takes_the_best_across_learners(prepared):
    """Headroom is what the *best* model achieves, not the average."""
    results = X.run_grid(
        prepared["X_train"], prepared["y_train"],
        prepared["X_val"], prepared["y_val"],
        prepared["X_test"], prepared["y_test"],
        learners=["decision_tree", "logistic_regression"],
        variants=["unweighted"], seeds=[0],
        severity=prepared["severity"], cache=False, verbose=False,
    )
    results["feature_set"] = "full"

    headroom = X.headroom_by_rung(results)
    assert headroom["full"] == pytest.approx(results.query("split == 'test'")["ner"].min())
    assert 0.0 <= headroom["full"] <= 1.0


def _check(pr_aucs: dict[str, float]) -> pd.DataFrame:
    rows = [{"learner": k, "config": "A", "split": "test", "pr_auc": v}
            for k, v in pr_aucs.items()]
    return X.benchmark_check(pd.DataFrame(rows)).set_index("learner")


def test_benchmark_flags_underperformance_as_a_bug():
    """Falling well below a published number on identical data means a bug."""
    check = _check({"random_forest": 0.10})
    assert not bool(check.loc["random_forest", "passes"])
    assert "BELOW" in check.loc["random_forest", "verdict"]


def test_benchmark_accepts_matching_results():
    check = _check({"xgboost": 0.92})
    assert bool(check.loc["xgboost", "passes"])


def test_benchmark_accepts_outperformance():
    """Richer features than the published study should beat it, not fail a gate."""
    check = _check({"logistic_regression": 0.8160})
    assert bool(check.loc["logistic_regression", "passes"])
    assert "above published" in check.loc["logistic_regression", "verdict"]


def test_ablated_rungs_are_not_judged_against_the_benchmark():
    """Below the benchmark is the *point* once features are removed on purpose.

    Reporting it as a suspected bug would bury the real signal under warnings
    that all mean 'the ablation worked'.
    """
    rows = [{"learner": "xgboost", "config": "A", "split": "test", "pr_auc": 0.57}]
    check = X.benchmark_check(pd.DataFrame(rows), comparable=False).set_index("learner")

    assert bool(check.loc["xgboost", "passes"])
    assert "not comparable" in check.loc["xgboost", "verdict"]


def test_the_same_score_is_a_bug_when_features_were_comparable():
    """The identical number must still be flagged where the comparison holds."""
    rows = [{"learner": "xgboost", "config": "A", "split": "test", "pr_auc": 0.57}]
    check = X.benchmark_check(pd.DataFrame(rows), comparable=True).set_index("learner")

    assert not bool(check.loc["xgboost", "passes"])
    assert "BELOW" in check.loc["xgboost", "verdict"]


def test_comparable_rungs_are_the_undeep_ones():
    from momo_fraud.features import FEATURE_SETS

    assert X.COMPARABLE_RUNGS <= set(FEATURE_SETS)
    assert "no_origin_balance" not in X.COMPARABLE_RUNGS


def test_benchmark_flags_near_perfection_as_leakage():
    """At 0.129% prevalence, a PR-AUC of 0.999 is a symptom, not an achievement."""
    check = _check({"xgboost": 0.999})
    assert not bool(check.loc["xgboost", "passes"])
    assert "SUSPICIOUSLY HIGH" in check.loc["xgboost", "verdict"]
