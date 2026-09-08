"""Tests for the repairs made after the methodology audit.

Each test here pins a defect the audit found, so a regression re-introduces a
known-bad procedure loudly rather than quietly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from momo_fraud import evaluate as E
from momo_fraud import experiment as X


# --- paired bootstrap (audit M1) ---------------------------------------------


@pytest.fixture
def paired_scores():
    """One clearly better ranker and one clearly worse, on the same labels."""
    rng = np.random.default_rng(0)
    n, n_pos = 20_000, 200
    y = np.zeros(n, dtype=int)
    y[rng.choice(n, n_pos, replace=False)] = 1

    strong = rng.uniform(0, 1, n) + y * 0.8       # separates well
    weak = rng.uniform(0, 1, n) + y * 0.15        # separates barely
    return y, strong, weak


def test_paired_delta_recovers_the_point_estimate(paired_scores):
    y, strong, weak = paired_scores
    out = E.paired_bootstrap_delta(y, strong, weak, average_precision_score, n_boot=200)

    expected = (average_precision_score(y, strong)
                - average_precision_score(y, weak))
    assert out["delta"] == pytest.approx(expected)


def test_paired_delta_detects_a_real_difference(paired_scores):
    y, strong, weak = paired_scores
    out = E.paired_bootstrap_delta(y, strong, weak, average_precision_score, n_boot=300)

    assert out["delta"] > 0
    assert out["excludes_zero"]
    assert out["ci_low"] > 0
    assert out["prop_crossing_zero"] == 0.0


def test_identical_models_produce_a_zero_delta(paired_scores):
    """The pairing must cancel exactly when both vectors are the same.

    This is the property the old marginal-CI procedure could not express: two
    identical models have overlapping marginal intervals *and* a delta that is
    identically zero, and only the second is informative.
    """
    y, strong, _ = paired_scores
    out = E.paired_bootstrap_delta(y, strong, strong, average_precision_score, n_boot=100)

    assert out["delta"] == 0.0
    assert out["ci_low"] == 0.0 and out["ci_high"] == 0.0
    assert not out["excludes_zero"]


def test_paired_delta_is_deterministic_for_a_seed(paired_scores):
    y, strong, weak = paired_scores
    kwargs = dict(n_boot=100, seed=3)
    a = E.paired_bootstrap_delta(y, strong, weak, average_precision_score, **kwargs)
    b = E.paired_bootstrap_delta(y, strong, weak, average_precision_score, **kwargs)
    assert a == b


def test_paired_delta_is_tighter_than_marginal_intervals(paired_scores):
    """Why the repair matters: pairing removes the shared observation noise.

    Two models scored on the same rows share most of their sampling variation.
    The paired interval on the difference must therefore be narrower than the
    sum of the two marginal interval widths -- which is what the old
    overlap-based rule was implicitly comparing against.
    """
    y, strong, weak = paired_scores
    paired = E.paired_bootstrap_delta(y, strong, weak, average_precision_score, n_boot=300)

    _, a_lo, a_hi = E.bootstrap_ci(y, strong, average_precision_score, n_boot=300)
    _, b_lo, b_hi = E.bootstrap_ci(y, weak, average_precision_score, n_boot=300)

    assert (paired["ci_high"] - paired["ci_low"]) < (a_hi - a_lo) + (b_hi - b_lo)


# --- validation-based selection (audit C3) -----------------------------------


@pytest.fixture
def rung_results():
    """Two rungs where validation and test disagree about which is viable."""
    return pd.DataFrame([
        {"feature_set": "full", "split": "val", "ner": 0.01, "learner": "a", "config": "C",
         "level": "decision", "threshold_rule": "ner_optimal"},
        {"feature_set": "full", "split": "test", "ner": 0.20, "learner": "a", "config": "C",
         "level": "decision", "threshold_rule": "ner_optimal"},
        {"feature_set": "ablated", "split": "val", "ner": 0.30, "learner": "a", "config": "C",
         "level": "decision", "threshold_rule": "ner_optimal"},
        {"feature_set": "ablated", "split": "test", "ner": 0.02, "learner": "a", "config": "C",
         "level": "decision", "threshold_rule": "ner_optimal"},
    ])


def test_headroom_defaults_to_validation(rung_results):
    """The default must be the safe split, not the one that was hard-coded."""
    headroom = X.headroom_by_rung(rung_results)
    assert headroom["full"] == 0.01 and headroom["ablated"] == 0.30


def test_headroom_can_still_report_on_test(rung_results):
    headroom = X.headroom_by_rung(rung_results, split="test")
    assert headroom["full"] == 0.20 and headroom["ablated"] == 0.02


def test_headroom_rejects_an_unknown_split(rung_results):
    with pytest.raises(ValueError, match="unknown split"):
        X.headroom_by_rung(rung_results, split="holdout")


def test_selection_split_changes_the_rung(rung_results):
    """Pins the defect: on this frame the two splits choose opposite rungs."""
    order = ["full", "ablated"]
    on_val, _ = X.choose_rung(X.headroom_by_rung(rung_results), order)
    on_test, _ = X.choose_rung(X.headroom_by_rung(rung_results, split="test"), order)

    assert on_val == "ablated"
    assert on_test == "full"


def test_best_config_is_selected_on_validation():
    results = pd.DataFrame([
        {"learner": "x", "config": "A", "split": "val", "ner": 0.9,
         "level": "none", "threshold_rule": "fixed"},
        {"learner": "x", "config": "A", "split": "test", "ner": 0.1,
         "level": "none", "threshold_rule": "fixed"},
        {"learner": "x", "config": "C", "split": "val", "ner": 0.2,
         "level": "decision", "threshold_rule": "ner_optimal"},
        {"learner": "x", "config": "C", "split": "test", "ner": 0.8,
         "level": "decision", "threshold_rule": "ner_optimal"},
    ])
    best = X.select_best_config(results)

    assert list(best["config"]) == ["C"]          # val winner, not the test winner
    assert "ner_val" in best.columns
    assert best["ner_val"].iloc[0] == 0.2


def test_best_config_rejects_a_split_with_no_rows():
    results = pd.DataFrame([
        {"learner": "x", "config": "A", "split": "val", "ner": 0.9,
         "level": "none", "threshold_rule": "fixed"},
    ])
    with pytest.raises(ValueError, match="no rows"):
        X.select_best_config(results, on="test")


# --- fast PR-AUC path must agree with the generic one ------------------------


def test_fast_pr_auc_path_agrees_with_the_generic_bootstrap(paired_scores):
    """The optimisation must not change the answer, only the runtime.

    Both paths draw from the same seeded generator in the same order, so the
    replicates are identical and the deltas should match to floating-point
    tie-handling noise.
    """
    y, strong, weak = paired_scores
    fast = E.paired_bootstrap_pr_auc(y, strong, weak, n_boot=100, seed=11)
    slow = E.paired_bootstrap_delta(y, strong, weak, average_precision_score,
                                    n_boot=100, seed=11)

    assert fast["delta"] == pytest.approx(slow["delta"], abs=1e-6)
    assert fast["ci_low"] == pytest.approx(slow["ci_low"], abs=1e-6)
    assert fast["ci_high"] == pytest.approx(slow["ci_high"], abs=1e-6)
    assert fast["excludes_zero"] == slow["excludes_zero"]


def test_fast_pr_auc_point_estimate_matches_sklearn(paired_scores):
    y, strong, weak = paired_scores
    out = E.paired_bootstrap_pr_auc(y, strong, weak, n_boot=10)
    expected = average_precision_score(y, strong) - average_precision_score(y, weak)
    assert out["delta"] == pytest.approx(expected, abs=1e-6)


def test_fast_pr_auc_is_zero_for_identical_vectors(paired_scores):
    y, strong, _ = paired_scores
    out = E.paired_bootstrap_pr_auc(y, strong, strong, n_boot=50)
    assert out["delta"] == 0.0 and not out["excludes_zero"]
