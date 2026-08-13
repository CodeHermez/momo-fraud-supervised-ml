"""Tests for the risk framework.

These guard the properties the whole evaluation rests on. If the NER baseline
or the threshold identity breaks, every number in the report moves.
"""

from __future__ import annotations

import numpy as np
import pytest

from momo_fraud import constants as C
from momo_fraud import risk as R


# --- NER anchors -------------------------------------------------------------


def test_flagging_nothing_scores_exactly_one():
    """NER is normalised by the do-nothing baseline, so it must be exactly 1.0."""
    y = np.array([0, 0, 1, 0, 1, 0, 0, 0])
    assert R.normalized_expected_risk(y, np.zeros_like(y), r=100.0) == pytest.approx(1.0)


def test_flagging_everything_matches_closed_form():
    y = np.array([0, 0, 1, 0, 1, 0, 0, 0])
    n_legit, n_fraud, r = 6, 2, 100.0

    ner = R.normalized_expected_risk(y, np.ones_like(y), r=r)
    assert ner == pytest.approx(n_legit / (n_fraud * r))


def test_trivial_strategies_break_even_at_recommended_ratio():
    """At R = N_neg/N_pos, flag-everything and flag-nothing both score 1.0.

    That is what makes the recommended ratio a meaningful anchor: any NER below
    1.0 is genuine skill, not an artifact of a lenient cost setting.
    """
    y = np.concatenate([np.zeros(900), np.ones(100)]).astype(int)
    r = 900 / 100

    assert R.normalized_expected_risk(y, np.zeros_like(y), r) == pytest.approx(1.0)
    assert R.normalized_expected_risk(y, np.ones_like(y), r) == pytest.approx(1.0)


def test_ner_undefined_without_positives():
    with pytest.raises(ValueError, match="No positive samples"):
        R.normalized_expected_risk(np.zeros(10, dtype=int), np.zeros(10, dtype=int), r=10.0)


def test_perfect_predictions_score_zero():
    y = np.array([0, 1, 0, 1, 1, 0])
    assert R.normalized_expected_risk(y, y, r=50.0) == pytest.approx(0.0)


# --- Threshold identity ------------------------------------------------------


def test_bayes_threshold_formula():
    assert R.bayes_threshold(1.0) == pytest.approx(0.5)
    assert R.bayes_threshold(9.0) == pytest.approx(0.1)
    assert R.bayes_threshold(C.R_DEFAULT) == pytest.approx(C.PREVALENCE, rel=1e-3)


def test_recommended_ratio_threshold_equals_class_prior():
    """Johnson & Khoshgoftaar's theory: at R = N_neg/N_pos, lambda is the prior."""
    assert R.bayes_threshold(C.R_DEFAULT) == pytest.approx(C.N_FRAUD / C.N_ROWS, rel=1e-9)


def test_empirical_optimum_recovers_the_bayes_threshold():
    """On calibrated scores the NER-minimising cut must land near 1/(1+R).

    This is verification item 6 in the plan: it ties the empirical optimiser to
    the closed form, so a divergence localises either a bug or a miscalibrated
    model rather than being silently absorbed.
    """
    rng = np.random.default_rng(0)
    n, r = 200_000, 9.0

    scores = rng.uniform(0, 1, n)  # calibrated by construction
    y = (rng.uniform(0, 1, n) < scores).astype(int)

    found = R.optimal_threshold(y, scores, r).threshold
    assert found == pytest.approx(R.bayes_threshold(r), abs=0.02)


def test_ner_optimal_equals_youden_at_the_class_ratio():
    """At R = N_neg/N_pos the two criteria are provably the same criterion.

    Minimising FP + R*FN reduces to maximising TPR - FPR at that ratio, so a
    "risk-optimal versus ROC-optimal" comparison carries no information there.
    Guarding it in a test stops the equivalence being rediscovered as a bug, or
    reported as a finding.
    """
    rng = np.random.default_rng(4)
    n = 60_000
    scores = rng.uniform(0, 1, n)
    y = (rng.uniform(0, 1, n) < scores * 0.02).astype(int)

    r_class = (len(y) - y.sum()) / y.sum()
    assert R.optimal_threshold(y, scores, r_class).threshold == pytest.approx(
        R.youden_threshold(y, scores))


def test_the_two_criteria_diverge_away_from_the_class_ratio():
    """Off that ratio they must differ, or the sweep would be meaningless."""
    rng = np.random.default_rng(4)
    n = 60_000
    scores = rng.uniform(0, 1, n)
    y = (rng.uniform(0, 1, n) < scores * 0.02).astype(int)

    r_class = (len(y) - y.sum()) / y.sum()
    youden = R.youden_threshold(y, scores)

    assert R.optimal_threshold(y, scores, r_class / 50).threshold > youden
    assert R.optimal_threshold(y, scores, r_class * 50).threshold < youden


def test_severity_weighting_breaks_the_equivalence():
    """Per-instance severity is what makes the comparison informative again."""
    rng = np.random.default_rng(4)
    n = 60_000
    scores = rng.uniform(0, 1, n)
    y = (rng.uniform(0, 1, n) < scores * 0.02).astype(int)
    severity = rng.uniform(0, 1, n)

    r_class = (len(y) - y.sum()) / y.sum()
    weighted = R.optimal_threshold(y, scores, r_class, severity).threshold
    assert weighted != pytest.approx(R.youden_threshold(y, scores))


def test_optimal_threshold_matches_brute_force():
    """The cumulative scan must agree with an exhaustive sweep."""
    rng = np.random.default_rng(7)
    n, r = 400, 20.0

    scores = rng.uniform(0, 1, n)
    y = (rng.uniform(0, 1, n) < scores * 0.4).astype(int)

    best = R.optimal_threshold(y, scores, r)
    brute = min(
        R.normalized_expected_risk(y, scores >= t, r)
        for t in np.unique(scores)
    )
    assert best.ner == pytest.approx(min(brute, 1.0), abs=1e-9)


def test_optimal_threshold_never_worse_than_doing_nothing():
    """Flag-nothing is always an available fallback, so NER cannot exceed 1.0."""
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 500)
    scores = rng.uniform(0, 1, 500)  # deliberately uninformative

    assert R.optimal_threshold(y, scores, r=2.0).ner <= 1.0 + 1e-12


def test_tied_scores_do_not_split_a_group():
    """A threshold cannot cut between two identical scores."""
    y = np.array([1, 0, 1, 0, 1, 0])
    scores = np.array([0.5, 0.5, 0.5, 0.2, 0.2, 0.2])

    result = R.optimal_threshold(y, scores, r=10.0)
    n_flagged_at_threshold = int((scores >= result.threshold).sum())
    assert n_flagged_at_threshold == result.n_flagged


# --- Per-instance severity ---------------------------------------------------


def test_per_transaction_threshold_lowers_the_bar_for_large_amounts():
    """Higher severity must produce a lower threshold, never a higher one."""
    severity = np.array([0.1, 0.5, 0.9])
    thresholds = R.per_transaction_thresholds(severity, r=100.0)

    assert np.all(np.diff(thresholds) < 0)
    assert thresholds[1] == pytest.approx(1.0 / (1.0 + 100.0 * 0.5))


def test_severity_scaler_is_monotone_and_bounded():
    rng = np.random.default_rng(11)
    amounts = rng.lognormal(8, 2, 50_000)

    scaler = R.SeverityScaler().fit(amounts)
    probe = np.array([amounts.min(), np.median(amounts), amounts.max()])
    severity = scaler.transform(probe)

    assert np.all((severity >= 0) & (severity <= 1))
    assert np.all(np.diff(severity) > 0)
    assert severity[1] == pytest.approx(0.5, abs=0.01)


def test_severity_scaler_round_trips():
    scaler = R.SeverityScaler().fit(np.arange(1000, dtype=float))
    restored = R.SeverityScaler.from_dict(scaler.to_dict())

    probe = np.array([10.0, 500.0, 990.0])
    np.testing.assert_allclose(scaler.transform(probe), restored.transform(probe))


def test_severity_weighted_ner_penalises_missing_large_transactions():
    """Missing a high-severity fraud must cost more than missing a small one."""
    y = np.array([0, 1, 1])
    severity = np.array([0.5, 0.1, 0.9])

    miss_small = R.total_risk(y, np.array([0, 0, 1]), r=100.0, severity=severity)
    miss_large = R.total_risk(y, np.array([0, 1, 0]), r=100.0, severity=severity)
    assert miss_large > miss_small


# --- Data-level expression of the same ratio ---------------------------------


def test_undersample_size_follows_the_cost_matrix():
    """N'_neg = N_neg / R -- Johnson & Khoshgoftaar Eq. 1."""
    assert R.undersample_size(1_000_000, 100.0) == 10_000
    assert R.undersample_size(1_000, 1.0) == 1_000
    assert R.undersample_size(10, 1000.0) == 1  # never empties the majority class


# --- Operational metrics -----------------------------------------------------


def test_recall_at_budget_counts_the_top_scores():
    y = np.array([1, 0, 1, 0, 0, 1])
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.05])

    assert R.recall_at_budget(y, scores, 1) == pytest.approx(1 / 3)
    assert R.recall_at_budget(y, scores, 3) == pytest.approx(2 / 3)
    assert R.recall_at_budget(y, scores, 6) == pytest.approx(1.0)


def test_precision_at_budget():
    y = np.array([1, 0, 1, 0, 0, 1])
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.05])

    assert R.precision_at_budget(y, scores, 2) == pytest.approx(0.5)
    assert R.precision_at_budget(y, scores, 3) == pytest.approx(2 / 3)


def test_budget_larger_than_dataset_is_clamped():
    y = np.array([1, 0, 1])
    scores = np.array([0.9, 0.5, 0.1])
    assert R.recall_at_budget(y, scores, 1_000) == pytest.approx(1.0)


# --- Bands -------------------------------------------------------------------


def test_bands_are_ordered_and_anchored_on_lambda():
    cutoffs = R.band_cutoffs(r=99.0)  # lambda = 0.01 -> 1.0 on the 0-100 scale

    assert cutoffs["Medium"] < cutoffs["High"] < cutoffs["Critical"]
    assert cutoffs["High"] == pytest.approx(1.0)


def test_band_assignment_covers_every_level():
    cutoffs = R.band_cutoffs(r=99.0)
    scores = np.array([0.01, 0.5, 5.0, 50.0])

    bands = R.assign_bands(scores, cutoffs)
    assert list(bands) == ["Low", "Medium", "High", "Critical"]


def test_risk_score_maps_probability_to_0_100():
    np.testing.assert_allclose(R.risk_score([0.0, 0.5, 1.0]), [0.0, 50.0, 100.0])
