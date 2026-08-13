"""Tests for splitting.

Split hygiene is the difference between this study and the work it critiques,
so the invariants are asserted rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pytest

from momo_fraud import constants as C
from momo_fraud.splits import Split, account_overlap, stratified_split, temporal_split


def test_partitions_are_disjoint(paysim_like):
    split = stratified_split(paysim_like["isFraud"])
    split.assert_disjoint()  # raises on overlap


def test_overlap_is_rejected_at_construction():
    """A leaking split must fail loudly rather than train on its own test set."""
    with pytest.raises(ValueError, match="leaking"):
        Split(np.array([0, 1, 2]), np.array([2, 3]), np.array([4]), "broken")


def test_every_row_is_used_exactly_once(paysim_like):
    split = stratified_split(paysim_like["isFraud"])
    combined = np.concatenate([split.train, split.val, split.test])

    assert len(combined) == len(paysim_like)
    np.testing.assert_array_equal(np.sort(combined), np.arange(len(paysim_like)))


def test_proportions_match_the_proposal(paysim_like):
    """70/15/15, as specified in section 4.4."""
    split = stratified_split(paysim_like["isFraud"])
    n = len(paysim_like)

    assert len(split.train) / n == pytest.approx(C.SPLIT_TRAIN, abs=0.01)
    assert len(split.val) / n == pytest.approx(C.SPLIT_VAL, abs=0.01)
    assert len(split.test) / n == pytest.approx(C.SPLIT_TEST, abs=0.01)


def test_stratification_preserves_prevalence(paysim_like):
    """Without this, test fraud counts would swing wildly between seeds."""
    y = paysim_like["isFraud"]
    split = stratified_split(y)
    population_rate = y.mean()

    for partition_rate in split.prevalence(y).values():
        assert partition_rate == pytest.approx(population_rate, rel=0.15)


def test_split_is_deterministic_for_a_seed(paysim_like):
    a = stratified_split(paysim_like["isFraud"], seed=7)
    b = stratified_split(paysim_like["isFraud"], seed=7)
    np.testing.assert_array_equal(a.test, b.test)


def test_different_seeds_give_different_splits(paysim_like):
    a = stratified_split(paysim_like["isFraud"], seed=1)
    b = stratified_split(paysim_like["isFraud"], seed=2)
    assert not np.array_equal(a.test, b.test)


# --- Temporal ----------------------------------------------------------------


def test_temporal_split_never_trains_on_the_future(paysim_like):
    """The whole point: no training row may occur after a test row."""
    steps = paysim_like["step"].to_numpy()
    split = temporal_split(steps)

    assert steps[split.train].max() <= steps[split.val].min()
    assert steps[split.val].max() <= steps[split.test].min()


def test_temporal_split_respects_boundaries(paysim_like):
    steps = paysim_like["step"].to_numpy()
    split = temporal_split(steps, train_end=520, val_end=632)

    assert steps[split.train].max() <= 520
    assert steps[split.test].min() > 632


def test_temporal_split_is_disjoint(paysim_like):
    temporal_split(paysim_like["step"].to_numpy()).assert_disjoint()


def test_empty_temporal_partition_is_rejected(paysim_like):
    """A boundary past the end of the simulation is a configuration error."""
    with pytest.raises(ValueError, match="empty test partition"):
        temporal_split(paysim_like["step"].to_numpy(), train_end=700, val_end=744)


def test_temporal_boundaries_land_near_the_target_proportions(paysim_like):
    """Defaults were chosen for ~70/15/15; confirm they are not wildly off."""
    split = temporal_split(paysim_like["step"].to_numpy())
    n = len(paysim_like)

    assert 0.6 < len(split.train) / n < 0.8
    assert 0.05 < len(split.test) / n < 0.25


# --- Account exposure --------------------------------------------------------


def test_account_overlap_is_quantified(paysim_like):
    """PaySim reuses destination accounts; notebook 02 reports the exposure."""
    split = stratified_split(paysim_like["isFraud"])
    overlap = account_overlap(paysim_like, split)

    assert 0.0 <= overlap["fraction_test_rows_seen"] <= 1.0
    assert overlap["n_test_accounts"] > 0
