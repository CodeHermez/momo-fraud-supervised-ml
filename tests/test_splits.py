"""Tests for splitting.

Split hygiene is the difference between this study and the work it critiques,
so the invariants are asserted rather than assumed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from momo_fraud import constants as C
from momo_fraud.splits import (
    Split,
    account_overlap,
    derive_temporal_boundaries,
    stratified_split,
    temporal_split,
)

REAL_PAYSIM = Path(__file__).resolve().parents[1] / "data" / "processed" / "paysim.parquet"
REAL_PAYSIM_CSV = (Path(__file__).resolve().parents[1] / "data" / "raw" / C.RAW_FILENAME)


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
    """The whole point: no training row may occur after a test row.

    ``tolerance=None`` because this fixture's steps are uniform: the shipped
    boundaries are derived from the real file's front-loaded shape and do not
    hit 70/15/15 here. Chronological ordering is what is under test, not the
    proportions -- those have their own test on a representative distribution.
    """
    steps = paysim_like["step"].to_numpy()
    split = temporal_split(steps, tolerance=None)

    assert steps[split.train].max() <= steps[split.val].min()
    assert steps[split.val].max() <= steps[split.test].min()


def test_temporal_split_respects_boundaries(paysim_like):
    steps = paysim_like["step"].to_numpy()
    split = temporal_split(steps, train_end=520, val_end=632, tolerance=None)

    assert steps[split.train].max() <= 520
    assert steps[split.test].min() > 632


def test_temporal_split_is_disjoint(paysim_like):
    temporal_split(paysim_like["step"].to_numpy(), tolerance=None).assert_disjoint()


def test_empty_temporal_partition_is_rejected(paysim_like):
    """A boundary past the end of the simulation is a configuration error."""
    with pytest.raises(ValueError, match="empty test partition"):
        temporal_split(paysim_like["step"].to_numpy(), train_end=700, val_end=744,
                       tolerance=None)


def test_temporal_boundaries_land_near_the_target_proportions(front_loaded_steps):
    """Defaults must hit ~70/15/15 on a *front-loaded* step distribution.

    The previous version of this test ran on ``paysim_like``, whose steps are
    drawn uniformly. Under a uniform distribution almost any pair of boundaries
    near the middle lands near the target, so it passed while the real
    boundaries realised 95.6 / 3.0 / 1.4. The fixture here reproduces the shape
    that actually broke them.
    """
    train_end, val_end = derive_temporal_boundaries(front_loaded_steps)
    split = temporal_split(front_loaded_steps, train_end=train_end, val_end=val_end)
    n = len(front_loaded_steps)

    assert split.sizes()["train"] / n == pytest.approx(C.SPLIT_TRAIN, abs=0.03)
    assert split.sizes()["val"] / n == pytest.approx(C.SPLIT_VAL, abs=0.03)
    assert split.sizes()["test"] / n == pytest.approx(C.SPLIT_TEST, abs=0.03)


def test_lopsided_boundaries_are_rejected(front_loaded_steps):
    """The failure that went unnoticed must now be loud.

    Boundaries deep in the tail of a front-loaded distribution put almost every
    row in train. That is exactly what 520/632 did to the real file, and it has
    to raise rather than return a Split.
    """
    high = int(np.quantile(front_loaded_steps, 0.999))
    with pytest.raises(ValueError, match="outside the .* tolerance"):
        temporal_split(front_loaded_steps, train_end=high - 1, val_end=high)


def test_tolerance_can_be_waived_deliberately(front_loaded_steps):
    """Exploring other boundaries stays possible, but must be explicit."""
    high = int(np.quantile(front_loaded_steps, 0.999))
    split = temporal_split(front_loaded_steps, train_end=high - 1, val_end=high,
                           tolerance=None)
    assert split.strategy == "temporal"


def test_derived_boundaries_respect_the_cumulative_count(front_loaded_steps):
    """Each boundary is the largest step whose cumulative share clears the target."""
    train_end, val_end = derive_temporal_boundaries(front_loaded_steps)

    assert train_end < val_end
    assert (front_loaded_steps <= train_end).mean() <= C.SPLIT_TRAIN
    assert (front_loaded_steps <= val_end).mean() <= C.SPLIT_TRAIN + C.SPLIT_VAL


@pytest.mark.skipif(not REAL_PAYSIM.exists() and not REAL_PAYSIM_CSV.exists(),
                    reason="canonical PaySim file not present")
def test_shipped_boundaries_hold_on_the_real_dataset():
    """The constants must be checked against the file they were derived from.

    This is the test whose absence let 520/632 through. It is skipped when the
    470 MB file is not present, but it is the only one that can speak to the
    numbers actually shipped in ``constants.py``.
    """
    from momo_fraud import data as D

    steps = D.load()["step"].to_numpy()
    split = temporal_split(steps)  # raises if the tolerance is breached
    n = len(steps)

    assert split.sizes()["train"] / n == pytest.approx(C.SPLIT_TRAIN, abs=0.03)
    assert split.sizes()["val"] / n == pytest.approx(C.SPLIT_VAL, abs=0.03)
    assert split.sizes()["test"] / n == pytest.approx(C.SPLIT_TEST, abs=0.03)
    assert (C.TEMPORAL_TRAIN_END, C.TEMPORAL_VAL_END) == derive_temporal_boundaries(steps)


@pytest.mark.skipif(not REAL_PAYSIM.exists() and not REAL_PAYSIM_CSV.exists(),
                    reason="canonical PaySim file not present")
def test_temporal_prevalence_is_non_stationary_on_the_real_dataset():
    """Hitting the row proportions does not equalise the class prior.

    Pinned deliberately: it is the reason the temporal arm cannot reuse
    ``R_DEFAULT`` and expect NER's anchor to hold, and a future change that
    quietly made the priors match would invalidate that reasoning.
    """
    from momo_fraud import data as D

    df = D.load()
    y = df[C.TARGET].to_numpy()
    split = temporal_split(df["step"].to_numpy())

    ratios = split.class_ratio(y)
    assert ratios["test"] < C.R_DEFAULT / 2
    assert split.prevalence(y)["test"] > split.prevalence(y)["train"] * 2


def test_class_ratio_matches_the_population_on_a_stratified_split(paysim_like):
    """Stratification is what makes the default cost ratio applicable."""
    y = paysim_like["isFraud"].to_numpy()
    split = stratified_split(y)
    population = (len(y) - y.sum()) / y.sum()

    for partition, ratio in split.class_ratio(y).items():
        assert ratio == pytest.approx(population, rel=0.15), partition


def test_class_ratio_needs_positives():
    split = Split(np.array([0, 1]), np.array([2]), np.array([3]), "manual")
    with pytest.raises(ValueError, match="no positives"):
        split.class_ratio(np.array([1, 1, 1, 0]))


# --- Account exposure --------------------------------------------------------


def test_account_overlap_is_quantified(paysim_like):
    """PaySim reuses destination accounts; notebook 02 reports the exposure."""
    split = stratified_split(paysim_like["isFraud"])
    overlap = account_overlap(paysim_like, split)

    assert 0.0 <= overlap["fraction_test_rows_seen"] <= 1.0
    assert overlap["n_test_accounts"] > 0
