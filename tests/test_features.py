"""Tests for feature engineering.

Two properties carry the weight: nothing is fit outside training data, and the
single-transaction path produces byte-identical features to the batch path. The
second is what stops the prototype drifting away from the notebooks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from momo_fraud.features import ARTIFACT_FEATURES, FeatureBuilder


def test_cutoff_comes_from_training_data_only(paysim_like):
    """The high-risk cutoff must not move when val or test is transformed."""
    train, holdout = paysim_like.iloc[:3000], paysim_like.iloc[3000:]

    builder = FeatureBuilder().fit(train)
    cutoff_after_fit = builder.high_risk_cutoff

    builder.transform(holdout)
    assert builder.high_risk_cutoff == cutoff_after_fit
    assert cutoff_after_fit == pytest.approx(np.percentile(train["amount"], 99.0))


def test_cutoff_ignores_holdout_extremes(paysim_like):
    """An enormous holdout transaction must not shift the training cutoff."""
    train = paysim_like.iloc[:3000].copy()
    holdout = paysim_like.iloc[3000:].copy()
    holdout.loc[holdout.index[0], "amount"] = 1e12

    builder = FeatureBuilder().fit(train)
    before = builder.high_risk_cutoff
    builder.transform(holdout)

    assert builder.high_risk_cutoff == before


def test_transform_requires_fit(paysim_like):
    with pytest.raises(RuntimeError, match="must be fit"):
        FeatureBuilder().transform(paysim_like)


def test_single_row_matches_batch(paysim_like):
    """The prototype's entry point must reproduce the notebook's features."""
    builder = FeatureBuilder().fit(paysim_like)
    batch = builder.transform(paysim_like)

    for row_position in (0, 17, 999):
        record = paysim_like.iloc[row_position].to_dict()
        single = builder.transform_one(record)

        assert list(single.columns) == list(batch.columns)
        np.testing.assert_allclose(
            single.to_numpy(dtype=float),
            batch.iloc[[row_position]].to_numpy(dtype=float),
            rtol=1e-6,
        )


def test_transform_one_rejects_incomplete_transactions(paysim_like):
    builder = FeatureBuilder().fit(paysim_like)
    with pytest.raises(ValueError, match="missing required fields"):
        builder.transform_one({"step": 1, "type": "TRANSFER", "amount": 100.0})


def test_type_columns_are_fixed_regardless_of_input(paysim_like):
    """A single TRANSFER must still emit all five type columns, not just one."""
    builder = FeatureBuilder().fit(paysim_like)
    single = builder.transform_one(paysim_like.iloc[0].to_dict())

    type_columns = [c for c in single.columns if c.startswith("type_")]
    assert len(type_columns) == 5
    assert single[type_columns].to_numpy().sum() == 1


def test_error_balance_captures_the_drain_signature(paysim_like):
    """Fraud drains the origin exactly, so its balance error is ~0."""
    builder = FeatureBuilder().fit(paysim_like)
    features = builder.transform(paysim_like)

    fraud = paysim_like["isFraud"] == 1
    assert features.loc[fraud, "errorBalanceOrig"].abs().max() < 1.0


def test_ablation_drops_only_the_artifact_features(paysim_like):
    """The without-artifacts run must remove exactly those two columns."""
    full = FeatureBuilder(include_artifacts=True).fit(paysim_like)
    ablated = FeatureBuilder(include_artifacts=False).fit(paysim_like)

    removed = set(full.feature_names_) - set(ablated.feature_names_)
    assert removed == set(ARTIFACT_FEATURES)


def test_feature_order_is_stable_across_transforms(paysim_like):
    builder = FeatureBuilder().fit(paysim_like)
    first = builder.transform(paysim_like.iloc[:100])
    second = builder.transform(paysim_like.iloc[100:200])

    assert list(first.columns) == list(second.columns) == builder.feature_names_


def test_builder_round_trips_through_dict(paysim_like):
    """Persistence must preserve the fitted state exactly, for the artifact."""
    builder = FeatureBuilder().fit(paysim_like)
    restored = FeatureBuilder.from_dict(builder.to_dict())

    assert restored.high_risk_cutoff == builder.high_risk_cutoff
    assert restored.feature_names_ == builder.feature_names_

    pd.testing.assert_frame_equal(
        builder.transform(paysim_like.iloc[:50]),
        restored.transform(paysim_like.iloc[:50]),
    )


def test_severity_is_unitless_and_bounded(paysim_like):
    builder = FeatureBuilder().fit(paysim_like)
    severity = builder.severity_of(paysim_like)

    assert severity.min() >= 0.0
    assert severity.max() <= 1.0
    assert len(severity) == len(paysim_like)


def test_feature_sets_are_strictly_nested(paysim_like):
    """Each level must remove a superset of the level before it."""
    from momo_fraud.features import FEATURE_SETS, columns_for

    builder = FeatureBuilder().fit(paysim_like)
    all_columns = builder.feature_names_

    sizes = {name: len(columns_for(name, all_columns)) for name in FEATURE_SETS}
    order = ["full", "no_error_features", "no_origin_balance", "transaction_only"]

    for earlier, later in zip(order, order[1:]):
        assert sizes[later] < sizes[earlier], f"{later} should be smaller than {earlier}"
        assert set(columns_for(later, all_columns)) < set(columns_for(earlier, all_columns))


def test_transaction_only_drops_every_balance_column(paysim_like):
    """The strictest set must hold nothing derived from account balances.

    Dropping the error features alone is insufficient: a tree reconstructs
    ``amount == oldbalanceOrg`` from the raw columns and recovers the
    simulator's fraud rule anyway.
    """
    from momo_fraud.features import columns_for

    builder = FeatureBuilder().fit(paysim_like)
    kept = columns_for("transaction_only", builder.feature_names_)

    assert not any("balance" in c.lower() for c in kept)
    assert not any("Zero" in c for c in kept)
    assert "amount" in kept and "hour" in kept


def test_unknown_feature_set_is_rejected(paysim_like):
    from momo_fraud.features import columns_for

    with pytest.raises(ValueError, match="unknown feature set"):
        columns_for("nonsense", ["amount"])


def test_hour_derives_from_step(paysim_like):
    builder = FeatureBuilder().fit(paysim_like)
    features = builder.transform(paysim_like)

    np.testing.assert_array_equal(
        features["hour"].to_numpy(), (paysim_like["step"] % 24).to_numpy()
    )
    assert features["hour"].between(0, 23).all()
