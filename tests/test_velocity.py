"""Tests for velocity.py -- the causal account-history features.

The whole point of this module is that it cannot leak, so most of these tests
prove specific structural guarantees (same-step rows don't see each other, no
row ever sees the future, feature values don't depend on row order or on
which split a row lands in) rather than just checking plausible-looking
numbers. A handful of hand-computed cases anchor the arithmetic itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from momo_fraud import splits as S
from momo_fraud.velocity import (
    NO_HISTORY_SENTINEL,
    VELOCITY_FEATURES,
    build_velocity_features,
    exclude_velocity,
)


def _txn(step, orig, dest):
    return {
        "step": step,
        "type": "TRANSFER",
        "amount": 100.0,
        "nameOrig": orig,
        "oldbalanceOrg": 100.0,
        "newbalanceOrig": 0.0,
        "nameDest": dest,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 100.0,
        "isFraud": 0,
        "isFlaggedFraud": 0,
    }


# --- hand-computed arithmetic -------------------------------------------------


def test_hand_computed_prior_count_and_recency():
    """C1 sends at steps 5, 5, 10 -- the module docstring's own example."""
    df = pd.DataFrame([
        _txn(5, "C1", "M1"),
        _txn(5, "C1", "M2"),
        _txn(10, "C1", "M1"),
    ])
    out = build_velocity_features(df)

    np.testing.assert_array_equal(out["vel_orig_prior_count"].to_numpy(), [0, 0, 2])
    np.testing.assert_array_equal(out["vel_orig_steps_since_last"].to_numpy(), [-1, -1, 5])
    np.testing.assert_array_equal(out["vel_orig_is_first_seen"].to_numpy(), [1, 1, 0])


def test_prior_count_is_monotonic_and_recency_tracks_last_distinct_step():
    df = pd.DataFrame([
        _txn(1, "C1", "M1"),
        _txn(5, "C1", "M2"),
        _txn(5, "C1", "M3"),
        _txn(9, "C1", "M1"),
        _txn(20, "C1", "M4"),
    ])
    out = build_velocity_features(df)

    counts = out["vel_orig_prior_count"].to_numpy()
    assert np.all(np.diff(counts) >= 0)
    np.testing.assert_array_equal(counts, [0, 1, 1, 3, 4])
    np.testing.assert_array_equal(
        out["vel_orig_steps_since_last"].to_numpy(), [-1, 4, 4, 4, 11]
    )


def test_rolling_window_excludes_events_outside_24_steps():
    df = pd.DataFrame([
        _txn(0, "C1", "M1"),
        _txn(20, "C1", "M2"),
        _txn(30, "C1", "M3"),
        _txn(50, "C1", "M4"),
    ])
    out = build_velocity_features(df)

    # At step 50 the window is (25, 49]: only the step-30 event qualifies --
    # step 20 is outside it, step 0 is far outside it.
    assert out.loc[3, "vel_orig_prior_count_24h"] == 1
    assert out.loc[3, "vel_orig_prior_count"] == 3


def test_fan_in_counts_distinct_senders_not_transactions():
    df = pd.DataFrame([
        _txn(1, "C1", "M1"),
        _txn(2, "C2", "M1"),
        _txn(3, "C1", "M1"),  # repeat sender -- must not add a new one
        _txn(4, "C3", "M1"),
    ])
    out = build_velocity_features(df)
    np.testing.assert_array_equal(out["vel_dest_unique_senders"].to_numpy(), [0, 1, 2, 2])


def test_fan_out_counts_distinct_recipients_not_transactions():
    df = pd.DataFrame([
        _txn(1, "C1", "M1"),
        _txn(2, "C1", "M2"),
        _txn(3, "C1", "M1"),  # repeat recipient -- must not add a new one
        _txn(4, "C1", "M3"),
    ])
    out = build_velocity_features(df)
    np.testing.assert_array_equal(out["vel_orig_unique_recipients"].to_numpy(), [0, 1, 2, 2])


def test_joint_factorization_keeps_one_account_code_across_roles():
    """A must be recognised as the same account whether it sends or receives.

    If nameOrig/nameDest were factorized separately, A-as-sender and
    A-as-receiver would get different integer codes and this would wrongly
    read 0.
    """
    df = pd.DataFrame([
        _txn(1, "A", "M1"),   # A acts as origin
        _txn(10, "B", "A"),   # A acts as destination
    ])
    out = build_velocity_features(df)

    assert out.loc[1, "vel_dest_prior_count"] == 1
    assert out.loc[1, "vel_dest_is_first_seen"] == 0


# --- structural leakage guarantees --------------------------------------------


def test_same_step_rows_never_see_each_other():
    df = pd.DataFrame([_txn(5, "C1", f"M{i}") for i in range(3)])
    out = build_velocity_features(df)

    assert (out["vel_orig_prior_count"] == 0).all()
    assert (out["vel_orig_is_first_seen"] == 1).all()
    assert (out["vel_orig_steps_since_last"] == NO_HISTORY_SENTINEL).all()


def test_row_order_invariance():
    """Shuffling row order must not change any row's own features."""
    rng = np.random.default_rng(0)
    accounts = [f"C{i}" for i in range(15)]
    dests = [f"M{i}" for i in range(8)]

    rows = []
    for i, step in enumerate(rng.integers(1, 100, size=200)):
        row = _txn(int(step), rng.choice(accounts), rng.choice(dests))
        row["txn_id"] = i
        rows.append(row)
    df = pd.DataFrame(rows)

    original = build_velocity_features(df)
    original["txn_id"] = df["txn_id"].to_numpy()

    shuffled_df = df.sample(frac=1.0, random_state=1).reset_index(drop=True)
    shuffled = build_velocity_features(shuffled_df)
    shuffled["txn_id"] = shuffled_df["txn_id"].to_numpy()

    original_aligned = original.sort_values("txn_id").reset_index(drop=True)
    shuffled_aligned = shuffled.sort_values("txn_id").reset_index(drop=True)

    pd.testing.assert_frame_equal(
        original_aligned[VELOCITY_FEATURES], shuffled_aligned[VELOCITY_FEATURES]
    )


def test_truncating_future_rows_does_not_change_past_features():
    """No row may ever be influenced by a transaction that happens after it."""
    rng = np.random.default_rng(2)
    accounts = [f"C{i}" for i in range(10)]
    dests = [f"M{i}" for i in range(6)]
    steps = rng.integers(1, 60, size=150)

    rows = [_txn(int(s), rng.choice(accounts), rng.choice(dests)) for s in steps]
    df = pd.DataFrame(rows).reset_index(drop=True)
    df["txn_id"] = np.arange(len(df))

    full = build_velocity_features(df)
    full["txn_id"] = df["txn_id"]

    cutoff = 30
    truncated_df = df[df["step"] <= cutoff].reset_index(drop=True)
    truncated = build_velocity_features(truncated_df)
    truncated["txn_id"] = truncated_df["txn_id"]

    merged = truncated.merge(full, on="txn_id", suffixes=("_trunc", "_full"))
    assert len(merged) == len(truncated)
    for col in VELOCITY_FEATURES:
        np.testing.assert_array_equal(
            merged[f"{col}_trunc"].to_numpy(),
            merged[f"{col}_full"].to_numpy(),
            err_msg=f"{col} changed when future rows (step > {cutoff}) were added",
        )


def test_features_are_deterministic_given_the_same_frame(paysim_like):
    """Recomputing on an identical frame must be bit-identical -- no hidden
    randomness or dependence on prior calls."""
    first = build_velocity_features(paysim_like)
    second = build_velocity_features(paysim_like)
    pd.testing.assert_frame_equal(first, second)


def test_full_frame_computation_differs_from_per_split_subset(paysim_like):
    """The reason this module runs on the full frame before any split exists:
    recomputing on a split's subset alone drops history from rows the split
    put elsewhere, silently changing causal features. This is what that costs.
    """
    df = paysim_like
    full = build_velocity_features(df)

    strat = S.stratified_split(df["isFraud"].to_numpy())
    train_only = build_velocity_features(df.iloc[strat.train].reset_index(drop=True))
    full_train_slice = full.iloc[strat.train].reset_index(drop=True)

    # Stratified split scatters rows across time, so a train-only computation
    # must disagree with the full-frame one for at least some rows: their true
    # prior history includes val/test rows a train-only pass never sees.
    assert not full_train_slice["vel_dest_prior_count"].equals(
        train_only["vel_dest_prior_count"]
    )


def test_features_do_not_depend_on_which_split_strategy_is_used(paysim_like):
    """Velocity features are computed once, before either split exists -- so
    they must be identical regardless of whether stratified or temporal
    splitting is used downstream."""
    df = paysim_like
    features = build_velocity_features(df)

    # Neither split object is even consulted by the computation; recomputing
    # after constructing both must give the exact same frame.
    S.stratified_split(df["isFraud"].to_numpy())
    S.temporal_split(df["step"], tolerance=None)  # uniform fixture steps
    features_again = build_velocity_features(df)

    pd.testing.assert_frame_equal(features, features_again)


# --- integration / smoke -------------------------------------------------------


def test_smoke_on_realistic_schema(paysim_like):
    out = build_velocity_features(paysim_like)

    assert len(out) == len(paysim_like)
    assert list(out.columns) == VELOCITY_FEATURES
    assert not out.isna().any().any()
    for col in VELOCITY_FEATURES:
        assert pd.api.types.is_numeric_dtype(out[col])

    # paysim_like reuses nameDest heavily (`M{i % 500}` over 4000 rows), so by
    # the end of the frame most destination accounts should be repeats.
    assert out["vel_dest_is_first_seen"].mean() < 0.2


def test_exclude_velocity_removes_exactly_the_velocity_columns():
    columns = ["amount", "hour"] + VELOCITY_FEATURES + ["type_TRANSFER"]
    kept = exclude_velocity(columns)
    assert set(columns) - set(kept) == set(VELOCITY_FEATURES)
