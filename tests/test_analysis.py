"""Tests for the EDA and leakage-audit computations.

The audit decides which features are admissible, so it is tested rather than
trusted. The fixture reproduces PaySim's fraud mechanism, which means the
artifact detectors should fire on it exactly as they will on the real data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from momo_fraud import analysis as A
from momo_fraud import constants as C
from momo_fraud.features import FeatureBuilder


# --- Descriptive -------------------------------------------------------------


def test_fraud_rate_by_type_covers_all_types(paysim_like):
    table = A.fraud_rate_by_type(paysim_like)

    assert set(table["type"]) == set(C.TRANSACTION_TYPES)
    assert table["n_transactions"].sum() == len(paysim_like)
    assert table["n_fraud"].sum() == paysim_like[C.TARGET].sum()


def test_fraud_concentrates_in_transfer_and_cash_out(paysim_like):
    """The fixture models the real property: fraud lives in a subset of types."""
    table = A.fraud_rate_by_type(paysim_like).set_index("type")
    fraud_bearing = table.loc[table["n_fraud"] > 0].index.tolist()

    assert set(fraud_bearing).issubset(set(C.FRAUD_BEARING_TYPES))


def test_shares_sum_to_one(paysim_like):
    table = A.fraud_rate_by_type(paysim_like)
    assert table["share_of_volume"].sum() == pytest.approx(1.0)
    assert table["share_of_fraud"].sum() == pytest.approx(1.0)


def test_hourly_profile_spans_the_day(paysim_like):
    profile = A.hourly_profile(paysim_like)

    assert profile["hour"].min() >= 0
    assert profile["hour"].max() <= 23
    assert profile["n_transactions"].sum() == len(paysim_like)


def test_amount_summary_reports_both_classes(paysim_like):
    summary = A.amount_summary(paysim_like)

    assert list(summary["class"]) == ["legitimate", "fraud"]
    assert (summary["p50"] <= summary["p95"]).all()
    assert summary["n"].sum() == len(paysim_like)


# --- Leakage audit -----------------------------------------------------------


def test_flagged_fraud_evidence_reports_precision(paysim_like):
    """With no flags set the precision is undefined, not silently zero."""
    evidence = A.flagged_fraud_evidence(paysim_like)

    assert evidence["n_flagged"] == 0
    assert np.isnan(evidence["precision_against_label"])


def test_flagged_fraud_precision_when_flags_fire(paysim_like):
    """A column that only ever fires on real fraud is an output, not an input."""
    df = paysim_like.copy()
    fraud_rows = df.index[df[C.TARGET] == 1][:5]
    df.loc[fraud_rows, "isFlaggedFraud"] = 1

    evidence = A.flagged_fraud_evidence(df)
    assert evidence["n_flagged"] == 5
    assert evidence["precision_against_label"] == pytest.approx(1.0)


def test_drain_signature_separates_the_classes(paysim_like):
    """Fraud empties the origin account; legitimate transactions do not."""
    table = A.drain_signature(paysim_like).set_index("class")

    assert table.loc["fraud", "fraction_amount_equals_balance"] > 0.95
    assert table.loc["legitimate", "fraction_amount_equals_balance"] < 0.05


def test_zero_balance_rates_flag_the_destination_artifact(paysim_like):
    """PaySim never credits the mule account on fraudulent transfers."""
    table = A.zero_balance_rates(paysim_like).set_index("condition")

    assert table.loc["dest_zero_both", "rate_fraud"] > 0.95
    assert table.loc["dest_zero_both", "rate_fraud"] > table.loc["dest_zero_both", "rate_legitimate"]


def test_all_rates_are_proportions(paysim_like):
    table = A.zero_balance_rates(paysim_like)
    numeric = table[["rate_legitimate", "rate_fraud"]].to_numpy()
    assert np.all((numeric >= 0.0) & (numeric <= 1.0))


# --- Rule recoverability -----------------------------------------------------


def test_rule_recoverability_reports_every_candidate(paysim_like):
    table = A.rule_recoverability(paysim_like)

    assert len(table) == 4
    assert table["precision"].between(0, 1).all()
    assert table["recall"].between(0, 1).all()


def test_conjunction_beats_its_parts_on_precision(paysim_like):
    """Adding clauses must sharpen precision, never blunt it."""
    table = A.rule_recoverability(paysim_like).set_index("rule")

    assert (table.loc["risky type AND drained", "precision"]
            >= table.loc["type in (TRANSFER, CASH_OUT)", "precision"])
    assert (table.loc["risky type AND drained AND emptied", "precision"]
            >= table.loc["risky type AND drained", "precision"])


def test_conjunction_never_increases_recall(paysim_like):
    """A stricter rule cannot catch more; that would be a logic error."""
    table = A.rule_recoverability(paysim_like).set_index("rule")

    assert (table.loc["risky type AND drained AND emptied", "recall"]
            <= table.loc["risky type AND drained", "recall"])


def test_rule_recovers_the_simulated_fraud_mechanism(paysim_like):
    """The fixture models PaySim's script, so the rule must recover it."""
    table = A.rule_recoverability(paysim_like).set_index("rule")
    best = table.loc["risky type AND drained AND emptied"]

    assert best["precision"] > 0.95
    assert best["recall"] > 0.90


# --- Single-feature probe ----------------------------------------------------


def test_single_feature_auc_ranks_every_feature(paysim_like):
    builder = FeatureBuilder().fit(paysim_like)
    X = builder.transform(paysim_like)

    table = A.single_feature_auc(X, paysim_like[C.TARGET], sample_size=0)

    assert len(table) == X.shape[1]
    assert table["auc"].between(0.0, 1.0).all()
    assert table["auc"].is_monotonic_decreasing  # sorted strongest first


def test_probe_detects_the_destination_artifact(paysim_like):
    """The probe must surface the artifact the fixture actually encodes.

    That artifact is the destination side: PaySim never credits the mule
    account on a fraudulent transfer, so both destination balances sit at zero.
    (``errorBalanceOrig`` is deliberately *not* asserted here -- in this fixture
    fraud and the balanced majority of legitimate rows both have an error of
    exactly zero, so no single split separates them. On the real file its
    pattern differs; notebook 02 reports the measured value rather than
    assuming one.)
    """
    builder = FeatureBuilder().fit(paysim_like)
    X = builder.transform(paysim_like)

    table = A.single_feature_auc(X, paysim_like[C.TARGET], sample_size=0)
    suspicious = A.flag_suspicious_features(table, threshold=0.90)

    assert {"destZeroBefore", "destZeroAfter"} <= set(suspicious)


def test_probe_ranks_artifacts_above_noise(paysim_like):
    """The general property: mechanism-encoding features outrank noise."""
    rng = np.random.default_rng(19)
    builder = FeatureBuilder().fit(paysim_like)
    X = builder.transform(paysim_like)
    X["noise"] = rng.normal(size=len(X))

    table = A.single_feature_auc(X, paysim_like[C.TARGET], sample_size=0).set_index("feature")
    assert table.loc["destZeroBefore", "auc"] > table.loc["noise", "auc"]


def test_probe_scores_pure_noise_near_chance(paysim_like):
    """A feature with no relationship to the label must not look predictive."""
    rng = np.random.default_rng(5)
    X = pd.DataFrame({"noise": rng.normal(size=len(paysim_like))})

    auc = A.single_feature_auc(X, paysim_like[C.TARGET], sample_size=0)["auc"].iloc[0]
    assert auc == pytest.approx(0.5, abs=0.1)


def test_probe_sampling_preserves_every_fraud_case(paysim_like):
    """Stratified sampling must not discard positives at 0.1% prevalence."""
    builder = FeatureBuilder().fit(paysim_like)
    X = builder.transform(paysim_like)

    sampled = A.single_feature_auc(X, paysim_like[C.TARGET], sample_size=500)
    full = A.single_feature_auc(X, paysim_like[C.TARGET], sample_size=0)

    assert set(sampled["feature"]) == set(full["feature"])


def test_flag_suspicious_respects_the_threshold():
    table = pd.DataFrame({"feature": ["a", "b", "c"], "auc": [0.99, 0.91, 0.62]})

    assert A.flag_suspicious_features(table, threshold=0.90) == ["a", "b"]
    assert A.flag_suspicious_features(table, threshold=0.95) == ["a"]
