"""Tests for the exported scoring artifact.

The property that matters most: a reloaded bundle must reproduce the original
model's scores exactly. If it does not, the prototype and the notebooks disagree
about what a transaction is worth, and nothing downstream can be trusted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from momo_fraud import constants as C
from momo_fraud.features import FeatureBuilder, columns_for
from momo_fraud.models import make_learner
from momo_fraud.predict import FraudScorer, build_thresholds
from momo_fraud.risk import bayes_threshold


@pytest.fixture
def scorer(paysim_like, tmp_path):
    """A fitted scorer, round-tripped through disk exactly as the prototype does."""
    y = paysim_like[C.TARGET].to_numpy()
    builder = FeatureBuilder().fit(paysim_like)
    X = builder.transform(paysim_like)
    cols = columns_for("no_origin_balance", list(X.columns))

    model = make_learner("xgboost", seed=C.RANDOM_SEED)
    model.fit(X[cols], y)

    thresholds = build_thresholds(
        decision_threshold=bayes_threshold(C.R_DEFAULT),
        r=C.R_DEFAULT, feature_names=cols, rung="no_origin_balance",
    )
    original = FraudScorer(model=model, builder=builder, thresholds=thresholds,
                           feature_names=cols)
    original.save(tmp_path, model_card={"note": "test"})

    return original, FraudScorer.load(tmp_path, with_explainer=False), paysim_like


def test_reloaded_bundle_reproduces_scores_exactly(scorer):
    """Export fidelity -- the check the plan calls for."""
    original, restored, df = scorer

    before = original.score_batch(df.head(500))["probability"].to_numpy()
    after = restored.score_batch(df.head(500))["probability"].to_numpy()
    np.testing.assert_allclose(before, after, rtol=1e-6)


def test_joblib_variant_scores_identically_to_the_json_bundle(scorer, tmp_path):
    """The pickle is a convenience copy; it must not be a different model."""
    original, restored, df = scorer

    path = original.save_joblib(tmp_path / "model.joblib")
    assert path.exists()

    from_pickle = FraudScorer.load_joblib(path)
    sample = df.head(500)

    np.testing.assert_allclose(
        restored.score_batch(sample)["probability"].to_numpy(),
        from_pickle.score_batch(sample)["probability"].to_numpy(),
        rtol=1e-6,
    )
    # The frozen decision configuration has to survive too -- a pickle that
    # scores the same but carries a different threshold flags different rows.
    assert from_pickle.thresholds == restored.thresholds
    assert from_pickle.feature_names == restored.feature_names


def test_joblib_is_written_only_when_asked(scorer, tmp_path):
    original, _, _ = scorer

    original.save(tmp_path / "plain")
    assert not (tmp_path / "plain" / "model.joblib").exists()

    original.save(tmp_path / "with_pickle", joblib=True)
    assert (tmp_path / "with_pickle" / "model.joblib").exists()


def test_load_joblib_rejects_a_pickle_of_something_else(tmp_path):
    """Loading an arbitrary pickle should fail loudly, not half-work."""
    import joblib

    decoy = tmp_path / "decoy.joblib"
    joblib.dump({"not": "a scorer"}, decoy)

    with pytest.raises(TypeError, match="not a FraudScorer"):
        FraudScorer.load_joblib(decoy)


def test_single_and_batch_paths_agree(scorer):
    """The prototype's one-transaction call must match the batch pipeline."""
    _, restored, df = scorer
    batch = restored.score_batch(df.head(20))

    for position in (0, 7, 19):
        single = restored.score(df.iloc[position].to_dict(), explain=False)
        assert single["probability"] == pytest.approx(
            batch["probability"].iloc[position], rel=1e-6)
        assert single["band"] == batch["band"].iloc[position]


def test_score_returns_the_prototype_contract(scorer):
    _, restored, df = scorer
    result = restored.score(df.iloc[0].to_dict(), explain=False)

    assert set(result) >= {"risk_score", "probability", "band", "action",
                           "flagged", "decision_threshold"}
    assert 0.0 <= result["risk_score"] <= 100.0
    assert result["action"] in set(C.BAND_ACTIONS.values())


def test_risk_score_is_the_probability_on_a_0_100_scale(scorer):
    _, restored, df = scorer
    result = restored.score(df.iloc[0].to_dict(), explain=False)
    assert result["risk_score"] == pytest.approx(result["probability"] * 100, rel=1e-4)


def test_flagged_follows_the_frozen_threshold(scorer):
    _, restored, df = scorer
    batch = restored.score_batch(df.head(200))
    threshold = restored.thresholds["decision_threshold"]

    expected = batch["probability"] >= threshold
    assert (batch["flagged"] == expected).all()


def test_missing_fields_are_rejected(scorer):
    _, restored, _ = scorer
    with pytest.raises(ValueError, match="missing required fields"):
        restored.score({"amount": 100.0, "type": "TRANSFER"})


def test_explanation_degrades_rather_than_failing(scorer):
    """No explainer should mean no explanation, not an outage."""
    _, restored, df = scorer
    restored.explainer = None

    result = restored.score(df.iloc[0].to_dict(), explain=True)
    assert result["top_factors"] == []
    assert "risk_score" in result


def test_thresholds_carry_the_band_configuration(scorer):
    _, restored, _ = scorer
    t = restored.thresholds

    assert set(t["band_cutoffs"]) == {"Medium", "High", "Critical"}
    assert t["bayes_threshold"] == pytest.approx(bayes_threshold(t["severity_ratio_R"]))
    assert t["feature_set"] == "no_origin_balance"


def test_bands_are_ordered_by_score(scorer):
    """A higher risk score can never land in a lower band."""
    _, restored, df = scorer
    batch = restored.score_batch(df.head(1000))

    rank = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
    ordered = batch.sort_values("risk_score")
    levels = [rank[b] for b in ordered["band"]]
    assert levels == sorted(levels)


def test_bundle_contains_every_file_the_prototype_needs(scorer, tmp_path):
    original, _, _ = scorer
    written = {p.name for p in tmp_path.iterdir()}

    assert {"model.json", "preprocessor.json", "thresholds.json",
            "model_card.json"} <= written
