"""The prototype must reproduce the experiment, not approximate it.

These lock down the properties that make the demo trustworthy: it loads the
frozen bundle, it scores one transaction exactly as it scores a million, and it
refuses malformed input without falling over.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROTOTYPE = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PROTOTYPE.parent
for entry in (str(PROTOTYPE), str(PROJECT_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from core import schema  # noqa: E402
from momo_fraud import constants as C  # noqa: E402
from momo_fraud.predict import FraudScorer  # noqa: E402

ARTIFACTS = PROJECT_ROOT / "artifacts"
DEMO_SAMPLE = PROTOTYPE / "data" / "demo_sample.parquet"

#: The probe ``predict.py`` itself round-trips on, and the value it produced when
#: this prototype was built. A change here means the bundle changed.
PROBE = {
    "step": 212, "type": "TRANSFER", "amount": 181.0,
    "oldbalanceOrg": 181.0, "newbalanceOrig": 0.0,
    "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
}
PROBE_PROBABILITY = 0.9892911314964294


@pytest.fixture(scope="module")
def scorer() -> FraudScorer:
    if not (ARTIFACTS / "model.json").exists():
        pytest.skip("no model bundle in artifacts/")
    return FraudScorer.load(ARTIFACTS, with_explainer=True)


@pytest.fixture(scope="module")
def sample() -> pd.DataFrame:
    if not DEMO_SAMPLE.exists():
        pytest.skip("demo sample not built")
    return pd.read_parquet(DEMO_SAMPLE)


def _as_input(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame[schema.REQUIRED_FIELDS].copy()
    rows["step"] = rows["step"].astype(int)
    rows["type"] = rows["type"].astype(str)
    return rows


# --- the bundle -------------------------------------------------------------

def test_loads_the_frozen_bundle(scorer: FraudScorer) -> None:
    assert scorer.feature_names == [
        "amount", "log_amount", "oldbalanceDest", "newbalanceDest", "hour", "day",
        "destZeroBefore", "destZeroAfter", "highRiskFlag",
        "type_CASH_IN", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER",
    ]
    assert scorer.thresholds["feature_set"] == "no_origin_balance"


def test_probe_scores_as_recorded(scorer: FraudScorer) -> None:
    result = scorer.score(PROBE)
    assert result["probability"] == pytest.approx(PROBE_PROBABILITY, abs=1e-9)
    assert result["band"] == "Critical"
    assert result["action"] == "block"
    assert result["flagged"]


def test_explanation_is_populated(scorer: FraudScorer) -> None:
    factors = scorer.score(PROBE)["top_factors"]
    assert factors, "SHAP explainer produced no factors"
    assert all(f["feature"] in scorer.feature_names for f in factors)


# --- train/serve skew -------------------------------------------------------

def test_single_and_batch_paths_agree(scorer: FraudScorer, sample: pd.DataFrame) -> None:
    """The guard ``predict.py`` was written to protect. It must hold here too."""
    rows = _as_input(sample.sample(25, random_state=1))

    batch = scorer.score_batch(rows)["probability"].to_numpy()
    single = np.array([
        scorer.score(rows.iloc[i].to_dict(), explain=False)["probability"]
        for i in range(len(rows))
    ])

    assert np.abs(batch - single).max() < 1e-12


def test_origin_balances_do_not_change_the_score(scorer: FraudScorer) -> None:
    """The 'no_origin_balance' rung, asserted rather than assumed.

    The UI tells users these two fields are ignored. If that ever stops being
    true, the UI is lying and this test should be the thing that notices.
    """
    baseline = scorer.score(PROBE, explain=False)["probability"]

    altered = PROBE | {"oldbalanceOrg": 987_654.0, "newbalanceOrig": 123_456.0}
    assert scorer.score(altered, explain=False)["probability"] == baseline


def test_shipped_model_tracks_the_recorded_experiment(
    scorer: FraudScorer, sample: pd.DataFrame
) -> None:
    """The bundle is a train+val refit, so it is close to but not identical to
    the train-only model the experiment scored. Pin 'close'."""
    subset = sample.sample(2_000, random_state=2)
    live = scorer.score_batch(_as_input(subset))["probability"].to_numpy()
    recorded = subset["score"].to_numpy()

    assert np.corrcoef(live, recorded)[0, 1] > 0.95
    agreement = (
        (live >= scorer.thresholds["decision_threshold"])
        == (recorded >= scorer.thresholds["decision_threshold"])
    ).mean()
    assert agreement > 0.97


# --- the input contract -----------------------------------------------------

def test_sample_profiles_are_valid_and_land_where_claimed(scorer: FraudScorer) -> None:
    for profile in schema.SAMPLE_PROFILES:
        clean, problems = schema.validate(pd.DataFrame([profile.transaction]))
        assert not problems, f"{profile.name}: {problems}"

        result = scorer.score(profile.transaction, explain=False)
        caught = result["flagged"]
        expected = profile.recorded_score >= scorer.thresholds["decision_threshold"]
        assert caught == expected, (
            f"{profile.name} was described as "
            f"{'flagged' if expected else 'not flagged'} but scored "
            f"{result['probability']:.8f}"
        )


def test_bundled_csv_validates_cleanly(scorer: FraudScorer) -> None:
    raw = pd.read_csv(PROTOTYPE / "assets" / "sample_transactions.csv")
    clean, problems = schema.validate(raw)

    assert not problems
    assert len(clean) == len(raw)
    assert len(scorer.score_batch(clean)) == len(raw)


def test_extra_columns_are_reported_not_used() -> None:
    raw = pd.read_csv(PROTOTYPE / "assets" / "sample_transactions.csv")
    extra = dict(schema.describe_extra_columns(raw))

    assert "isFraud" in extra and "nameOrig" in extra
    assert not set(extra) & set(schema.REQUIRED_FIELDS)


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("type", "WIRE", "type"),
        ("amount", -5.0, "amount"),
        ("oldbalanceDest", "not-a-number", "oldbalanceDest"),
        ("step", 9_999, "step"),
    ],
)
def test_bad_values_are_reported_not_raised(field: str, value: object, expected: str) -> None:
    row = schema.blank_row() | {field: value}
    clean, problems = schema.validate(pd.DataFrame([row]))

    assert clean.empty
    assert expected in {p.column for p in problems}


def test_missing_columns_are_named() -> None:
    clean, problems = schema.validate(pd.DataFrame([{"step": 1, "type": "TRANSFER"}]))

    assert clean.empty
    missing = {p.column for p in problems}
    assert "amount" in missing and "oldbalanceDest" in missing


def test_good_rows_survive_alongside_bad_ones() -> None:
    frame = pd.DataFrame([
        schema.blank_row() | {"type": "TRANSFER", "amount": 100.0},
        schema.blank_row() | {"type": "NONSENSE"},
    ])
    clean, problems = schema.validate(frame)

    assert len(clean) == 1
    assert problems


def test_type_is_normalised() -> None:
    clean, problems = schema.validate(
        pd.DataFrame([schema.blank_row() | {"type": " transfer "}])
    )
    assert not problems
    assert clean.iloc[0]["type"] == "TRANSFER"


def test_empty_frame_does_not_raise() -> None:
    clean, problems = schema.validate(pd.DataFrame(columns=schema.REQUIRED_FIELDS))
    assert clean.empty
    assert problems


def test_every_model_feature_has_a_readable_phrase(scorer: FraudScorer) -> None:
    from core import explain

    unmapped = set(scorer.feature_names) - set(explain.FEATURE_PHRASES)
    assert not unmapped, f"no plain-English phrase for: {sorted(unmapped)}"


def test_every_band_has_an_action_and_a_meaning() -> None:
    from core import explain

    for band in C.BAND_ACTIONS:
        assert band in explain.BAND_MEANING
        assert band in explain.BAND_COLOURS
