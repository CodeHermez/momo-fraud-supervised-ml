"""Tests for the frozen baseline and the guards that hold it in place.

Each test pins a control the freeze-phase consistency checks depend on. If one
of these regresses, the baseline has silently stopped being frozen, and the
checks would go on reporting PASS against a weakened guard.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from momo_fraud import baseline as B
from momo_fraud import constants as C
from momo_fraud import experiment as X
from momo_fraud.features import FEATURE_SETS


# --- the freeze itself --------------------------------------------------------


def test_frozen_rung_is_a_real_feature_set():
    assert B.FROZEN_RUNG in FEATURE_SETS


def test_frozen_rung_is_not_a_diagnostic_rung():
    assert B.FROZEN_RUNG not in B.DIAGNOSTIC_RUNGS


def test_diagnostic_rungs_are_excluded_from_the_selectable_ladder():
    assert not set(B.SELECTABLE_RUNGS) & B.DIAGNOSTIC_RUNGS
    assert "no_balance_state" in B.DIAGNOSTIC_RUNGS


def test_the_claim_is_not_overstated():
    """The wording is load-bearing, so it is asserted rather than trusted."""
    claim = B.FROZEN_RUNG_CLAIM.lower()
    assert "reduced" in claim
    for overstatement in ("artefact-free", "artifact-free", "simulator-independent",
                          "free of", "eliminates"):
        assert overstatement not in claim


def test_lightgbm_is_excluded_from_the_core_comparison():
    assert "lightgbm" not in B.FROZEN_LEARNERS
    assert "lightgbm" in B.EXCLUDED_LEARNERS


def test_core_learners_are_the_declared_quartet():
    assert B.FROZEN_LEARNERS == list(C.LEARNERS)
    assert len(B.FROZEN_LEARNERS) == 4


# --- guard: the frozen rung ---------------------------------------------------


def test_assert_frozen_rung_accepts_the_frozen_rung():
    B.assert_frozen_rung(B.FROZEN_RUNG)     # must not raise


def test_assert_frozen_rung_rejects_a_diagnostic_rung():
    with pytest.raises(ValueError, match="DIAGNOSTIC"):
        B.assert_frozen_rung("no_balance_state")


def test_assert_frozen_rung_rejects_any_other_rung():
    with pytest.raises(ValueError, match="frozen rung"):
        B.assert_frozen_rung("full")


# --- guard: diagnostic rungs cannot be chosen ---------------------------------


@pytest.fixture
def headroom_favouring_the_diagnostic_rung():
    """Headroom where the diagnostic rung would win if it were selectable.

    `no_balance_state` has more headroom than `no_origin_balance`, so a ladder
    that admits it and is ordered most-ablated-first would pick it.
    """
    return pd.Series({
        "full": 0.003,
        "no_error_features": 0.007,
        "no_balance_state": 0.184,
        "no_origin_balance": 0.115,
    })


def test_choose_rung_excludes_diagnostic_rungs_by_default(
        headroom_favouring_the_diagnostic_rung):
    order = ["full", "no_error_features", "no_balance_state", "no_origin_balance"]
    chosen, _ = X.choose_rung(headroom_favouring_the_diagnostic_rung, order)
    assert chosen == "no_origin_balance"


def test_choose_rung_can_admit_them_deliberately(
        headroom_favouring_the_diagnostic_rung):
    order = ["full", "no_error_features", "no_balance_state", "no_origin_balance"]
    chosen, _ = X.choose_rung(headroom_favouring_the_diagnostic_rung, order,
                              exclude=frozenset())
    assert chosen == "no_balance_state"


def test_choose_rung_raises_when_nothing_selectable_remains():
    with pytest.raises(ValueError, match="no selectable rungs"):
        X.choose_rung(pd.Series({"no_balance_state": 0.2}), ["no_balance_state"])


def test_choose_rung_raises_when_headroom_covers_no_selectable_rung():
    with pytest.raises(ValueError, match="no values for any selectable rung"):
        X.choose_rung(pd.Series({"something_else": 0.2}), ["full"])


# --- guard: superseded artifacts ----------------------------------------------


@pytest.fixture
def results_dir(tmp_path):
    """A miniature results directory with one current and one superseded file."""
    pd.DataFrame({"ner": [0.1]}).to_csv(tmp_path / "current.csv", index=False)
    pd.DataFrame({"ner": [0.9]}).to_csv(tmp_path / "stale.csv", index=False)
    (tmp_path / "experiment_index.json").write_text(json.dumps({
        "artifacts": [
            {"artifact": "current", "status": "VALID", "note": None},
            {"artifact": "stale", "status": "SUPERSEDED",
             "note": "replaced by current"},
        ]
    }), encoding="utf-8")
    return tmp_path


def test_current_artifacts_load_normally(results_dir):
    frame = B.load_results("current", results_dir=results_dir)
    assert frame["ner"].iloc[0] == 0.1


def test_superseded_artifacts_are_refused(results_dir):
    with pytest.raises(B.SupersededArtifactError, match="SUPERSEDED"):
        B.load_results("stale", results_dir=results_dir)


def test_the_refusal_explains_what_replaced_it(results_dir):
    with pytest.raises(B.SupersededArtifactError, match="replaced by current"):
        B.load_results("stale", results_dir=results_dir)


def test_superseded_artifacts_can_be_read_deliberately(results_dir):
    frame = B.load_results("stale", allow_superseded=True, results_dir=results_dir)
    assert frame["ner"].iloc[0] == 0.9


def test_unlisted_artifacts_are_not_blocked(tmp_path):
    """An artifact with no index entry is unknown, not superseded."""
    pd.DataFrame({"ner": [0.5]}).to_csv(tmp_path / "novel.csv", index=False)
    assert B.load_results("novel", results_dir=tmp_path)["ner"].iloc[0] == 0.5


# --- guard: the experiment schema ---------------------------------------------


def test_schema_requires_the_two_cost_ratios_separately():
    """Conflating R_train and R_eval is how a decision-layer effect gets
    attributed to training, which is exactly what happened to the R-sweep."""
    assert "R_train" in B.REQUIRED_EXPERIMENT_FIELDS
    assert "R_eval" in B.REQUIRED_EXPERIMENT_FIELDS


def test_schema_requires_both_seed_kinds_separately():
    assert "model_seed" in B.REQUIRED_EXPERIMENT_FIELDS
    assert "split_seed" in B.REQUIRED_EXPERIMENT_FIELDS
    assert set(B.SEED_KINDS) == {"model_seed", "split_seed"}


def test_validate_reports_every_missing_field():
    frame = pd.DataFrame([{"learner": "xgboost", "feature_set": B.FROZEN_RUNG}])
    missing = B.validate_experiment_frame(frame)
    assert "R_train" in missing and "R_eval" in missing
    assert "model_seed" in missing and "split_seed" in missing
    assert "pr_auc" in missing and "ner" in missing


def test_validate_passes_a_complete_record():
    row = {f: 0 for f in B.REQUIRED_EXPERIMENT_FIELDS + B.REQUIRED_METRIC_FIELDS}
    assert B.validate_experiment_frame(pd.DataFrame([row])) == []


def test_metrics_can_be_waived_for_a_configuration_only_record():
    row = {f: 0 for f in B.REQUIRED_EXPERIMENT_FIELDS}
    assert B.validate_experiment_frame(pd.DataFrame([row]), require_metrics=False) == []


# --- the evaluation protocol --------------------------------------------------


def test_accuracy_is_excluded_from_the_metric_set():
    assert "accuracy" in B.FROZEN_METRICS["excluded"]
    assert "accuracy" not in B.FROZEN_METRICS["also_reported"]
    assert B.FROZEN_METRICS["primary_ranking"] == "pr_auc"
    assert B.FROZEN_METRICS["primary_decision_risk"] == "ner"


def test_the_required_metrics_are_all_produced_by_full_report():
    """The schema must ask only for metrics the evaluation layer actually emits."""
    import numpy as np
    from momo_fraud import evaluate as E

    y = np.array([0, 0, 1, 0, 1, 0, 0, 1])
    scores = np.array([0.1, 0.2, 0.9, 0.3, 0.8, 0.05, 0.15, 0.7])
    produced = set(E.full_report(y, scores, threshold=0.5, r=C.R_DEFAULT))

    assert set(B.REQUIRED_METRIC_FIELDS) <= produced


# --- the CSL taxonomy ---------------------------------------------------------


def test_config_f_is_classified_as_sensitivity_analysis_not_algorithm_level():
    sensitivity = B.CSL_TAXONOMY["sensitivity_analysis"]
    assert sensitivity["configs"] == ["F"]
    assert "instance_weighted" in sensitivity["variants"]

    algorithm = B.CSL_TAXONOMY["algorithm_and_data_level"]
    assert "F" not in algorithm["configs"]
    assert "instance_weighted" not in algorithm["variants"]


def test_config_f_objective_mismatch_is_stated():
    objective = B.CSL_TAXONOMY["sensitivity_analysis"]["objective"]
    assert "R * s_i" in objective
    assert "C_FN = R" in objective


def test_the_forbidden_severity_claim_is_recorded():
    assert B.CONFIG_F_FORBIDDEN_CLAIM in B.FORBIDDEN_CLAIMS


def test_every_taxonomy_variant_is_a_real_fit_variant():
    declared = {v for group in B.CSL_TAXONOMY.values() for v in group["variants"]}
    assert declared == set(X.FIT_VARIANTS)


def test_every_taxonomy_config_is_a_real_config():
    declared = {c for group in B.CSL_TAXONOMY.values() for c in group["configs"]}
    assert declared == set(C.CONFIGS)


# --- conclusions --------------------------------------------------------------


def test_every_supported_conclusion_cites_evidence():
    for entry in B.SUPPORTED_CONCLUSIONS:
        assert entry["claim"] and entry["evidence"]


def test_the_nine_frozen_conclusions_are_all_present():
    assert len(B.SUPPORTED_CONCLUSIONS) == 9


def test_forbidden_claims_cover_the_known_overstatements():
    joined = " ".join(B.FORBIDDEN_CLAIMS).lower()
    for topic in ("artefact-free", "severity weighting", "csl never helps",
                  "r_train", "prevalence", "seeds or splits",
                  "empirical security evaluation"):
        assert topic in joined


def test_the_authentication_design_claim_is_bounded():
    """RQ 2.1.1 is a design contribution; the guard must say so explicitly.

    The failure mode this catches is the design ladder being cited as evidence
    that step-up authentication works. See CHANGE-06 and
    docs/authentication_design_contribution.md section D.
    """
    assert B.AUTH_DESIGN_FORBIDDEN_CLAIM in B.FORBIDDEN_CLAIMS
    claim = B.AUTH_DESIGN_FORBIDDEN_CLAIM.lower()
    assert "not an empirical security evaluation" in claim
    assert "derived" in claim          # what may be said
    assert "no control was tested" in claim   # what may not


# --- the manifest -------------------------------------------------------------


def test_manifest_declares_the_freeze():
    payload = B.manifest()
    assert payload["status"] == "FROZEN"
    assert payload["feature_set"]["frozen_rung"] == B.FROZEN_RUNG
    assert payload["feature_set"]["claim"] == B.FROZEN_RUNG_CLAIM


def test_manifest_records_both_cost_ratios_and_the_split_seed():
    payload = B.manifest()
    assert payload["cost"]["R_default"] == pytest.approx(C.R_DEFAULT)
    assert "R_train_vs_R_eval" in payload["cost"]
    assert payload["split"]["split_seed"] == C.RANDOM_SEED


def test_manifest_carries_the_rung_rationale():
    rationale = " ".join(B.manifest()["feature_set"]["rationale"]).lower()
    assert "origin-side" in rationale
    assert "destination-side" in rationale
    assert "no evidence-based reason" in rationale


def test_written_manifest_matches_the_module(tmp_path):
    """The file on disk must not drift from the definition in code."""
    try:
        written = B.load_manifest()
    except FileNotFoundError:
        pytest.skip("manifest not yet written; run scripts/freeze/write_manifest.py")

    assert written["status"] == "FROZEN"
    assert written["feature_set"]["frozen_rung"] == B.FROZEN_RUNG
    assert written["split"]["split_seed"] == C.RANDOM_SEED
    assert written["models"]["learners"] == B.FROZEN_LEARNERS
    assert written["consistency_checks"]["verdict"] == "PASS"
