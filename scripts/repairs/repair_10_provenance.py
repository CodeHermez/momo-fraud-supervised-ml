"""REPAIR 10 -- machine-readable environment and configuration record.

The audit (m1) found that ``requirements.txt`` pins what *should* be installed
while nothing records what actually ran: no interpreter version, no resolved
package versions, no commit SHA tied to any result file. The first time a number
fails to reproduce there is then no way to separate a methodological difference
from a dependency that moved.

This writes two things:

``results/experiment_environment.json``
    One snapshot of the interpreter, platform, resolved package versions, git
    state and dataset identity.

``results/experiment_index.json``
    For every result artifact: the notebook or script that produces it, the
    configuration it was produced under, and whether the audit left it valid,
    superseded, or in need of a rerun.

**No result is modified.** This is provenance only. Where a configuration is not
recoverable from the repository it is recorded as ``null`` rather than guessed --
a missing field is more useful than a plausible invention.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from momo_fraud import constants as C            # noqa: E402
from momo_fraud import evaluate as E             # noqa: E402
from momo_fraud import experiment as X           # noqa: E402
from momo_fraud import models as M               # noqa: E402
from momo_fraud import provenance as P           # noqa: E402

RESULTS = PROJECT_ROOT / "results"

STRATIFIED_SPLIT = {
    "strategy": "stratified",
    "train": C.SPLIT_TRAIN, "val": C.SPLIT_VAL, "test": C.SPLIT_TEST,
    "seed": C.RANDOM_SEED,
    "stratified_on": C.TARGET,
    "implementation": "splits.stratified_split",
}

TEMPORAL_SPLIT = {
    "strategy": "temporal",
    "train_end": C.TEMPORAL_TRAIN_END, "val_end": C.TEMPORAL_VAL_END,
    "tolerance": C.TEMPORAL_PROPORTION_TOLERANCE,
    "derivation": "splits.derive_temporal_boundaries on the empirical cumulative row count",
    "implementation": "splits.temporal_split",
}

THRESHOLD_PROTOCOL = {
    "rule": "ner_optimal",
    "selected_on": "val",
    "applied_to": "test",
    "note": ("At R = N_neg/N_pos the NER-optimal and Youden thresholds coincide "
             "exactly; configs B and C are one condition at the default R."),
}


def learner_hyperparameters() -> dict:
    """The actual constructor arguments, read off the estimators themselves."""
    out = {}
    for name in C.LEARNERS:
        estimator = M.make_learner(name, seed=C.RANDOM_SEED)
        if hasattr(estimator, "steps"):          # logistic regression is a Pipeline
            out[name] = {
                "pipeline": [step for step, _ in estimator.steps],
                **{k: v for k, v in estimator.named_steps["clf"].get_params().items()
                   if k in {"max_iter", "solver", "class_weight", "random_state"}},
            }
        else:
            keep = {"n_estimators", "max_depth", "min_samples_leaf", "max_samples",
                    "criterion", "learning_rate", "subsample", "colsample_bytree",
                    "tree_method", "scale_pos_weight", "eval_metric", "class_weight",
                    "random_state"}
            out[name] = {k: v for k, v in estimator.get_params().items() if k in keep}
    return out


#: (artifact stem, produced by, status, note). Status is one of VALID,
#: SUPERSEDED, DIAGNOSTIC.
ARTIFACTS = [
    ("00_dataset_provenance", "notebooks/00_data_acquisition.ipynb", "VALID", None),
    ("01_fraud_by_type", "notebooks/01_eda.ipynb", "VALID", None),
    ("01_amount_summary", "notebooks/01_eda.ipynb", "VALID", None),
    ("01_hourly_profile", "notebooks/01_eda.ipynb", "VALID", None),
    ("02_flagged_fraud_evidence", "notebooks/02_leakage_audit.ipynb", "VALID",
     "descriptive; n_flagged = 16"),
    ("02_drain_signature", "notebooks/02_leakage_audit.ipynb", "VALID",
     "descriptive property of the generator, computed on all rows"),
    ("02_zero_balance_rates", "notebooks/02_leakage_audit.ipynb", "VALID", None),
    ("02_rule_recoverability", "notebooks/02_leakage_audit.ipynb", "VALID",
     "in-sample descriptive statistic; not a held-out detector result"),
    ("02_single_feature_auc", "notebooks/02_leakage_audit.ipynb", "VALID",
     "computed on all rows including test; see 09_destination_single_feature_auc "
     "for the training-rows-only version"),
    ("02_account_overlap", "notebooks/02_leakage_audit.ipynb", "VALID", None),
    ("02_feature_decision", "notebooks/02_leakage_audit.ipynb", "VALID",
     "declares a 2-feature ablation; the study ran a 6-feature one - see CHANGE-01"),
    ("03_baselines_all_rungs", "notebooks/03_baselines.ipynb", "VALID",
     "carries both val and test rows; the basis for the repaired rung decision"),
    ("03_ladder_headroom", "notebooks/03_baselines.ipynb", "SUPERSEDED",
     "test-selected; replaced by 03_ladder_headroom_v2"),
    ("03_experimental_rung", "notebooks/03_baselines.ipynb", "SUPERSEDED",
     "test-selected; replaced by 03_experimental_rung_v2 (same rung)"),
    ("03_ladder_headroom_v2", "scripts/repairs/repair_02_validation_selection.py",
     "VALID", "validation-selected"),
    ("03_experimental_rung_v2", "scripts/repairs/repair_02_validation_selection.py",
     "VALID", "validation-selected; outcome unchanged"),
    ("03_benchmark_check", "notebooks/03_baselines.ipynb", "VALID", None),
    ("04_cost_sensitive", "notebooks/04_cost_sensitive.ipynb", "VALID",
     "the main grid; metrics verified against raw predictions"),
    ("04_h1_pr_auc_ci", "notebooks/04_cost_sensitive.ipynb", "SUPERSEDED",
     "marginal CIs, n_boot=100; replaced by 04_h1_paired_bootstrap"),
    ("04_h1_verdict", "notebooks/04_cost_sensitive.ipynb", "SUPERSEDED",
     "CI-overlap verdicts; replaced by 04_h1_verdict_v2"),
    ("04_h1_paired_bootstrap", "scripts/repairs/repair_03_paired_bootstrap.py",
     "VALID", "paired stratified bootstrap, n_boot=2000"),
    ("04_h1_verdict_v2", "scripts/repairs/repair_03_paired_bootstrap.py", "VALID", None),
    ("04_best_config_per_learner", "notebooks/04_cost_sensitive.ipynb", "SUPERSEDED",
     "test-selected; replaced by 04_best_config_per_learner_v2"),
    ("04_best_config_per_learner_v2",
     "scripts/repairs/repair_02_validation_selection.py", "VALID",
     "validation-selected; differs for 3 of 4 learners"),
    ("05_replication_arms", "notebooks/05_lokanan_replication.ipynb", "VALID",
     "see 05_arm_interpretation for what the arms do and do not establish"),
    ("05_precision_collapse", "notebooks/05_lokanan_replication.ipynb", "VALID",
     "the A-to-B contrast does not isolate evaluation prevalence"),
    ("05_replication_verdict", "notebooks/05_lokanan_replication.ipynb", "VALID", None),
    ("05_arm_interpretation", "scripts/repairs (documentation)", "VALID", None),
    ("05_builder_fit_repair", "scripts/repairs/repair_07_notebook05_builder.py",
     "VALID", "shows the M7 leak was inert for the modelled columns"),
    ("06_threshold_markers", "notebooks/06_risk_analysis.ipynb", "VALID", None),
    ("06_threshold_rule_comparison", "notebooks/06_risk_analysis.ipynb", "VALID", None),
    ("06_r_sweep", "notebooks/06_risk_analysis.ipynb", "SUPERSEDED",
     "misleading name; replaced by 06_decision_cost_sweep (same NER values)"),
    ("06_decision_cost_sweep", "scripts/repairs/repair_04_06_relabel.py", "VALID",
     "decision-cost sweep at fixed R_train = 773.70"),
    ("06_decision_cost_sweep_margins", "scripts/repairs/repair_04_06_relabel.py",
     "VALID", None),
    ("06_recall_at_budget", "notebooks/06_risk_analysis.ipynb", "VALID", None),
    ("06_risk_bands", "notebooks/06_risk_analysis.ipynb", "VALID", None),
    ("06_per_transaction_threshold", "notebooks/06_risk_analysis.ipynb", "VALID", None),
    ("06_risk_summary", "notebooks/06_risk_analysis.ipynb", "VALID", None),
    ("07_permutation_importance", "notebooks/07_interpretability.ipynb", "VALID",
     "computed on test; distorted by amount/log_amount duplication and the "
     "mutually exclusive type_* one-hots"),
    ("07_shap_importance", "notebooks/07_interpretability.ipynb", "VALID", None),
    ("07_gain_importance", "notebooks/07_interpretability.ipynb", "VALID", None),
    ("07_vs_published", "notebooks/07_interpretability.ipynb", "VALID", None),
    ("07_interpretability_summary", "notebooks/07_interpretability.ipynb", "VALID", None),
    ("07_learner_selection_v2", "scripts/repairs/repair_02_validation_selection.py",
     "VALID", "validation-selected; unchanged at xgboost"),
    ("09_velocity_vs_baseline", "notebooks/09_velocity_features.ipynb", "SUPERSEDED",
     "temporal rows invalid (split realised 95.6/3.0/1.4); stratified rows remain "
     "valid and are carried into 09_temporal_repaired for comparison"),
    ("09_temporal_repaired", "scripts/repairs/repair_01_temporal_split.py", "VALID",
     "derived boundaries; three cost ratios reported"),
    ("09_temporal_prevalence", "scripts/repairs/repair_01_temporal_split.py", "VALID",
     "shows the class prior is non-stationary in simulated time"),
    ("09_velocity_permutation_importance", "notebooks/09_velocity_features.ipynb",
     "VALID", "stratified arm; same correlated-feature caveat as 07"),
    ("09_destination_artefact_rates", "scripts/repairs/repair_09_destination_diagnostic.py",
     "DIAGNOSTIC", None),
    ("09_destination_single_feature_auc",
     "scripts/repairs/repair_09_destination_diagnostic.py", "DIAGNOSTIC",
     "training rows only"),
    ("09_destination_ablation", "scripts/repairs/repair_09_destination_diagnostic.py",
     "DIAGNOSTIC", "no_balance_state is not promoted to the main rung"),
    ("10_xgboost_hpo_candidates", "notebooks/10_advanced_optimization.ipynb", "VALID",
     "val PR-AUC is optimistically biased: the same val set served early stopping, "
     "candidate ranking and threshold selection"),
    ("10_xgboost_hpo_winner", "notebooks/10_advanced_optimization.ipynb", "VALID", None),
    ("10_learner_comparison", "notebooks/10_advanced_optimization.ipynb", "VALID",
     "contains the failed LightGBM run"),
    ("10_stacking_results", "notebooks/10_advanced_optimization.ipynb", "VALID", None),
    ("10_multiseed_ner", "notebooks/10_advanced_optimization.ipynb", "VALID",
     "4 seeds, unweighted config C only; LR and DT are deterministic so the "
     "effective seed sensitivity covers 2 learners; split held fixed"),
    ("10_summary", "notebooks/10_advanced_optimization.ipynb", "SUPERSEDED",
     "presented a failed run as a comparison; replaced by 10_summary_v2"),
    ("10_summary_v2", "scripts/repairs/repair_04_06_relabel.py", "VALID", None),
    ("10_failed_runs", "scripts/repairs/repair_04_06_relabel.py", "VALID",
     "the LightGBM run, marked as failed rather than compared"),
]


def main() -> None:
    environment = P.experiment_record(
        name="repository_environment_snapshot",
        seed=C.RANDOM_SEED,
        split={"stratified": STRATIFIED_SPLIT, "temporal": TEMPORAL_SPLIT},
        model={"learners": C.LEARNERS,
               "hyperparameters": learner_hyperparameters(),
               "selection_metric": "PR-AUC",
               "note": "hyperparameters are fixed, not tuned, in notebooks 03-09"},
        csl={"configs": C.CONFIGS,
             "variants": list(X.FIT_VARIANTS),
             "class_weight": "w+ = R, w- = 1",
             "instance_weight": "w+_i = R * s_i, w- = 1 (see docs/config_f_cost_objectives.md)",
             "rus_target": "N_neg / R"},
        cost_ratio={"R_default": float(C.R_DEFAULT),
                    "derivation": "N_legit / N_fraud over the full dataset",
                    "R_sweep_decision_layer": [float(r) for r in C.R_SWEEP]},
        thresholds=THRESHOLD_PROTOCOL,
        evaluation={"primary_metric": "ner",
                    "ranking_metric": "pr_auc",
                    "accuracy_reported": False,
                    "bootstrap": {"paired": True, "n_boot": 2000, "stratified": True}},
        notes=("Provenance only; no result was modified. Fields that cannot be "
               "recovered from the repository are null rather than guessed."),
    )
    P.write(environment, "experiment_environment")

    index = {
        "generated_by": "scripts/repairs/repair_10_provenance.py",
        "environment": "results/experiment_environment.json",
        "change_log": "results/methodology_change_log.json",
        "status_legend": {
            "VALID": "survives the audit; safe to cite",
            "SUPERSEDED": "retained as a record; cite the replacement instead",
            "DIAGNOSTIC": "informs a future research decision; not a study result",
        },
        "artifacts": [
            {"artifact": stem, "produced_by": producer, "status": status,
             "note": note, "present": (RESULTS / f"{stem}.csv").exists()
                                      or (RESULTS / f"{stem}.json").exists()}
            for stem, producer, status, note in ARTIFACTS
        ],
    }
    E.save_json(index, "experiment_index")

    counts: dict[str, int] = {}
    missing = []
    for row in index["artifacts"]:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
        if not row["present"]:
            missing.append(row["artifact"])

    print("results/experiment_environment.json written")
    print(f"  python {environment['environment']['python_version']}  "
          f"git {environment['git']['commit'][:10]} "
          f"(dirty={environment['git']['dirty']})")
    print("results/experiment_index.json written")
    for status, n in sorted(counts.items()):
        print(f"  {status:12s} {n}")
    if missing:
        print(f"  not yet on disk: {missing}")


if __name__ == "__main__":
    main()
