"""E3 aggregation, paired deltas and quality control.

Three sources of uncertainty are computed and reported **separately**, because
each answers a different question and none substitutes for another:

* **test-observation** -- paired stratified bootstrap within one fixed
  (split seed, model seed) cell. Says how much of a delta could be an accident of
  which transactions landed in the test partition.
* **training-seed** -- spread across model seeds at a fixed split. Says how much
  is an accident of estimator stochasticity.
* **split** -- spread across split seeds. Says how much is an accident of which
  rows became train, validation and test.

A bootstrap interval is never described as covering the other two.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results"

import sys                                                     # noqa: E402
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from momo_fraud import baseline as B                           # noqa: E402
from momo_fraud import constants as C                          # noqa: E402
from momo_fraud import evaluate as E                           # noqa: E402
from momo_fraud import experiment as X                         # noqa: E402
from momo_fraud import provenance as P                         # noqa: E402

EXPERIMENT_ID = "E3_multiseed_multisplit"
FROZEN_SHA = "b2e6f624"

#: The core class-constant comparison. Config F is handled separately.
CORE_CSL_VARIANTS = ["weighted", "rus"]
SENSITIVITY_VARIANT = "instance_weighted"
BASELINE_VARIANT = "unweighted"

#: Reported configs. B and C coincide at the default R (a theorem), so C stands
#: for the decision-level arm and B is reported alongside for completeness.
CONFIG_OF_VARIANT = {"unweighted": "C", "weighted": "E",
                     "instance_weighted": "F", "rus": "G"}


def designated_config_rows(test: pd.DataFrame) -> pd.DataFrame:
    """One config per fit variant, so a variant is not averaged over three thresholds.

    Each fit variant serves several configurations that differ only in threshold
    rule. PR-AUC is threshold-free and identical across them, but NER is not --
    the unweighted fit yields NER 0.659 at config A (t=0.5) and 0.250 at config C
    (NER-optimal) from the very same model. Grouping by variant without picking a
    config would average those together and call the result "the unweighted NER".

    The designated config puts every arm on the **same threshold rule**
    (NER-optimal), so a CSL-vs-baseline delta isolates the training-level
    intervention rather than confounding it with the decision rule:

        unweighted -> C,  weighted -> E,  instance_weighted -> F,  rus -> G
    """
    keep = [(v, c) for v, c in CONFIG_OF_VARIANT.items()]
    mask = pd.Series(False, index=test.index)
    for variant, config in keep:
        mask |= (test["csl_variant"] == variant) & (test["config"] == config)
    return test[mask].copy()


def load_runs() -> pd.DataFrame:
    cells = sorted((RESULTS / "E3_cells").glob("*.json"))
    if not cells:
        raise SystemExit("no completed E3 cells found.")
    rows = []
    for path in cells:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") == "complete":
            rows.extend(payload["rows"])
    return pd.DataFrame(rows)


# --- aggregation ---------------------------------------------------------------


def _spread(group: pd.Series) -> dict:
    return {
        "mean": float(group.mean()), "median": float(group.median()),
        "std": float(group.std(ddof=1)) if len(group) > 1 else 0.0,
        "min": float(group.min()), "max": float(group.max()),
        "range": float(group.max() - group.min()), "n": int(len(group)),
    }


def robustness_summary(test: pd.DataFrame) -> pd.DataFrame:
    """mean / median / std / min / max per learner x variant, for both metrics."""
    rows = []
    for (learner, variant), group in test.groupby(["learner", "csl_variant"]):
        row = {"learner": learner, "csl_variant": variant,
               "config": CONFIG_OF_VARIANT[variant],
               "deterministic_in_model_seed": bool(group["deterministic_in_model_seed"].iloc[0]),
               "n_cells": len(group),
               "n_split_seeds": group["split_seed"].nunique(),
               "n_model_seeds": group["model_seed"].nunique()}
        for metric in ("pr_auc", "ner"):
            row.update({f"{metric}_{k}": v for k, v in _spread(group[metric]).items()
                        if k != "n"})
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["learner", "csl_variant"])


def variance_decomposition(test: pd.DataFrame) -> pd.DataFrame:
    """Split variability and training-seed variability, kept apart.

    Split spread is measured at a fixed model seed; training-seed spread at a
    fixed split. Pooling them would report one number for two different
    questions.
    """
    rows = []
    for (learner, variant), group in test.groupby(["learner", "csl_variant"]):
        deterministic = bool(group["deterministic_in_model_seed"].iloc[0])
        for metric in ("pr_auc", "ner"):
            # Training-seed spread within each split, then averaged.
            within = group.groupby("split_seed")[metric].std(ddof=1).dropna()
            # Split spread within each model seed, then averaged.
            across = group.groupby("model_seed")[metric].std(ddof=1).dropna()
            rows.append({
                "learner": learner, "csl_variant": variant, "metric": metric,
                "deterministic_in_model_seed": deterministic,
                "training_seed_std": 0.0 if deterministic else
                    (float(within.mean()) if len(within) else np.nan),
                "training_seed_std_is_exact_zero_by_construction": deterministic,
                "split_std": float(across.mean()) if len(across) else np.nan,
                "total_std": float(group[metric].std(ddof=1)),
            })
    return pd.DataFrame(rows)


def paired_deltas(test: pd.DataFrame) -> pd.DataFrame:
    """Delta = CSL - baseline, matched on learner, split seed and model seed.

    Matching is exact wherever both sides vary. Where the *baseline* is
    deterministic in the model seed (logistic regression, non-RUS), its score is
    identical at every model seed by construction, so pairing a stochastic CSL
    cell against the seed-42 baseline is exact rather than approximate. That case
    is flagged so it is never mistaken for an unmatched comparison.
    """
    rows = []
    for learner, by_learner in test.groupby("learner"):
        base_all = by_learner[by_learner["csl_variant"] == BASELINE_VARIANT]
        if base_all.empty:
            continue
        base_deterministic = bool(base_all["deterministic_in_model_seed"].iloc[0])

        for variant in (CORE_CSL_VARIANTS + [SENSITIVITY_VARIANT]):
            csl = by_learner[by_learner["csl_variant"] == variant]
            for _, cell in csl.iterrows():
                match = base_all[
                    (base_all["split_seed"] == cell["split_seed"])
                    & (base_all["model_seed"] == cell["model_seed"])
                ]
                exact_seed_match = not match.empty
                if match.empty and base_deterministic:
                    match = base_all[base_all["split_seed"] == cell["split_seed"]]
                if match.empty:
                    continue
                base = match.iloc[0]

                rows.append({
                    "learner": learner, "csl_variant": variant,
                    "config": CONFIG_OF_VARIANT[variant],
                    "arm": ("sensitivity" if variant == SENSITIVITY_VARIANT
                            else "core_class_constant"),
                    "split_seed": int(cell["split_seed"]),
                    "model_seed": int(cell["model_seed"]),
                    "baseline_model_seed": int(base["model_seed"]),
                    "exact_seed_match": exact_seed_match,
                    "matched_by_determinism": not exact_seed_match,
                    "same_test_observations": True,   # same split seed -> same rows
                    "baseline_pr_auc": float(base["pr_auc"]),
                    "csl_pr_auc": float(cell["pr_auc"]),
                    "delta_pr_auc": float(cell["pr_auc"] - base["pr_auc"]),
                    "baseline_ner": float(base["ner"]),
                    "csl_ner": float(cell["ner"]),
                    "delta_ner": float(cell["ner"] - base["ner"]),
                    "R_train_baseline": float(base["R_train"]),
                    "R_train_csl": float(cell["R_train"]),
                    "R_eval": float(cell["R_eval"]),
                })
    return pd.DataFrame(rows)


def delta_summary(deltas: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (learner, variant), group in deltas.groupby(["learner", "csl_variant"]):
        row = {"learner": learner, "csl_variant": variant,
               "config": CONFIG_OF_VARIANT[variant],
               "arm": group["arm"].iloc[0], "n_pairs": len(group)}
        for metric in ("delta_pr_auc", "delta_ner"):
            row.update({f"{metric}_{k}": v for k, v in _spread(group[metric]).items()
                        if k != "n"})
            row[f"{metric}_n_negative"] = int((group[metric] < 0).sum())
            row[f"{metric}_all_same_sign"] = bool(
                (group[metric] < 0).all() or (group[metric] > 0).all())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["arm", "learner", "csl_variant"])


def test_observation_uncertainty(rung: str, r: float) -> pd.DataFrame:
    """Paired bootstrap within ONE fixed cell, for scale comparison only.

    Deliberately computed at a single (split seed, model seed) so it cannot be
    confused with the across-seed spreads. It answers: within one partition and
    one training run, how much of the delta could be test-sampling noise?
    """
    split_seed, model_seed = 42, 42
    tag = f"{rung}__s{split_seed}"
    rows = []
    for learner in C.LEARNERS:
        try:
            base = E.load_predictions(learner, BASELINE_VARIANT, "test",
                                      model_seed, tag=tag)
        except FileNotFoundError:
            continue
        y = base["y_true"].to_numpy()
        for variant in CORE_CSL_VARIANTS + [SENSITIVITY_VARIANT]:
            try:
                other = E.load_predictions(learner, variant, "test", model_seed, tag=tag)
            except FileNotFoundError:
                continue
            out = E.paired_bootstrap_pr_auc(y, other["score"].to_numpy(),
                                            base["score"].to_numpy(), n_boot=2000)
            rows.append({"learner": learner, "csl_variant": variant,
                         "split_seed": split_seed, "model_seed": model_seed,
                         "uncertainty_source": "test_observation_only",
                         **out})
    return pd.DataFrame(rows)


def aggregate(rung: str, r: float, git: dict, environment: dict) -> None:
    runs = load_runs()
    E.save_results(runs.to_dict("records"), "E3_runs")

    # All test rows are kept in E3_runs.csv; the summaries below use one
    # config per variant so NER is never averaged across threshold rules.
    test = designated_config_rows(runs.query("split == 'test'").copy())
    E.save_results(test.to_dict("records"), "E3_designated_config_rows")

    prevalence = (runs.drop_duplicates("split_seed")[
        ["split_seed", "train_prevalence", "val_prevalence", "test_prevalence",
         "train_n_positive", "train_n_negative", "val_n_positive", "val_n_negative",
         "test_n_positive", "test_n_negative"]].sort_values("split_seed"))
    E.save_results(prevalence.to_dict("records"), "E3_split_prevalence")

    summary = robustness_summary(test)
    E.save_results(summary.to_dict("records"), "E3_robustness_summary")

    decomposition = variance_decomposition(test)
    E.save_results(decomposition.to_dict("records"), "E3_variance_decomposition")

    deltas = paired_deltas(test)
    E.save_results(deltas.to_dict("records"), "E3_paired_deltas")

    summary_deltas = delta_summary(deltas)
    E.save_results(summary_deltas.to_dict("records"), "E3_delta_summary")

    bootstrap = test_observation_uncertainty(rung, r)
    if not bootstrap.empty:
        E.save_results(bootstrap.to_dict("records"), "E3_test_observation_bootstrap")

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "status": "COMPLETE",
        "frozen_baseline_sha": FROZEN_SHA,
        "frozen_baseline_manifest": "results/frozen_experimental_baseline.json",
        "feature_set": rung,
        "split_strategy": "stratified 70/15/15",
        "split_seeds": sorted(runs["split_seed"].unique().tolist()),
        "model_seeds": sorted(runs["model_seed"].unique().tolist()),
        "learners": sorted(runs["learner"].unique().tolist()),
        "fit_variants": sorted(runs["csl_variant"].unique().tolist()),
        "configurations": sorted(runs["config"].unique().tolist()),
        "cost": {"C_FP": 1.0, "R_train_csl": float(r),
                 "R_train_unweighted": 1.0, "R_eval": float(r),
                 "R_varied_in_E3": False},
        "threshold_protocol": {
            "order": "TRAIN -> FIT -> VAL -> SELECT -> LOCK -> TEST -> REPORT",
            "selected_on": "val", "rules": sorted(runs["threshold_rule"].unique().tolist()),
        },
        "metrics": {"primary_ranking": "pr_auc", "primary_decision_risk": "ner",
                    "accuracy_computed": False},
        "uncertainty_sources_reported_separately": [
            "test_observation (paired bootstrap within one fixed cell)",
            "training_seed (across model seeds at fixed split)",
            "split (across split seeds)",
        ],
        "deterministic_cells": {
            "learner": "logistic_regression",
            "variants": ["unweighted", "weighted", "instance_weighted"],
            "reason": ("lbfgs does not consume random_state; std_pr_auc is exactly "
                       "0.0 across four seeds on the full training set"),
            "omitted_fits": 36,
            "omitted_are": "deterministic duplicates, not missing observations",
            "training_seed_variance": 0.0,
        },
        "config_f": {
            "variant": SENSITIVITY_VARIANT,
            "role": "sensitivity analysis",
            "not_merged_into_core_comparison": True,
            "objective_mismatch": ("trains on C_FN = R * s_i; thresholded and "
                                   "scored on C_FN = R"),
        },
        "n_cells_planned": 220,
        "n_cells_completed": int(runs["cell_id"].nunique()),
        "execution_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": git.get("commit"),
        "python_version": environment["python_version"],
        "artifacts": [
            "results/E3_runs.csv", "results/E3_split_prevalence.csv",
            "results/E3_robustness_summary.csv", "results/E3_variance_decomposition.csv",
            "results/E3_paired_deltas.csv", "results/E3_delta_summary.csv",
            "results/E3_test_observation_bootstrap.csv", "results/E3_qc.json",
        ],
    }
    E.save_json(manifest, "E3_multiseed_multisplit_manifest")

    P.write(P.experiment_record(
        name=EXPERIMENT_ID, seed=42,
        split={"strategy": "stratified", "seeds": manifest["split_seeds"]},
        model={"learners": manifest["learners"], "model_seeds": manifest["model_seeds"]},
        csl={"variants": manifest["fit_variants"], "taxonomy": list(B.CSL_TAXONOMY)},
        cost_ratio=manifest["cost"], thresholds=manifest["threshold_protocol"],
        evaluation=manifest["metrics"],
        notes="Robustness experiment. No tuning, no feature selection, R not varied.",
    ), "E3_multiseed_multisplit_provenance")

    print(f"\naggregated {runs['cell_id'].nunique()} cells -> {len(runs)} rows")
    print("  results/E3_runs.csv, E3_robustness_summary.csv, E3_paired_deltas.csv,")
    print("  E3_delta_summary.csv, E3_variance_decomposition.csv,")
    print("  E3_split_prevalence.csv, E3_multiseed_multisplit_manifest.json")


# --- quality control ------------------------------------------------------------


def quality_control(rung: str, r: float) -> bool:
    runs = load_runs()
    test = designated_config_rows(runs.query("split == 'test'"))
    deltas = pd.read_csv(RESULTS / "E3_paired_deltas.csv")

    from E3_multiseed_multisplit import (                        # noqa: PLC0415
        MODEL_SEEDS, SPLIT_SEEDS, _frozen_fingerprint, planned_cells,
    )

    planned = {c[0] + "__" + c[1] + f"__split{c[2]}__model{c[3]}" for c in planned_cells()}
    present = set(runs["cell_id"].unique())

    checks: list[tuple[str, bool, str]] = [
        ("every result uses the frozen rung",
         bool((runs["feature_set"] == rung).all()), rung),
        ("no diagnostic rung entered the experiment",
         not bool(runs["feature_set"].isin(B.DIAGNOSTIC_RUNGS).any()),
         f"diagnostic rungs: {sorted(B.DIAGNOSTIC_RUNGS)}"),
        ("split seeds are exactly the pre-declared values",
         sorted(runs["split_seed"].unique()) == SPLIT_SEEDS,
         str(sorted(runs["split_seed"].unique()))),
        ("model seeds are within the pre-declared values",
         set(runs["model_seed"].unique()) <= set(MODEL_SEEDS),
         str(sorted(runs["model_seed"].unique()))),
        ("every required cell exists", planned <= present,
         f"{len(present & planned)}/{len(planned)} present"),
        ("no unplanned cell present", present <= planned,
         f"{len(present - planned)} unplanned"),
        ("all thresholds validation-selected",
         bool((runs["threshold_selected_on"] == "val").all()), "val"),
        ("R_train present on every row", not bool(runs["R_train"].isna().any()),
         f"values {sorted(runs['R_train'].unique())}"),
        ("R_eval present and equal to the frozen default",
         bool((runs["R_eval"] == r).all()), f"{r:.4f}"),
        ("R not varied in E3", runs["R_eval"].nunique() == 1, "1 distinct R_eval"),
        ("seed provenance present on every row",
         not bool(runs[["split_seed", "model_seed"]].isna().any().any()), "complete"),
        ("git SHA recorded on every row",
         not bool(runs["git_sha"].isna().any()), str(runs["git_sha"].iloc[0])[:8]),
        ("timestamp recorded on every row",
         not bool(runs["timestamp_utc"].isna().any()), "present"),
        ("accuracy never computed", "accuracy" not in runs.columns, "absent"),
        ("all paired comparisons matched on split seed",
         bool((deltas["same_test_observations"]).all()), "same test rows"),
        ("paired comparisons matched on model seed or deterministic",
         bool((deltas["exact_seed_match"] | deltas["matched_by_determinism"]).all()),
         f"{int(deltas['exact_seed_match'].sum())} exact, "
         f"{int(deltas['matched_by_determinism'].sum())} via determinism"),
        ("every arm uses the NER-optimal threshold rule",
         bool((test["threshold_rule"] == "ner_optimal").all()),
         str(sorted(test["threshold_rule"].unique()))),
        ("one config per variant in the summaries",
         bool((test.groupby("csl_variant")["config"].nunique() == 1).all()),
         str(test.groupby("csl_variant")["config"].first().to_dict())),
        ("config F kept out of the core arm",
         set(deltas.query("arm == 'core_class_constant'")["csl_variant"])
         == set(CORE_CSL_VARIANTS),
         str(sorted(set(deltas.query("arm == 'core_class_constant'")["csl_variant"])))),
        ("schema validation passes on the full frame",
         not B.validate_experiment_frame(runs), "all required fields present"),
    ]

    fingerprint_path = RESULTS / "E3_baseline_fingerprint.json"
    current = _frozen_fingerprint()
    if fingerprint_path.exists():
        before = json.loads(fingerprint_path.read_text(encoding="utf-8"))
        checks.append(("frozen baseline artifacts unchanged", before == current,
                       "sha256 match" if before == current else "MODIFIED"))
    else:
        fingerprint_path.write_text(json.dumps(current, indent=1), encoding="utf-8")
        checks.append(("frozen baseline fingerprint recorded", True, "baseline stored"))

    # A locked threshold must be identical on val and test within each cell/config.
    locked = runs.groupby(["cell_id", "config"])["threshold"].nunique()
    checks.append(("locked threshold identical on val and test",
                   bool((locked == 1).all()), f"{int((locked == 1).sum())} cell-configs"))

    width = max(len(n) for n, _, _ in checks)
    print("\nE3 quality control\n")
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<{width}}  {detail}")

    passed = all(ok for _, ok, _ in checks)
    print(f"\n  {sum(ok for _, ok, _ in checks)}/{len(checks)} -> "
          f"{'PASS' if passed else 'FAIL'}")

    E.save_json({
        "experiment_id": EXPERIMENT_ID,
        "verdict": "PASS" if passed else "FAIL",
        "n_cells": int(runs["cell_id"].nunique()),
        "checks": [{"check": n, "status": "PASS" if ok else "FAIL", "detail": d}
                   for n, ok, d in checks],
    }, "E3_qc")
    return passed
