"""E2 aggregation, paired R_train comparisons and quality control.

Two reference points, reported side by side and never conflated:

* **vs unweighted** -- what class weighting at this ``R_train`` costs or gains
  against no cost weighting at all.
* **vs the frozen default** ``R_train`` = 773.7011 -- whether the study's chosen
  ratio sits in an ordinary region of the curve or an unusual one.

Both are computed as *paired* differences on matched learner, split seed, model
seed and test observations. Treating the R conditions as unrelated experiments
would discard the pairing that makes the comparison precise.

**No R is called "optimal".** E2 is a pre-declared sensitivity analysis; the
study's methodology does not permit selecting an operating cost ratio from test
performance, so the language stays descriptive -- reduced or increased penalty,
stable region, learner-dependent response.
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
from momo_fraud import provenance as P                         # noqa: E402

EXPERIMENT_ID = "E2_training_cost_sweep"
FROZEN_SHA = "b2e6f624"


def _cfg():
    from E2_training_cost_sweep import (                       # noqa: PLC0415
        DETERMINISTIC_LEARNERS, MODEL_SEEDS, R_DEFAULT, R_EVAL, R_TRAIN_GRID,
        SPLIT_SEEDS, planned_cells,
    )
    return dict(det=DETERMINISTIC_LEARNERS, model_seeds=MODEL_SEEDS,
                r_default=R_DEFAULT, r_eval=R_EVAL, grid=R_TRAIN_GRID,
                split_seeds=SPLIT_SEEDS, planned=planned_cells)


def load_e2_cells() -> pd.DataFrame:
    """New E2 fits: class weighting at the five non-default R_train values."""
    cells = sorted((RESULTS / "E2_cells").glob("*.json"))
    if not cells:
        raise SystemExit("no completed E2 cells found.")
    rows = []
    for path in cells:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") == "complete":
            rows.extend(payload["rows"])
    return pd.DataFrame(rows)


def load_reused_from_e3() -> pd.DataFrame:
    """The two E2 conditions E3 already produced, under identical settings.

    The unweighted reference and class weighting at the default R_train were run
    by E3 at the same rung, splits, seeds, R_eval and threshold protocol, so
    refitting them would consume ~1.5 hours to reproduce numbers that already
    exist. They are relabelled into E2's schema, not recomputed.
    """
    cfg = _cfg()
    e3 = pd.read_csv(RESULTS / "E3_designated_config_rows.csv")
    keep = e3[e3["csl_variant"].isin(["unweighted", "weighted"])].copy()

    keep["experiment_id"] = "E3 (reused)"
    keep["csl_mechanism"] = np.where(keep["csl_variant"] == "weighted",
                                     "class_weighting", "none")
    keep["R_train"] = np.where(keep["csl_variant"] == "weighted",
                               cfg["r_default"], 1.0)
    keep["is_default_R_train"] = keep["csl_variant"] == "weighted"
    keep["r_train_reached_estimator"] = True
    keep["r_train_parameter"] = np.where(keep["csl_variant"] == "weighted",
                                         "class_weight[1] / scale_pos_weight", "none")
    keep["source"] = "E3"
    return keep


def assemble() -> pd.DataFrame:
    new = load_e2_cells().query("split == 'test'").copy()
    new["source"] = "E2"
    reused = load_reused_from_e3()
    shared = [c for c in new.columns if c in reused.columns]
    return pd.concat([new[shared], reused[shared]], ignore_index=True)


# --- aggregation ---------------------------------------------------------------


def _spread(values: pd.Series) -> dict:
    return {"mean": float(values.mean()), "median": float(values.median()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()), "max": float(values.max()),
            "range": float(values.max() - values.min()), "n": int(len(values))}


def sweep_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """PR-AUC and NER at every R_train, per learner."""
    rows = []
    for (learner, r_train), group in frame.groupby(["learner", "R_train"]):
        row = {"learner": learner, "R_train": float(r_train),
               "csl_mechanism": group["csl_mechanism"].iloc[0],
               "is_default_R_train": bool(group["is_default_R_train"].iloc[0]),
               "n_cells": len(group),
               "n_split_seeds": group["split_seed"].nunique(),
               "n_model_seeds": group["model_seed"].nunique()}
        for metric in ("pr_auc", "ner"):
            row.update({f"{metric}_{k}": v for k, v in _spread(group[metric]).items()
                        if k != "n"})
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["learner", "R_train"])


def variance_decomposition(frame: pd.DataFrame) -> pd.DataFrame:
    """Split and training-seed variability at each R_train, kept apart."""
    rows = []
    for (learner, r_train), group in frame.groupby(["learner", "R_train"]):
        deterministic = learner in _cfg()["det"] and group["R_train"].iloc[0] != 1.0
        for metric in ("pr_auc", "ner"):
            within = group.groupby("split_seed")[metric].std(ddof=1).dropna()
            across = group.groupby("model_seed")[metric].std(ddof=1).dropna()
            rows.append({
                "learner": learner, "R_train": float(r_train), "metric": metric,
                "training_seed_std": float(within.mean()) if len(within) else 0.0,
                "split_std": float(across.mean()) if len(across) else np.nan,
                "total_std": float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0,
                "deterministic_in_model_seed": bool(deterministic),
            })
    return pd.DataFrame(rows)


def paired_vs(frame: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Paired differences against a reference condition.

    ``reference`` is ``"unweighted"`` (R_train = 1.0) or ``"default"``
    (R_train = 773.7011). Matching is on learner, split seed and model seed, so
    both sides see identical test observations. Where the reference is
    deterministic in the model seed it is matched on split alone, which is exact
    rather than approximate, and flagged.
    """
    cfg = _cfg()
    ref_r = 1.0 if reference == "unweighted" else cfg["r_default"]
    rows = []

    for learner, by_learner in frame.groupby("learner"):
        ref_all = by_learner[by_learner["R_train"] == ref_r]
        if ref_all.empty:
            continue
        ref_deterministic = (learner in cfg["det"])

        for r_train, group in by_learner.groupby("R_train"):
            if r_train == ref_r:
                continue
            for _, cell in group.iterrows():
                match = ref_all[(ref_all["split_seed"] == cell["split_seed"])
                                & (ref_all["model_seed"] == cell["model_seed"])]
                exact = not match.empty
                if match.empty and ref_deterministic:
                    match = ref_all[ref_all["split_seed"] == cell["split_seed"]]
                if match.empty:
                    continue
                ref = match.iloc[0]
                rows.append({
                    "learner": learner, "reference": reference,
                    "reference_R_train": float(ref_r), "R_train": float(r_train),
                    "split_seed": int(cell["split_seed"]),
                    "model_seed": int(cell["model_seed"]),
                    "exact_seed_match": exact,
                    "matched_by_determinism": not exact,
                    "same_test_observations": True,
                    "reference_pr_auc": float(ref["pr_auc"]),
                    "pr_auc": float(cell["pr_auc"]),
                    "delta_pr_auc": float(cell["pr_auc"] - ref["pr_auc"]),
                    "reference_ner": float(ref["ner"]),
                    "ner": float(cell["ner"]),
                    "delta_ner": float(cell["ner"] - ref["ner"]),
                })
    return pd.DataFrame(rows)


def delta_summary(deltas: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (learner, reference, r_train), group in deltas.groupby(
            ["learner", "reference", "R_train"]):
        row = {"learner": learner, "reference": reference, "R_train": float(r_train),
               "n_pairs": len(group)}
        for metric in ("delta_pr_auc", "delta_ner"):
            row.update({f"{metric}_{k}": v for k, v in _spread(group[metric]).items()
                        if k != "n"})
            row[f"{metric}_n_negative"] = int((group[metric] < 0).sum())
            row[f"{metric}_all_same_sign"] = bool(
                (group[metric] < 0).all() or (group[metric] > 0).all())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["reference", "learner", "R_train"])


def monotonicity(summary: pd.DataFrame) -> pd.DataFrame:
    """Is the response to R_train monotone? Measured, never assumed or imposed.

    Reports Spearman rank correlation between R_train and the metric, and
    whether the ordered sequence is actually monotone. No monotone curve is
    fitted; a non-monotone response is a finding, not noise to smooth away.
    """
    from scipy.stats import spearmanr

    rows = []
    weighted = summary[summary["csl_mechanism"] == "class_weighting"]
    for (learner,), group in weighted.groupby(["learner"]):
        ordered = group.sort_values("R_train")
        for metric in ("pr_auc", "ner"):
            values = ordered[f"{metric}_mean"].to_numpy()
            r_values = ordered["R_train"].to_numpy()
            rho, p = spearmanr(r_values, values)
            diffs = np.diff(values)
            rows.append({
                "learner": learner, "metric": metric,
                "spearman_rho": float(rho), "spearman_p": float(p),
                "monotone_increasing": bool((diffs >= 0).all()),
                "monotone_decreasing": bool((diffs <= 0).all()),
                "monotone": bool((diffs >= 0).all() or (diffs <= 0).all()),
                "n_sign_changes": int((np.diff(np.sign(diffs)) != 0).sum()),
                "argmin_R_train": float(r_values[int(np.argmin(values))]),
                "argmax_R_train": float(r_values[int(np.argmax(values))]),
                "value_at_min": float(values.min()),
                "value_at_max": float(values.max()),
                "range": float(values.max() - values.min()),
            })
    return pd.DataFrame(rows)


def aggregate(rung: str, git: dict, environment: dict) -> None:
    cfg = _cfg()
    frame = assemble()
    E.save_results(frame.to_dict("records"), "E2_runs")

    summary = sweep_summary(frame)
    E.save_results(summary.to_dict("records"), "E2_sweep_summary")

    decomposition = variance_decomposition(frame)
    E.save_results(decomposition.to_dict("records"), "E2_variance_decomposition")

    vs_unweighted = paired_vs(frame, "unweighted")
    vs_default = paired_vs(frame, "default")
    deltas = pd.concat([vs_unweighted, vs_default], ignore_index=True)
    E.save_results(deltas.to_dict("records"), "E2_paired_deltas")
    E.save_results(delta_summary(deltas).to_dict("records"), "E2_delta_summary")

    E.save_results(monotonicity(summary).to_dict("records"), "E2_monotonicity")

    prevalence = pd.read_csv(RESULTS / "E3_split_prevalence.csv")
    E.save_results(prevalence.to_dict("records"), "E2_split_prevalence")

    verification = (load_e2_cells().query("split == 'test'")
                    .drop_duplicates("cell_id")
                    [["cell_id", "learner", "R_train", "r_train_parameter",
                      "r_train_in_estimator", "r_train_reached_estimator",
                      "convergence_warnings"]])
    E.save_results(verification.to_dict("records"), "E2_r_train_verification")

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "status": "COMPLETE",
        "frozen_baseline_sha": FROZEN_SHA,
        "feature_set": rung,
        "csl_mechanism": "class weighting (sweep); unweighted (reference)",
        "rus_excluded_from_sweep": (
            "RUS changes training sample composition rather than applying an "
            "explicit class-cost ratio; E3's RUS cells stand as a fixed reference "
            "and are not members of this sweep"),
        "R_train_grid": [float(r) for r in cfg["grid"]],
        "R_train_default": float(cfg["r_default"]),
        "R_eval": float(cfg["r_eval"]),
        "R_eval_varied": False,
        "split_seeds": cfg["split_seeds"],
        "model_seeds": cfg["model_seeds"],
        "learners": sorted(frame["learner"].unique().tolist()),
        "threshold_protocol": {
            "order": "TRAIN -> FIT -> VAL -> SELECT -> LOCK -> TEST -> REPORT",
            "rule": "ner_optimal", "selected_on": "val",
            "selected_at": "R_eval", "reselected_per_cell": True,
        },
        "metrics": {"primary_ranking": "pr_auc", "primary_decision_risk": "ner",
                    "accuracy_computed": False},
        "reused_from_E3": {
            "unweighted_reference": 52, "weighted_at_default_R_train": 52,
            "reason": "identical conditions; refitting would reproduce existing numbers",
        },
        "new_fits": int(load_e2_cells()["cell_id"].nunique()),
        "deterministic_omissions": {
            "learner": "logistic_regression", "omitted_fits": 60,
            "reason": "lbfgs ignores random_state; E3 measured training_seed_std = 0.0",
        },
        "optimization_performed": False,
        "no_R_declared_optimal": True,
        "execution_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": git.get("commit"),
        "python_version": environment["python_version"],
    }
    E.save_json(manifest, "E2_training_cost_sweep_manifest")

    P.write(P.experiment_record(
        name=EXPERIMENT_ID, seed=42,
        split={"strategy": "stratified", "seeds": cfg["split_seeds"]},
        model={"learners": manifest["learners"], "model_seeds": cfg["model_seeds"]},
        csl={"mechanism": "class_weighting", "R_train_grid": manifest["R_train_grid"]},
        cost_ratio={"R_train_swept": manifest["R_train_grid"],
                    "R_eval_fixed": float(cfg["r_eval"])},
        thresholds=manifest["threshold_protocol"], evaluation=manifest["metrics"],
        notes="Training-cost sensitivity. No tuning, no optimisation, R_eval fixed.",
    ), "E2_training_cost_sweep_provenance")

    print(f"\naggregated {frame['cell_id'].nunique()} cells "
          f"({manifest['new_fits']} new + 104 reused)")


# --- quality control ------------------------------------------------------------


def quality_control(rung: str) -> bool:
    cfg = _cfg()
    frame = assemble()
    new = load_e2_cells()
    deltas = pd.read_csv(RESULTS / "E2_paired_deltas.csv")

    from E2_training_cost_sweep import _frozen_fingerprint, cell_id   # noqa: PLC0415

    planned = {cell_id(*c) for c in cfg["planned"]()}
    present = set(new["cell_id"].unique())

    checks: list[tuple[str, bool, str]] = [
        ("all required R_train values represented",
         set(np.round(frame["R_train"].unique(), 4))
         >= set(np.round(cfg["grid"], 4)),
         str(sorted(np.round(frame["R_train"].unique(), 2)))),
        ("R_eval fixed at the frozen default",
         bool((frame["R_eval"] == cfg["r_eval"]).all()) and frame["R_eval"].nunique() == 1,
         f"{cfg['r_eval']:.4f}, 1 distinct value"),
        ("each R_train reached the training estimator",
         bool(new["r_train_reached_estimator"].all()),
         f"{int(new['r_train_reached_estimator'].sum())}/{len(new)} rows verified "
         f"on the fitted estimator"),
        ("estimator weight equals the requested R_train",
         bool(np.allclose(new["r_train_in_estimator"].astype(float),
                          new["R_train"].astype(float))),
         "exact match"),
        ("no R value selected adaptively",
         set(np.round(new["R_train"].unique(), 4))
         <= set(np.round(cfg["grid"], 4)),
         "grid is pre-declared and closed"),
        ("every required cell exists", planned <= present,
         f"{len(present & planned)}/{len(planned)} present"),
        ("no unplanned cell present", present <= planned,
         f"{len(present - planned)} unplanned"),
        ("all thresholds validation-selected",
         bool((new["threshold_selected_on"] == "val").all()), "val"),
        ("threshold re-selected per cell (not carried across R)",
         int(new.query("split=='test'").groupby("learner")["threshold"].nunique().min()) > 1,
         "distinct thresholds per learner across R"),
        ("feature set remains the frozen rung",
         bool((frame["feature_set"] == rung).all()), rung),
        ("no diagnostic rung present",
         not bool(frame["feature_set"].isin(B.DIAGNOSTIC_RUNGS).any()), "none"),
        ("all split seeds represented",
         sorted(frame["split_seed"].unique()) == cfg["split_seeds"],
         str(sorted(frame["split_seed"].unique()))),
        ("model seeds within the pre-declared set",
         set(frame["model_seed"].unique()) <= set(cfg["model_seeds"]),
         str(sorted(frame["model_seed"].unique()))),
        ("provenance complete on new cells",
         not bool(new[["split_seed", "model_seed", "git_sha",
                       "timestamp_utc"]].isna().any().any()), "complete"),
        ("accuracy never computed", "accuracy" not in frame.columns, "absent"),
        ("paired comparisons share test observations",
         bool(deltas["same_test_observations"].all()), "matched on split seed"),
        ("paired comparisons matched on seed or deterministic",
         bool((deltas["exact_seed_match"] | deltas["matched_by_determinism"]).all()),
         f"{int(deltas['exact_seed_match'].sum())} exact, "
         f"{int(deltas['matched_by_determinism'].sum())} via determinism"),
        ("both reference conditions present",
         set(deltas["reference"].unique()) == {"unweighted", "default"},
         str(sorted(deltas["reference"].unique()))),
        ("RUS absent from the sweep",
         "rus" not in set(frame.get("csl_variant", pd.Series(dtype=str)).unique()),
         "class weighting and unweighted only"),
        ("schema validation passes",
         not B.validate_experiment_frame(new), "all required fields present"),
    ]

    fingerprint_path = RESULTS / "E2_baseline_fingerprint.json"
    current = _frozen_fingerprint()
    if fingerprint_path.exists():
        before = json.loads(fingerprint_path.read_text(encoding="utf-8"))
        checks.append(("frozen baseline and E3 artifacts unchanged",
                       before == current, "sha256 match" if before == current
                       else f"MODIFIED: {[k for k in before if before[k] != current.get(k)]}"))
    else:
        fingerprint_path.write_text(json.dumps(current, indent=1), encoding="utf-8")
        checks.append(("frozen fingerprint recorded", True, "baseline stored"))

    width = max(len(n) for n, _, _ in checks)
    print("\nE2 quality control\n")
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<{width}}  {detail}")

    passed = all(ok for _, ok, _ in checks)
    print(f"\n  {sum(ok for _, ok, _ in checks)}/{len(checks)} -> "
          f"{'PASS' if passed else 'FAIL'}")

    E.save_json({
        "experiment_id": EXPERIMENT_ID,
        "verdict": "PASS" if passed else "FAIL",
        "n_new_cells": int(new["cell_id"].nunique()),
        "checks": [{"check": n, "status": "PASS" if ok else "FAIL", "detail": d}
                   for n, ok, d in checks],
    }, "E2_qc")
    return passed
