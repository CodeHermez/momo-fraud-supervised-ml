"""REPAIR 4 and 6 -- relabel the R-sweep, and separate a failed run from a comparison.

**Repair 4.** ``results/06_r_sweep.csv`` reads as a cost-sensitive-learning
sensitivity analysis. It is not one. Every weighted / instance-weighted / RUS
model in all 36 cells was trained once, at R = 773.70 (notebook 04 calls
``run_grid(..., r=C.R_DEFAULT)``); only the threshold moves with R. The sweep
therefore varies the *decision* cost ratio against a fixed *training* cost
ratio, and a phrase like "CSL wins at R = 2000" attributes to training an effect
that was produced at the decision layer.

The data are correct and are preserved unchanged. What changes is the naming and
an explicit ``R_train`` column, so the distinction cannot be lost again by
reading the file without its notebook.

**Repair 6.** ``results/10_summary.csv`` lists LightGBM at PR-AUC 0.043 -- a fit
that failed, not a model that lost. Presenting it in a table of approaches
invites the reader to treat it as a comparison. It moves to its own file marked
as a failed run, and is dropped from the summary.

Neither repair refits anything.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from momo_fraud import constants as C            # noqa: E402
from momo_fraud import evaluate as E             # noqa: E402
from momo_fraud import provenance as P           # noqa: E402

RESULTS = PROJECT_ROOT / "results"
R_TRAIN_USED = float(C.R_DEFAULT)


def repair_r_sweep() -> None:
    sweep = pd.read_csv(RESULTS / "06_r_sweep.csv")

    renamed = sweep.rename(columns={"R": "R_decision"})
    renamed.insert(1, "R_train", R_TRAIN_USED)
    renamed["training_cost_ratio_varied"] = False
    renamed["sweep_layer"] = "decision"

    E.save_results(renamed.to_dict("records"), "06_decision_cost_sweep")

    # The count the write-up quotes, restated with the correct attribution.
    piv = renamed.pivot_table(index=["learner", "R_decision"], columns="variant",
                              values="ner")
    csl_cols = ["weighted", "instance_weighted", "rus"]
    wins = (piv[csl_cols].min(axis=1) < piv["unweighted"])

    margin = (piv["unweighted"] - piv[csl_cols].min(axis=1)).rename("margin")
    summary = pd.concat([wins.rename("csl_lower_ner"), margin], axis=1).reset_index()
    summary["R_train"] = R_TRAIN_USED
    E.save_results(summary.to_dict("records"), "06_decision_cost_sweep_margins")

    print(f"decision-cost sweep: CSL-trained models reach lower NER in "
          f"{int(wins.sum())} of {len(wins)} cells")
    print("  (all such models were trained at a single R_train = "
          f"{R_TRAIN_USED:.2f}; only the decision threshold varies)\n")

    P.write(P.experiment_record(
        name="06_decision_cost_sweep",
        seed=C.RANDOM_SEED,
        split={"strategy": "stratified", "seed": C.RANDOM_SEED},
        model={"source": "cached vectors from notebook 04; no refit"},
        csl={"training_cost_ratio": R_TRAIN_USED,
             "training_cost_ratio_varied": False,
             "variants": ["unweighted", "weighted", "instance_weighted", "rus"]},
        cost_ratio={"R_decision_swept": [float(r) for r in C.R_SWEEP],
                    "R_train_fixed": R_TRAIN_USED},
        thresholds={"rule": "ner_optimal", "selected_on": "val", "applied_to": "test"},
        evaluation={"metric": "ner"},
        notes=("Renames 06_r_sweep.csv. This is a decision-cost / threshold "
               "evaluation sweep, NOT a training-cost CSL sweep. A training-R "
               "sweep remains an open experiment (audit E2)."),
    ), "06_decision_cost_sweep_provenance")


def repair_lightgbm() -> None:
    summary = pd.read_csv(RESULTS / "10_summary.csv")
    is_lgbm = summary["approach"].str.contains("LightGBM", case=False, na=False)

    failed = summary[is_lgbm].copy()
    failed["status"] = "FAILED RUN - not a model comparison"
    failed["reason"] = (
        "PR-AUC 0.043 at 0.129% prevalence indicates the fit did not learn the "
        "minority class under matched-budget hyperparameters, rather than a "
        "learner that underperformed. Not diagnosed and not rerun: the audit "
        "found no research question depending on it."
    )
    E.save_results(failed.to_dict("records"), "10_failed_runs")

    kept = summary[~is_lgbm].drop(columns=["beats_baseline"], errors="ignore").copy()
    kept["beats_baseline"] = kept["ner"] < kept.loc[
        kept["approach"].str.contains("shipped baseline"), "ner"].iloc[0]
    E.save_results(kept.to_dict("records"), "10_summary_v2")

    print("10_summary_v2.csv (LightGBM removed to 10_failed_runs.csv):")
    print(kept.round(4).to_string(index=False))


if __name__ == "__main__":
    repair_r_sweep()
    repair_lightgbm()
