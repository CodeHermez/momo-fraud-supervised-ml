"""REPAIR 2 -- move rung and configuration selection off the test set.

The audit found three decisions taken on test NER: the experimental rung
(notebook 03), the best configuration per learner (notebook 04), and the
interpretability learner (notebook 07). The protocol the study declares is
TRAIN -> FIT -> VALIDATION -> SELECT -> LOCK -> TEST -> REPORT; these read the
last step before taking the fourth.

The pre-declared criterion is unchanged: **best NER >= 0.05, take the
least-ablated viable rung.** It is applied here to validation NER instead of
test NER. Nothing else about it moves -- in particular the floor is not retuned
because the numbers changed.

No model is refitted. ``03_baselines_all_rungs.csv`` already carries both splits,
so this is a re-read of existing results under the correct selection rule.

Writes ``*_v2`` artifacts; the originals are left in place as the record of what
the superseded procedure produced.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from momo_fraud import constants as C            # noqa: E402
from momo_fraud import evaluate as E             # noqa: E402
from momo_fraud import experiment as X           # noqa: E402
from momo_fraud import provenance as P           # noqa: E402
from momo_fraud.features import FEATURE_SETS, DIAGNOSTIC_FEATURE_SETS  # noqa: E402

HEADROOM_FLOOR = 0.05      # pre-declared in notebook 03; unchanged by this repair
RESULTS = PROJECT_ROOT / "results"


def main() -> None:
    baselines = pd.read_csv(RESULTS / "03_baselines_all_rungs.csv")
    ladder_order = [r for r in FEATURE_SETS if r not in DIAGNOSTIC_FEATURE_SETS]

    # Config C only: the NER-optimal threshold is what "best achievable NER"
    # means, and mixing in the fixed-0.5 configs would understate every rung.
    config_c = baselines.query("config == 'C'")

    val_headroom = X.headroom_by_rung(config_c, split="val").reindex(ladder_order).dropna()
    test_headroom = X.headroom_by_rung(config_c, split="test").reindex(ladder_order).dropna()

    chosen, rationale = X.choose_rung(val_headroom, ladder_order, floor=HEADROOM_FLOOR)
    chosen_on_test, _ = X.choose_rung(test_headroom, ladder_order, floor=HEADROOM_FLOOR)

    ladder = pd.DataFrame({
        "feature_set": val_headroom.index,
        "best_ner_val": val_headroom.values,
        "best_ner_test": test_headroom.reindex(val_headroom.index).values,
        "viable_on_val": val_headroom.values >= HEADROOM_FLOOR,
    })
    E.save_results(ladder.to_dict("records"), "03_ladder_headroom_v2")
    print(ladder.round(4).to_string(index=False))

    decision = {
        "experimental_rung": chosen,
        "selected_on": "val",
        "best_ner_at_rung_val": float(val_headroom[chosen]),
        "best_ner_at_rung_test": float(test_headroom[chosen]),
        "headroom_floor": HEADROOM_FLOOR,
        "rationale": rationale,
        "features_dropped": FEATURE_SETS[chosen],
        "supersedes": "03_experimental_rung.json (selected on test NER)",
        "rung_if_selected_on_test": chosen_on_test,
        "selection_split_changes_the_rung": chosen != chosen_on_test,
        "criterion_unchanged_from_original": True,
    }
    E.save_json(decision, "03_experimental_rung_v2")

    print(f"\nrung on VALIDATION : {chosen}  (val NER {val_headroom[chosen]:.4f})")
    print(f"rung on TEST (old) : {chosen_on_test}")
    print("UNCHANGED -- the audit's C3 has no effect on which rung was used."
          if chosen == chosen_on_test else
          "CHANGED -- every downstream result at the old rung must be rerun.")

    # --- Repair 2b: best configuration per learner, selected on validation ---
    cs = pd.read_csv(RESULTS / "04_cost_sensitive.csv").query("feature_set == @chosen")
    best_val = X.select_best_config(cs, metric="ner", on="val")

    test_rows = cs.query("split == 'test'")[
        ["learner", "config", "ner", "recall", "precision", "n_flagged", "pr_auc"]
    ].rename(columns={"ner": "ner_test"})
    best = best_val.merge(test_rows, on=["learner", "config"], how="left")

    # What the superseded procedure would have picked, for comparison only.
    best["config_if_selected_on_test"] = [
        cs.query("split == 'test' and learner == @lr").nsmallest(1, "ner")["config"].iloc[0]
        for lr in best["learner"]
    ]
    best["selection_split_changes_config"] = best["config"] != best["config_if_selected_on_test"]
    E.save_results(best.to_dict("records"), "04_best_config_per_learner_v2")

    print("\nLowest-risk configuration per learner (selected on validation):\n")
    print(best.round(4).to_string(index=False))

    # --- Repair 2c: interpretability learner, selected on validation ---------
    interp_learner = best_val.nsmallest(1, "ner_val")["learner"].iloc[0]
    interp_on_test = (cs.query("split == 'test' and config == 'C'")
                      .nsmallest(1, "ner")["learner"].iloc[0])
    E.save_json({
        "interpretability_learner": interp_learner,
        "selected_on": "val",
        "metric": "ner",
        "learner_if_selected_on_test": interp_on_test,
        "selection_split_changes_learner": interp_learner != interp_on_test,
        "supersedes": "notebook 07 LEARNER constant ('lowest NER in notebook 04', test)",
    }, "07_learner_selection_v2")
    print(f"\ninterpretability learner on VALIDATION: {interp_learner} "
          f"(on test it would be {interp_on_test})")

    P.write(P.experiment_record(
        name="validation_based_selection_v2",
        split={"strategy": "stratified", "train": C.SPLIT_TRAIN,
               "val": C.SPLIT_VAL, "test": C.SPLIT_TEST, "seed": C.RANDOM_SEED},
        model={"learners": C.LEARNERS},
        csl={"configs": list(C.CONFIGS)},
        cost_ratio=float(C.R_DEFAULT),
        thresholds={"rule": "ner_optimal", "selected_on": "val"},
        evaluation={"selection_split": "val", "reporting_split": "test",
                    "criterion": f"best NER >= {HEADROOM_FLOOR}, least-ablated viable rung"},
        notes="No refitting: re-reads 03_baselines_all_rungs.csv and 04_cost_sensitive.csv.",
    ), "03_experimental_rung_v2_provenance")


if __name__ == "__main__":
    main()
