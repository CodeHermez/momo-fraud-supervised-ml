"""REPAIR 1 (part b) -- the stratified vs temporal comparison, restated honestly.

``results/09_velocity_vs_baseline.csv`` compared stratified against temporal on
a temporal split that realised 95.6 / 3.0 / 1.4 with a test partition at 1.391%
prevalence. It reported temporal as roughly three times *better* (XGBoost NER
0.0398 against 0.1314), which is the opposite of what an honest time-ordered
split should show.

With the boundaries derived rather than assumed, the comparison reverses:
temporal is harder for three of four learners, which is what training on the
past and testing on the future is supposed to cost.

This assembles the comparison table from the repaired temporal run and the
existing stratified results. It refits nothing.
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
RUNG = "no_origin_balance"

METRICS = ["ner", "pr_auc", "precision", "recall", "n_flagged"]


def main() -> None:
    temporal = pd.read_csv(RESULTS / "09_temporal_repaired.csv")
    stratified = pd.read_csv(RESULTS / "04_cost_sensitive.csv").query(
        "split == 'test' and feature_set == @RUNG and config == 'C'")
    strat_velocity = pd.read_csv(RESULTS / "09_velocity_vs_baseline.csv").query(
        "split_strategy == 'stratified'")
    superseded = pd.read_csv(RESULTS / "09_velocity_vs_baseline.csv").query(
        "split_strategy == 'temporal'")

    rows = []

    # Stratified, no velocity -- from the main grid.
    for _, r in stratified.iterrows():
        rows.append({"learner": r["learner"], "split_strategy": "stratified",
                     "with_velocity": False, "cost_ratio_name": "R_default",
                     "R": float(C.R_DEFAULT),
                     **{m: r[m] for m in METRICS}})

    # Stratified, with velocity -- from the notebook-09 stratified arm, which the
    # audit left valid.
    for _, r in strat_velocity.query("with_velocity").iterrows():
        rows.append({"learner": r["learner"], "split_strategy": "stratified",
                     "with_velocity": True, "cost_ratio_name": "R_default",
                     "R": float(C.R_DEFAULT),
                     **{m: r.get(m) for m in METRICS}})

    # Temporal -- repaired, at all three reported cost ratios.
    for _, r in temporal.iterrows():
        rows.append({"learner": r["learner"], "split_strategy": "temporal",
                     "with_velocity": bool(r["with_velocity"]),
                     "cost_ratio_name": r["cost_ratio_name"], "R": float(r["R"]),
                     **{m: r[m] for m in METRICS}})

    comparison = pd.DataFrame(rows)

    # Velocity delta within each (split strategy, cost ratio) cell -- comparing
    # across strategies would confound the feature question with the split.
    def velocity_delta(row):
        if not row["with_velocity"]:
            return None
        match = comparison[
            (comparison["learner"] == row["learner"])
            & (comparison["split_strategy"] == row["split_strategy"])
            & (comparison["cost_ratio_name"] == row["cost_ratio_name"])
            & (~comparison["with_velocity"])
        ]
        return None if match.empty else row["ner"] - match["ner"].iloc[0]

    comparison["delta_ner_vs_no_velocity"] = comparison.apply(velocity_delta, axis=1)
    E.save_results(comparison.to_dict("records"), "09_split_strategy_comparison")

    # The headline correction, at the study's declared cost ratio.
    headline = (comparison.query("cost_ratio_name == 'R_default' and not with_velocity")
                .pivot_table(index="learner", columns="split_strategy", values="ner"))
    headline["temporal_harder"] = headline["temporal"] > headline["stratified"]
    headline = headline.merge(
        superseded.query("~with_velocity").set_index("learner")["ner"]
                  .rename("temporal_superseded"),
        left_index=True, right_index=True, how="left")

    E.save_results(headline.reset_index().to_dict("records"),
                   "09_temporal_correction")

    print("Stratified vs temporal at R = 773.70 (config C, no velocity):\n")
    print(headline.round(4).to_string())
    print(f"\ntemporal harder for {int(headline['temporal_harder'].sum())} "
          f"of {len(headline)} learners")
    print("\nThe superseded column came from boundaries 520/632, which realised "
          "95.6/3.0/1.4\nwith a test partition at 1.391% prevalence -- a base-rate "
          "artefact, not a finding.")

    P.write(P.experiment_record(
        name="09_split_strategy_comparison",
        seed=C.RANDOM_SEED,
        split={"stratified": {"train": C.SPLIT_TRAIN, "val": C.SPLIT_VAL,
                              "test": C.SPLIT_TEST, "seed": C.RANDOM_SEED},
               "temporal": {"train_end": C.TEMPORAL_TRAIN_END,
                            "val_end": C.TEMPORAL_VAL_END,
                            "realised_shares": [0.6968, 0.1530, 0.1502]}},
        model={"learners": C.LEARNERS, "variant": "unweighted", "config": "C"},
        cost_ratio={"R_default": float(C.R_DEFAULT),
                    "temporal_also_reported_at": ["R_train_ratio", "R_test_ratio"]},
        thresholds={"rule": "ner_optimal", "selected_on": "val"},
        evaluation={"rung": RUNG, "assembled_from": [
            "09_temporal_repaired.csv", "04_cost_sensitive.csv",
            "09_velocity_vs_baseline.csv (stratified rows only)"]},
        notes="No refitting. Supersedes the temporal rows of 09_velocity_vs_baseline.csv.",
    ), "09_split_strategy_comparison_provenance")


if __name__ == "__main__":
    main()
