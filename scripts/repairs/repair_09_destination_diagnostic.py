"""REPAIR 9 -- diagnostic: how much does the destination-side balance state carry?

The audit (M3) found the ablation ladder asymmetric. Origin balances were removed
with the argument that a tree reconstructs the drain rule from the raw columns
even after the derived error feature is gone. The same argument applies verbatim
to the destination side, where the simulator never credits the receiving account
at all, but it was not applied there. The experimental rung therefore retains
oldbalanceDest, newbalanceDest, destZeroBefore and destZeroAfter, and
results/09_velocity_permutation_importance.csv ranks destZeroAfter second overall.

**This is a diagnostic, not a change of feature set.** no_balance_state is not
promoted to the main experimental rung here: that would change the feature basis
of every existing CSL result, and it is a research decision to be taken
explicitly on evidence rather than as a side effect of a repair pass.

Three questions, in order:

1. Is the destination-side signal a bookkeeping artefact? Measured the way
   notebook 02 measured the origin side -- class-conditional rates and a
   single-feature out-of-sample stump AUC -- but computed on **training rows
   only**, unlike the original screen which used all 6.36M rows.
2. What does removing it cost? no_origin_balance against no_balance_state on the
   same learners, thresholds selected on validation.
3. Does the deeper rung still leave headroom for a cost-sensitive comparison?
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from momo_fraud import analysis as A             # noqa: E402
from momo_fraud import constants as C            # noqa: E402
from momo_fraud import data as D                 # noqa: E402
from momo_fraud import evaluate as E             # noqa: E402
from momo_fraud import experiment as X           # noqa: E402
from momo_fraud import provenance as P           # noqa: E402
from momo_fraud import splits as S               # noqa: E402
from momo_fraud.features import (                # noqa: E402
    DEST_BALANCE_FEATURES, FeatureBuilder, columns_for,
)

SEED = C.RANDOM_SEED
R = C.R_DEFAULT
RUNGS = ["no_origin_balance", "no_balance_state"]


def main() -> None:
    df = D.load()
    y = df[C.TARGET].to_numpy()
    split = S.stratified_split(y)

    builder = FeatureBuilder().fit(df.iloc[split.train])
    X_all = builder.transform(df)

    # --- Q1: is it a bookkeeping artefact? ----------------------------------
    train_df = df.iloc[split.train]
    train_y = y[split.train]
    fraud = train_y.astype(bool)

    old_dest = train_df["oldbalanceDest"].to_numpy()
    new_dest = train_df["newbalanceDest"].to_numpy()
    amount = train_df["amount"].to_numpy()

    checks = {
        "dest_zero_before": old_dest == 0,
        "dest_zero_after": new_dest == 0,
        "dest_zero_both": (old_dest == 0) & (new_dest == 0),
        "dest_not_credited": np.abs(old_dest + amount - new_dest) > 1.0,
        "dest_balance_unchanged": np.abs(new_dest - old_dest) <= 1.0,
    }
    rates = pd.DataFrame([
        {"condition": name,
         "rate_legitimate": float(mask[~fraud].mean()),
         "rate_fraud": float(mask[fraud].mean()),
         "lift_fraud_over_legit": float(mask[fraud].mean() / max(mask[~fraud].mean(), 1e-12))}
        for name, mask in checks.items()
    ])
    E.save_results(rates.to_dict("records"), "09_destination_artefact_rates")
    print("Destination-side class-conditional rates (TRAINING rows only):")
    print(rates.round(4).to_string(index=False))
    print()

    # Solo stump AUC on training rows only. The original screen in notebook 02
    # ran on all 6.36M rows, which the audit flagged as a feature-admissibility
    # decision computed partly on the test partition.
    dest_cols = [c for c in DEST_BALANCE_FEATURES if c in X_all.columns]
    auc = A.single_feature_auc(X_all.iloc[split.train][dest_cols], train_y)
    E.save_results(auc.to_dict("records"), "09_destination_single_feature_auc")
    print("Solo out-of-sample stump AUC (training rows only):")
    print(auc.round(4).to_string(index=False))
    print()

    # --- Q2/Q3: what does removing it cost? ---------------------------------
    labels = {k: y[getattr(split, k)] for k in ("train", "val", "test")}
    rows = []
    for rung in RUNGS:
        cols = columns_for(rung, list(X_all.columns))
        parts = {k: X_all.iloc[getattr(split, k)][cols] for k in ("train", "val", "test")}
        print(f"--- {rung} ({len(cols)} features) ---")
        for learner in C.LEARNERS:
            start = time.perf_counter()
            fit = X.fit_one(learner, "unweighted", parts["train"], labels["train"],
                            parts["val"], parts["test"], r=R, seed=SEED)
            chosen = E.select_threshold(labels["val"], fit.val_scores, "ner_optimal", R)
            for split_name, y_true, scores in (
                ("val", labels["val"], fit.val_scores),
                ("test", labels["test"], fit.test_scores),
            ):
                rows.append(E.full_report(
                    y_true, scores, threshold=chosen.threshold, r=R,
                    learner=learner, config="C", variant="unweighted", level="decision",
                    threshold_rule="ner_optimal", split=split_name, seed=SEED,
                    feature_set=rung, fit_seconds=fit.fit_seconds, n_train=fit.n_train,
                ))
            print(f"  {learner:20s} {time.perf_counter() - start:6.0f}s  "
                  f"test NER {rows[-1]['ner']:.4f}  PR-AUC {rows[-1]['pr_auc']:.4f}")

    out = pd.DataFrame(rows)
    E.save_results(out.to_dict("records"), "09_destination_ablation")

    test = out.query("split == 'test'").pivot_table(
        index="learner", columns="feature_set", values=["ner", "pr_auc"])
    print("\n--- test NER / PR-AUC by rung ---")
    print(test.round(4).to_string())

    headroom = X.headroom_by_rung(out, split="val")
    print("\n--- validation headroom ---")
    print(headroom.round(4).to_string())
    print(f"\nboth rungs clear the 0.05 floor: {bool((headroom >= 0.05).all())}")
    print("\nno_balance_state is NOT promoted to the main rung by this script.")

    P.write(P.experiment_record(
        name="09_destination_diagnostic",
        seed=SEED,
        split={"strategy": "stratified", "train": C.SPLIT_TRAIN,
               "val": C.SPLIT_VAL, "test": C.SPLIT_TEST, "seed": SEED},
        model={"learners": C.LEARNERS, "variant": "unweighted",
               "hyperparameters": "models.make_learner defaults"},
        csl={"level": "decision only", "method": "ner_optimal threshold"},
        cost_ratio=float(R),
        thresholds={"rule": "ner_optimal", "selected_on": "val", "applied_to": "test"},
        evaluation={"rungs": RUNGS, "purpose": "diagnostic only",
                    "promoted_to_main_rung": False},
        notes=("Diagnostic for audit finding M3. The solo-AUC screen here runs on "
               "training rows only, unlike the original in notebook 02."),
    ), "09_destination_diagnostic_provenance")


if __name__ == "__main__":
    main()
