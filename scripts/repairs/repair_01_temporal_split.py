"""REPAIR 1 -- regenerate the temporal robustness check on a valid split.

The audit found the temporal split realised 95.6 / 3.0 / 1.4 with a test
prevalence of 1.391%, invalidating every temporal number in
``results/09_velocity_vs_baseline.csv``. ``constants.TEMPORAL_TRAIN_END`` /
``TEMPORAL_VAL_END`` are now derived from the file's cumulative row count
(322 / 377 -> 69.68 / 15.30 / 15.02) and ``splits.temporal_split`` asserts it.

Hitting the row proportions does not fix the class prior: PaySim's fraud
prevalence is non-stationary, so the partitions carry 0.082% / 0.059% / 0.420%
fraud and the test partition's class ratio is 237.3 against a population 773.7.
NER scores "flag nothing" at 1.0 for any R, but "flag everything" reaches 1.0 --
and Youden coincides with the NER-optimal threshold -- only at R = N_neg/N_pos of
the evaluated partition. So this reports **three** cost ratios rather than
silently picking one:

    R_default    = 773.70  population ratio; the study's declared cost
                            assumption, and the only one comparable with the
                            stratified arm
    R_train      = 1219.40 the temporal *training* partition's ratio; a
                            split-specific choice that uses no test information
    R_test       =  237.34 the temporal *test* partition's ratio; restores NER's
                            break-even anchor, at the cost of reading a marginal
                            count off the evaluation partition

Thresholds are selected on validation at whichever R is being reported, frozen,
then applied once to test.

Writes ``09_temporal_repaired.csv`` and ``09_temporal_prevalence.csv``. The
superseded file is left untouched.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from momo_fraud import constants as C            # noqa: E402
from momo_fraud import data as D                 # noqa: E402
from momo_fraud import evaluate as E             # noqa: E402
from momo_fraud import experiment as X           # noqa: E402
from momo_fraud import provenance as P           # noqa: E402
from momo_fraud import splits as S               # noqa: E402
from momo_fraud.features import FeatureBuilder, columns_for   # noqa: E402
from momo_fraud.velocity import VELOCITY_FEATURES, build_velocity_features  # noqa: E402

RUNG = "no_origin_balance"
SEED = C.RANDOM_SEED


def main() -> None:
    df = D.load()
    y = df[C.TARGET].to_numpy()

    derived = S.derive_temporal_boundaries(df["step"])
    print(f"derived boundaries {derived} (shipped: "
          f"{C.TEMPORAL_TRAIN_END}/{C.TEMPORAL_VAL_END})")

    split = S.temporal_split(df["step"])          # raises if outside tolerance
    sizes, prevalence, ratios = split.sizes(), split.prevalence(y), split.class_ratio(y)
    n = len(df)

    prevalence_rows = [
        {"partition": part, "n_rows": sizes[part], "share": sizes[part] / n,
         "n_fraud": int(y[getattr(split, part)].sum()),
         "prevalence": prevalence[part], "class_ratio": ratios[part]}
        for part in ("train", "val", "test")
    ]
    print(pd.DataFrame(prevalence_rows).to_string(index=False))
    E.save_results(prevalence_rows, "09_temporal_prevalence")

    cost_ratios = {
        "R_default": float(C.R_DEFAULT),
        "R_train_ratio": float(ratios["train"]),
        "R_test_ratio": float(ratios["test"]),
    }
    print(f"\ncost ratios reported: {cost_ratios}\n")

    # Fit on temporal training rows only -- the builder's high-risk cutoff and
    # severity grid are learned state and must not see val or test.
    builder = FeatureBuilder().fit(df.iloc[split.train])
    X_all = builder.transform(df)
    cols_base = columns_for(RUNG, list(X_all.columns))

    vel = build_velocity_features(df)             # row-causal; no fitted state
    X_vel = pd.concat([X_all, vel], axis=1)

    arms = {
        "no_velocity": (X_all, cols_base),
        "velocity": (X_vel, cols_base + VELOCITY_FEATURES),
    }
    labels = {k: y[getattr(split, k)] for k in ("train", "val", "test")}

    rows = []
    for arm, (frame, cols) in arms.items():
        parts = {k: frame.iloc[getattr(split, k)][cols] for k in ("train", "val", "test")}
        for learner in C.LEARNERS:
            start = time.perf_counter()
            fit = X.fit_one(learner, "unweighted", parts["train"], labels["train"],
                            parts["val"], parts["test"], r=C.R_DEFAULT, seed=SEED)
            E.save_predictions(labels["val"], fit.val_scores, learner=learner,
                               config="unweighted", split="val", seed=SEED,
                               tag=f"{RUNG}_temporal_{arm}")
            E.save_predictions(labels["test"], fit.test_scores, learner=learner,
                               config="unweighted", split="test", seed=SEED,
                               tag=f"{RUNG}_temporal_{arm}")

            for r_name, r in cost_ratios.items():
                # Threshold on validation at this R, frozen, applied to test.
                chosen = E.select_threshold(labels["val"], fit.val_scores, "ner_optimal", r)
                rows.append(E.full_report(
                    labels["test"], fit.test_scores, threshold=chosen.threshold, r=r,
                    learner=learner, config="C", variant="unweighted", level="decision",
                    threshold_rule="ner_optimal", split="test", seed=SEED,
                    split_strategy="temporal", with_velocity=(arm == "velocity"),
                    cost_ratio_name=r_name, feature_set=f"{RUNG}_temporal_{arm}",
                    fit_seconds=fit.fit_seconds, n_train=fit.n_train,
                ))
            print(f"  {arm:12s} {learner:20s} {time.perf_counter() - start:6.0f}s")

    out = pd.DataFrame(rows)
    E.save_results(out.to_dict("records"), "09_temporal_repaired")

    P.write(P.experiment_record(
        name="09_temporal_repaired",
        seed=SEED,
        split={"strategy": "temporal", "train_end": C.TEMPORAL_TRAIN_END,
               "val_end": C.TEMPORAL_VAL_END, "derived_boundaries": list(derived),
               "tolerance": C.TEMPORAL_PROPORTION_TOLERANCE,
               "sizes": sizes, "prevalence": prevalence, "class_ratio": ratios},
        model={"learners": C.LEARNERS, "variant": "unweighted",
               "hyperparameters": "models.make_learner defaults"},
        csl={"level": "decision only", "method": "ner_optimal threshold"},
        cost_ratio=cost_ratios,
        thresholds={"rule": "ner_optimal", "selected_on": "val", "applied_to": "test"},
        evaluation={"feature_sets": list(arms), "rung": RUNG,
                    "primary_metric": "ner", "reported": "test only"},
        notes=("Supersedes the temporal rows of 09_velocity_vs_baseline.csv, which "
               "used boundaries 520/632 realising 95.6/3.0/1.4."),
    ), "09_temporal_repaired_provenance")

    print("\n--- NER by cost ratio (test) ---")
    print(out.pivot_table(index=["learner", "with_velocity"],
                          columns="cost_ratio_name", values="ner").round(4).to_string())


if __name__ == "__main__":
    main()
