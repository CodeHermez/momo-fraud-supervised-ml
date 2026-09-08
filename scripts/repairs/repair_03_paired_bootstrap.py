"""REPAIR 3 -- replace CI-overlap verdicts with a paired bootstrap of the delta.

The audit found the H1 verdict declared a difference when two *marginal* 95%
bootstrap CIs failed to overlap. Non-overlap does imply a difference, but
overlap does not imply its absence: the marginal comparison is strictly weaker
than the paired one and cannot support "indistinguishable" in either direction.
Four of the twelve comparisons were decided that way.

Both models in every comparison were scored on the identical test set, so the
correct procedure resamples once per replicate and evaluates both vectors on
those same rows, cancelling the observation-level noise they share.

No refitting: this reads the cached probability vectors.

Writes ``04_h1_paired_bootstrap.csv`` and ``04_h1_verdict_v2.csv``. The
superseded ``04_h1_pr_auc_ci.csv`` / ``04_h1_verdict.csv`` are left in place.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from momo_fraud import constants as C            # noqa: E402
from momo_fraud import evaluate as E             # noqa: E402
from momo_fraud import provenance as P           # noqa: E402

N_BOOT = 2000
SEED = C.RANDOM_SEED
CSL_VARIANTS = ["weighted", "instance_weighted", "rus"]


def verdict_for(row: dict) -> str:
    """Plain-language reading of the paired interval.

    Deliberately avoids the phrase "statistically significant": this is a
    bootstrap achieved significance level on one test set with one training
    seed, not a test over independent replications of the experiment.
    """
    if not row["excludes_zero"]:
        return "no detectable difference (paired CI spans zero)"
    return ("lower PR-AUC than unweighted" if row["delta"] < 0
            else "higher PR-AUC than unweighted")


def main() -> None:
    rung = json.loads(
        (PROJECT_ROOT / "results" / "03_experimental_rung_v2.json").read_text()
    )["experimental_rung"]
    print(f"rung {rung} | n_boot {N_BOOT} | paired, stratified, shared resamples\n")

    rows = []
    for learner in C.LEARNERS:
        base = E.load_predictions(learner, "unweighted", "test", SEED, tag=rung)
        y = base["y_true"].to_numpy()

        for variant in CSL_VARIANTS:
            other = E.load_predictions(learner, variant, "test", SEED, tag=rung)
            start = time.perf_counter()
            out = E.paired_bootstrap_pr_auc(
                y, other["score"].to_numpy(), base["score"].to_numpy(),
                n_boot=N_BOOT, seed=SEED,
            )
            out.update(learner=learner, variant=variant, comparison=f"{variant} - unweighted")
            out["verdict"] = verdict_for(out)
            rows.append(out)
            print(f"  {learner:20s} {variant:18s} delta {out['delta']:+.4f}  "
                  f"CI [{out['ci_low']:+.4f}, {out['ci_high']:+.4f}]  "
                  f"cross {out['prop_crossing_zero']:.3f}  ({time.perf_counter() - start:.0f}s)")

    paired = pd.DataFrame(rows)[[
        "learner", "variant", "comparison", "delta", "ci_low", "ci_high",
        "n_boot", "n_crossing_zero", "prop_crossing_zero", "p_two_sided",
        "excludes_zero", "verdict",
    ]]
    E.save_results(paired.to_dict("records"), "04_h1_paired_bootstrap")

    # Side-by-side with the superseded verdict, so the change is auditable.
    old = pd.read_csv(PROJECT_ROOT / "results" / "04_h1_verdict.csv")
    merged = paired.merge(
        old.rename(columns={"verdict": "verdict_old_ci_overlap",
                            "delta_pr_auc": "delta_old"}),
        on=["learner", "variant"], how="left",
    )
    merged["verdict_changed"] = (
        merged["verdict_old_ci_overlap"].str.contains("indistinguishable")
        != ~merged["excludes_zero"]
    )
    E.save_results(merged.to_dict("records"), "04_h1_verdict_v2")

    print("\n--- summary ---")
    print(paired["verdict"].value_counts().to_string())
    print(f"\nverdicts changed by the repair: {int(merged['verdict_changed'].sum())} "
          f"of {len(merged)}")
    print(merged.loc[merged["verdict_changed"],
                     ["learner", "variant", "delta", "verdict_old_ci_overlap",
                      "verdict"]].to_string(index=False))

    P.write(P.experiment_record(
        name="04_h1_paired_bootstrap",
        seed=SEED,
        split={"strategy": "stratified", "train": C.SPLIT_TRAIN, "val": C.SPLIT_VAL,
               "test": C.SPLIT_TEST, "seed": SEED},
        model={"learners": C.LEARNERS, "source": "cached probability vectors, no refit"},
        csl={"variants": CSL_VARIANTS, "baseline": "unweighted"},
        cost_ratio=float(C.R_DEFAULT),
        thresholds={"note": "PR-AUC is threshold-free; no threshold involved"},
        evaluation={"metric": "PR-AUC", "procedure": "paired stratified bootstrap",
                    "n_boot": N_BOOT, "alpha": 0.05, "rung": rung},
        notes=("Supersedes 04_h1_pr_auc_ci.csv / 04_h1_verdict.csv, which compared "
               "marginal CIs for overlap."),
    ), "04_h1_paired_bootstrap_provenance")


if __name__ == "__main__":
    main()
