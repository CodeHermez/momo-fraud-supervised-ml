"""REPAIR 7 -- fit notebook 05's FeatureBuilder on a training partition only.

The audit (M7) found ``notebooks/05_lokanan_replication.ipynb`` calls
``FeatureBuilder().fit(subset)`` before ``subset`` is divided, so the builder's
learned state -- the 99th-percentile ``amount`` cutoff and the severity quantile
grid -- is fitted on rows that later become Arm A's and Arm C's evaluation
partitions. It is the one place in the codebase that breaks the fit-on-train-only
invariant, and it lands in Arm C, which is presented as the *correct* protocol.

This script does two things:

1. Establishes whether the leak moved any number, by building the model matrix
   with a builder fit on everything and with one fit on the training partition
   only, and comparing the columns notebook 05 actually models.

2. Records the result either way, so the fix is accompanied by a measurement of
   what it was worth rather than an assumption.

The expectation going in is that it moved nothing, because ``LOKANAN_RAW_FEATURES``
contains no feature derived from fitted state -- ``highRiskFlag`` is the only
column that uses ``high_risk_cutoff`` and it is not in that list, and severity is
never passed to ``full_report`` in that notebook. Whether that holds is what this
checks rather than asserts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from momo_fraud import constants as C            # noqa: E402
from momo_fraud import data as D                 # noqa: E402
from momo_fraud import evaluate as E             # noqa: E402
from momo_fraud import provenance as P           # noqa: E402
from momo_fraud.features import LOKANAN_RAW_FEATURES, FeatureBuilder  # noqa: E402

REPLICATION_N = 200_000
TRAIN_FRACTION = 0.60
SEED = C.RANDOM_SEED


def main() -> None:
    df = D.load()
    y_full = df[C.TARGET].to_numpy()

    keep, _ = train_test_split(np.arange(len(df)), train_size=REPLICATION_N,
                               stratify=y_full, random_state=SEED)
    subset = df.iloc[np.sort(keep)].reset_index(drop=True)
    y = subset[C.TARGET].to_numpy()

    idx_tr, idx_te = train_test_split(np.arange(len(y)), train_size=TRAIN_FRACTION,
                                      stratify=y, random_state=SEED)

    leaky = FeatureBuilder().fit(subset)                    # what notebook 05 did
    clean = FeatureBuilder().fit(subset.iloc[idx_tr])       # what it should do

    X_leaky = leaky.transform(subset)[LOKANAN_RAW_FEATURES]
    X_clean = clean.transform(subset)[LOKANAN_RAW_FEATURES]

    identical = X_leaky.equals(X_clean)
    differing = [c for c in LOKANAN_RAW_FEATURES if not X_leaky[c].equals(X_clean[c])]

    # The fitted state itself does differ -- that is the leak. What matters is
    # whether any *modelled* column carries it.
    state = {
        "high_risk_cutoff_fit_on_all": float(leaky.high_risk_cutoff),
        "high_risk_cutoff_fit_on_train": float(clean.high_risk_cutoff),
        "cutoff_relative_difference": float(
            abs(leaky.high_risk_cutoff - clean.high_risk_cutoff)
            / max(leaky.high_risk_cutoff, 1e-9)),
        "severity_grid_identical": bool(
            np.array_equal(leaky.severity.quantiles, clean.severity.quantiles)),
    }

    fitted_state_columns = sorted(
        set(leaky.feature_names_) - set(LOKANAN_RAW_FEATURES)
    )

    record = {
        "repair": "REPAIR 7 -- notebook 05 FeatureBuilder fitted on training rows only",
        "audit_finding": "M7",
        "modelled_columns": LOKANAN_RAW_FEATURES,
        "modelled_matrix_identical": bool(identical),
        "columns_that_differ": differing,
        "fitted_state": state,
        "columns_excluded_from_the_lokanan_feature_set": fitted_state_columns,
        "severity_used_in_notebook_05": False,
        "conclusion": (
            "The leak was real but inert: the builder's fitted state reaches only "
            "highRiskFlag and the severity grid, neither of which is in "
            "LOKANAN_RAW_FEATURES or passed to full_report in notebook 05. The "
            "model matrix is bit-identical either way, so no Arm A, B or C number "
            "changes and no rerun is required."
            if identical else
            "The leak moved the model matrix. Arms A and C must be rerun."
        ),
        "rerun_required": not identical,
    }
    E.save_json(record, "05_builder_fit_repair")

    print(f"modelled matrix identical      : {identical}")
    print(f"columns that differ            : {differing or 'none'}")
    print(f"high-risk cutoff (all rows)    : {state['high_risk_cutoff_fit_on_all']:,.2f}")
    print(f"high-risk cutoff (train only)  : {state['high_risk_cutoff_fit_on_train']:,.2f}")
    print(f"  relative difference          : {state['cutoff_relative_difference']:.4%}")
    print(f"severity grids identical       : {state['severity_grid_identical']}")
    print(f"\nfitted-state columns not modelled by notebook 05: {fitted_state_columns}")
    print(f"\nrerun required: {record['rerun_required']}")

    P.write(P.experiment_record(
        name="05_builder_fit_repair",
        seed=SEED,
        split={"strategy": "stratified subsample then 60/40",
               "replication_n": REPLICATION_N, "train_fraction": TRAIN_FRACTION},
        model={"note": "no model fitted; feature-matrix comparison only"},
        evaluation={"compared": "FeatureBuilder fit on all rows vs training rows only",
                    "columns": LOKANAN_RAW_FEATURES},
        notes="Measures whether audit finding M7 moved any reported number.",
    ), "05_builder_fit_repair_provenance")


if __name__ == "__main__":
    main()
