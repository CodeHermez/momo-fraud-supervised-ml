"""Build the small labelled slice the prototype ships with.

The raw PaySim CSV is 470 MB and gitignored, so the prototype cannot read it at
demo time -- and a Docker image should not carry it. This script distils the
held-out test split down to a few megabytes: every fraud, plus a random sample
of legitimate traffic, each row carrying the score the experiment actually
recorded for it.

Run once, from the repo root, after the raw data is in place:

    python prototype/scripts/build_demo_sample.py

The alignment this depends on is worth stating plainly. ``stratified_split``
returns *sorted* index arrays, and the prediction parquet was written from the
test split in that order with the index reset. So row *i* of the parquet is row
``split.test[i]`` of the raw frame. That is asserted below rather than trusted:
a silent misalignment would relabel every row in the explorer, which is the one
failure mode that would make the demo actively lie.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from momo_fraud import constants as C  # noqa: E402
from momo_fraud import data as D  # noqa: E402
from momo_fraud import splits  # noqa: E402

PREDICTIONS = (
    PROJECT_ROOT
    / "predictions"
    / "xgboost__unweighted__test__seed42__no_origin_balance.parquet"
)
OUT_PATH = PROJECT_ROOT / "prototype" / "data" / "demo_sample.parquet"

#: Legitimate rows to keep. Enough that the fraud rate stays visibly small and
#: false positives are worth exploring, small enough to load instantly.
N_LEGIT = 40_000
SEED = 42


def main() -> int:
    if not PREDICTIONS.exists():
        print(f"missing {PREDICTIONS.relative_to(PROJECT_ROOT)}", file=sys.stderr)
        return 1

    print("loading PaySim ...")
    df = D.load()

    print("rebuilding the seed-42 stratified split ...")
    split = splits.stratified_split(df[C.TARGET], seed=C.RANDOM_SEED)

    test = df.iloc[split.test].reset_index(drop=True)
    preds = pd.read_parquet(PREDICTIONS)

    if len(test) != len(preds):
        print(
            f"split has {len(test):,} rows but predictions have {len(preds):,}",
            file=sys.stderr,
        )
        return 1

    # The whole script rests on this. Check it, do not assume it.
    if not np.array_equal(test[C.TARGET].to_numpy(), preds["y_true"].to_numpy()):
        print(
            "FAILED: labels disagree between the split and the prediction file.\n"
            "The row alignment assumption is wrong -- do not ship this sample.",
            file=sys.stderr,
        )
        return 1
    print(f"alignment verified across {len(test):,} rows")

    test = test.assign(
        score=preds["score"].to_numpy(),
        severity=preds["severity"].to_numpy(),
    )

    fraud = test[test[C.TARGET] == 1]
    legit = test[test[C.TARGET] == 0].sample(
        n=min(N_LEGIT, (test[C.TARGET] == 0).sum()),
        random_state=SEED,
    )

    sample = (
        pd.concat([fraud, legit])
        .sort_index()
        .reset_index(drop=True)
        .drop(columns=C.LEAK_COLUMNS)
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(OUT_PATH, engine="pyarrow", compression="snappy", index=False)

    size_mb = OUT_PATH.stat().st_size / 1e6
    print(
        f"wrote   {OUT_PATH.relative_to(PROJECT_ROOT)}  "
        f"({len(sample):,} rows, {len(fraud):,} fraud, {size_mb:.1f} MB)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
