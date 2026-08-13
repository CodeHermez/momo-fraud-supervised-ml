"""Train / validation / test splitting.

Two strategies, both reported:

**Stratified** (proposal section 4.4) -- 70/15/15, comparable with the
literature, which all uses random splits.

**Temporal** -- train on early ``step`` values, test on later ones. The proposal
frames deployment as real-time fraud screening, and a random split lets the
model learn from transactions that occur *after* the ones it is scored on. The
stratified split stays primary for comparability; the temporal split is the
robustness check that says whether the comparison survives honest time ordering.

The invariant that matters throughout: **thresholds are selected on validation,
frozen, and applied once to test.** Everything else in this study is a defence
against the kind of leakage that invalidated the prior work.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

from . import constants as C


@dataclass
class Split:
    """Row indices for one train/val/test partition."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    strategy: str

    def __post_init__(self) -> None:
        self.assert_disjoint()

    def assert_disjoint(self) -> None:
        """No row may appear in two partitions. Cheap, and catches real bugs."""
        pairs = [("train", "val"), ("train", "test"), ("val", "test")]
        for a, b in pairs:
            overlap = np.intersect1d(getattr(self, a), getattr(self, b))
            if overlap.size:
                raise ValueError(
                    f"{a}/{b} overlap by {overlap.size} rows -- split is leaking."
                )

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}

    def prevalence(self, y: pd.Series | np.ndarray) -> dict[str, float]:
        """Fraud rate per partition. Should track the population rate closely."""
        y = np.asarray(y)
        return {name: float(y[getattr(self, name)].mean()) for name in ("train", "val", "test")}


def stratified_split(
    y: pd.Series | np.ndarray,
    *,
    train_size: float = C.SPLIT_TRAIN,
    val_size: float = C.SPLIT_VAL,
    seed: int = C.RANDOM_SEED,
) -> Split:
    """70/15/15 stratified split (proposal section 4.4).

    Stratification is not optional at 0.129% prevalence -- an unstratified
    15% test split would vary by hundreds of fraud cases between seeds, and
    that variance would swamp the effects being measured.
    """
    y = np.asarray(y)
    idx = np.arange(len(y))

    train_idx, holdout_idx = train_test_split(
        idx, train_size=train_size, stratify=y, random_state=seed
    )
    # val_size is a fraction of the whole, so rescale within the holdout.
    val_fraction = val_size / (1.0 - train_size)
    val_idx, test_idx = train_test_split(
        holdout_idx,
        train_size=val_fraction,
        stratify=y[holdout_idx],
        random_state=seed,
    )
    return Split(np.sort(train_idx), np.sort(val_idx), np.sort(test_idx), "stratified")


def temporal_split(
    steps: pd.Series | np.ndarray,
    *,
    train_end: int = C.TEMPORAL_TRAIN_END,
    val_end: int = C.TEMPORAL_VAL_END,
) -> Split:
    """Split on simulation time -- train on the past, test on the future.

    Boundaries default to steps 520 and 632, chosen to land near 70/15/15 by
    row count. Call ``Split.sizes()`` to confirm on the real data; PaySim's
    hourly volume is not uniform, so exact proportions drift a little.
    """
    steps = np.asarray(steps)
    idx = np.arange(len(steps))

    train_idx = idx[steps <= train_end]
    val_idx = idx[(steps > train_end) & (steps <= val_end)]
    test_idx = idx[steps > val_end]

    for name, part in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        if part.size == 0:
            raise ValueError(f"temporal split produced an empty {name} partition.")

    return Split(train_idx, val_idx, test_idx, "temporal")


def repeated_cv(seed: int = C.RANDOM_SEED) -> RepeatedStratifiedKFold:
    """5x5 repeated stratified CV with 95% CIs.

    Johnson & Khoshgoftaar use 6x5 and Mienye & Sun 3x10; 5x5 sits between
    them and gives 25 estimates per cell, enough for the paired tests in
    notebook 04 without making the grid unaffordable.
    """
    return RepeatedStratifiedKFold(
        n_splits=C.CV_FOLDS, n_repeats=C.CV_REPEATS, random_state=seed
    )


def account_overlap(df: pd.DataFrame, split: Split, column: str = "nameDest") -> dict[str, float]:
    """How many test accounts were already seen in training.

    PaySim reuses destination accounts, so a model can in principle memorise
    specific mule accounts rather than learn fraud behaviour. We drop the raw
    identifiers, but this quantifies the exposure for the notebook-02 write-up.
    """
    train_accounts = set(df[column].iloc[split.train].unique())
    test_accounts = df[column].iloc[split.test]

    seen = test_accounts.isin(train_accounts)
    return {
        "n_test_accounts": int(test_accounts.nunique()),
        "n_test_rows_with_seen_account": int(seen.sum()),
        "fraction_test_rows_seen": float(seen.mean()),
    }
