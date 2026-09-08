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
        """Fraud rate per partition.

        Tracks the population rate closely under ``stratified_split`` -- by
        construction. Under ``temporal_split`` it does not: PaySim's fraud
        prevalence is non-stationary, so this is the number that tells a caller
        whether the population cost ratio still applies.
        """
        y = np.asarray(y)
        return {name: float(y[getattr(self, name)].mean()) for name in ("train", "val", "test")}

    def class_ratio(self, y: pd.Series | np.ndarray) -> dict[str, float]:
        """``N_neg / N_pos`` per partition -- the ratio at which NER is anchored.

        ``normalized_expected_risk`` scores "flag nothing" at 1.0 for any ``R``,
        but "flag everything" reaches 1.0 only when ``R`` equals the evaluated
        partition's class ratio, and the Youden/NER-optimal threshold equivalence
        holds only there too. On a stratified split every partition returns the
        population ratio and the study default applies unchanged; on a temporal
        split they diverge, and this is how a caller finds out by how much.
        """
        y = np.asarray(y)
        out = {}
        for name in ("train", "val", "test"):
            part = y[getattr(self, name)]
            n_pos = int(part.sum())
            if n_pos == 0:
                raise ValueError(f"{name} partition has no positives; class ratio undefined.")
            out[name] = float((len(part) - n_pos) / n_pos)
        return out


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


def derive_temporal_boundaries(
    steps: pd.Series | np.ndarray,
    *,
    train_size: float = C.SPLIT_TRAIN,
    val_size: float = C.SPLIT_VAL,
) -> tuple[int, int]:
    """Boundaries that land nearest the target proportions on *these* steps.

    For each target cumulative share, take the largest step whose cumulative row
    count does not exceed it. A boundary cannot split an hour, so this is the
    closest achievable cut from below.

    This exists so the constants in ``constants.py`` are *derived* from the file
    rather than guessed at it. Run it against the real data and the pair it
    returns is what belongs in ``TEMPORAL_TRAIN_END`` / ``TEMPORAL_VAL_END``.
    """
    steps = np.asarray(steps)
    values, counts = np.unique(steps, return_counts=True)
    cumulative = np.cumsum(counts) / len(steps)

    def largest_below(target: float) -> int:
        eligible = np.flatnonzero(cumulative <= target)
        return int(values[eligible[-1]] if eligible.size else values[0])

    return largest_below(train_size), largest_below(train_size + val_size)


def temporal_split(
    steps: pd.Series | np.ndarray,
    *,
    train_end: int = C.TEMPORAL_TRAIN_END,
    val_end: int = C.TEMPORAL_VAL_END,
    tolerance: float | None = C.TEMPORAL_PROPORTION_TOLERANCE,
) -> Split:
    """Split on simulation time -- train on the past, test on the future.

    Boundaries default to steps 322 and 377, **derived** from the canonical
    file's cumulative row count by ``derive_temporal_boundaries``. On the real
    data they realise 69.68 / 15.30 / 15.02.

    The realised proportions are now asserted rather than assumed. PaySim's
    hourly volume is heavily front-loaded, and the previous boundaries
    (520 / 632) were chosen as if it were uniform: they realised 95.6 / 3.0 / 1.4
    and nothing caught it, because the only test guarding them ran on a synthetic
    frame with uniformly drawn steps. Pass ``tolerance=None`` to explore other
    boundaries deliberately.

    Hitting the row proportions does **not** equalise the class prior: fraud
    prevalence is non-stationary in PaySim, so these partitions carry
    0.082% / 0.059% / 0.420% fraud. A caller evaluating NER on a temporal
    partition must supply a cost ratio appropriate to that partition rather than
    the population default -- see ``Split.class_ratio``.
    """
    steps = np.asarray(steps)
    idx = np.arange(len(steps))

    train_idx = idx[steps <= train_end]
    val_idx = idx[(steps > train_end) & (steps <= val_end)]
    test_idx = idx[steps > val_end]

    for name, part in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        if part.size == 0:
            raise ValueError(f"temporal split produced an empty {name} partition.")

    if tolerance is not None:
        targets = {"train": C.SPLIT_TRAIN, "val": C.SPLIT_VAL, "test": C.SPLIT_TEST}
        realised = {
            "train": len(train_idx) / len(steps),
            "val": len(val_idx) / len(steps),
            "test": len(test_idx) / len(steps),
        }
        off = {
            name: (realised[name], targets[name])
            for name in targets
            if abs(realised[name] - targets[name]) > tolerance
        }
        if off:
            detail = "; ".join(
                f"{name} {got:.2%} vs target {want:.0%}"
                for name, (got, want) in sorted(off.items())
            )
            raise ValueError(
                f"temporal boundaries {train_end}/{val_end} realise proportions outside "
                f"the {tolerance:.0%} tolerance: {detail}. "
                f"Re-derive them with splits.derive_temporal_boundaries(steps)."
            )

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
