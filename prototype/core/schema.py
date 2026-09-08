"""What counts as a transaction, and what to do when it doesn't.

Every input path in the prototype -- the typed form, the editable grid, the
uploaded CSV -- narrows to the same seven raw fields and the same validator.
The field list mirrors ``FeatureBuilder.transform_one``; it is duplicated here
only so the UI can name the fields before the scorer is asked to reject them.

The sample profiles are real rows from the held-out test split, carrying their
real labels and the scores the experiment recorded. Inventing plausible-looking
numbers would have been easier and would have made the demo a puppet show.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from momo_fraud import constants as C  # noqa: E402

#: The raw fields ``FeatureBuilder.transform_one`` requires (features.py:190).
REQUIRED_FIELDS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

NUMERIC_FIELDS = [f for f in REQUIRED_FIELDS if f != "type"]
AMOUNT_FIELDS = [f for f in NUMERIC_FIELDS if f != "step"]

#: Accepted in an upload and ignored. Two different reasons, both worth saying
#: out loud in the UI rather than dropping silently.
IGNORED_FIELDS = {
    "nameOrig": "account identifier -- dropped, PaySim reuses accounts",
    "nameDest": "account identifier -- dropped, PaySim reuses accounts",
    "isFlaggedFraud": "the simulator's own rule, computed from the label -- leakage",
    "isFraud": "ground-truth label -- used to score accuracy, never as an input",
}

#: Collected because the feature builder validates their presence, then dropped
#: before the model sees them. This is the 'no_origin_balance' rung: these two
#: columns let a tree recover PaySim's fraud rule outright (notebook 02).
UNUSED_BY_MODEL = ["oldbalanceOrg", "newbalanceOrig"]

FIELD_HELP = {
    "step": "Hours since the simulation started (1-743). Sets the hour of day and day number.",
    "type": "The kind of transaction. Fraud only ever occurs in TRANSFER and CASH_OUT.",
    "amount": "Value moved in this transaction.",
    "oldbalanceOrg": "Sender's balance before. Collected, but not used by this model.",
    "newbalanceOrig": "Sender's balance after. Collected, but not used by this model.",
    "oldbalanceDest": "Recipient's balance before.",
    "newbalanceDest": "Recipient's balance after.",
}


@dataclass(frozen=True)
class Profile:
    """A real test-split transaction, with what actually happened to it."""

    name: str
    blurb: str
    transaction: dict
    actual_fraud: bool
    recorded_score: float


#: Five rows chosen to walk the whole confusion matrix: a clear pass, a
#: borderline pass, a caught fraud, a false alarm, and a miss.
SAMPLE_PROFILES: list[Profile] = [
    Profile(
        name="Routine payment",
        blurb="A small merchant payment. Should sail through untouched.",
        transaction={
            "step": 601, "type": "PAYMENT", "amount": 1976.42,
            "oldbalanceOrg": 0.0, "newbalanceOrig": 0.0,
            "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
        },
        actual_fraud=False,
        recorded_score=6.630429e-09,
    ),
    Profile(
        name="Large legitimate cash-out",
        blurb="R761k moved out, and genuine. Lands just under the alert line -- "
              "the clearest illustration of how fine the cut is.",
        transaction={
            "step": 134, "type": "CASH_OUT", "amount": 761879.44,
            "oldbalanceOrg": 0.0, "newbalanceOrig": 0.0,
            "oldbalanceDest": 1618968.38, "newbalanceDest": 2380847.75,
        },
        actual_fraud=False,
        recorded_score=0.001570,
    ),
    Profile(
        name="Account-drain transfer (fraud)",
        blurb="The classic signature: the whole balance transferred out at once, "
              "leaving the account empty. Caught.",
        transaction={
            "step": 53, "type": "TRANSFER", "amount": 3640271.25,
            "oldbalanceOrg": 3640271.25, "newbalanceOrig": 0.0,
            "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
        },
        actual_fraud=True,
        recorded_score=1.0,
    ),
    Profile(
        name="False alarm",
        blurb="Legitimate, but the model flags it. This is what the other 94% of "
              "alerts look like -- worth seeing, not hiding.",
        transaction={
            "step": 413, "type": "CASH_OUT", "amount": 55437.40,
            "oldbalanceOrg": 0.0, "newbalanceOrig": 0.0,
            "oldbalanceDest": 562986.81, "newbalanceDest": 618424.19,
        },
        actual_fraud=False,
        recorded_score=0.289043,
    ),
    Profile(
        name="Missed fraud",
        blurb="Real fraud that scores just below the line and is let through. "
              "One of the 11% the model does not catch.",
        transaction={
            "step": 539, "type": "CASH_OUT", "amount": 79594.91,
            "oldbalanceOrg": 79594.91, "newbalanceOrig": 0.0,
            "oldbalanceDest": 2009435.0, "newbalanceDest": 2089030.0,
        },
        actual_fraud=True,
        recorded_score=0.001828,
    ),
]


@dataclass(frozen=True)
class Problem:
    """One thing wrong with one place in the input."""

    row: str
    column: str
    issue: str


def blank_row() -> dict:
    """An empty transaction, for seeding the grid and the form."""
    return {
        "step": 1, "type": "TRANSFER", "amount": 0.0,
        "oldbalanceOrg": 0.0, "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
    }


def validate(df: pd.DataFrame) -> tuple[pd.DataFrame, list[Problem]]:
    """Coerce a frame towards the input contract and report what is wrong.

    Never raises. A demo that dies on a malformed upload is a demo that ends
    early, so every problem is returned as data for the page to render, and the
    rows that are fine are still returned ready to score.

    Returns:
        The subset of rows that are safe to score, and every problem found.
        Both can be non-empty at once: bad rows are dropped, good rows survive.
    """
    problems: list[Problem] = []

    missing = [c for c in REQUIRED_FIELDS if c not in df.columns]
    if missing:
        for column in missing:
            problems.append(Problem("(file)", column, "required column is missing"))
        return df.iloc[0:0], problems

    if df.empty:
        problems.append(Problem("(file)", "-", "no data rows"))
        return df, problems

    out = df.copy()
    bad = pd.Series(False, index=out.index)

    # 'type' must be one of the five PaySim kinds; anything else has no one-hot
    # column and would score as if it were a sixth, unseen type.
    types = out["type"].astype("string").str.strip().str.upper()
    unknown = ~types.isin(C.TRANSACTION_TYPES) | types.isna()
    for idx in out.index[unknown]:
        problems.append(Problem(
            str(idx), "type",
            f"{df.at[idx, 'type']!r} is not one of {', '.join(C.TRANSACTION_TYPES)}",
        ))
    out["type"] = types
    bad |= unknown

    for column in NUMERIC_FIELDS:
        values = pd.to_numeric(out[column], errors="coerce")

        not_numeric = values.isna()
        for idx in out.index[not_numeric]:
            problems.append(Problem(str(idx), column, f"{df.at[idx, column]!r} is not a number"))

        negative = values.notna() & (values < 0)
        for idx in out.index[negative]:
            problems.append(Problem(str(idx), column, f"{values[idx]:,.2f} is negative"))

        out[column] = values
        bad |= not_numeric | negative

    # 'step' is an hour index into a 31-day simulation. Outside that range the
    # derived hour/day are meaningless rather than merely unusual.
    steps = pd.to_numeric(out["step"], errors="coerce")
    out_of_range = steps.notna() & ((steps < 1) | (steps > C.MAX_STEP))
    for idx in out.index[out_of_range]:
        problems.append(Problem(
            str(idx), "step",
            f"{steps[idx]:,.0f} is outside 1-{C.MAX_STEP} (the simulation's length)",
        ))
    bad |= out_of_range

    clean = out.loc[~bad, REQUIRED_FIELDS]
    if not clean.empty:
        clean = clean.astype({c: "float64" for c in AMOUNT_FIELDS})
        clean["step"] = clean["step"].astype("int64")

    return clean, problems


def describe_extra_columns(df: pd.DataFrame) -> list[tuple[str, str]]:
    """Columns present in an upload that the model will not read."""
    known = set(REQUIRED_FIELDS)
    return [
        (c, IGNORED_FIELDS.get(c, "not part of the input contract -- ignored"))
        for c in df.columns
        if c not in known
    ]
