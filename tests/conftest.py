"""Shared fixtures.

The synthetic frame mimics PaySim's schema and its fraud mechanism (an account
drained by a TRANSFER whose amount equals the origin balance, into a
destination whose balance the simulator never updates) so tests exercise the
same code paths the real data will, without needing the 470 MB file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from momo_fraud import constants as C


@pytest.fixture
def paysim_like() -> pd.DataFrame:
    """A small frame with PaySim's schema, columns and fraud signature."""
    rng = np.random.default_rng(42)
    n = 4_000
    n_fraud = 40

    amount = rng.lognormal(mean=8.0, sigma=1.5, size=n).astype("float32")
    old_org = (amount * rng.uniform(1.0, 5.0, n)).astype("float32")

    df = pd.DataFrame({
        "step": rng.integers(1, C.MAX_STEP + 1, n).astype("int16"),
        "type": pd.Categorical(
            rng.choice(C.TRANSACTION_TYPES, n), categories=C.TRANSACTION_TYPES
        ),
        "amount": amount,
        "nameOrig": [f"C{i}" for i in range(n)],
        "oldbalanceOrg": old_org,
        "newbalanceOrig": (old_org - amount).astype("float32"),
        "nameDest": [f"M{i % 500}" for i in range(n)],
        "oldbalanceDest": rng.lognormal(9.0, 1.5, n).astype("float32"),
        "newbalanceDest": rng.lognormal(9.0, 1.5, n).astype("float32"),
        "isFraud": np.zeros(n, dtype="int8"),
        "isFlaggedFraud": np.zeros(n, dtype="int8"),
    })

    # Real PaySim leaves the origin balance at zero for a large share of rows
    # (the simulator does not track balances for every account type), so the
    # balance arithmetic does *not* reconcile for those. Without this the
    # fixture would be unrealistically tidy and errorBalanceOrig would carry no
    # signal at all -- which is not what the real file looks like.
    unbalanced = rng.choice(n, size=int(n * 0.4), replace=False)
    df.loc[unbalanced, "oldbalanceOrg"] = 0.0
    df.loc[unbalanced, "newbalanceOrig"] = 0.0

    # Fraud: drain the origin account via TRANSFER, destination balances stay 0.
    fraud_idx = rng.choice(n, size=n_fraud, replace=False)
    df.loc[fraud_idx, "isFraud"] = 1
    df.loc[fraud_idx, "type"] = "TRANSFER"
    df.loc[fraud_idx, "oldbalanceOrg"] = amount[fraud_idx]
    df.loc[fraud_idx, "amount"] = amount[fraud_idx]
    df.loc[fraud_idx, "newbalanceOrig"] = 0.0
    df.loc[fraud_idx, "oldbalanceDest"] = 0.0
    df.loc[fraud_idx, "newbalanceDest"] = 0.0

    return df


@pytest.fixture
def front_loaded_steps() -> np.ndarray:
    """Steps shaped like the real file's: most volume early, a long thin tail.

    ``paysim_like`` draws steps uniformly, which is what let boundaries realising
    95.6 / 3.0 / 1.4 on the real data pass a proportions test. PaySim's hourly
    volume collapses after roughly the first half of the simulation, so any test
    that claims to check split proportions has to run against that shape rather
    than a flat one.
    """
    rng = np.random.default_rng(7)
    n = 60_000
    # Beta(1.4, 4) puts ~70% of mass in the first ~30% of the range, which is
    # the front-loading regime that broke the old constants.
    steps = (rng.beta(1.4, 4.0, size=n) * C.MAX_STEP).astype("int64")
    return np.clip(steps, 0, C.MAX_STEP)
