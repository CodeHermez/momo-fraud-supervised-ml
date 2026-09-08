"""Row-causal account-history ("velocity") features.

``nameOrig``/``nameDest`` are dropped as raw identifiers before modelling
(``C.ID_COLUMNS``) and never otherwise used, even though 63.28% of test rows
involve an account already seen in training (``02_account_overlap.json``).
This module turns that history into features -- without ever letting a row
see its own future.

**Why this cannot leak.** Every column here is a deterministic function of
transactions strictly before the row's own ``step`` -- the same information a
live production system streaming transactions in real time would have. Which
partition a row is *later* assigned to (train/val/test, stratified or
temporal) never enters the computation, so there is nothing for the split to
leak into. Contrast this with ``FeatureBuilder`` (``features.py``), which fits
quantities (the high-risk cutoff, the severity grid) from whichever rows land
in train -- a genuinely different kind of state, and the reason that module
must never be fit on val/test. This module has no ``fit`` step at all: it is
computed once, on the full ordered table, before any split exists.

**Tie-break rule.** Transactions sharing the same ``step`` never see each
other -- every lookback here queries strictly ``step - 1`` and earlier, never
the current step. This is deliberate: PaySim's ``step`` is hour-granularity,
so many thousands of rows share a step, and their relative order in the CSV is
not a real arrival-time signal. A same-step-aware design would smuggle in an
unstated ordering assumption; this one does not, and is provably invariant to
row order (see ``tests/test_velocity.py``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Rolling-window size in steps (PaySim's ``step`` is hours, so 24 = 1 day).
WINDOW_STEPS = 24

#: Sentinel for "no prior activity" on the steps-since-last-activity columns.
#: Paired with the ``*_is_first_seen`` flag rather than left as ``NaN`` so a
#: tree-based learner can split cleanly on "first-time account" without a
#: numeric value that could plausibly collide with a real steps-since value.
NO_HISTORY_SENTINEL = -1

VELOCITY_FEATURES: list[str] = [
    "vel_orig_prior_count",
    "vel_orig_prior_count_24h",
    "vel_orig_unique_recipients",
    "vel_orig_steps_since_last",
    "vel_orig_is_first_seen",
    "vel_dest_prior_count",
    "vel_dest_prior_count_24h",
    "vel_dest_unique_senders",
    "vel_dest_steps_since_last",
    "vel_dest_is_first_seen",
]


def exclude_velocity(columns: list[str]) -> list[str]:
    """The ablation control: the same columns with the velocity family removed."""
    velocity = set(VELOCITY_FEATURES)
    return [c for c in columns if c not in velocity]


# --- the causal lookup primitive --------------------------------------------


def _asof_lookup(
    accounts: np.ndarray,
    query_steps: np.ndarray,
    right_account: np.ndarray,
    right_step: np.ndarray,
    right_value: np.ndarray,
) -> np.ndarray:
    """For each ``(account, query_step)``, the ``right_value`` at the largest
    ``right_step <= query_step`` for that same account, or ``NaN`` if none.

    A backward as-of join keyed on ``account``. Callers pass ``query_steps =
    step - 1`` (or earlier) so the lookup can never see the row's own step,
    which is what makes every derived feature causal by construction rather
    than by convention.
    """
    n = len(accounts)
    left = pd.DataFrame({
        "_pos": np.arange(n),
        "account": accounts,
        "step": query_steps,
    })
    right = pd.DataFrame({
        "account": right_account,
        "step": right_step,
        "value": right_value,
    })

    # merge_asof requires both frames sorted by the "on" key.
    left_sorted = left.sort_values("step", kind="mergesort")
    right_sorted = right.sort_values("step", kind="mergesort")

    merged = pd.merge_asof(
        left_sorted, right_sorted,
        on="step", by="account", direction="backward",
    )
    return merged.sort_values("_pos", kind="mergesort")["value"].to_numpy()


def _cumulative_by_account_step(account: np.ndarray, step: np.ndarray) -> pd.DataFrame:
    """One row per distinct ``(account, step)``, with the cumulative *inclusive*
    count of events for that account through that step.

    The building block every prior-activity feature reduces to: querying this
    table at ``step - 1`` (or earlier) gives "how much activity happened
    strictly before this point," and querying at two offsets and subtracting
    gives a rolling window -- both via the same as-of mechanism.
    """
    counts = (
        pd.DataFrame({"account": account, "step": step})
        .groupby(["account", "step"], sort=True)
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["account", "step"], kind="mergesort")
    )
    counts["cum_incl"] = counts.groupby("account")["count"].cumsum()
    return counts


def build_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-causal account-history features, computed once on the full ordered
    table, before any split. See the module docstring for why this is safe.

    Args:
        df: A frame with PaySim's schema (``nameOrig``, ``nameDest``, ``step``
            at minimum). Row order does not matter -- only ``step`` values do.

    Returns:
        A frame indexed like ``df``, with the columns in ``VELOCITY_FEATURES``.
    """
    n = len(df)
    step = df["step"].to_numpy().astype("int64")

    # Joint factorization: the same physical account must get the same code
    # whether it appears as sender or receiver, or every cross-role aggregate
    # (e.g. "this account's total prior activity") silently undercounts.
    codes, _ = pd.factorize(
        pd.concat([df["nameOrig"], df["nameDest"]], ignore_index=True), sort=False
    )
    codes = codes.astype("int32")
    orig = codes[:n]
    dest = codes[n:]

    activity_account = np.concatenate([orig, dest])
    activity_step = np.concatenate([step, step])
    activity_counts = _cumulative_by_account_step(activity_account, activity_step)
    ac = activity_counts["account"].to_numpy()
    as_ = activity_counts["step"].to_numpy()
    av = activity_counts["cum_incl"].to_numpy()
    a_step_only = activity_counts["step"].to_numpy()  # for steps-since-last

    # New-partner events: one row per (orig, dest) pair, at the step it first
    # occurred. A per-pair minimum, so which of several same-step, same-pair
    # rows "represents" the pair never matters -- there is no tie to break.
    pairs = pd.DataFrame({"orig": orig, "dest": dest, "step": step})
    unique_pairs = pairs.groupby(["orig", "dest"], as_index=False)["step"].min()

    fanout_counts = _cumulative_by_account_step(
        unique_pairs["orig"].to_numpy(), unique_pairs["step"].to_numpy()
    )
    fanin_counts = _cumulative_by_account_step(
        unique_pairs["dest"].to_numpy(), unique_pairs["step"].to_numpy()
    )

    out = pd.DataFrame(index=df.index)

    for role, code in (("orig", orig), ("dest", dest)):
        prior = _asof_lookup(code, step - 1, ac, as_, av)
        prior = np.nan_to_num(prior, nan=0.0)
        out[f"vel_{role}_prior_count"] = prior.astype("int32")

        prior_before_window = _asof_lookup(code, step - 1 - WINDOW_STEPS, ac, as_, av)
        prior_before_window = np.nan_to_num(prior_before_window, nan=0.0)
        out[f"vel_{role}_prior_count_{WINDOW_STEPS}h"] = (
            prior - prior_before_window
        ).astype("int32")

        last_step = _asof_lookup(code, step - 1, ac, as_, a_step_only)
        is_first = np.isnan(last_step)
        steps_since = np.where(is_first, NO_HISTORY_SENTINEL, step - last_step)
        out[f"vel_{role}_steps_since_last"] = steps_since.astype("int32")
        out[f"vel_{role}_is_first_seen"] = is_first.astype("int8")

    partner_prior_orig = _asof_lookup(
        orig, step - 1,
        fanout_counts["account"].to_numpy(), fanout_counts["step"].to_numpy(),
        fanout_counts["cum_incl"].to_numpy(),
    )
    out["vel_orig_unique_recipients"] = np.nan_to_num(partner_prior_orig, nan=0.0).astype("int32")

    partner_prior_dest = _asof_lookup(
        dest, step - 1,
        fanin_counts["account"].to_numpy(), fanin_counts["step"].to_numpy(),
        fanin_counts["cum_incl"].to_numpy(),
    )
    out["vel_dest_unique_senders"] = np.nan_to_num(partner_prior_dest, nan=0.0).astype("int32")

    return out[VELOCITY_FEATURES]
