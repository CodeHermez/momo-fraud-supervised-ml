# Destination-side balance state: diagnostic

**Status:** diagnostic only. `no_balance_state` was **not** promoted to the main
experimental rung. Raised as audit finding M3.

Produced by `scripts/repairs/repair_09_destination_diagnostic.py`.
Artifacts: `results/09_destination_artefact_rates.csv`,
`results/09_destination_single_feature_auc.csv`,
`results/09_destination_ablation.csv`.

## The concern

The origin-side balances were removed with a specific argument: dropping the
derived `errorBalanceOrig` is not enough, because a tree reconstructs
`amount == oldbalanceOrg` from the raw columns that remain. The same argument
applies verbatim to the destination side — `oldbalanceDest + amount -
newbalanceDest` is equally reconstructible, and PaySim never credits the mule
account on a fraudulent transfer — but it was not applied there. The
`no_origin_balance` rung therefore retains `oldbalanceDest`, `newbalanceDest`,
`destZeroBefore` and `destZeroAfter`, and the notebook-09 permutation table
ranks `destZeroAfter` second overall.

The asymmetry in the *reasoning* was real. The question this diagnostic answers
is whether the asymmetry in the *evidence* is comparable.

## Q1 — Is the destination side the same kind of artefact?

Class-conditional rates, training rows only:

| Condition | Legitimate | Fraud | Lift |
|---|---|---|---|
| `dest_zero_before` | 0.4246 | 0.6519 | **1.54×** |
| `dest_zero_after` | 0.3833 | 0.4990 | 1.30× |
| `dest_zero_both` | 0.3640 | 0.4973 | 1.37× |
| `dest_balance_unchanged` | 0.3640 | 0.4983 | 1.37× |
| `dest_not_credited` | 0.6021 | 0.5150 | **0.86×** |

Single-feature out-of-sample stump AUC, training rows only:

| Feature | AUC |
|---|---|
| `oldbalanceDest` | 0.6237 |
| `destZeroBefore` | 0.6141 |
| `errorBalanceDest` | 0.5714 |
| `destZeroAfter` | 0.5565 |
| `newbalanceDest` | 0.5564 |

Set against the origin side, which is what motivated the ablation in the first
place:

| | Origin side | Destination side |
|---|---|---|
| Strongest solo AUC | `errorBalanceOrig` **0.9020** | `oldbalanceDest` **0.6237** |
| Signature | `amount == oldbalanceOrg` in 97.82% of fraud vs 0.0013% of legit | `dest_zero_before` in 65% of fraud vs 42% of legit |
| Median gap | exactly **0.0** for fraud | not applicable |
| Deterministic? | yes — a three-clause rule recovers the label at P 0.9999 | no |

**Answer: no.** The destination side is not the same kind of object. The origin
side is a deterministic generator rule that a boolean expression recovers almost
exactly. The destination side is diffuse, probabilistic association at lifts
between 1.3× and 1.5×, with no single feature above 0.63 alone.

The sharpest result is `dest_not_credited`, which was the specific mechanism the
concern named — "the simulator never credits the receiving account". Measured, it
runs at **0.60 for legitimate transactions and 0.52 for fraud**: a lift of 0.86,
pointing the *wrong way*. PaySim's destination bookkeeping fails to reconcile for
the majority of legitimate transactions too, so non-credit does not identify
fraud. The premise behind extending the origin-side argument to the destination
side does not survive measurement.

## Q2 — What does removing it cost?

Test-set results, config C, thresholds selected on validation:

| Learner | NER `no_origin_balance` | NER `no_balance_state` | PR-AUC `no_origin_balance` | PR-AUC `no_balance_state` |
|---|---|---|---|---|
| Logistic regression | 0.2499 | 0.3906 | 0.5040 | **0.0784** |
| Decision tree | 0.1527 | 0.2312 | 0.7297 | 0.3822 |
| Random forest | 0.1324 | 0.1959 | 0.7967 | 0.5082 |
| XGBoost | 0.1314 | 0.1931 | 0.8173 | 0.5697 |

Removing the destination balance state costs 0.06–0.14 NER and 0.25–0.43 PR-AUC.
For logistic regression it is close to total: PR-AUC 0.078 is barely above the
0.00129 prevalence floor.

## Q3 — Does the deeper rung still leave headroom?

Yes. Validation headroom is 0.1147 at `no_origin_balance` and 0.1841 at
`no_balance_state`; both clear the pre-declared 0.05 floor.

That matters for the selection rule rather than against it. The criterion is
*least-ablated rung clearing the floor*, so adding `no_balance_state` to the
ladder does not change the outcome — `no_origin_balance` is less ablated and
already viable. The pre-declared criterion selects the same rung whether or not
this rung exists.

## Conclusion

Three findings, in order of what they settle:

1. **The reasoning asymmetry the audit identified was genuine** and is now
   documented rather than implicit.
2. **The evidence is not symmetric.** The origin side is a deterministic
   simulator rule; the destination side is weak probabilistic association, and
   the specific non-credit mechanism that motivated the concern is anti-correlated
   with fraud. Removing it would discard genuine — if simulator-specific — signal
   rather than an artefact.
3. **The main rung does not change,** and would not change even if
   `no_balance_state` were admitted to the ladder, because the pre-declared
   criterion prefers the least-ablated viable rung.

So `no_origin_balance` stands as the experimental rung, now for a measured reason
rather than an unexamined one.

## What is still open

`destZeroAfter` ranking second in permutation importance while scoring 0.5565
alone is not a contradiction — it is what a feature that is useful *in
combination* looks like. But the notebook-07 and notebook-09 permutation tables
share a known distortion (`amount`/`log_amount` are a deterministic pair and the
`type_*` one-hots are mutually exclusive), so the ranking itself should not be
read too closely. A grouped permutation importance, permuting correlated
families together, would settle what the destination block contributes as a
block. That is listed as an optional experiment for the next phase and is not
run here.
