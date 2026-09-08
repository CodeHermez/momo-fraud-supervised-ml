# Hájek et al. (2023) — comparability extraction sheet

**The question this answers is not "what did they find?" but "are their
experimental conditions comparable enough that the apparent disagreement with
configuration G is meaningful?"**

Until every row below is filled, §8.1 of the report must retain its cautious
reconciliation language, and the four candidate explanations must stay labelled
as hypotheses.

**Reference.** Hajek, P., Abedin, M. Z., & Sivarajah, U. (2023). Fraud detection
in mobile payment systems using an XGBoost-based framework. *Information Systems
Frontiers* **25**(5), 1985–2003. doi:10.1007/s10796-022-10346-6. Paywalled, no
green OA; needs UP institutional access.

**How to use this.** Fill the right-hand column from the paper only. Do not infer
a value from the abstract or from a secondary source — an unfilled row is more
useful than a guessed one, because it keeps the hypothesis visibly open.

---

## The eleven dimensions

| # | Dimension | This study | Hájek et al. | Comparable? |
|---|---|---|---|---|
| 1 | **Feature set** | `no_origin_balance`, 14 features. `errorBalanceOrig`, `errorBalanceDest`, `oldbalanceOrg`, `newbalanceOrig`, `origZeroBefore`, `origZeroAfter` **removed**. Destination-side balances retained. | *(the decisive row — did they retain the origin balances?)* | |
| 2 | **Preprocessing** | Type one-hot; `log_amount`; `amount` p99 cutoff fitted on training rows only; no imputation needed. | | |
| 3 | **Train/test protocol** | Stratified 70/15/15. Transaction-level split; **no account partitioning**; 63.28% test-row account overlap. Test retains natural prevalence. | | |
| 4 | **Imbalance treatment** | Seven configurations across data (RUS), algorithm (class weighting, per-instance severity), decision (thresholding). Resampling confined to training rows. | *(RUS ratio? applied to training only, or before splitting?)* | |
| 5 | **Cost formulation** | Unitless. `R = N₋/N₊ = 773.70`; `NER = (FP + R·FN)/(R·N₊)`, so 1.0 = flag nothing. No currency. | *(is "cost saving" a currency figure? what cost matrix?)* | |
| 6 | **Threshold selection** | Validation-selected, locked, applied once to test. Order enforced: TRAIN→FIT→VAL→SELECT→LOCK→TEST→REPORT. | *(fixed 0.5? tuned? tuned on test?)* | |
| 7 | **Evaluation metric** | Primary ranking PR-AUC; primary decision NER; also recall@budget, precision, recall, F1, MCC, Brier, ROC-AUC. Accuracy excluded. | *(cost saving + what else?)* | |
| 8 | **XGBoost configuration** | Frozen; `scale_pos_weight = R` in weighted arms only. | *(depth, estimators, learning rate, tuning protocol)* | |
| 9 | **RUS implementation** | `N'₋ = N₋/R`; training rows only; stochastic, seed-controlled. E3 measured DT/RUS σ = 0.042 PR-AUC — the largest training-seed variance in the study. | *(target ratio? repeated draws or one?)* | |
| 10 | **Prevalence** | 0.129% throughout; evaluation never rebalanced. | Stated as 0.13%; *(is the evaluation set rebalanced?)* | |
| 11 | **Seeds / repetitions** | E3: 4 split seeds × 4 model seeds, 220 fits. Split σ ≈ 0.010–0.016 PR-AUC; seed σ ≈ 0.001–0.003. | *(single run or repeated? variance reported?)* | |

## The decisive question

> **Is their "cost saving" commensurable with NER?**

They are not the same object unless three things hold:

1. The cost saving is computed from a **class-constant** cost matrix (not
   example-dependent on transaction amount).
2. It is evaluated at **natural prevalence**, not on a rebalanced test set — E4
   established that evaluated prevalence alone moves precision substantially, so
   a saving computed on a rebalanced set is not comparable to one computed at
   0.129%.
3. It is computed at a **fixed operating point** across the compared
   configurations. If the threshold is retuned per configuration, then the
   reported saving mixes the decision layer into a data-layer comparison, and the
   RUS attribution is confounded — which is candidate reconciliation 2, and the
   same conflation this study had to repair as audit finding C2.

If any of the three fails, the disagreement is **not** a disagreement about
random under-sampling, and the report should say which one failed.

## Rows already answerable, and what they imply

Two rows can be partially filled from open sources and both narrow the gap:

- **Prevalence (row 10)** matches at 0.13%, so the comparison is at least at the
  same imbalance.
- **Dataset** matches — PaySim, >6M transactions — so this is not a
  cross-dataset comparison.

That is precisely what makes row 1 decisive. Same data and same prevalence, so
if the feature sets differ, the difference in feature sets is doing the work; and
this study's own full-feature benchmark reaches PR-AUC 0.9977 at the unablated
rung, against 0.817 at the ablated one. A gap of that size between rungs on
identical data dwarfs any plausible effect of a resampling choice.

## What to write once it is filled

- **If they retained the origin balances:** the results are not comparable, the
  disagreement dissolves, and the report says their models were scoring against a
  generator rule that this study removed. State it without triumphalism — the
  leak was undocumented (the PaySim paper excludes fraud generation from its
  scope), so using those features was the reasonable default.
- **If they ablated them too and still found RUS best:** the disagreement is
  real and interesting, and the next question is row 6 — whether their saving is
  a thresholding effect. Report it as an open discrepancy, not as their error.
- **If their cost saving is example-dependent on amount:** it is measuring the
  objective this study's configuration F was built for and could not fairly
  evaluate. That is a limitation of this study, and should be reported as one.

## Note on the second dataset

They also report on **BankSim**. Establish whether the RUS + XGBoost conclusion
holds on both datasets or only PaySim; a result that holds on two simulators is a
stronger claim than one that holds on the one whose label this study has shown to
be recoverable by three boolean clauses.
