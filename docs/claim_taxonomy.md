# Claim taxonomy — what kind of thing each finding is

**Purpose.** Every claim this study makes belongs to exactly one epistemic
category, and the categories are not interchangeable. A mathematical
demonstration is not a measurement; a measured association is not a mechanism;
an artifact of one simulator is not a fact about mobile money. Writing the report
against this table is what makes it hard to challenge — an examiner who disputes
a claim has to dispute its *category* first, and the category is stated.

Use it as the checklist when drafting: for every sentence that asserts something,
find it here, and write it at the strength the category permits — no more, and
**no less**.

---

## The five categories

| Category | What it means | What defeats it |
|---|---|---|
| **1. Demonstrated mathematically** | Follows by derivation. True for every model and every dataset, independent of what was run. | An error in the derivation. Not new data. |
| **2. Empirically validated** | A mathematical claim, confirmed in measurement across the study's grid. The strongest category the study has: proof *and* evidence. | Neither alone — both would have to fail. |
| **3. Measured** | Directly observed on PaySim under a stated protocol. True of this data. | A protocol error, or a scope claim beyond the grid actually run. |
| **4. Associated** | A statistical relationship observed without an established mechanism. Correlation, lift, or importance ranking. | A confound. These are the claims most exposed to alternative explanation. |
| **5. Cannot be claimed** | Outside what PaySim or this design can support, however plausible. | Nothing — these are the boundaries, and asserting them anyway is the failure. |

---

## 1. Demonstrated mathematically

Hold for every model. State them as derivations, not results.

| Claim | Where derived |
|---|---|
| At `R = N₋/N₊`, the risk-optimal and ROC-optimal thresholds are the **same threshold**. `FP + R·FN = N₋·(FPR + 1 − TPR)` at that ratio, whose minimiser is the maximiser of `TPR − FPR`. | `risk.youden_threshold` docstring |
| The Bayes-optimal decision threshold is `λ = 1/(1+R)`. | `risk.bayes_threshold`; canonically Elkan (2001) — **cite him, position against him** |
| `NER = 1.0` exactly for a model that flags nothing, by construction of the denominator `N_fraud × R`. | `risk.normalized_expected_risk` |
| All three CSL levels derive from one parameter: `N'_neg = N₋/R` (data), `scale_pos_weight = R` (algorithm), `λ = 1/(1+R)` (decision). | proposal §3.3 taxonomy; `constants` |
| The band cutoffs are `λ/10`, `λ`, `10λ` — so one `R` sets both the alerting threshold and the authentication ladder. | `risk.band_cutoffs`; CHANGE-06 |

> **Writing note.** Claim 1 is the study's most defensible single contribution
> because it cannot be attacked on data grounds. Do not bury it among the
> empirical results.

## 2. Empirically validated

Derived *and* confirmed. Say "demonstrated and measured", and give both.

| Claim | Derivation | Confirmation |
|---|---|---|
| The threshold equivalence holds exactly. | as above | **E6: 104/104 cells, max threshold gap `0.000e+00`, max ΔNER `0.000e+00`**; smoke test re-derives `argmax(TPR−FPR) = argmin(FP+R·FN)` at the same index (22756) |
| The equivalence survives the training/validation ratio distinction (773.7011 vs 773.6688, a factor of 1.0000417). | — | E6: 104/104 agree at both |
| `NER` equals its closed form `(FP + R·FN)/(R·N₊)`. | definition | audit recomputation to 1e-11; E6 to **2.220e-16** |
| `B ≡ C` is degenerate at the study's operating point. | as above | E6 §C; audit finding M5 |

## 3. Measured

True of PaySim under the stated protocol. **Always state the grid** — a measured
claim without its scope is an overclaim.

| Claim | Artifact | Scope actually run |
|---|---|---|
| CSL training never improved ranking quality. | `04_h1_paired_bootstrap.csv`; **E3** | 12/12 negative at one seed/split; **E3: 116/116 negative over 4 splits × 4 seeds** |
| The CSL effect on decision risk is learner-dependent. | **E3** | LR + DT benefit 64/64; RF + XGB harmed 61/64. Reproduces in every split and seed |
| Threshold optimisation removes 49–62% of risk. | `06_threshold_markers.csv` | 4 learners, one split |
| Learner ranking XGB > RF > DT > LR. | **E2** | **0 violations in 112 cells** |
| Evaluated prevalence drives precision. | **E4** | 104 cells, 62,504 rows, model and threshold frozen, positives never resampled |
| Threshold-rule divergence grows with distance from the class ratio, asymmetrically. | **E6** | below 0.127→0.217→0.282; above 0.114→0.160→0.161 |
| The ROC-optimal threshold can be worse than flagging nothing. | **E6** | NER > 1.0 in 100/832 rows, peak 6.3756, at `R` below the class ratio |
| Fraud occurs in 2 of 5 transaction types; zero in the other three across 3.59M rows. | `01_fraud_by_type.csv` | full dataset |
| Three boolean clauses recover the label at P 0.9999 / R 0.9770. | `02_rule_recoverability.csv` | **in-sample descriptive statistic, all rows — not a held-out detector result** (audit m5) |
| Split variability exceeds training-seed variability by roughly an order of magnitude. | **E3** | σ 0.010–0.016 vs 0.001–0.003 PR-AUC |
| The authentication ladder leaves 97.04% of traffic unescalated while the escalated 2.96% carries 90.2% of fraud. | `06_risk_bands.csv`; `auth_band_load` | XGBoost, test split — **friction cost only, not effectiveness** |

## 4. Associated

Relationships without established mechanism. **These are the claims an examiner
will press.** Write them with the alternative explanation already named.

| Claim | Artifact | The confound to name first |
|---|---|---|
| Transaction `type` is the most important feature (0.701 vs 0.447 for `amount`). | `07_permutation_importance.csv` | **Audit M9 is unresolved**: permutation importance is distorted by correlated features, and grouped permutation importance was never run. State the ranking as provisional. |
| Fraud is nocturnal — hours 00–04 carry 1.74% of volume and 19.85% of fraud. | `01_hourly_profile.csv` | A simulator scheduling artefact is at least as likely as a behavioural one. |
| Destination-side balance columns carry fraud signal. | `09_destination_artefact_rates.csv` | Lifts of only 1.30–1.54×, strongest solo AUC 0.6237 — **diffuse association, explicitly not the deterministic rule the origin side is** |
| Fraud transactions are larger (mean 1,467,967 vs 178,197). | `01_amount_summary.csv` | Generator parameter, not observed behaviour. |
| 63.28% of test rows involve an account seen in training. | `02_account_overlap.json` | History exists but is not exposed as features; no claim about whether it leaks. |

> **Writing note.** The `type` importance result contradicts prior work, which
> makes it attractive to state strongly. Resist: M9 is open, and a contradiction
> asserted on a distorted metric is the easiest finding in the report to attack.

## 5. Cannot be claimed from PaySim

The boundaries. Each is enforced somewhere in code or documented in a report.

| Cannot claim | Why | Guard |
|---|---|---|
| Anything empirical about authentication vulnerabilities. | No session, device, SIM, PIN, channel or location field. | `SUPPORTED_CONCLUSIONS` #9 |
| That step-up authentication controls work. | No control was tested; PaySim fraud is account-draining, not authentication compromise. | `AUTH_DESIGN_FORBIDDEN_CLAIM`; CHANGE-06 |
| That the feature set is simulator-independent or artefact-free. | Retained destination columns carry real generator association. | `FROZEN_RUNG_CLAIM` + a test |
| That per-instance severity weighting does not work. | Config F trains on `R·sᵢ` and is scored on `R` — objective mismatch. | `CONFIG_F_FORBIDDEN_CLAIM`; `docs/config_f_cost_objectives.md` |
| That CSL never helps. | Learner- and cost-dependent; only ranking quality is uniformly negative. | `FORBIDDEN_CLAIMS` #3 |
| Per-typology conclusions. | `isFraud` is binary. | `RESULTS_SYNTHESIS` §6.2 |
| Seasonality or salary-cycle effects. | 743 steps ≈ 31 days. | §6.5 |
| Transfer to real mobile money fraud. | Fraud confined to TRANSFER and CASH_OUT **by construction**. | §6.6 |
| A comparison against published results obtained on rebalanced test sets. | Not directly comparable. | E4 §J.3 |
| That notebook 05's fold-drop ordering holds. | Superseded by E4. | E4 §G |

---

## Staleness audit — the freeze predates the experiments

`src/momo_fraud/baseline.py` contains **zero references to E2, E3, E4 or E6**.
The frozen baseline was written before all four ran and its claim lists have not
been revisited since. Building this table surfaced three problems. They matter
because the guards exist to prevent *over*-claiming and are now, in places,
forcing *under*-claiming — which throws away the study's most expensive evidence.

**One guard is now factually false.**

> `FORBIDDEN_CLAIMS` #6: "Any CSL delta is stable across seeds or splits. **Every
> one rests on a single model seed and a single split seed.**"

The second sentence is untrue. E3 ran 220 fits over 4 split seeds × 4 model
seeds and found 116/116 paired comparisons negative, with the learner-dependent
NER pattern reproducing in every split and seed. Read literally, this guard tells
the report's author to disclaim the robustness E3 was run to establish — 4 hours
40 minutes of compute and the study's strongest reply to a reviewer.

The guard should be **narrowed, not deleted**: it remains correct for the
notebook-04 grid, which is single-seed. The corrected form distinguishes the two
grids and points at E3 for stability claims.

**Two guards are stale but not false.**

- #4 ("the decision-cost sweep... every CSL model in it was trained at
  `R_train = 773.70`") is still true *of that sweep*, but E2 has since swept
  `R_train` over `{10, 50, 100, 773.7, 2000, 5000}`. The guard should say where
  training-ratio claims may now legitimately be made.
- #5 ("evaluation prevalence alone accounts for the precision collapse") is still
  true *of the notebook 05 arms*, but E4 isolated prevalence exactly — one model,
  one threshold, positives retained, negatives subsampled. The guard should
  distinguish what notebook 05 cannot support from what E4 can.

**Four supported conclusions are understated.** #4 (CSL PR-AUC deltas) cites only
the 12-comparison single-seed result when E3 offers 116/116 across 16 cells; #5
(learner- and cost-dependence) cites the decision-cost sweep when E2 and E3 are
far stronger; #6 (threshold optimisation) predates E6's equivalence result; #7
(prevalence and precision) carries a caveat that E4 has since discharged.

**Resolved under CHANGE-07.** The four conclusions now cite E2/E3/E4/E6 and the three guards are scoped to the grid they are true of; counts are unchanged at 9 and 7. The audit below is retained as the record of what was found.

**Nothing here was a reason to change a number.** The measurements are unaffected.
What is stale is the study's record of *what it is entitled to say* about them —
which is exactly the record the report will be written from.

---

## How to use this when writing

1. **Every assertion gets a category before it gets a sentence.** If you cannot
   place it, you do not yet know what you are claiming.
2. **Category 1 and 2 claims lead.** They are the least attackable and the most
   distinctive. Most fraud-detection papers have none.
3. **Category 3 claims carry their grid.** "Across 4 splits and 4 seeds" is not
   padding; it is the claim's scope, and omitting it converts a measurement into
   an overclaim.
4. **Category 4 claims name their confound in the same paragraph.** Pre-empting
   the objection is stronger than surviving it.
5. **Category 5 goes in Methodology, not only in Limitations.** A boundary
   declared late reads as a concession; declared early it reads as design.
6. **When the freeze and an experiment report disagree, the experiment report is
   newer.** Until the staleness above is resolved, check `docs/E*_report.md`
   before quoting `baseline.py`.
