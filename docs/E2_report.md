# E2 — Training-cost-ratio (`R_train`) sweep

**Closure status: PASS.** 364/364 analysis cells present, all seven pre-declared
`R_train` conditions represented, quality control 21/21 PASS.

Frozen baseline `b2e6f624` unchanged throughout. `R_eval` held at the frozen
default 773.7011 for every cell; the feature rung stayed `no_origin_balance`;
thresholds were validation-selected and re-selected per cell.

> **Research question.** Within the tested PaySim configuration, does varying the
> training cost ratio `R_train` systematically affect PR-AUC and NER, and does
> that effect differ by learner?

This document is a **reporting and closure verification** of an execution that had
already been accepted provisionally. No model was re-fitted to produce it, and no
file under `results/` was modified. Every figure below was recomputed read-only
from `results/E2_runs.csv` and cross-checked against the stored summaries.

Section letters A–H correspond to the requested closure-verification outputs.

---

## A. Closure status — PASS

### A.1 All seven pre-declared conditions are present

The interim summary displayed only five of the seven `R_train` levels. **This was
a presentation omission, not a gap in the analytical dataset.** R = 50 and
R = 2000 are both fully populated, each carrying a complete 52-cell block
identical in structure to every other weighted level.

| `R_train` | Role | Cells | DT | RF | XGB | LR |
|---|---|---|---|---|---|---|
| 1 | unweighted reference | 52 | 16 | 16 | 16 | 4 |
| 10 | weighted | 52 | 16 | 16 | 16 | 4 |
| **50** | weighted (**omitted from interim summary**) | **52** | 16 | 16 | 16 | 4 |
| 100 | weighted | 52 | 16 | 16 | 16 | 4 |
| 773.7011 | weighted, frozen default | 52 | 16 | 16 | 16 | 4 |
| **2000** | weighted (**omitted from interim summary**) | **52** | 16 | 16 | 16 | 4 |
| 5000 | weighted | 52 | 16 | 16 | 16 | 4 |

No condition is missing. No stop condition was triggered.

### A.2 Composition audit

| | |
|---|---|
| Analysis cells | 364 |
| — new E2 cells | 260 |
| — reused E3 unweighted reference cells | 52 |
| — reused E3 default-weighted cells | 52 |
| Duplicate `cell_id` | 0 |
| Distinct `R_eval` values | 1 (773.7011) |
| Split seeds | 42, 123, 456, 789 |
| Model seeds | 42, 123, 456, 789 (LR: 42 only, deterministic) |

52 + 52 + 260 = **364**, matching `results/E2_runs.csv` exactly. The `source`
column confirms the split as 260 `E2` / 104 `E3`.

Cell counts are 16 per learner per level for the three stochastic learners
(4 split seeds × 4 model seeds) and 4 for logistic regression, which is
deterministic in the model seed and therefore not replicated across it.

### A.3 E3 reuse — verified numerically identical

The 104 reused cells were matched against `results/E3_runs.csv` on the full key
`(cell_id, config, level, threshold_rule, split)`. **`cell_id` alone is not
unique in `E3_runs.csv`** — 220 unique ids across 752 rows — so an id-only join
silently mismatches rows; the full key joins 1:1 with zero unmatched.

On that join all 104 cells are **bit-exact** on every decision-relevant field:
`pr_auc`, `roc_auc`, `brier`, `threshold`, `tp`/`fp`/`fn`/`tn`, `precision`,
`recall`, `f1`, `mcc`, `n_flagged`, `alert_rate`, `R`, `ner`, and all prevalence
and count fields.

Two fields differ at float round-trip precision only: `total_risk` on 1 of 104
cells (max |Δ| = 2.9 × 10⁻¹¹) and `fit_seconds` on 1 of 104 (2.8 × 10⁻¹⁴). These
are CSV decimal round-trip artefacts, not recomputation.

The reused rows are therefore **numerically identical but not byte-identical**:
the E2 rows carry four additional columns (`csl_mechanism`, `is_default_R_train`,
`r_train_parameter`, `source`) and relabelled `config` / `level` /
`threshold_rule` fields. Reuse is confirmed; recomputation did not occur.

### A.4 A counting note that should not resurface

`results/E2_qc.json` reports "520/520 rows verified on the fitted estimator"
while `results/E2_r_train_verification.csv` contains 260 rows. Both are correct
with different denominators, and no discrepancy exists:

- the QC check at `scripts/experiments/E2_analysis.py:335` uses
  `new = load_e2_cells()` **unfiltered**, i.e. 260 cells × {validation, test}
  = 520 evaluation rows;
- the persisted CSV at `scripts/experiments/E2_analysis.py:266` applies
  `.query("split == 'test'").drop_duplicates("cell_id")`, giving 260.

`results/E2_runs.csv` is likewise test-only (all 364 rows `split == 'test'`).

---

## B. The seven-level `R_train` table

Machine-readable copy: `report/tables/E2_r_train_full.csv` (28 rows), verified
equal to `results/E2_sweep_summary.csv` to floating-point precision on all eight
statistics. R = 1 is the unweighted reference; R = 773.7011 is the frozen default.

### XGBoost (n = 16 per level)

| `R_train` | PR-AUC mean | sd | min | max | NER mean | sd | min | max |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.8337 | 0.0104 | 0.8159 | 0.8440 | 0.1171 | 0.0091 | 0.1076 | 0.1340 |
| 10 | 0.8345 | 0.0112 | 0.8166 | 0.8461 | 0.1200 | 0.0101 | 0.1089 | 0.1355 |
| 50 | 0.8283 | 0.0116 | 0.8091 | 0.8405 | 0.1211 | 0.0102 | 0.1093 | 0.1376 |
| 100 | 0.8254 | 0.0117 | 0.8053 | 0.8385 | 0.1213 | 0.0097 | 0.1091 | 0.1381 |
| 773.7011 | 0.8158 | 0.0121 | 0.7937 | 0.8283 | 0.1282 | 0.0117 | 0.1152 | 0.1494 |
| 2000 | 0.8098 | 0.0119 | 0.7893 | 0.8206 | 0.1335 | 0.0103 | 0.1214 | 0.1522 |
| 5000 | 0.8036 | 0.0118 | 0.7836 | 0.8174 | 0.1393 | 0.0105 | 0.1283 | 0.1595 |

### Random forest (n = 16 per level)

| `R_train` | PR-AUC mean | sd | min | max | NER mean | sd | min | max |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.8166 | 0.0126 | 0.7947 | 0.8284 | 0.1152 | 0.0085 | 0.1036 | 0.1324 |
| 10 | 0.8109 | 0.0138 | 0.7877 | 0.8226 | 0.1105 | 0.0078 | 0.1018 | 0.1255 |
| 50 | 0.8077 | 0.0134 | 0.7839 | 0.8207 | 0.1109 | 0.0074 | 0.1006 | 0.1218 |
| 100 | 0.8020 | 0.0140 | 0.7761 | 0.8156 | 0.1122 | 0.0086 | 0.1023 | 0.1258 |
| 773.7011 | 0.7701 | 0.0145 | 0.7431 | 0.7885 | 0.1364 | 0.0099 | 0.1225 | 0.1527 |
| 2000 | 0.7498 | 0.0158 | 0.7204 | 0.7693 | 0.1530 | 0.0117 | 0.1330 | 0.1751 |
| 5000 | 0.7231 | 0.0171 | 0.6962 | 0.7482 | 0.1805 | 0.0086 | 0.1687 | 0.1967 |

### Decision tree (n = 16 per level)

| `R_train` | PR-AUC mean | sd | min | max | NER mean | sd | min | max |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.7513 | 0.0141 | 0.7297 | 0.7675 | 0.1791 | 0.0162 | 0.1527 | 0.1931 |
| 10 | 0.7657 | 0.0178 | 0.7407 | 0.7862 | 0.1381 | 0.0110 | 0.1259 | 0.1544 |
| 50 | 0.7603 | 0.0104 | 0.7431 | 0.7690 | 0.1321 | 0.0104 | 0.1173 | 0.1424 |
| 100 | 0.7587 | 0.0095 | 0.7505 | 0.7744 | 0.1310 | 0.0109 | 0.1177 | 0.1445 |
| 773.7011 | 0.7360 | 0.0148 | 0.7113 | 0.7453 | 0.1275 | 0.0095 | 0.1171 | 0.1408 |
| 2000 | 0.6810 | 0.0067 | 0.6738 | 0.6883 | 0.1338 | 0.0103 | 0.1199 | 0.1478 |
| 5000 | 0.6685 | 0.0098 | 0.6543 | 0.6783 | 0.1357 | 0.0097 | 0.1222 | 0.1485 |

### Logistic regression (n = 4 per level; deterministic in model seed)

| `R_train` | PR-AUC mean | sd | min | max | NER mean | sd | min | max |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.5236 | 0.0143 | 0.5040 | 0.5345 | 0.2287 | 0.0147 | 0.2163 | 0.2499 |
| 10 | 0.4904 | 0.0167 | 0.4662 | 0.5022 | 0.2180 | 0.0176 | 0.2079 | 0.2442 |
| 50 | 0.3874 | 0.0177 | 0.3609 | 0.3979 | 0.2102 | 0.0124 | 0.2019 | 0.2287 |
| 100 | 0.3222 | 0.0174 | 0.2966 | 0.3332 | 0.2076 | 0.0110 | 0.1998 | 0.2239 |
| 773.7011 | 0.1578 | 0.0073 | 0.1487 | 0.1657 | 0.1957 | 0.0124 | 0.1854 | 0.2130 |
| 2000 | 0.1386 | 0.0100 | 0.1278 | 0.1493 | 0.1900 | 0.0097 | 0.1796 | 0.2019 |
| 5000 | 0.1297 | 0.0114 | 0.1180 | 0.1421 | 0.1906 | 0.0078 | 0.1820 | 0.2008 |

---

## C. Monotonicity verification

The interim summary stated: *"PR-AUC falls monotonically with `R_train` for all
four learners."* **That statement is overstated in two distinct ways** and is
corrected here.

### C.1 (D) The ordered mean PR-AUC sequences

Ordered R = 1 → 10 → 50 → 100 → 773.7011 → 2000 → 5000:

| Learner | Sequence |
|---|---|
| XGBoost | 0.8337 → **0.8345** → 0.8283 → 0.8254 → 0.8158 → 0.8098 → 0.8036 |
| Random forest | 0.8166 → 0.8109 → 0.8077 → 0.8020 → 0.7701 → 0.7498 → 0.7231 |
| Decision tree | 0.7513 → **0.7657** → 0.7603 → 0.7587 → 0.7360 → 0.6810 → 0.6685 |
| Logistic regression | 0.5236 → 0.4904 → 0.3874 → 0.3222 → 0.1578 → 0.1386 → 0.1297 |

XGBoost and the decision tree both **rise** from the unweighted reference to
R = 10 before falling; the bolded value is each one's grid maximum.

### C.2 (A) Aggregated-mean monotonicity

| Learner | Strictly decreasing over all 7 levels | Over the weighted sub-grid (R ≥ 10) | Sign changes (7 levels) |
|---|---|---|---|
| XGBoost | **No** | Yes | 1 |
| Random forest | Yes | Yes | 0 |
| Decision tree | **No** | Yes | 1 |
| Logistic regression | Yes | Yes | 0 |

### C.3 (B) Cell-level monotonicity — it does **not** hold

Counting individual split × model-seed trajectories that are strictly decreasing:

| Learner | Over all 7 levels | Over R ≥ 10 only |
|---|---|---|
| XGBoost | **5 / 16** | 15 / 16 |
| Random forest | 14 / 16 | 14 / 16 |
| Decision tree | **0 / 16** | 8 / 16 |
| Logistic regression | 4 / 4 | 4 / 4 |

**The aggregate and cell levels differ, materially so for the decision tree and
XGBoost.** Monotonicity in this experiment is a property of the aggregated means;
it is not a property of the individual cells, and must not be reported as one.

### C.4 (C) Spearman rank correlation, `R_train` vs PR-AUC

| Learner | Means, n = 7 | p | Means, n = 6 (R ≥ 10) | Pooled cells | p |
|---|---|---|---|---|---|
| XGBoost | −0.9643 | 4.54 × 10⁻⁴ | −1.0000 | −0.6954 | 1.8 × 10⁻¹⁷ |
| Random forest | −1.0000 | — | −1.0000 | −0.8893 | 3.7 × 10⁻³⁹ |
| Decision tree | −0.7857 | 0.0362 | −1.0000 | −0.7380 | 1.6 × 10⁻²⁰ |
| Logistic regression | −1.0000 | — | −1.0000 | −0.9793 | 1.4 × 10⁻¹⁹ |

**Two corrections to the stored artefact `results/E2_monotonicity.csv`:**

1. **It covers the six weighted levels only.** `scripts/experiments/E2_analysis.py:220`
   filters `summary[summary["csl_mechanism"] == "class_weighting"]`, and the
   unweighted R = 1 reference carries `csl_mechanism == "none"`, so it is dropped
   before the Spearman and monotonicity computation. The filter is defensible in
   itself, but the emitted CSV carries no column recording the restriction — that
   is what allowed the interim summary to read it as covering all seven
   conditions. The artefact's own values are honest: it correctly reports
   `monotone = False` for decision-tree and logistic-regression NER, and its
   docstring states that "a non-monotone response is a finding, not noise to
   smooth away." **The overclaim was introduced in the prose summary, not in the
   computation.**
2. **Its `spearman_p = 0.0` entries are a SciPy large-sample-approximation
   artefact,** not a true zero. For a perfect rank correlation the exact
   two-sided permutation p is 2/6! = **0.00278** at n = 6 and 2/7! = **0.000397**
   at n = 7. Those are the values to quote.

### C.5 Corrected wording

> Within the tested grid, mean PR-AUC decreases monotonically in `R_train` across
> the weighted levels (R ≥ 10) for all four learners. Including the unweighted
> reference, monotonicity holds for random forest and logistic regression only:
> XGBoost and the decision tree both attain their highest mean PR-AUC at R = 10
> rather than at R = 1. Monotonicity is a property of the aggregated means;
> individual split × seed cells are frequently non-monotonic (0/16 decision-tree
> and 5/16 XGBoost trajectories are strictly decreasing over the full grid).

---

## D. NER minimum verification

Reported as the **minimum observed NER in the tested grid**. No `R_train` value is
described as optimal: the grid is coarse, closed, and pre-declared, and for two
learners the minimum sits at a grid boundary where the true minimiser is not
bracketed.

| Learner | Min observed NER | at `R_train` | Boundary / interior | Monotonic over 7 levels | Exact ties |
|---|---|---|---|---|---|
| XGBoost | 0.1171 | **1** (unweighted) | boundary (low) | yes — **increasing**, ρ = +1.000 | none |
| Random forest | 0.1105 | **10** (lowest weighted) | boundary (low) | no — 1 sign change, ρ = +0.786, p = 0.036 | none |
| Decision tree | 0.1275 | **773.7011** | **interior** | no — 1 sign change, ρ = −0.429, p = 0.337 (n.s.) | none |
| Logistic regression | 0.1900 | **2000** | **interior** | no — 1 sign change, ρ = −0.964, p = 4.5 × 10⁻⁴ | none |

**Ties.** There are no exact ties among the means. There are, however, levels
statistically indistinguishable from the minimum — those falling within one
within-level standard deviation of it:

| Learner | Levels within 1 sd of the minimum |
|---|---|
| XGBoost | 1, 10, 50, 100 |
| Random forest | 1, 10, 50, 100 |
| Decision tree | 50, 100, 773.7011, 2000, 5000 |
| Logistic regression | 773.7011, 2000, 5000 |

For the decision tree, five of the seven levels are within one standard deviation
of the minimum; its NER response over the weighted grid is not distinguishable
from flat.

**Per-cell argmin distributions** confirm the aggregate picture is not an artefact
of averaging, while showing it is not unanimous either:

| Learner | Distribution of the per-cell NER-minimising `R_train` |
|---|---|
| XGBoost | R = 1: 12, R = 10: 2, R = 50: 1, R = 100: 1 |
| Random forest | R = 10: 7, R = 50: 6, R = 100: 3 |
| Decision tree | R = 773.7011: 12, R = 50: 4 |
| Logistic regression | R = 2000: 2, R = 5000: 2 |

Logistic regression's four cells split evenly between two levels, so its interior
minimum is weakly determined and should be reported as such.

---

## E. Variance verification

Three uncertainty sources are distinguished. They are not interchangeable, and
the E2 sweep speaks to only the second and third; the first is carried over from
E3's bootstrap.

**1. Test-observation uncertainty** — resampling test rows, holding the fitted
model fixed. From `results/E3_test_observation_bootstrap.csv` (n_boot = 2000),
95% CI half-widths on paired PR-AUC deltas:

| Learner | min | median | max |
|---|---|---|---|
| XGBoost | 0.0061 | 0.0061 | 0.0117 |
| Random forest | 0.0090 | 0.0099 | 0.0113 |
| Decision tree | 0.0101 | 0.0106 | 0.0178 |
| Logistic regression | 0.0209 | 0.0220 | 0.0224 |

**2. Training-seed variability** — refitting on the same split with a different
model seed. **3. Split variability** — refitting on independently drawn
stratified splits. From `results/E2_variance_decomposition.csv`, across all seven
levels:

| Learner | Metric | Training-seed sd | Split sd | Ratio (split / seed) |
|---|---|---|---|---|
| XGBoost | PR-AUC | 9.4 × 10⁻⁴ – 1.9 × 10⁻³ | 0.0116 – 0.0135 | 6.9 – 13.3× |
| Random forest | PR-AUC | 1.2 × 10⁻³ – 3.3 × 10⁻³ | 0.0141 – 0.0189 | 5.8 – 12.6× |
| Decision tree | PR-AUC | 2.6 × 10⁻⁷ – 2.5 × 10⁻⁴ | 0.0075 – 0.0199 | 64 – 42,510× |
| Logistic regression | PR-AUC | 0 exactly | 0.0073 – 0.0177 | undefined |
| XGBoost | NER | 1.6 × 10⁻³ – 3.3 × 10⁻³ | 0.0101 – 0.0129 | 3.2 – 7.1× |
| Random forest | NER | 1.9 × 10⁻³ – 4.4 × 10⁻³ | 0.0082 – 0.0123 | 2.2 – 4.4× |
| Decision tree | NER | 3.0 × 10⁻⁷ – 1.2 × 10⁻⁴ | 0.0106 – 0.0181 | 105 – 38,137× |
| Logistic regression | NER | 0 exactly | 0.0078 – 0.0176 | undefined |

### E.1 The "two orders of magnitude" claim is not supported across learners

It holds for the decision tree (two to four orders) and trivially for logistic
regression, which is exactly deterministic in the model seed. **For XGBoost and
random forest the margin is roughly 2–13×, i.e. well under one order of
magnitude.** Corrected wording:

> Split variability exceeds training-seed variability for every learner tested.
> The margin is under one order of magnitude for XGBoost and random forest
> (≈ 3–13×) and two to four orders for the decision tree, whose fit is
> near-deterministic given the split; logistic regression is exactly
> deterministic in the model seed, so the ratio is undefined. A blanket
> "two orders of magnitude" statement is not supported across learners.

### E.2 The binding uncertainty is the split

Split sd (0.0073 – 0.0199) is of the same order as, and frequently larger than,
the test-observation CI half-width (0.0061 – 0.0224). Test-set size is not the
limiting factor on precision in this design; the number of independent splits is.
Only four split seeds were used.

---

## F. Interpretation

Language throughout this section is deliberately associational. E2 varied
`R_train` while holding the split protocol, feature rung, threshold rule and
`R_eval` fixed, which supports attributing the within-E2 differences to
`R_train`; it does not license causal claims beyond that design, and no result is
claimed to hold outside the tested PaySim configuration and R grid.

### F.1 The default R = 773.7011 in context

Paired within split × model seed; deltas are default **minus** comparator. Every
PR-AUC delta below is unanimous in sign across all matched cells (16/16, or 4/4
for logistic regression).

| Learner | vs unweighted (R = 1) | vs lowest tested R = 10 | vs highest tested R = 5000 |
|---|---|---|---|
| | ΔPR-AUC / ΔNER | ΔPR-AUC / ΔNER | ΔPR-AUC / ΔNER |
| XGBoost | −0.0179 / +0.0110 | −0.0187 / +0.0081 | +0.0122 / −0.0111 |
| Random forest | −0.0465 / +0.0212 | −0.0408 / +0.0259 | +0.0470 / −0.0441 |
| Decision tree | −0.0152 / −0.0516 | −0.0297 / −0.0106 | +0.0675 / −0.0082 |
| Logistic regression | −0.3658 / −0.0330 | −0.3326 / −0.0223 | +0.0281 / +0.0050 |

Rank of the default within the seven-level grid:

| Learner | PR-AUC rank (1 = highest) | NER rank (1 = lowest) |
|---|---|---|
| XGBoost | 5 / 7 | 5 / 7 |
| Random forest | 5 / 7 | 5 / 7 |
| Decision tree | 5 / 7 | **1 / 7** |
| Logistic regression | 5 / 7 | 3 / 7 |

**Positioning.** Within the tested grid the default is **intermediate-to-
aggressive**: 5th of 7 on PR-AUC for all four learners, placing it in the upper
(more aggressive) half — more aggressive than R = 1, 10, 50 and 100, less so than
2000 and 5000. Its NER character is learner-dependent: aggressive and costly for
XGBoost and random forest, whose NER at the default is worse than at every lower R
tested, but at or near the observed NER floor for the decision tree (rank 1) and
logistic regression (rank 3). This describes position within the tested grid only.

### F.2 The learner ranking claim — confirmed, and stronger than stated

The interim summary stated *"XGB > RF > DT > LR at every `R_train`."* **This claim
survives verification at both levels** and can be stated at full strength.

| `R_train` | XGB | RF | DT | LR | Holds | XGB − RF gap |
|---|---|---|---|---|---|---|
| 1 | 0.8337 | 0.8166 | 0.7513 | 0.5236 | yes | +0.0171 |
| 10 | 0.8345 | 0.8109 | 0.7657 | 0.4904 | yes | +0.0236 |
| 50 | 0.8283 | 0.8077 | 0.7603 | 0.3874 | yes | +0.0206 |
| 100 | 0.8254 | 0.8020 | 0.7587 | 0.3222 | yes | +0.0234 |
| 773.7011 | 0.8158 | 0.7701 | 0.7360 | 0.1578 | yes | +0.0457 |
| 2000 | 0.8098 | 0.7498 | 0.6810 | 0.1386 | yes | +0.0600 |
| 5000 | 0.8036 | 0.7231 | 0.6685 | 0.1297 | yes | +0.0805 |

Beyond the aggregate means, the ordering also holds in **every individual
split × seed cell: 0 violations out of 112 matched cells**, with all three
pairwise dominance rates equal to 1.00 at all seven levels. (Logistic regression
carries a single model seed, so cells were matched on `split_seed`.) This is
invariant individual-cell ordering, not merely invariant aggregate ordering.

Worth recording alongside it: the XGBoost−random-forest gap **widens** with
`R_train`, from +0.0171 at R = 1 to +0.0805 at R = 5000.

### F.3 The four questions

**Q1 — Does increasing `R_train` systematically affect PR-AUC?**
Yes, within the tested grid. Across the weighted levels every learner's mean
PR-AUC decreases monotonically in `R_train`, with unanimous-sign paired deltas.
The magnitude is strongly learner-dependent: total range across the grid is 0.031
(XGBoost), 0.094 (random forest), 0.097 (decision tree) and 0.361 (logistic
regression). For XGBoost the R = 10 versus unweighted delta is +0.0008
(sd 0.0023, 5/16 cells negative) — indistinguishable from zero — so light
weighting is associated with essentially no PR-AUC cost for XGBoost, whereas
heavy weighting is.

**Q2 — Does increasing `R_train` systematically affect NER?**
Not in a common direction. NER rises with `R_train` for XGBoost (ρ = +1.00) and
random forest (ρ = +0.79); it falls for logistic regression (ρ = −0.96); and for
the decision tree the relationship is non-monotonic and not distinguishable from
flat over the weighted grid (ρ = −0.086, p = 0.87). **The direction of the NER
response is itself learner-dependent** — a single summary statement about NER and
`R_train` would misrepresent at least one learner.

**Q3 — Does the effect differ by learner?**
Yes, on both axes, and this is the clearest E2 result. Ranked by PR-AUC
sensitivity to `R_train`: LR ≫ DT ≈ RF ≫ XGB. Ranked by where NER is minimised
in the tested grid: XGBoost at R = 1, random forest at R = 10, decision tree at
R = 773.7011, logistic regression at R = 2000. Within this configuration, the
higher-capacity ensembles are associated with the least PR-AUC degradation and
with NER minima at *low* R, while the linear model shows the opposite pattern on
both axes.

**Q4 — Does E2 evidence that the default R = 773.7011 contributes to the
learner-dependent CSL behaviour observed in E3?**
E2 provides evidence **consistent with** that account, but the design supports
association rather than causation. At the default, the paired PR-AUC cost
relative to unweighted is −0.018 (XGBoost), −0.047 (random forest), −0.015
(decision tree) and −0.366 (logistic regression) — an approximately twenty-fold
spread across learners at a single fixed R. Because the default sits in the upper
half of the tested grid, it falls in a region where the learners' responses have
already diverged sharply, whereas at R = 10 those responses are much closer
together. The choice of R is therefore **associated with** the magnitude of the
learner split observed in E3.

It does not follow that `R_train` is the sole or primary driver of E3's
behaviour. E3 also varied the CSL mechanism (weighted, instance-weighted, RUS),
which E2 held fixed at class weighting; that factor is confounded out of E2 and
remains a live alternative explanation.

---

## G. Wording corrections carried forward

None of the following appear in any file on disk — the interim summary was prose
only — but they are recorded so they do not enter the thesis text.

| Do not write | Write instead |
|---|---|
| "R = 10 is optimal" | "R = 10 gives the minimum observed NER in the tested grid for random forest" |
| "CSL does not work" | "higher `R_train` is associated with lower test PR-AUC within the tested grid" |
| "XGBoost is immune to cost weighting" | "XGBoost shows the smallest PR-AUC response to `R_train` within the tested grid (range 0.031)" |
| "cost weighting causes overfitting" | "higher `R_train` is associated with lower test PR-AUC" |
| "the default R is universally wrong" | "the default sits in the upper half of the tested grid and is associated with a learner-dependent PR-AUC cost" |
| "the results generalize beyond PaySim" | "within the tested PaySim configuration and R grid" |

Of the three substantive claims made in the interim summary: the learner ranking
is **confirmed and strengthened** (F.2); the monotonicity claim is **corrected**
(C.5); the variance claim is **corrected** (E.1).

---

## H. Remaining methodological limitations

Only limitations still live after E2 are listed.

1. **Boundary minima.** XGBoost's and random forest's NER minima sit at the low
   edge of the grid (R = 1 and R = 10). The tested range does not bracket them;
   no R between 1 and 10 was tested, so a lower minimiser cannot be excluded.
2. **The grid is coarse and log-spaced.** The 100 → 773.7011 → 2000 span is where
   the decision tree and random forest change fastest and is covered by three
   points.
3. **Logistic regression has n = 4 cells per level** (single model seed,
   deterministic). Its interior NER minimum rests on a 2–2 split of four cells
   and is weakly determined.
4. **Split variability is the binding uncertainty** (sd 0.0073 – 0.0199,
   comparable to the test-observation CI half-width) and only four split seeds
   were used.
5. **Mechanism is confounded out, not tested.** E2 varies R under class weighting
   only; it cannot speak to instance weighting or RUS, both of which E3 covered.
   This directly limits the strength of the Q4 answer.
6. **Single dataset and single configuration.** PaySim, one frozen feature rung
   (`no_origin_balance`), one threshold rule (`ner_optimal`, validation-selected).
   No external-validity claim is available.
7. **`R_eval` held fixed at 773.7011 throughout**, so train/eval cost-ratio
   mismatch is uninvestigated — that is E6's question, not E2's.

---

## I. Recommendation on E4 — proceed

**E2 closure passes and does not block E4.** All seven pre-declared conditions are
present, the composition arithmetic is exact, the E3 reuse chain is verified
numerically identical, and the three narrative issues identified above are
documentation corrections rather than data defects. Nothing in E2 needs re-running.

Two points E2 contributes to E4's design:

1. **Hold `R_train` fixed at a declared value in E4** and state it. Since the
   `R_train` effect is strongly learner-dependent, leaving R free would confound
   any prevalence effect with the learner-specific R response documented here.
2. **Reuse the same four split seeds.** Split variability is the binding noise
   source (§E.2), so matching E2's split protocol lets E4's deltas be judged
   against the same noise floor rather than a new one.

E4 has **not** been executed. Neither has E6. This document ends at the
recommendation.
