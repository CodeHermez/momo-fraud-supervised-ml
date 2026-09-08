# E3 — Multi-seed × multi-split CSL robustness

**Execution status: PASS.** 220/220 cells complete, 0 failed, 0 retried.
Quality control 22/22 PASS. Smoke test 16/16 PASS.

Frozen baseline `b2e6f624` unchanged throughout. Nothing was tuned, no feature
was selected, no baseline re-chosen, and R was not varied.

> **Research question.** Does the observed effect of cost-sensitive learning on
> PR-AUC and NER persist across independently varied training seeds and
> stratified train/validation/test splits?

---

## A. Execution status — PASS

| | |
|---|---|
| Cells planned | 220 |
| Completed | 220 |
| Failed | 0 |
| Retried | 0 |
| Omitted (deterministic duplicates) | 36 |
| Quality control | 22/22 PASS |
| Wall clock | ≈ 4h 40m |

**Baseline reproduction.** At the cell that overlaps the frozen baseline
(split seed 42, model seed 42, config C), all four learners reproduce it to
floating-point equality:

| Learner | PR-AUC frozen | PR-AUC E3 | NER frozen | NER E3 |
|---|---|---|---|---|
| Logistic regression | 0.503969 | 0.503969 | 0.249887 | 0.249887 |
| Decision tree | 0.729743 | 0.729743 | 0.152678 | 0.152678 |
| Random forest | 0.796750 | 0.796750 | 0.132355 | 0.132355 |
| XGBoost | 0.817258 | 0.817258 | 0.131376 | 0.131376 |

## B. Smoke test — PASS

Cell: **`xgboost__rus__split123__model456`** — a stochastic learner, a non-default
split seed, a non-default model seed, and the variant that exercises
training-only resampling.

All 16 checks passed. The two that carry the most weight:

- **Threshold locked, not re-fitted on test.** Validation selected 0.6055159; the
  test-optimal threshold would have been 0.5595563. The validation value was the
  one applied. The lock is demonstrably real, not merely asserted.
- **Resampling confined to training rows.** 11,498 rows drawn from 4,453,834
  training rows; validation and test untouched.

Also verified: rung, both seeds, 70/15/15 proportions, preprocessing fitted on
training rows only (cutoff equals the training p99 exactly), `R_train` and
`R_eval` both explicit, PR-AUC / NER / confusion counts recomputed from the
cached vectors, provenance complete, frozen artifacts unchanged by SHA-256.

## C. Matrix completion

**220 fits = 4 learners × 4 fit variants × 4 split seeds × (1 or 4 model seeds).**

The 36 omitted cells are logistic regression × {unweighted, weighted,
instance_weighted} × model seeds {123, 456, 789} × 4 split seeds. These are
**deterministic duplicates, not missing observations**: `lbfgs` never consumes
`random_state`, and `std_pr_auc` is exactly 0.0 across four seeds on the full
training set. Running them would have produced four identical numbers and
presented zero training variance as four independent replicates.

RUS is stochastic for *every* learner including logistic regression — the
undersampling draw consumes the model seed — so all 16 LR RUS cells were run.

Seven configurations come from four fits: thresholding is post-hoc.

## D. Exact experimental matrix

| Axis | Values |
|---|---|
| Learners | logistic regression, decision tree, random forest, XGBoost |
| Fit variants | unweighted, weighted, instance_weighted, rus |
| Split seeds | 42, 123, 456, 789 |
| Model seeds | 42, 123, 456, 789 (1 for deterministic LR cells) |
| Configs reported | C (unweighted), E (weighted), F (instance_weighted), G (rus) |
| Feature set | `no_origin_balance` (frozen) |
| `R_train` | 773.7011 for CSL arms; 1.0 for unweighted (no cost weighting) |
| `R_eval` | 773.7011 throughout — **R was not varied** |
| Threshold rule | NER-optimal in every reported arm |

**Every arm uses the same threshold rule**, so a CSL-vs-baseline delta isolates
the training-level intervention rather than confounding it with the decision
rule. This matters: the same unweighted model yields NER 0.659 at config A
(t = 0.5) and 0.250 at config C (NER-optimal).

All four splits carry identical prevalence (0.1291% in every partition) and
identical test composition (1,232 positives, 953,162 negatives). Stratification
holds the class balance fixed while membership varies, so split effects are not
confounded with prevalence effects.

## E. Split robustness

Baseline (config C) PR-AUC by split seed:

| Learner | 42 | 123 | 456 | 789 | range |
|---|---|---|---|---|---|
| Logistic regression | 0.5040 | 0.5338 | 0.5221 | 0.5345 | 0.0305 |
| Decision tree | 0.7298 | 0.7671 | 0.7511 | 0.7572 | 0.0373 |
| Random forest | 0.7964 | 0.8275 | 0.8181 | 0.8243 | 0.0311 |
| XGBoost | 0.8169 | 0.8422 | 0.8353 | 0.8405 | 0.0253 |

Split seed 42 — the one every prior result in this study used — is the **lowest**
of the four for all four learners. Absolute performance figures quoted from it
are therefore mildly pessimistic, by roughly 0.02–0.04 PR-AUC.

## F. Training-seed robustness

Variance decomposition, computed separately and never pooled — training-seed
spread is measured within a split, split spread within a model seed:

| Learner / variant | training-seed σ (PR-AUC) | split σ (PR-AUC) | ratio |
|---|---|---|---|
| XGBoost / unweighted | 0.00131 | 0.01164 | **8.9×** |
| Random forest / unweighted | 0.00168 | 0.01406 | **8.4×** |
| Decision tree / unweighted | 0.00024 | 0.01581 | **66×** |
| Logistic regression / unweighted | **0.00000** | 0.01427 | — |
| Decision tree / rus | 0.04168 | 0.02867 | **0.7×** |

**Split variability exceeds training-seed variability by roughly an order of
magnitude in almost every cell.** The single exception is decision tree under
RUS, where the draw of 11,498 training rows matters more to a single tree than
the partition does.

This is itself a methodological finding: the study's earlier multi-seed work
(`10_multiseed_ner.csv`) varied only the model seed at a fixed split, and was
therefore measuring the smaller of the two components.

**Logistic regression, non-RUS variants: training-seed variance = 0.**
This is a deterministic property of `lbfgs`, not an estimate obtained by
repeating identical fits.

## G. PR-AUC results — core class-constant CSL

ΔPR-AUC = PR-AUC(CSL) − PR-AUC(baseline), paired within learner, split seed and
model seed. Negative means CSL ranks worse.

| Learner | Variant | pairs | mean | median | σ | min | max | negative |
|---|---|---|---|---|---|---|---|---|
| Logistic regression | weighted | 4 | −0.3658 | −0.3646 | 0.0101 | −0.3787 | −0.3553 | **4/4** |
| Logistic regression | rus | 16 | −0.3835 | −0.3843 | 0.0235 | −0.4172 | −0.3344 | **16/16** |
| Decision tree | weighted | 16 | −0.0152 | −0.0155 | 0.0054 | −0.0221 | −0.0080 | **16/16** |
| Decision tree | rus | 16 | −0.5153 | −0.5199 | 0.0443 | −0.5845 | −0.4557 | **16/16** |
| Random forest | weighted | 16 | −0.0465 | −0.0451 | 0.0054 | −0.0544 | −0.0399 | **16/16** |
| Random forest | rus | 16 | −0.0641 | −0.0653 | 0.0108 | −0.0871 | −0.0464 | **16/16** |
| XGBoost | weighted | 16 | −0.0179 | −0.0179 | 0.0025 | −0.0222 | −0.0130 | **16/16** |
| XGBoost | rus | 16 | −0.0737 | −0.0745 | 0.0079 | −0.0844 | −0.0577 | **16/16** |

**116 of 116 paired comparisons are negative.** Every learner, every core CSL
mechanism, every split seed, every model seed. No cell crosses zero, and the
maximum (least negative) value in every row is still negative.

## H. NER results — core class-constant CSL

ΔNER = NER(CSL) − NER(baseline). Negative means CSL carries **lower** risk, i.e.
CSL is better.

| Learner | Variant | pairs | mean | median | σ | min | max | negative | direction |
|---|---|---|---|---|---|---|---|---|---|
| Logistic regression | weighted | 4 | −0.0330 | −0.0369 | 0.0084 | −0.0377 | −0.0204 | **4/4** | CSL better |
| Logistic regression | rus | 16 | −0.0320 | −0.0336 | 0.0077 | −0.0420 | −0.0191 | **16/16** | CSL better |
| Decision tree | weighted | 16 | −0.0516 | −0.0620 | 0.0239 | −0.0704 | −0.0119 | **16/16** | CSL better |
| Decision tree | rus | 16 | −0.0437 | −0.0513 | 0.0216 | −0.0686 | −0.0046 | **16/16** | CSL better |
| Random forest | weighted | 16 | +0.0212 | +0.0202 | 0.0051 | +0.0142 | +0.0301 | 0/16 | CSL worse |
| Random forest | rus | 16 | +0.0099 | +0.0097 | 0.0028 | +0.0053 | +0.0159 | 0/16 | CSL worse |
| XGBoost | weighted | 16 | +0.0110 | +0.0104 | 0.0037 | +0.0059 | +0.0179 | 0/16 | CSL worse |
| XGBoost | rus | 16 | +0.0028 | +0.0033 | 0.0031 | −0.0043 | +0.0074 | 3/16 | mostly worse |

**The NER effect is sharply learner-dependent, and consistently so.** The two
weaker learners (logistic regression, decision tree) benefit from CSL on
end-to-end risk in 64 of 64 pairs. The two stronger learners (random forest,
XGBoost) are harmed in 61 of 64. Only XGBoost/RUS is genuinely mixed — 3 of 16
pairs favour CSL, and its mean of +0.0028 sits inside its own split spread.

## I. Configuration F — sensitivity analysis, reported separately

Config F trains on `C_FN = R·sᵢ` but is thresholded and scored on `C_FN = R`.
It is **not** a like-for-like implementation of the class-constant objective and
is not merged into the core result above.

| Learner | pairs | ΔPR-AUC mean | σ | negative | ΔNER mean | σ | negative |
|---|---|---|---|---|---|---|---|
| Logistic regression | 4 | −0.2621 | 0.0054 | 4/4 | −0.0138 | 0.0099 | 4/4 |
| Decision tree | 16 | −0.0149 | 0.0044 | 16/16 | −0.0471 | 0.0228 | 16/16 |
| Random forest | 16 | −0.0390 | 0.0041 | 16/16 | +0.0166 | 0.0056 | 0/16 |
| XGBoost | 16 | −0.0189 | 0.0025 | 16/16 | +0.0117 | 0.0036 | 0/16 |

F tracks the core arms closely: uniformly negative on PR-AUC, learner-split on
NER. **This does not establish that severity weighting works or does not work.**
Training and evaluation use different cost objectives, and F carries roughly half
config D's effective positive weight, so the comparison cannot settle that
question in either direction. E3 establishes only what happens under this
specific implementation and this evaluation objective.

## J. Robustness conclusion

> **Does the previously observed negative CSL PR-AUC effect persist across
> different data splits and training seeds?**

### Ranking quality (PR-AUC): **robust.**

116 of 116 paired comparisons negative, across 4 learners × 2 core mechanisms ×
4 splits × 4 model seeds. Every per-row maximum is still negative, so no
combination of split and seed tested produces a CSL model that ranks better than
its unweighted counterpart. The effect sizes vary in magnitude (−0.015 to −0.52)
but never in sign.

The strength of this claim is bounded by what was varied: four stratified splits
of one dataset, four model seeds, one cost ratio, one feature set. It is not a
claim about cost-sensitive learning in general.

### Decision risk (NER): **learner-dependent, and robustly so.**

The learner-dependence is itself stable — it reproduces in every split and every
seed. Logistic regression and decision tree benefit (64/64 pairs); random forest
and XGBoost are harmed (61/64). This is not a case of a finding dissolving under
replication; it is a finding that *is* a dependence.

The one qualification: XGBoost/RUS at +0.0028 mean is small relative to its own
split spread and should be described as **indistinguishable from no effect**
rather than as a harm.

### Consequently

The earlier conclusion — *CSL does not improve ranking quality, while its effect
on end-to-end risk depends on the learner* — **survives replication and is now
supported by 220 fits over 4 splits and 4 seeds rather than a single run.**

## K. Remaining methodological uncertainty

Kept separate, because each answers a different question:

**Test-observation uncertainty.** Paired stratified bootstrap (n = 2000) within
the single fixed cell split42/model42. All 12 comparisons exclude zero. This
covers only which transactions landed in that one test partition — **it does not
quantify seed or split variability**, and is reported at one cell precisely so it
cannot be mistaken for either.

**Training-seed variability.** σ ≈ 0.001–0.003 PR-AUC for RF and XGBoost;
essentially zero for DT and exactly zero for LR (non-RUS). The exception is
DT/RUS at σ = 0.042. Small relative to the effects being measured, except there.

**Split variability.** σ ≈ 0.010–0.016 PR-AUC — roughly an order of magnitude
larger than training-seed variability. Split seed 42 is the lowest-scoring of
the four for every learner.

**Still unresolved, and not addressed by E3:**

- Only one cost ratio was used. `R_train` was fixed at 773.70 throughout; nothing
  here speaks to CSL at other training cost ratios.
- One dataset, one feature set, one simulator. Split resampling does not create
  independent data.
- Four splits and four seeds give a coarse variance estimate; the σ values are
  themselves uncertain.
- Config F's objective mismatch is unresolved by design.
- The `no_origin_balance` rung carries reduced, not eliminated, simulator
  artefact exposure.

## L. Recommendation on E2 — proceed

**E2 (training-cost-ratio sweep) should proceed, and is now the highest-value
next experiment.** Three reasons, all grounded in E3:

1. **E3 removes the main alternative explanation.** Before E3, a negative CSL
   result could have been an artefact of one split or one seed. It is not. The
   effect is stable enough that varying the training cost ratio will measure the
   cost ratio rather than noise.

2. **E3 gives E2 its noise floor.** We now know split σ ≈ 0.010–0.016 PR-AUC and
   training-seed σ ≈ 0.001–0.003. Any `R_train` effect E2 reports can be judged
   against measured variability instead of assumed variability. Without E3, an
   `R_train` effect of 0.01 would have been uninterpretable.

3. **E3 sharpens the question E2 should ask.** The learner-split in the NER
   result — weak learners helped, strong learners harmed, at a single
   `R_train` = 773.70 — is exactly the pattern that a training-cost sweep can
   explain or refute. The obvious hypothesis is that 773.70 over-weights the
   minority class for high-capacity learners and under-weights it for
   low-capacity ones. E2 tests that directly.

**Scope note.** E2 need not repeat E3's full seed×split grid. A defensible design
is the `R_train × R_eval` grid at a reduced replication (2 splits × 2 seeds),
using E3's variance estimates to size the comparison — but that is a design
decision for the next authorization, not something to assume here.

**Not recommended before E2:** E4 (prevalence-only) and E6 (threshold divergence)
are both cheap and both answer narrower questions. They can run after E2 or in
parallel; neither blocks it.
