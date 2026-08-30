# E4 — Arm D: prevalence-only manipulation

**Execution status: PASS.** 104/104 designated cells evaluated, 0 skipped.
62,504 evaluation rows. Quality control 14/14 PASS. Smoke test 7/7 PASS.

Frozen baseline `b2e6f624` unchanged. No model was fitted: the experiment reads
the probability vectors E3 cached and re-evaluates them on subsampled negative
sets. The script imports no model library, so the no-retraining guarantee is
structural rather than asserted.

> **Research question.** How much of the precision collapse reported in the
> Lokanan replication is attributable to *evaluated prevalence alone*, holding
> the model, the operating point and the real positives fixed?

---

## A. Design

| | |
|---|---|
| Manipulation | test-set negatives subsampled without replacement |
| Positives | all 1,232 real positives retained; none synthesised |
| Model | frozen per cell; no refitting at any level |
| Threshold | E3 validation-selected, locked across every level |
| `R_eval` | 773.7011 throughout |
| Cells | 104 (unweighted reference + weighted at default R; 4 learners) |
| Prevalence levels | 7, pre-declared and closed |
| Replicates | 100 per cell per stochastic level; natural is deterministic |

**On the grid.** The audit pre-declared "real negatives subsampled to
{50%, 10%, 1%, 0.129%}". That phrasing admits two readings — *target evaluated
prevalence* or *retained fraction of negatives* — which give different grids.
Rather than guess, the executed grid is the **union**, so both readings are
reported in full and neither is privileged.

| Level | Negative retention | Evaluated prevalence | Reading |
|---|---|---|---|
| natural | 1.0 | 0.1291% | A (no-op control) |
| 0.258% | 0.5 | 0.2581% | B |
| 1% | 0.12879 | 1.0000% | A |
| 1.28% | 0.1 | 1.2765% | B |
| 10% | 0.011620 | 10.000% | A |
| 11.3% | 0.01 | 11.446% | B |
| 50% | 0.00129087 | 50.000% | A and B |

Levels within a replicate are **nested** — the smaller subsample is a subset of
the larger — which pairs the levels and removes sampling noise from between-level
comparisons. Each level's marginal remains a uniform random subset of the correct
size.

**What cannot move, and did not.** Positives, the threshold, TP, FN and therefore
recall are invariant by construction. QC asserts this exactly: 0 cells with a
moving recall, 0 TP violations, 0 FN violations, and 1,232 positives at every
level of every cell.

---

## B. Verification

Four checks establish that the manipulation did what it claims.

1. **The natural level reproduces the stored E3 cell.** Max |diff| across all 104
   cells: PR-AUC 3.33 × 10⁻¹⁶, NER 1.11 × 10⁻¹⁶, precision exactly 0. The control
   is the same evaluation E3 already reported.
2. **Average precision matches scikit-learn.** The tie-aware fast implementation
   agrees with `average_precision_score` to < 10⁻¹² on both the full test set and
   a balanced subsample.
3. **Observed precision matches its closed form.** Under uniform negative
   subsampling at rate *r* with the threshold fixed, E[FP] = *r*·FP and TP is
   unchanged, so precision must converge to TP/(TP + *r*·FP). Max deviation across
   all cells and levels: 2.33 × 10⁻³. The subsample is uniform.
4. **Achieved prevalence equals the target**, max |difference| 0.

### B.1 A bug this gate caught

The first execution failed QC on check 1. E3 saves cached vectors as
`seed{model_seed}__{rung}__s{split_seed}`
(`scripts/experiments/E3_multiseed_multisplit.py:190-194`); the first draft of
E4 had the two seeds transposed. This loaded a *real* cached vector for every
cell — just the wrong one, pairing each cell's threshold with another cell's
scores. It reproduced correctly on the 26 cells where `split_seed == model_seed`
and failed on the other 78 in swapped pairs.

Two things are worth recording. The smoke-test cell (split 42, model 42) sits on
that diagonal, so the smoke test passed while the experiment was wrong — a
single well-chosen cell is not sufficient verification. And the failure produced
entirely plausible prevalence curves; only the reproduction check against stored
E3 values exposed it. The corrected run reproduces all 104 cells to 3.3 × 10⁻¹⁶.

---

## C. Precision is a function of prevalence

Weighted arm (default R = 773.7011), mean over 4 split seeds × 4 model seeds ×
100 replicates. **The model and threshold are identical in every row of each
column; only the evaluated negative count changes.**

| Evaluated prevalence | XGBoost | Random forest | Decision tree | Logistic regression |
|---|---|---|---|---|
| natural 0.129% | 0.0444 | 0.0287 | 0.0301 | 0.0163 |
| 0.258% | 0.0848 | 0.0558 | 0.0584 | 0.0320 |
| 1% | 0.2623 | 0.1868 | 0.1941 | 0.1144 |
| 1.28% | 0.3115 | 0.2269 | 0.2353 | 0.1419 |
| 10% | 0.7873 | 0.7123 | 0.7216 | 0.5864 |
| 11.3% | 0.8112 | 0.7421 | 0.7507 | 0.6224 |
| **50% (balanced)** | **0.9702** | **0.9567** | **0.9587** | **0.9272** |
| **fold change** | **21.8×** | **33.3×** | **31.9×** | **57.0×** |

Recall is bit-identical down every column (0.8997, 0.9047, 0.9121, 0.8732
respectively). Precision rises by between 21.8× and 57.0× while the classifier's
behaviour on any individual transaction is unchanged.

Unweighted arm, for completeness: 22.8× (XGBoost), 28.9× (random forest), 6.7×
(decision tree), 64.8× (logistic regression).

**This is the arm's core result.** Within the tested PaySim configuration, the
precision collapse reported in notebook 05 is fully reproducible as an artefact
of evaluated prevalence. No appeal to model quality, overfitting or feature
leakage is required to explain it.

---

## D. PR-AUC is also prevalence-dependent — the more consequential finding

PR-AUC is routinely offered as the prevalence-appropriate alternative to ROC-AUC.
E4 measures that it is not prevalence-*invariant*. Weighted arm, mean PR-AUC:

| Evaluated prevalence | XGBoost | Random forest | Decision tree | Logistic regression |
|---|---|---|---|---|
| natural 0.129% | 0.8158 | 0.7701 | 0.7360 | 0.1578 |
| 1% | 0.8737 | 0.8407 | 0.8406 | 0.5268 |
| 10% | 0.9438 | 0.9336 | 0.9302 | 0.8568 |
| 50% (balanced) | 0.9875 | 0.9855 | 0.9817 | 0.9705 |
| change | +0.1717 (+21.0%) | +0.2154 (+28.0%) | +0.2457 (+33.4%) | +0.8127 (+515.1%) |

Logistic regression moves from 0.158 to 0.971 — from unusable to apparently
excellent — with an unchanged model. **A PR-AUC figure is only interpretable
alongside the prevalence it was computed at, and PR-AUC values from evaluations
at different prevalences are not comparable.** Any cross-study comparison that
comes from a rebalanced test set is measuring the rebalancing.

---

## E. NER is the most prevalence-robust of the three

| Weighted arm | natural | 50% balanced | relative change |
|---|---|---|---|
| XGBoost | 0.1282 | 0.1003 | −21.7% |
| Random forest | 0.1364 | 0.0954 | −30.1% |
| Decision tree | 0.1275 | 0.0879 | −31.1% |
| Logistic regression | 0.1957 | 0.1269 | −35.1% |

NER moves by 22–35% where precision moves by 2,000–5,600%. It is not
prevalence-invariant either — nothing at a fixed threshold can be, since FP
shrinks with the negative count — but it degrades gracefully rather than
collapsing.

**An exact analytic anchor.** NER = (FP + R·FN)/(R·P). As negatives are removed
the FP term vanishes and NER → FN/P = 1 − recall. Measured against that floor
across all eight learner × variant combinations, agreement is within
1.2 × 10⁻⁵ to 1.0 × 10⁻⁴. At balanced prevalence NER is, to four decimal places,
simply one minus recall — which is worth stating plainly, because it means a
balanced evaluation reduces the cost-sensitive metric to a threshold-free recall
statistic and discards the false-positive term the metric exists to price.

---

## F. Balanced evaluation compresses the learner ranking

E2 established that XGB > RF > DT > LR held in every one of 112 matched cells at
natural prevalence. E4 shows that ordering is a property of the evaluated
prevalence, not of the learners alone.

**Spread in mean PR-AUC (best learner minus worst):**

| Evaluated prevalence | Weighted | Unweighted |
|---|---|---|
| natural 0.129% | 0.6580 | 0.3101 |
| 1% | 0.3469 | 0.2073 |
| 10% | 0.0870 | 0.1027 |
| 50% (balanced) | **0.0171** | **0.0633** |

**Fraction of matched cells in which the full XGB > RF > DT > LR ordering holds:**

| Evaluated prevalence | Weighted | Unweighted | XGB > RF (unweighted) |
|---|---|---|---|
| natural | 100% | 100% | 100% |
| 1% | 54.8% | 100% | 100% |
| 10% | 83.0% | 14.0% | 72.0% |
| 50% (balanced) | 96.2% | **5.8%** | **56.8%** |

In the unweighted arm at balanced prevalence the ordering survives in 5.8% of
cells and XGBoost beats random forest 56.8% of the time — indistinguishable from
a coin flip. Under balanced evaluation the four learners' PR-AUC values converge
into a band narrower than the split-to-split noise E2 measured, so the evaluation
can no longer discriminate between them.

This is a direct mechanism for a pattern the study set out to examine: balanced
test sets make weak and strong models look alike, at high headline numbers.

*Pairing note.* Cross-learner matching uses `(split_seed, model_seed, replicate)`.
Logistic regression is deterministic in the model seed and carries only
`model_seed = 42`, so the natural-prevalence row rests on 4 matched cells; the
stochastic levels have 400 each.

---

## G. Relationship to notebook 05

`results/05_precision_collapse.csv` reported a two-point contrast. E4 measures
the curve those two points sit on, but the numbers are **not directly
comparable**: notebook 05's arms came from the Lokanan replication protocol,
whereas E4 uses the frozen designated configuration and the locked `ner_optimal`
threshold.

| Learner | nb05 arm_a | nb05 arm_b | nb05 fold | E4 balanced | E4 natural | E4 fold |
|---|---|---|---|---|---|---|
| Logistic regression | 0.9663 | 0.0314 | 30.8× | 0.9272 | 0.0163 | 57.0× |
| Decision tree | 0.9915 | 0.0975 | 10.2× | 0.9587 | 0.0301 | 31.9× |
| Random forest | 0.9979 | 0.2426 | 4.1× | 0.9567 | 0.0287 | 33.3× |
| XGBoost | 0.9995 | 0.4035 | 2.5× | 0.9702 | 0.0444 | 21.8× |

E4's fold changes are larger throughout and re-order the learners, because
notebook 05's `arm_b` operated at a different point on the precision-recall
curve. E4 should be cited as the measured version; notebook 05's ordering of the
fold drops should not be carried forward.

---

## H. Interpretation

Language is associational and scoped to the tested configuration.

**Does evaluated prevalence affect precision?** Yes, decisively and by
construction — E4 measures the size of the effect rather than establishing its
existence. Within the tested grid, precision rises 6.7× to 64.8× from natural to
balanced prevalence with the model and threshold frozen.

**Does it affect PR-AUC?** Yes. PR-AUC is not prevalence-invariant; it rises by
+0.17 to +0.81 over the same range. This limits what any single PR-AUC figure can
support, including several reported in this study, all of which are quoted at
natural prevalence and should be labelled as such.

**Does it affect NER?** Yes, but far less: 22–35% relative, against 2,000%+ for
precision. Of the three, NER is the most robust to prevalence and is the metric
whose value transfers best between evaluations at different prevalences — with
the caveat that at balanced prevalence it degenerates to 1 − recall.

**Does the effect differ by learner?** Yes in magnitude, not in direction. Every
learner moves the same way; the weakest model at natural prevalence (logistic
regression, PR-AUC 0.158) gains the most (+515%). The manipulation is therefore
**associated with** compression of the learner ranking, and at balanced
prevalence in the unweighted arm the ranking is no longer recoverable.

**What E4 does not show.** It does not show that a genuinely different
population — another operator, another period, another country — would behave
this way. E4 removes negatives *uniformly at random*, which isolates the
arithmetic component of prevalence change and nothing else. A real prevalence
difference would be structured, changing which negatives are present and not only
how many, and could move the model's ranking behaviour as well as the base rate.
E4 measures the floor of the effect, not its full extent.

---

## I. Limitations still live after E4

1. **Uniform subsampling only.** As above: the arithmetic component is isolated;
   structured prevalence differences are untested.
2. **The threshold is held fixed by design.** A practitioner facing a different
   prevalence would re-select it. E4 deliberately does not, because re-selecting
   would confound the evaluation effect with a decision effect — but this means
   E4 does not report what a well-operated system would achieve at 10%
   prevalence. That is a clean follow-up and is listed below, not run.
3. **Two CSL conditions only** — the unweighted reference and weighted at the
   default R. E4 says nothing about instance weighting or RUS, and nothing about
   other `R_train` values.
4. **Single dataset and configuration.** PaySim, one frozen rung
   (`no_origin_balance`), one threshold rule (`ner_optimal`), one `R_eval`.
5. **Logistic regression contributes one model seed** (it is deterministic), so
   cross-learner pairing at the natural level rests on 4 matched cells.
6. **Positives are never resampled**, so positive-side sampling uncertainty is
   not represented; the E3 test-observation bootstrap remains the reference for
   that.

---

## J. What this changes in the write-up

1. Every PR-AUC, precision and NER figure in the study should carry the
   prevalence it was computed at. All current headline figures are at natural
   0.129% prevalence.
2. The precision-collapse claim can now be stated as measured rather than
   suggestive, with the curve in §C and the closed form in §B (check 3) as
   support.
3. Comparisons against published results obtained on rebalanced test sets should
   be described as not directly comparable, with §D as the evidence.
4. Notebook 05's fold-drop *ordering* should not be carried forward (§G).

## K. Recommended next, not run

- **E6 (threshold divergence).** Unchanged in priority; runs from cache.
- **Threshold re-selection under prevalence shift** — the natural complement to
  E4's fixed-threshold design (limitation 2). Also cache-only.

Neither has been executed.
