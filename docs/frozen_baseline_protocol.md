# Frozen experimental baseline

**Status: FROZEN.** Machine-readable equivalent:
`results/frozen_experimental_baseline.json`. Verification:
`results/frozen_baseline_checks.json` (9/9 PASS).

This is the fixed reference point for every experiment from here on. It records
decisions already taken and evidenced in `docs/repair_report.md`; nothing in it
was computed by the freeze phase, and no model was fitted to produce it.

**Unfreezing.** Changing anything here requires a new entry in
`docs/methodology_change_log.md` stating what changed, why, whether it was
decided before or after seeing downstream results, and which experiments are
affected. *A better score is not a reason.*

---

## 1. Dataset

| | |
|---|---|
| Identifier | `ealaxi/paysim1` |
| SHA-256 | `16910f90577b0d981bf8ff289714510bb89bc71bff7d3f220f024e287e4eea6b` |
| Rows | 6,362,620 |
| Fraud | 8,213 (0.129%) |
| Validated | on every load, by `data.validate` — shape, columns and fraud count |

## 2. Feature set — FROZEN

**`no_origin_balance`** (14 features). Drops `errorBalanceOrig`,
`errorBalanceDest`, `oldbalanceOrg`, `newbalanceOrig`, `origZeroBefore`,
`origZeroAfter`.

Selected on **validation** NER (0.1147) under the unchanged pre-declared
criterion: *best NER ≥ 0.05, take the least-ablated viable rung.*

### Why this rung, on the evidence

1. **The origin-side evidence justified removal.** A three-clause boolean
   expression over those columns recovers the label at precision 0.99988 /
   recall 0.97699; `errorBalanceOrig` reaches AUC 0.9020 alone; the drain
   signature holds for 97.82% of fraud against 0.0013% of legitimate
   transactions, with a median gap of **exactly 0.0**. That is a deterministic
   simulator rule, not transaction behaviour.
2. **The destination-side diagnostic did not establish an equivalent artefact.**
   Strongest destination feature: 0.6237 alone. Lifts of 1.3–1.5×, not
   near-deterministic. The specific mechanism the concern named — the simulator
   not crediting the receiving account — measures **0.60 for legitimate against
   0.52 for fraud, a lift of 0.86, pointing the wrong way**. See
   `docs/destination_artefact_diagnostic.md`.
3. **Validation-based selection still returns this rung**, so the repair to the
   selection split did not disturb it.
4. **Therefore there is no evidence-based reason to change the main rung.**

### What may be claimed about it

> **reduced simulator artefact exposure**

Not *"simulator-independent"* and not *"artefact-free"*. PaySim is a simulator
throughout; fraud occurs only in TRANSFER and CASH_OUT by construction; and the
retained destination columns carry weak but real generator-specific association.
`baseline.FROZEN_RUNG_CLAIM` holds the permitted wording, and a test asserts it
contains no overstatement.

### Diagnostic rung

`no_balance_state` is retained **for diagnosis only**. It is excluded from the
selection ladder by default in `experiment.choose_rung`, and
`baseline.assert_frozen_rung` raises if it is used as a main feature set.

## 3. Split protocol — FROZEN

Stratified **70 / 15 / 15**, split seed **42**, stratified on `isFraud`,
via `splits.stratified_split`.

Invariants, each verified by the consistency checks:

- the same train/validation/test partitions across directly compared configurations
- `FeatureBuilder` fitted on **training rows only**
- resampling (RUS, SMOTE) applied to **training rows only**
- threshold selected on **validation**
- threshold **locked** before any test evaluation
- test set read **once**, for final reporting

### Two seeds, never conflated

| Seed | What it moves | Currently |
|---|---|---|
| **model seed** | estimator init, bagging, column subsampling, the RUS draw | varied across {42, 7, 13, 99} for the *unweighted* arm only |
| **split seed** | which rows land in which partition | **fixed at 42 for every result produced so far** |

Logistic regression and decision tree are deterministic here, so the model seed
moves only random forest and XGBoost. Split variability is **unmeasured**. Every
CSL delta rests on one model seed and one split seed.

The temporal split (boundaries 322/377, realising 69.68/15.30/15.02) is a
separate declared arm, not part of the main protocol.

## 4. Cost framework — FROZEN

```
R          = "missing one fraud is as bad as R false alarms"
C_FP       = 1
C_FN       = R
R_default  = N_legit / N_fraud = 773.7011
lambda     = 1 / (1 + R)
NER        = TotalCost / (N_fraud * R)
```

**`R_train` and `R_eval` are recorded separately, always.** `R_train` enters
`scale_pos_weight`, the RUS target and the severity weights; `R_eval` enters
threshold selection and the NER computation. They are equal at the default R in
every experiment run so far — and the existing decision-cost sweep varies
`R_eval` alone at a fixed `R_train = 773.70`, which is precisely why conflating
them is not allowed.

## 5. Threshold protocol — FROZEN

```
TRAIN -> FIT -> VALIDATION -> SELECT -> LOCK -> TEST -> REPORT
```

Rules: `fixed` (0.5), `youden`, `ner_optimal`. Selected on validation via
`evaluate.select_threshold`, applied unchanged to test.

At `R = N_neg/N_pos` the Youden and NER-optimal thresholds are **provably
identical** for every model, so configs B and C are one condition at the default
R — **six** distinct conditions, not seven.

## 6. Models — FROZEN

Logistic regression · decision tree · random forest · XGBoost.
Hyperparameters are `models.make_learner` defaults: **fixed, not tuned**.

Additional learners need separate justification; they are not part of the
research question as posed. **LightGBM remains excluded** from the valid model
comparison — PR-AUC 0.043 at 0.129% prevalence is a failed fit, recorded in
`results/10_failed_runs.csv`.

## 7. CSL taxonomy — the four mechanisms kept separate

| Group | Variants | Configs | Objective |
|---|---|---|---|
| **Baseline** | `unweighted` | A | unweighted loss |
| **Algorithm / data level** | `weighted`, `rus` | D, E, G | class-constant `C_FN = R` |
| **Decision level** | `unweighted` + threshold | B, C | class-constant `C_FN = R` |
| **Sensitivity analysis** | `instance_weighted` | F | **trains** on `C_FN = R·sᵢ`; **thresholded and scored** on `C_FN = R` |

### Configuration F is not a like-for-like implementation of D/G

F trains under an example-dependent weighting scheme and is evaluated under the
class-constant risk objective. Since `sᵢ` is a percentile rank with `E[sᵢ] ≈ 0.5`,
it also carries roughly **half** D's effective positive weight. It must not be
described as a directly equivalent implementation of the same class-constant
objective.

> **The existing F result does not support the claim that severity weighting
> does not work.** Training and evaluation use different cost objectives, so the
> comparison cannot settle that question in either direction.

Full statement: `docs/config_f_cost_objectives.md`.

## 8. Evaluation protocol — FROZEN

**Primary ranking metric: PR-AUC. Primary decision-risk metric: NER.**

Also reported: precision, recall, F1, MCC, FP, FN, number flagged, alert rate,
Brier score, ROC-AUC.

**Accuracy is excluded** and never computed — at 0.129% prevalence a
constant-negative predictor scores 0.9987. ROC-AUC is reported for comparability
with the prior work only and is not used to rank models.

### Uncertainty

Paired stratified bootstrap of the delta on shared resamples, `n_boot = 2000`
(`evaluate.paired_bootstrap_pr_auc`). **CI overlap must not be used as a
significance decision** — non-overlap implies a difference, overlap does not
imply its absence. Report *"the paired interval excludes zero"*, not
*"statistically significant"*.

### Required fields on every experiment record

`learner`, `feature_set`, `csl_variant`, `R_train`, `R_eval`, `threshold_rule`,
`threshold`, `model_seed`, `split_seed`, `split_strategy`, `split`,
`val_prevalence`, `test_prevalence` — plus `pr_auc`, `ner`, `precision`,
`recall`, `f1`, `mcc`, `fp`, `fn`, `n_flagged`, `alert_rate`.

Enforced by `baseline.validate_experiment_frame`.

## 9. What the evidence currently supports

Stated at the strength the evidence carries, no further.

1. PaySim contains highly recoverable simulator-specific fraud signatures.
2. `errorBalanceOrig` is unsuitable as an unrestricted feature; its predictive
   strength is tied to simulator bookkeeping.
3. The main experiments therefore use `no_origin_balance`.
4. CSL methods show consistently negative PR-AUC deltas against the unweighted
   baseline — 12 of 12, every paired CI excluding zero.
5. The effect of CSL on NER is learner- and cost-dependent.
6. Threshold optimisation produces substantial risk reductions relative to a
   fixed 0.5 threshold.
7. Evaluation prevalence has a major effect on precision.
8. Temporal generalisation is harder for 3 of 4 learners.
9. PaySim cannot empirically answer the authentication-vulnerability research
   question, because those variables do not exist in the dataset.

Each is stored with its supporting artifact in
`baseline.SUPPORTED_CONCLUSIONS`.

## 10. Claims the evidence does not reach

- The feature set is simulator-independent or artefact-free.
- Severity weighting does not work (from the F result).
- CSL never helps — only the *ranking-quality* result is uniformly negative.
- The decision-cost sweep shows CSL winning at a given **training** cost ratio.
- Evaluation prevalence alone accounts for the precision collapse.
- Any CSL delta is stable across seeds or splits.

## 11. Guards in force

| Guard | Mechanism |
|---|---|
| Superseded results can't be loaded as current | `baseline.load_results` raises `SupersededArtifactError` |
| Diagnostic rung can't be selected | `experiment.choose_rung` excludes it by default |
| Main experiments can't run off-rung | `baseline.assert_frozen_rung` |
| Records can't omit R_train/R_eval/seeds | `baseline.validate_experiment_frame` |
| Rung resolves to the validation-selected decision | `experiment.load_experimental_rung` |

Re-verify at any time:

```bash
python scripts/freeze/consistency_checks.py --json
python -m pytest tests/test_baseline.py -q
```
