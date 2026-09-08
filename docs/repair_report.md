# Repair phase report

Scope: repair the measurement system identified by the audit. No new research
experiment was designed or run, no hyperparameter was tuned, no result was
removed, and the research question is unchanged.

All 147 tests pass. Superseded artifacts are retained; regenerated ones are
versioned and name what they supersede.

---

## A. Repair summary

| Issue | Status | Changed files | Rerun required | Reason |
|---|---|---|---|---|
| **C1** Temporal split realised 95.6/3.0/1.4, test prevalence 1.391% | **FIXED** | `constants.py`, `splits.py`, `tests/test_splits.py`, `tests/conftest.py`, `notebooks/09`, `scripts/repairs/repair_01*.py` | **Yes — done** | Boundaries derived from the empirical cumulative row count (322/377 → 69.68/15.30/15.02); realised proportions now asserted; guarding test runs on a front-loaded fixture *and* the real file |
| **C2** R-sweep read as a training-cost CSL sweep | **FIXED** (relabel) | `scripts/repairs/repair_04_06_relabel.py`, `docs/methodology_change_log.md` | No | NER values are unchanged; `06_decision_cost_sweep.csv` adds explicit `R_train` / `R_decision` columns. A training-R sweep has not been run |
| **C3** Rung / best-config / learner selected on test NER | **FIXED** | `experiment.py`, `notebooks/03,04,07`, `scripts/repairs/repair_02*.py` | **Yes — done** (no refit) | `headroom_by_rung` now defaults to `split="val"`; `select_best_config` selects on validation by construction |
| **M1** CI-overlap used as a significance rule | **FIXED** | `evaluate.py`, `scripts/repairs/repair_03_paired_bootstrap.py` | **Yes — done** (no refit) | Paired stratified bootstrap of ΔPR-AUC on shared resamples, `n_boot = 2000` |
| **M2** RF/XGB R-sweep "wins" inside seed noise | **FIXED** (documented) | `repair_04_06_relabel.py` (margins table) | No | `06_decision_cost_sweep_margins.csv` records the margin per cell so it can be read against `10_multiseed_ner.csv` |
| **M3** Ablation ladder asymmetric (destination side retained) | **FIXED** (diagnosed) | `features.py`, `repair_09_destination_diagnostic.py`, `docs/destination_artefact_diagnostic.md` | **Yes — done** | Diagnostic run; `no_balance_state` added as a diagnostic rung and deliberately **not** promoted |
| **M4** Config F trains on `R·sᵢ`, scored on `R` | **FIXED** (documented) | `models.py`, `experiment.py`, `docs/config_f_cost_objectives.md` | No | Three objectives stated explicitly; F reclassified as a sensitivity-analysis training variant |
| **M5** Configs B ≡ C at the default R | **FIXED** (documented) | already in `risk.py`; restated in the protocol | No | A theorem, verified: val `N_neg/N_pos` = 773.669 vs `R` = 773.701, ratio 1.00004 |
| **M6** Notebook 05 arms do not isolate prevalence | **FIXED** (documented) | `results/05_arm_interpretation.json` | No | Permitted and non-permitted claims recorded; Arm D deferred |
| **M7** Notebook 05 builder fit on the whole subset | **FIXED** | `notebooks/05`, `repair_07_notebook05_builder.py` | **No — measured inert** | Fitted state reaches only `highRiskFlag` and the severity grid, neither modelled; matrix bit-identical |
| **M8** Single seed for all CSL results | **NOT FIXED** | — | Deferred | Fixing it is a new experiment (multi-seed × multi-split grid), which this phase excludes |
| **M9** Permutation importance distorted by correlated features | **NOT FIXED** (documented) | `results/experiment_index.json` | Deferred | Grouped permutation importance is a new experiment |
| **M10** Validation reused three times in HPO | **NOT FIXED** (documented) | `results/experiment_index.json` | No | Test numbers remain clean; the val PR-AUC is flagged as optimistic |
| **m1** No runtime environment record | **FIXED** | `provenance.py`, `repair_10_provenance.py` | No | `experiment_environment.json` + `experiment_index.json` |
| **m2** `predictions/` and `tests/` gitignored | **DEFERRED** | — | No | Changing what is committed is the user's call, not a methodology repair |
| **m3** Declared ablation ≠ executed ablation | **FIXED** | `docs/methodology_change_log.md`, `results/methodology_change_log.json` | No | CHANGE-01 records both, and why |
| **m4** Chronology cannot be proven | **FIXED** (documented) | same | No | Recorded as `"chronology": "UNPROVEN"`; **not** described as pre-registered |
| **m5** Three-clause rule is in-sample | **FIXED** (documented) | `experiment_index.json` | No | Marked as a descriptive property of the generator |
| **m6** Two rung-selection code paths | **FIXED** | `experiment.py`, notebooks 03–10 | No | Notebooks call `headroom_by_rung` / `choose_rung` / `load_experimental_rung` |
| **m7** Dead assignment in `fit_one` | **NOT FIXED** | — | No | Cosmetic; left alone per the instruction not to rewrite working components |
| **m8** Failed LightGBM row in a summary table | **FIXED** | `repair_04_06_relabel.py` | No | Moved to `10_failed_runs.csv`; `10_summary_v2.csv` excludes it |
| **m9** `isFlaggedFraud` evidence is n=16 | **FIXED** (documented) | `experiment_index.json` | No | Noted; the column stays dropped on the stronger argument |
| **m10** Brier round-trips through float32 | **FIXED** (documented) | `evaluate.py` docstring | No | ~1e-11 relative; noted where the fast path is defined |

---

## B. Results affected — superseded, not deleted

Eight artifacts are superseded. All remain on disk; `results/experiment_index.json`
carries the status of every artifact in the project.

| Superseded | Replaced by | Why |
|---|---|---|
| `03_ladder_headroom.csv` | `03_ladder_headroom_v2.csv` | selected on test NER |
| `03_experimental_rung.json` | `03_experimental_rung_v2.json` | selected on test NER (**same rung**) |
| `04_h1_pr_auc_ci.csv` | `04_h1_paired_bootstrap.csv` | marginal CIs, `n_boot=100` |
| `04_h1_verdict.csv` | `04_h1_verdict_v2.csv` | CI-overlap verdicts |
| `04_best_config_per_learner.csv` | `04_best_config_per_learner_v2.csv` | test-selected; differs for 3 of 4 learners |
| `06_r_sweep.csv` | `06_decision_cost_sweep.csv` | misleading name (NER values identical) |
| `09_velocity_vs_baseline.csv` (temporal rows only) | `09_split_strategy_comparison.csv`, `09_temporal_repaired.csv` | invalid split; **stratified rows remain valid** |
| `10_summary.csv` | `10_summary_v2.csv` | presented a failed run as a comparison |

---

## C. Results still valid

Unchanged and confirmed by the audit's independent recomputation:

- **The main CSL grid** — `04_cost_sensitive.csv`. All twelve metrics reproduce
  exactly from the raw cached predictions (only Brier differs, at 1e-11 through
  float32 storage).
- **The NER framework** — `NER(flag nothing) = 1.0` exactly;
  `total_risk = FP + R·FN` to 1e-11.
- **The threshold protocol** — validation-selected, frozen, applied once to test,
  at every call site in every notebook.
- **All PaySim artefact findings** — three-clause rule P 0.99988 / R 0.97699;
  `errorBalanceOrig` stump AUC 0.9020; drain signature 97.82% vs 0.0013%;
  `isFlaggedFraud` 16/16.
- **The experimental rung** — `no_origin_balance`, and now on validation.
- **The interpretability learner** — `xgboost`, and now on validation.
- **Velocity feature causality** — strictly `step − 1` lookback, no fitted state.
- **Notebook 05 Arms A, B and C** — all three numerically unchanged; the builder
  leak was measured inert.
- **The stratified velocity arm** of notebook 09.
- **H1's direction** — and now *more* strongly than before (see below).

### What the repairs changed substantively

**1. The temporal robustness check reversed.** At `R = 773.70`, config C, no velocity:

| Learner | Stratified | Temporal (superseded) | Temporal (repaired) |
|---|---|---|---|
| Decision tree | 0.1527 | 0.2089 | **0.1851** |
| Logistic regression | 0.2499 | 0.0482 | **0.2089** |
| Random forest | 0.1324 | 0.0403 | **0.1763** |
| XGBoost | 0.1314 | 0.0398 | **0.1495** |

Temporal is now harder for **3 of 4** learners. The previous result was a base-rate
artefact. A new finding fell out: PaySim's fraud prevalence is **strongly
non-stationary** (train 0.082%, val 0.059%, test 0.420%), so the repaired run
reports three cost ratios — `R_default` 773.70 (comparable with the stratified
arm), `R_train_ratio` 1219.40 (split-specific, no test information), and
`R_test_ratio` 237.34 (restores NER's break-even anchor, at the cost of reading a
marginal count off the evaluation partition).

**2. H1 strengthened.** Under the paired bootstrap, **all 12** CSL comparisons show
lower PR-AUC than unweighted, every CI excluding zero (max proportion of
replicates crossing zero: 0.0015). The four verdicts the old rule called
"indistinguishable" — decision tree and XGBoost under weighting and instance
weighting — resolve as detectable, small, negative differences (−0.016 to −0.020).

| Learner | weighted | instance_weighted | rus |
|---|---|---|---|
| Logistic regression | −0.3518 [−0.3732, −0.3293] | −0.2650 [−0.2855, −0.2436] | −0.3848 [−0.4069, −0.3620] |
| Decision tree | −0.0196 [−0.0301, −0.0088] | −0.0159 [−0.0258, −0.0057] | −0.5693 [−0.5861, −0.5504] |
| Random forest | −0.0500 [−0.0600, −0.0402] | −0.0420 [−0.0514, −0.0333] | −0.0677 [−0.0794, −0.0568] |
| XGBoost | −0.0195 [−0.0259, −0.0136] | −0.0191 [−0.0252, −0.0131] | −0.0756 [−0.0871, −0.0637] |

The correct phrasing is that the paired interval excludes zero — not "statistically
significant". This is one test set and one training seed; it speaks to
observation-level sampling variation, not to variation across replications of the
experiment. **That is exactly what M8 leaves open.**

**3. Best configuration per learner changed for 3 of 4 learners**, and the
validation-selected configurations report *higher* test NER (e.g. XGBoost G at
0.1355 rather than B at 0.1314). That is the optimism the repair removes.

**4. The destination-side concern did not survive measurement.** My own audit
finding M3 asserted the retained destination features were the same kind of
artefact as the origin side. They are not:

| | Origin | Destination |
|---|---|---|
| Strongest solo AUC | `errorBalanceOrig` **0.9020** | `oldbalanceDest` **0.6237** |
| Signature | `amount == oldbalanceOrg`, 97.82% of fraud vs 0.0013% of legit, median gap exactly 0.0 | `dest_zero_before`, 65% vs 42% — lift 1.54× |
| Deterministic | yes | no |

The specific mechanism the concern named — "the simulator never credits the
receiving account" — measures at **0.60 for legitimate and 0.52 for fraud, a lift
of 0.86, pointing the wrong way**. PaySim's destination bookkeeping fails to
reconcile for most legitimate transactions too. Removing that block costs
0.25–0.43 PR-AUC (logistic regression falls to 0.078). The reasoning asymmetry
was real and is now documented; the case for acting on it is not. `no_balance_state`
stays a diagnostic, and the pre-declared criterion selects `no_origin_balance`
whether or not that rung is in the ladder.

---

## D. Protocol after repair

```
DATA        canonical ealaxi/paysim1, sha256 16910f90…, 6,362,620 rows,
            8,213 fraud (0.129%), asserted on every load by data.validate

SPLIT       stratified 70/15/15, seed 42, stratified on isFraud
            temporal   boundaries derived from the empirical cumulative row
                       count (322 / 377 → 69.68/15.30/15.02); realised
                       proportions asserted against a 3-point tolerance;
                       per-partition prevalence and class ratio recorded

FEATURES    FeatureBuilder fitted on TRAINING ROWS ONLY (every notebook,
            including 05 Arm C). Only fitted state: the 99th-percentile amount
            cutoff and the severity quantile grid.
            Velocity features computed on the full ordered table before any
            split — no fitted state, strictly step−1 lookback.

RUNG        selected on VALIDATION NER, criterion unchanged: best NER ≥ 0.05,
            least-ablated viable rung → no_origin_balance (val NER 0.1147).
            Resolved at runtime by experiment.load_experimental_rung().
            no_balance_state exists as a DIAGNOSTIC rung and is excluded from
            the selection ladder.

COST        R = N_legit / N_fraud = 773.70 (population, declared once).
            Temporal arm additionally reports R_train_ratio and R_test_ratio,
            because NER's break-even anchor and the Youden ≡ NER-optimal
            equivalence hold only at the evaluated partition's class ratio.

CSL         class weighting     w⁺ = R,        w⁻ = 1
            severity weighting  w⁺ᵢ = R·sᵢ,    w⁻ = 1   (trained on R·sᵢ,
                                thresholded and scored on R — see
                                docs/config_f_cost_objectives.md)
            RUS                 N′_neg = N_neg / R
            thresholding        λ = argmin NER

THRESHOLD   TRAIN → FIT → VALIDATION → SELECT → LOCK → TEST → REPORT.
            Never selected on test, at any call site.
            At R = N_neg/N_pos the Youden and NER-optimal thresholds are
            provably identical, so configs B and C are ONE condition at the
            default R — six distinct conditions, not seven.

SELECTION   every model/config/feature-set choice on VALIDATION;
            test read once, after the decision is locked.

STATS       paired stratified bootstrap of the delta on shared resamples,
            n_boot = 2000. CI-overlap is not used as a decision rule.
            "the paired interval excludes zero", never "statistically
            significant".

METRICS     PR-AUC primary for ranking; NER primary for decision risk;
            precision/recall/F1/MCC/Brier/FP/FN/alert rate at the locked
            threshold. Positive class = fraud throughout.
            Accuracy is never computed.

PROVENANCE  results/experiment_environment.json  (python, packages, git SHA,
                                                  dataset, seeds, configs)
            results/experiment_index.json        (per-artifact status)
            results/methodology_change_log.json  (every protocol change)
```

---

## E. Outstanding methodological decisions

These need a deliberate research decision before new experiments begin. None is
a bug; each is a fork the repair phase deliberately did not take.

1. **Is configuration F meant to be an example-dependent experiment?** As built
   it trains on `C_FN = R·sᵢ` and is scored on `C_FN = R`, so it is a
   sensitivity-analysis training variant, and its negative result cannot be read
   as evidence about severity weighting on its own terms. Making it a true
   example-dependent arm means a single objective at all three stages *and*
   re-expressing D at matched effective weight (since `E[sᵢ] ≈ 0.5` means F
   currently carries about half D's weight). **Decision needed:** keep F as a
   sensitivity variant, or commit to the dedicated arm.

2. **Which cost ratio governs the temporal arm?** Prevalence is non-stationary,
   so no single R serves both comparability and NER's anchor. All three are
   currently reported. **Decision needed:** which is primary in the write-up.

3. **Does the study want a training-R sweep at all?** The existing sweep is a
   decision-cost sweep. The proposal's §3.3 taxonomy implies a training-cost one.
   It is affordable but it is a new experiment.

4. **Is one seed and one split acceptable for the headline CSL claim?** The
   paired bootstrap covers observation-level variation only. `10_multiseed_ner.csv`
   covers 4 seeds for the *unweighted* arm alone, and logistic regression and
   decision tree are deterministic, so effective seed coverage is two learners.
   Split variability is entirely unmeasured. Every CSL delta still rests on one
   seed and one split.

5. **How is the authentication research question resolved?** PaySim has no
   session, device, SIM, PIN, channel or location field. No experiment on this
   dataset can answer it. **Decision needed:** reframe as a design contribution
   grounded in the `band_cutoffs` → `BAND_ACTIONS` step-up-auth mapping, or state
   the limitation and drop the empirical claim.

6. **Should `predictions/` and `tests/` be committed?** Currently gitignored, so
   the results are not independently regenerable from the repository, and the
   invariant suite is not part of the record. This is a repository-policy call.

---

## F. Recommended next phase — ranked, not run

### Required validation experiments

These close a methodological gap in a claim the study already makes.

1. **Multi-seed × multi-split CSL comparison** (audit E3, decision 4 above).
   5 splits × 5 seeds, paired bootstrap plus Wilcoxon across the 25 paired cells.
   `evaluate.paired_test` already implements the second and is currently unused.
   Without this, the headline CSL claim rests on one seed and one split.

2. **Training-cost-ratio sweep** (audit E2, decision 3).
   `R_train ∈ {10, 50, 100, 773.7, 2000, 5000}` entering `scale_pos_weight`, the
   RUS target and the severity weights, with the full `R_train × R_eval` grid
   reported. This is the experiment the proposal's taxonomy calls for, and the
   only way to state anything about CSL "at cost ratio R".

3. **Arm D — prevalence-only manipulation** (audit E4, M6).
   One model, one threshold, the same real positives, real negatives subsampled
   to {50%, 10%, 1%, 0.129%}. No synthetic points, no ENN, no retraining.
   Converts the precision-collapse finding from suggestive to measured. Cheap.

### High-value research experiments

4. **Threshold-rule comparison away from the class ratio** (audit E6).
   B ≡ C is degenerate at `R = 773.70`. Sweeping R across the class ratio makes
   the comparison informative and demonstrates the theorem empirically — it
   dissolves a published disagreement rather than citing it. Runs entirely from
   cache.

5. **Dedicated example-dependent cost-sensitive arm** (decision 1, M4).
   Only if decision 1 goes that way.

### Optional

6. **Grouped permutation importance** (M9) — permuting `amount`/`log_amount` and
   the `type_*` one-hots as families, on validation rather than test. Would settle
   what the destination block contributes as a block.

7. **Diagnose or drop the LightGBM arm** (m8) — PR-AUC 0.043 is a failed fit. No
   research question currently depends on it.

**Not recommended:** promoting `no_balance_state` to the main rung. The diagnostic
does not support it, and the pre-declared criterion does not select it.
