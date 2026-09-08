# Methodology change log

A record of changes to the experimental protocol after it was first written
down, kept so the study's chronology can be audited rather than asserted.

**Principle: nothing here is rewritten.** Superseded artifacts stay in
`results/` exactly as produced. Regenerated ones are versioned (`_v2`, or given
a new name) and name what they supersede.

---

## CHANGE-01 - The ablation was deepened between notebooks 02 and 03

### What was originally declared

`results/02_feature_decision.json`, written by `notebooks/02_leakage_audit.ipynb`,
declares `ablation_removes: [errorBalanceOrig, errorBalanceDest]` and
`n_features_ablated: 18`.

That is a **two-feature** ablation, corresponding to the `no_error_features`
rung. It followed a stated screening rule: single-feature cross-validated stump
AUC >= 0.90, which only `errorBalanceOrig` met (0.9020; next highest
`oldbalanceOrg` at 0.7635).

### What was subsequently used

`results/03_experimental_rung.json` selects `no_origin_balance`, a **six-feature**
ablation that additionally removes `oldbalanceOrg`, `newbalanceOrig`,
`origZeroBefore` and `origZeroAfter` (20 -> 14 features). Every cost-sensitive
result in notebooks 04, 06, 07, 09 and 10 was produced at that rung.

### Why the change was made

The solo-AUC screen tests features one at a time and therefore cannot see a rule
that is reconstructible from a *combination*. Dropping `errorBalanceOrig` does
not remove the drain signature, because a tree can rebuild
`amount == oldbalanceOrg` from the raw balance columns that remain. The evidence
is in `results/03_ladder_headroom.csv`: at `no_error_features` the best
achievable NER is **0.0094**, meaning ~99% of the do-nothing risk is already
removable and every cost-sensitive configuration lands in the same sliver. A
comparison run there measures nothing.

A headroom floor of `best NER >= 0.05` was declared in
`notebooks/03_baselines.ipynb` and the least-ablated rung clearing it was taken.
That reasoning is sound and is retained unchanged.

### Was the change made before or after downstream results were seen?

**The repository cannot prove either way, and this document does not claim it.**

- Notebooks 03, 04 and 05 all entered version control in a single commit
  (`ab1358c`, 2026-08-13T17:30:41+02:00). Notebook 02 arrived 19 seconds earlier
  in `c2ee031`.
- There is no commit in which `03_experimental_rung.json` exists and
  `04_cost_sensitive.csv` does not.
- The *mechanism* is auditable and correctly ordered in code: notebook 03 writes
  the decision file, and notebooks 04, 06, 07, 09 and 10 each read the rung from
  it rather than hard-coding one. The selection criterion is expressed in code,
  not applied by hand.
- What cannot be shown is the *chronology* - that the criterion was fixed before
  the CSL deltas were observed.

Accordingly, the rung selection is described as **criterion-driven and
mechanically enforced**, and is **not** described as pre-registered.

### Experiments affected

Every result at the `no_origin_balance` rung: `04_cost_sensitive.csv`,
`04_h1_*`, `06_*`, `07_*`, `09_*`, `10_*`.

Full-feature and intermediate-rung results remain available in
`results/03_baselines_all_rungs.csv` as the diagnostic evidence of the artefact
they measure.

### Status after repair

The rung decision was recomputed on **validation** NER
(`results/03_experimental_rung_v2.json`, audit finding C3). The criterion was
not altered. The outcome is **unchanged**: `no_origin_balance`, validation NER
0.1147, the least-ablated rung clearing the 0.05 floor. The original
test-selected decision file is retained.

---

## CHANGE-02 - Temporal split boundaries were derived rather than assumed

### What changed

`TEMPORAL_TRAIN_END` / `TEMPORAL_VAL_END` moved from `520 / 632` to `322 / 377`.

### Why

The original values were chosen as if PaySim's hourly volume were uniform. It is
heavily front-loaded, so they realised **95.6 / 3.0 / 1.4** rather than the
70/15/15 the docstring claimed, with a test partition of 89,458 rows at 1.391%
prevalence. The new values are computed from the file's empirical cumulative row
count by `splits.derive_temporal_boundaries` and realise **69.68 / 15.30 / 15.02**.

The claim in the previous `constants.py` that the boundaries were "verified
against the real distribution" was false; no such verification existed. The test
that appeared to guard them ran on a synthetic frame with uniformly drawn steps.

### Experiments affected

The temporal rows of `results/09_velocity_vs_baseline.csv` (8 of 16). Superseded
by `results/09_temporal_repaired.csv`. The stratified rows are unaffected.

### Status

Fixed. `splits.temporal_split` now asserts realised proportions against
`TEMPORAL_PROPORTION_TOLERANCE`, and the guarding test runs against both a
front-loaded synthetic distribution and the real file.

---

## CHANGE-03 - Selection moved from test to validation

### What changed

Three decisions were taken on test NER and are now taken on validation NER:
the experimental rung, the best configuration per learner, and the
interpretability learner.

### Why

The protocol the study declares is TRAIN -> FIT -> VALIDATION -> SELECT -> LOCK
-> TEST -> REPORT. These three read the test partition before the decision was
locked, which is the defect the study criticises in the prior work.

### Outcome

| Decision | On test (superseded) | On validation (current) | Changed |
|---|---|---|---|
| Experimental rung | `no_origin_balance` | `no_origin_balance` | no |
| Interpretability learner | `xgboost` | `xgboost` | no |
| Best config - decision tree | E | F | **yes** |
| Best config - logistic regression | G | E | **yes** |
| Best config - random forest | B | B | no |
| Best config - XGBoost | B | G | **yes** |

The rung is unchanged, so the feature basis of every existing CSL result stands.
The per-learner "best configuration" table changes for three of four learners,
and the validation-selected configurations report *higher* test NER than the
test-selected ones did - which is the optimism the repair exists to remove.

### Status

Fixed. `experiment.headroom_by_rung` defaults to `split="val"`;
`experiment.select_best_config` selects on validation by construction.
Superseded files retained.

---

## CHANGE-04 - Statistical procedure for the H1 deltas

### What changed

The H1 verdict no longer decides differences by whether two marginal bootstrap
confidence intervals overlap. It uses a paired bootstrap of
`delta PR-AUC = PR-AUC_CSL - PR-AUC_unweighted` on shared stratified resamples,
`n_boot = 2000`.

### Why

Non-overlap of marginal intervals implies a difference; overlap does **not**
imply its absence. The old rule could not support the four "indistinguishable"
verdicts it issued. Both models were scored on the identical test set, so the
paired procedure was available all along and is strictly more powerful.

### Status

Fixed. `results/04_h1_paired_bootstrap.csv` and `04_h1_verdict_v2.csv`
supersede `04_h1_pr_auc_ci.csv` and `04_h1_verdict.csv`, which are retained.

---

## CHANGE-05 - The R-sweep was relabelled, not re-run

### What changed

`results/06_r_sweep.csv` is superseded by `results/06_decision_cost_sweep.csv`,
with explicit `R_train` and `R_decision` columns. The NER values are unchanged.

### Why

Every weighted / instance-weighted / RUS model in all 36 cells was trained once
at `R = 773.70`; only the decision threshold varies with R. The file name and
column heading implied a training-cost sweep. A genuine training-R sweep is an
open experiment and has not been run.

### Status

Documented. Original retained.

---

## CHANGE-06 - RQ 2.1.1 was resolved as a design contribution

### What changed

Research question 2.1.1 - "key authentication vulnerabilities in telco mobile
money" - is answered as a **design contribution** rather than an empirical
result: a mapping from transaction-risk bands to step-up authentication
requirements, explicitly distinguished from a security evaluation.

`docs/authentication_design_contribution.md` records the argument.
`baseline.AUTH_DESIGN_FORBIDDEN_CLAIM` records what may not be said about it,
and `tests/test_baseline.py` asserts that guard is present.

### Why

PaySim has no authentication, session, device, SIM, PIN, channel or location
field, so no experiment on this dataset can answer the question. This was
already a supported conclusion in the frozen baseline. The fork left open in
`docs/repair_report.md` section E, decision 5, was between reframing as a design
contribution and dropping the empirical claim outright.

The reframing was chosen because the material for it already exists and is
derived rather than invented: `risk.band_cutoffs` anchors every band boundary on
the Bayes threshold `lambda = 1/(1+R)`, and `constants.BAND_ACTIONS` already
names the `High` band `step_up_auth`. The contribution is that **one declared
parameter R sets both the alerting threshold and the authentication escalation
ladder**, so an institution states its risk appetite once instead of tuning a
detection threshold and a friction policy against each other.

Manufacturing an experiment around variables the dataset does not contain was
rejected as stretching PaySim past what it holds.

### What was NOT done

No experiment was run, no model fitted, no threshold reselected, and no result
changed. The band populations quoted are those `notebooks/06_risk_analysis.ipynb`
already produced. Nothing in the frozen baseline was unfrozen: the change adds a
forbidden claim, which tightens the guard rather than loosening it.

### Experiments affected

None. The change is to how an existing artifact is framed and reported.

### Status

Decided and documented. `docs/repair_report.md` section E decision 5 is closed;
the entry there is left as written, since this log is the record of what was
decided afterwards.
