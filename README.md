# momo-fraud-supervised-ml

Risk-based cost-sensitive fraud detection on mobile money transactions, supporting _Financial Fraud Detection in Mobile Money Networks Using Supervised Machine Learning_ (COS700).

Evaluation is **unitless throughout** — a severity ratio and a 0–100 risk score, never a currency.

## The question

Does cost-sensitive learning actually improve fraud detection on severely imbalanced mobile money data, and _which level_ of it does the work — data, algorithm, or decision?

Johnson & Khoshgoftaar (2022) showed on Medicare Part B claims (8,669,497 rows, 0.0456% positive) that class weighting and undersampling **reduce** a learner's discriminative power, and that thresholding alone outperforms both. They asked for replication on other severely imbalanced datasets. PaySim, at 0.129% prevalence, is that dataset — and no prior mobile money study reports any risk analysis at all.

## Findings

**PaySim's fraud is recoverable by three boolean clauses.**

```
type ∈ {TRANSFER, CASH_OUT}  AND  amount == oldbalanceOrg  AND  newbalanceOrig == 0
```

Precision **0.9999**, recall **0.9770**, on all 6,362,620 transactions. Every published machine-learning result we found on this dataset scores _below_ a rule with no parameters and no training. Results on the full feature set therefore measure the simulator, not fraud detection — which is why the experiment runs on an ablated feature set (`notebooks/02`, `notebooks/03`).

This is an in-sample descriptive property of the released dataset, not a held-out detector result. The PaySim paper places fraud generation explicitly outside its scope, so there is no documented generative rule to appeal to.

**Cost-sensitive _training_ does not improve ranking quality; cost-sensitive _thresholding_ reduces risk.**

Across four learners and twelve paired comparisons — class weighting, per-instance severity weighting, and undersampling — **none improved PR-AUC over the unweighted baseline**. All twelve deltas were negative (−0.0159 to −0.5693), every paired bootstrap CI excluding zero at `n_boot = 2000`. Varying the split and the training seed does not change this: **116 of 116** paired comparisons are negative across 4 split seeds × 4 model seeds.

Moving the threshold instead removes **49–62%** of the risk (NER 0.6591 → 0.2499 for logistic regression, 0.3206 → 0.1324 for random forest).

**On end-to-end risk the picture is learner-dependent, and that dependence is itself stable.** Logistic regression and the decision tree benefit from cost-sensitive training in 64 of 64 paired cells; random forest and XGBoost are harmed in 61 of 64. Only the ranking-quality result is uniform. The blunt claim "cost-sensitive learning never helps" is not supported and is explicitly forbidden by the frozen baseline.

**At `R = N_neg/N_pos`, risk-optimal and ROC-optimal thresholds are the same threshold.**

Minimising `FP + R·FN` reduces to maximising `TPR − FPR` at that ratio — provably, for every model. Measured across 104 frozen model cells, the two rules select a **bit-identical** threshold in every one, maximum gap `0.000e+00`. Away from the ratio they diverge, and below it the ROC-optimal threshold can be worse than flagging nothing. This reframes the apparent "statistical versus business threshold" disagreement in the literature as a disagreement between two _cost ratios_.

**`amount` is not the most important feature.**

Once the simulator's origin-balance bookkeeping is removed, transaction **type** dominates (permutation importance 0.701 versus 0.447 for amount). Prior work reporting amount as the top predictor did so with those balance features present. This result is **provisional**: permutation importance is distorted by correlated features and grouped permutation importance has not been run.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pip install -e .          # so `import momo_fraud` works outside notebooks/
.venv/Scripts/python -m pytest -q                 # 239 tests, library + prototype
```

The editable install is what lets a prototype elsewhere on the machine load the model. Without it only the notebooks work, because they patch `sys.path`.

Then run `notebooks/00_data_acquisition.ipynb`, which downloads `ealaxi/paysim1` (needs `~/.kaggle/kaggle.json`, or download it manually — the notebook prints instructions either way).

## Notebooks

|                                           |                                                                                   |
| ----------------------------------------- | --------------------------------------------------------------------------------- |
| `00_data_acquisition`                     | Download and validate. Refuses to proceed on anything but the canonical file.     |
| `01_eda`                                  | Class balance, where fraud lives, time-of-day signal, the fraud mechanism.        |
| `02_leakage_audit`                        | Decides what is admissible as a feature. **Run this before trusting any result.** |
| `03_baselines` – `05_lokanan_replication` | The experiment grid and the three-arm replication.                                |
| `06_risk_analysis`                        | Thresholds, risk sweeps, analyst review budgets.                                  |
| `07_interpretability`                     | SHAP and feature importance.                                                      |
| `08_model_export`                         | Freezes the artifact the prototype loads.                                         |
| `09_velocity_features`                    | Account-level velocity features and the repaired temporal split.                  |
| `10_advanced_optimization`                | Learner comparison, hyperparameter search, stacking.                              |

Notebooks orchestrate; the analysis primitives live in `src/momo_fraud/` so metric definitions cannot drift between them, and so they can be tested.

## The risk framework

One interpretable knob replaces every currency figure:

> **R** — "missing one fraud is as bad as R false alarms."

All three cost-sensitive levels in the proposal's section 3.3 taxonomy derive from it:

| Level     | Technique       | Derived from R         |
| --------- | --------------- | ---------------------- |
| Data      | undersampling   | `N'_neg = N_neg / R`   |
| Algorithm | class weighting | `scale_pos_weight = R` |
| Decision  | thresholding    | `λ = 1 / (1 + R)`      |

For PaySim `R = N_neg/N_pos = 773.70`, which puts `λ` exactly at the class prior (0.00129) — as the theory predicts. Johnson & Khoshgoftaar use the same parameterisation, so the replication is like-for-like at the level of the cost model and not only at the level of the headline claim.

Results are reported as **PR-AUC** (model quality), **Recall@budget** ("if analysts review _k_ transactions a day, what share of fraud is caught?"), and **Normalized Expected Risk** — the proposal's `TotalCost` divided by the do-nothing baseline, so it lands in [0,1] with 1.0 meaning "no better than ignoring the problem". Accuracy is excluded, and the exclusion is enforced in code.

## Audit, repair and freeze

The study audited its own measurement system before drawing conclusions. Twenty findings; sixteen repaired, four deferred with stated reasons. The three that changed conclusions:

- a temporal split that realised **95.6/3.0/1.4** instead of 70/15/15, with test prevalence 1.391% against a true 0.129%;
- rung, best-configuration and learner selection performed on **test** NER;
- confidence-interval overlap used as a significance rule, replaced by a paired stratified bootstrap.

Superseded artifacts are retained, versioned, and indexed by status; nothing was deleted. The configuration was then **frozen** — dataset checksum, feature set, splits, cost ratio, threshold protocol, metrics — and verified by nine consistency checks that read the notebooks and library rather than trusting a declaration. Unfreezing requires a change-log entry; a better score is not a reason.

`docs/repair_report.md`, `docs/frozen_baseline_protocol.md`, `docs/methodology_change_log.md`.

## Experiments

Four experiments ran against the frozen baseline — 480 distinct model fits, reported over 792 evaluation cells once cached fits are reused. Each carries a quality-control report and an independent smoke test, and each states what it does not establish.

| | Question | Scale | Outcome |
| --- | --- | --- | --- |
| **E2** | Does the result hold at other _training_ cost ratios? | 364 cells | Learner ranking XGB > RF > DT > LR, 0 violations in 112 |
| **E3** | Does it survive seeds and splits? | 220 fits | 116/116 negative; NER effect learner-dependent and stably so |
| **E4** | How much of the precision collapse is prevalence alone? | 104 cells | Model and threshold frozen; measured, not suggestive |
| **E6** | Where is the threshold comparison informative at all? | 104 cells | Equivalence exact in 104/104; divergence asymmetric |

Reports in `docs/E2_report.md`, `docs/E3_report.md`, `docs/E4_report.md` and `docs/E6_report.md`. Scripts in `scripts/experiments/`.

## Risk bands and authentication

The dataset carries no authentication, session, device, SIM, PIN, channel or location field, so the study's authentication research question **cannot be answered empirically** and no experiment was manufactured to pretend otherwise. It is resolved instead as a design contribution: the same declared `R` that sets the alerting threshold also sets an authentication escalation ladder, because `band_cutoffs` anchors every boundary on `λ`.

What is measured is the ladder's operational load — **97.04%** of transactions see no additional friction while the two escalated bands, 2.96% of traffic, carry **90.2%** of the fraud. What is **not** claimed is that any escalation works: no control was tested, and PaySim's fraud is an account-draining signature rather than an authentication compromise. That boundary is enforced in code and asserted by a test.

`docs/authentication_design_contribution.md`.

## Layout

```
src/momo_fraud/     analysis library — metrics, splits, risk, experiment protocol, guards
notebooks/          00–10, orchestration
scripts/            experiments/, repairs/, freeze/, report/ (table generators)
results/            every artifact, each with provenance and a VALID/SUPERSEDED/DIAGNOSTIC status
figures/            32 figures, PNG and vector PDF
docs/               experiment reports, the repair record, the freeze protocol, the change log
prototype/          Streamlit demo; all inference goes through src/momo_fraud/predict.py
tests/              239 tests across the library and the prototype
```

`Dockerfile` builds the prototype; see `prototype/README.md` for what it needs on disk.

## Data

Not in the repository (~470 MB, gitignored). PaySim is synthetic; `02_leakage_audit` identifies which features encode the _simulator_ rather than fraud behaviour, and every headline result is produced twice — with and without them.

The supportable claim about the resulting feature set is **reduced simulator artefact exposure** — not "simulator-independent" and not "artefact-free". A test asserts that wording is not overstated.
