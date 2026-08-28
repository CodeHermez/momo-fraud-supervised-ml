# momo-fraud-supervised-ml

Risk-based cost-sensitive fraud detection on mobile money transactions, supporting _Financial Fraud Detection in Mobile Money Networks Using Supervised Machine Learning_ (COS700).

Evaluation is **unitless throughout** a severity ratio and a 0–100 risk score, never a currency.

## The question

Does cost-sensitive learning actually improve fraud detection on severely imbalanced mobile money data, and _which level_ of it does the work data, algorithm, or decision?

Johnson & Khoshgoftaar (2022) showed on Medicare claims (8.7M rows, 0.046% positive) that class weighting and undersampling **reduce** a learner's discriminative power, and that thresholding alone outperforms both. They asked for replication on other severely imbalanced datasets. PaySim, at 0.129% prevalence, is that dataset and no prior mobile money study reports any risk analysis at all.

## Findings

**PaySim's fraud is recoverable by three boolean clauses.**

```
type ∈ {TRANSFER, CASH_OUT}  AND  amount == oldbalanceOrg  AND  newbalanceOrig == 0
```

Precision **0.9999**, recall **0.9770**, on all 6,362,620 transactions. Every published machine-learning result we found on this dataset scores _below_ a rule with no parameters and no training. Results on the full feature set therefore measure the simulator, not fraud detection which is why the experiment runs on an ablated feature set (`notebooks/02`, `notebooks/03`).

**Cost-sensitive _training_ does not help; cost-sensitive _thresholding_ does.**

Across four learners and twelve comparisons class weighting, per-instance severity weighting, and undersampling **none improved PR-AUC over the unweighted baseline** and all twelve deltas were negative (−0.014 to −0.536). Moving the threshold instead removes **49–62%** of the risk (NER 0.2590 → 0.1314 for XGBoost, 0.6591 → 0.2499 for logistic regression).

For the two strongest learners the tuned threshold beats _every_ cost-sensitive variant outright XGBoost 0.1314 against 0.1355–0.1464, random forest 0.1324 against 0.1399–0.2518. This replicates Johnson & Khoshgoftaar (2022) in a new domain at 0.129% prevalence, which is the replication they called for.

**At `R = N_neg/N_pos`, risk-optimal and ROC-optimal thresholds are the same threshold.**

Minimising `FP + R·FN` reduces to maximising `TPR − FPR` at that ratio provably, for every model. The two criteria only diverge away from the class ratio, which reframes the apparent "statistical versus business threshold" disagreement in the literature as a disagreement between two _cost ratios_.

**`amount` is not the most important feature.**

Once the simulator's origin-balance bookkeeping is removed, transaction **type** dominates (permutation importance 0.701 versus 0.447 for amount). Prior work reporting amount as the top predictor did so with those balance features present.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pip install -e .          # so `import momo_fraud` works outside notebooks/
.venv/Scripts/python -m pytest tests/ -q
```

The editable install is what lets a prototype elsewhere on the machine load the
model. Without it only the notebooks work, because they patch `sys.path`.

Then run `notebooks/00_data_acquisition.ipynb`, which downloads `ealaxi/paysim1` (needs `~/.kaggle/kaggle.json`, or download it manually the notebook prints instructions either way).

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

Notebooks orchestrate; the analysis primitives live in `src/momo_fraud/` so metric definitions cannot drift between them, and so they can be tested.

## The risk framework

One interpretable knob replaces every currency figure:

> **R** "missing one fraud is as bad as R false alarms."

All three cost-sensitive levels in the proposal's section 3.3 taxonomy derive from it:

| Level     | Technique       | Derived from R         |
| --------- | --------------- | ---------------------- |
| Data      | undersampling   | `N'_neg = N_neg / R`   |
| Algorithm | class weighting | `scale_pos_weight = R` |
| Decision  | thresholding    | `λ = 1 / (1 + R)`      |

For PaySim `R = N_neg/N_pos = 773.70`, which puts `λ` exactly at the class prior (0.00129) as the theory predicts.

Results are reported as **PR-AUC** (model quality), **Recall@budget** ("if analysts review _k_ transactions a day, what share of fraud is caught?"), and **Normalized Expected Risk** the proposal's `TotalCost` divided by the do-nothing baseline, so it lands in [0,1] with 1.0 meaning "no better than ignoring the problem".

## Data

Not in the repository (~470 MB, gitignored). PaySim is synthetic; `02_leakage_audit` identifies which features encode the _simulator_ rather than fraud behaviour, and every headline result is produced twice with and without them.
