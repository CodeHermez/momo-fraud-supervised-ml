"""Dataset facts and experiment-wide constants.

Everything here is either a property of the canonical PaySim file
(``ealaxi/paysim1``) or a design constant fixed by the plan. Values that
describe the data are asserted against the real file in ``data.load_raw``
rather than trusted.
"""

from __future__ import annotations

# --- Canonical PaySim (ealaxi/paysim1) -------------------------------------
# The file Lokanan (2023) and Thar & Wai (2025) both used.

RAW_FILENAME = "PS_20174392719_1491204439457_log.csv"

N_ROWS = 6_362_620
N_FRAUD = 8_213
N_LEGIT = N_ROWS - N_FRAUD  # 6_354_407
PREVALENCE = N_FRAUD / N_ROWS  # 0.001291 -> 0.129%

# Note the inconsistent spelling in the source file: the origin account's
# *old* balance is "oldbalanceOrg" (no 'i') while its *new* balance is
# "newbalanceOrig" (with 'i'). This is a quirk of PaySim, not a typo here.
RAW_COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]

TARGET = "isFraud"

#: Dropped before modelling. ``isFlaggedFraud`` is the simulator's own rule
#: firing on the label, not an input signal -- notebook 02 evidences this.
#: Account identifiers are dropped per Lokanan's preprocessing.
LEAK_COLUMNS = ["isFlaggedFraud"]
ID_COLUMNS = ["nameOrig", "nameDest"]

TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

#: Fraud occurs only in these two types. Verified in notebook 01; also
#: reported by Thar & Wai (2025), Fig 2.
FRAUD_BEARING_TYPES = ["TRANSFER", "CASH_OUT"]

#: ``step`` is hours since simulation start. The simulator was configured for
#: 744 hours (24 * 31) but the log stops at 743, giving 30.96 days of activity.
STEPS_PER_DAY = 24
MAX_STEP = 743


# --- Risk framework ---------------------------------------------------------

#: Severity ratio: "missing one fraud is as bad as R false alarms."
#: Johnson & Khoshgoftaar (2022) recommend C(0,1) = N_neg / N_pos, which for
#: PaySim is 6_354_407 / 8_213 = 773.7. The resulting Bayes threshold
#: lambda = 1/(1+R) then coincides with the class prior, as their theory predicts.
R_DEFAULT = N_LEGIT / N_FRAUD

#: Swept in the sensitivity analysis (notebook 06).
R_SWEEP = [1, 10, 50, 100, 500, R_DEFAULT, 1000, 2000, 5000]

#: Analyst review capacities for Recall@budget, as transactions per day.
#: The test split covers a fraction of the 30-day simulation, so notebook 06
#: scales these to the split's day count before applying them.
REVIEW_BUDGETS = [100, 500, 1000, 5000]

#: Operational bands mapped to the Fraud Management System in the proposal's
#: Figure 1. Cutoffs are derived from lambda in notebook 06, not fixed here;
#: these are the actions each band triggers.
BAND_ACTIONS = {
    "Low": "allow",
    "Medium": "monitor",
    "High": "step_up_auth",
    "Critical": "block",
}


# --- Experiment protocol ----------------------------------------------------

RANDOM_SEED = 42
CV_FOLDS = 5
CV_REPEATS = 5

#: Stratified split proportions (proposal section 4.4).
SPLIT_TRAIN = 0.70
SPLIT_VAL = 0.15
SPLIT_TEST = 0.15

#: Temporal split boundaries on ``step``, chosen to land near 70/15/15 by row
#: count. Verified against the real distribution in ``splits.temporal_split``.
TEMPORAL_TRAIN_END = 520
TEMPORAL_VAL_END = 632

LEARNERS = ["logistic_regression", "decision_tree", "random_forest", "xgboost"]

#: Configuration grid. Levels refer to the proposal's section 3.3 taxonomy.
CONFIGS = {
    "A": "baseline, t=0.5",
    "B": "threshold from ROC (Youden)",
    "C": "threshold minimising NER",
    "D": "scale_pos_weight = R, t=0.5",
    "E": "weighted + NER threshold",
    "F": "per-instance sample_weight ~ severity",
    "G": "RUS at N'_neg, natural test set",
}


# --- External benchmarks ----------------------------------------------------

#: Thar & Wai (2025), Table 4 -- PaySim PR-AUC. Used in notebook 03 as a
#: pipeline correctness check: large divergence means a bug, not a discovery.
THAR_WAI_PR_AUC = {
    "logistic_regression": 0.7120,
    "random_forest": 0.8905,
    "xgboost": 0.9160,
    "hmm": 0.7050,
    "hybrid_hmm_xgboost": 0.9689,
}

#: Lokanan (2023), Tables 6 and 8 -- measured on a ~50/50 resampled test set,
#: which is the defect notebook 05 demonstrates. Targets for replication Arm A.
LOKANAN_RESAMPLED = {
    "balanced_n": 1_187_716,
    "random_forest": {"precision": 0.96, "recall": 0.80, "f1": 0.87},
    "decision_tree": {"precision": 1.00, "recall": 0.72, "f1": 0.84},
    "logistic_regression": {"precision": 0.76, "recall": 0.96, "f1": 0.85},
}
