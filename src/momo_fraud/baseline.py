"""The frozen experimental baseline, and the guards that keep it frozen.

Everything downstream of the repair phase is compared against one fixed
reference point. This module is that reference point expressed as code, plus the
three control mechanisms the freeze-phase consistency checks found missing:

1. **Superseded artifacts cannot be loaded as current.** ``load_results`` refuses
   any artifact the experiment index marks SUPERSEDED unless the caller says so
   explicitly. Documenting a file's status in a JSON index does not stop anyone
   reading it with ``pd.read_csv``; refusing it does.

2. **The diagnostic rung cannot be selected by accident.** ``choose_rung``
   accepted whatever order it was handed, so ``no_balance_state`` entering the
   ladder was a matter of the caller remembering to filter it out. It is now
   filtered by default.

3. **Every experiment record carries the fields needed to interpret it.**
   ``R_train`` and ``R_eval`` were not distinguished anywhere except the
   decision-cost sweep, and seed was missing from most artifacts.
   ``validate_experiment_frame`` states the required schema and checks it.

**This module does not compute anything about the data.** It records decisions
already taken and evidenced elsewhere, and refuses configurations that depart
from them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import constants as C
from . import evaluate as E
from .features import DIAGNOSTIC_FEATURE_SETS, FEATURE_SETS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"

MANIFEST_NAME = "frozen_experimental_baseline"
INDEX_NAME = "experiment_index"


# --- The freeze ---------------------------------------------------------------

#: The main experimental feature set. Frozen after the repair phase.
#:
#: Evidence, in the order it was established:
#:
#: * The origin-side balance columns encode a deterministic simulator rule. A
#:   three-clause boolean expression over them recovers the label at precision
#:   0.99988 / recall 0.97699, ``errorBalanceOrig`` scores 0.9020 alone, and the
#:   drain signature holds for 97.82% of fraud against 0.0013% of legitimate
#:   transactions with a median gap of exactly 0.0. Removal is justified.
#: * The destination-side diagnostic did **not** establish an equivalent
#:   artefact. The strongest destination feature scores 0.6237 alone against
#:   0.9020, lifts run 1.3-1.5x rather than near-deterministic, and the specific
#:   mechanism the concern named -- the simulator not crediting the receiving
#:   account -- measures at 0.60 for legitimate against 0.52 for fraud, a lift of
#:   0.86 pointing the wrong way. See docs/destination_artefact_diagnostic.md.
#: * Validation-based rung selection still returns this rung (val NER 0.1147),
#:   under the unchanged pre-declared criterion.
#:
#: So there is no evidence-based reason to change the main rung, and it is fixed.
FROZEN_RUNG = "no_origin_balance"

#: What the freeze claims, in the words it is allowed to claim it in.
#:
#: The rung removes the balance-derived features whose predictive strength is
#: tied to simulator bookkeeping. It does **not** make the feature set
#: simulator-independent: PaySim is a simulator throughout, fraud occurs only in
#: TRANSFER and CASH_OUT by construction, and the retained destination-side
#: columns carry weak but real generator-specific association. "Reduced simulator
#: artefact exposure" is the supportable claim; "artefact-free" is not.
FROZEN_RUNG_CLAIM = "reduced simulator artefact exposure"

#: Retained for diagnosis only. Never a candidate for the main rung.
DIAGNOSTIC_RUNGS = frozenset(DIAGNOSTIC_FEATURE_SETS)

#: The ablation ladder that rung selection may choose from.
SELECTABLE_RUNGS = [r for r in FEATURE_SETS if r not in DIAGNOSTIC_RUNGS]

#: Core learners. Additions need separate justification -- they are not part of
#: the research question as posed.
FROZEN_LEARNERS = list(C.LEARNERS)

#: Excluded from the valid model comparison: the fit did not learn the minority
#: class (PR-AUC 0.043 at 0.129% prevalence). A failed run, not a comparison.
EXCLUDED_LEARNERS = {"lightgbm": "failed fit; see results/10_failed_runs.csv"}


# --- The CSL taxonomy ---------------------------------------------------------

#: Four mechanisms, kept conceptually separate. Collapsing them is how a
#: decision-layer effect gets attributed to training, and how a variant measured
#: on someone else's objective gets read as a verdict on its own.
CSL_TAXONOMY: dict[str, dict] = {
    "baseline": {
        "variants": ["unweighted"],
        "configs": ["A"],
        "level": "none",
        "objective": "unweighted log-loss / Gini",
        "description": "No cost sensitivity anywhere.",
    },
    "algorithm_and_data_level": {
        "variants": ["weighted", "rus"],
        "configs": ["D", "E", "G"],
        "level": "algorithm / data",
        "objective": "class-constant: C_FP = 1, C_FN = R",
        "description": (
            "Class weighting (w+ = R, w- = 1) and random undersampling "
            "(N'_neg = N_neg / R). Both express the same class-constant cost "
            "matrix, one in the loss and one in the sample."
        ),
    },
    "decision_level": {
        "variants": ["unweighted"],
        "configs": ["B", "C"],
        "level": "decision",
        "objective": "class-constant: C_FP = 1, C_FN = R",
        "description": (
            "Threshold optimisation on an unweighted model. At R = N_neg/N_pos "
            "the Youden and NER-optimal thresholds are provably identical, so B "
            "and C are ONE condition at the default R."
        ),
    },
    "sensitivity_analysis": {
        "variants": ["instance_weighted"],
        "configs": ["F"],
        "level": "algorithm (example-dependent training only)",
        "objective": "TRAINS on C_FN = R * s_i; THRESHOLDED and SCORED on C_FN = R",
        "description": (
            "Instance/severity weighting. NOT a directly equivalent "
            "implementation of the class-constant objective that D and G use: it "
            "optimises a different loss from the one it is judged on, and since "
            "E[s_i] ~ 0.5 it carries roughly half D's effective positive weight. "
            "Its result is evidence about this variant under the common "
            "evaluation objective, and is NOT a standalone verdict on severity "
            "weighting. See docs/config_f_cost_objectives.md."
        ),
    },
}

#: Stated separately because it is the sentence most likely to be written by
#: accident when summarising the F result.
CONFIG_F_FORBIDDEN_CLAIM = (
    "The existing configuration F result does NOT support the claim that "
    "severity weighting does not work. Training and evaluation use different "
    "cost objectives, so the comparison cannot settle that question in either "
    "direction. A dedicated example-dependent experiment would be required."
)

#: Stated separately for the same reason: RQ 2.1.1 is answered as a *design*
#: contribution (docs/authentication_design_contribution.md, CHANGE-06), and the
#: band-to-step-up mapping is the sentence most likely to be promoted by
#: accident from a design argument into a security result.
AUTH_DESIGN_FORBIDDEN_CLAIM = (
    "The risk-band to step-up authentication mapping is NOT an empirical "
    "security evaluation. No control was tested: nothing here shows that a "
    "biometric step-up prevents a vished transaction or that out-of-band "
    "verification defeats a SIM swap. PaySim carries no session, device, SIM, "
    "PIN, channel or location field, and its fraud is an account-draining "
    "signature rather than an authentication compromise, so the band "
    "populations describe friction cost on that process only. The supportable "
    "claim is that the escalation ladder is DERIVED from the declared cost "
    "ratio R, not that the escalations work."
)


# --- Protocol -----------------------------------------------------------------

FROZEN_SPLIT = {
    "strategy": "stratified",
    "train": C.SPLIT_TRAIN,
    "val": C.SPLIT_VAL,
    "test": C.SPLIT_TEST,
    "split_seed": C.RANDOM_SEED,
    "stratified_on": C.TARGET,
    "implementation": "splits.stratified_split",
    "invariants": [
        "same train/val/test partitions across directly compared configurations",
        "FeatureBuilder fitted on training rows only",
        "resampling (RUS, SMOTE) applied to training rows only",
        "threshold selected on validation",
        "threshold locked before any test evaluation",
        "test set read once, for final reporting",
    ],
}

#: Two different seeds, and conflating them is how "we ran four seeds" comes to
#: describe an experiment whose split never moved.
SEED_KINDS = {
    "model_seed": (
        "Seeds estimator initialisation, bagging, column subsampling and the RUS "
        "draw. Varying it measures training stochasticity. Note that "
        "LogisticRegression and DecisionTreeClassifier are deterministic here, so "
        "it moves only random forest and XGBoost."
    ),
    "split_seed": (
        "Seeds stratified_split. Varying it measures which rows landed in which "
        "partition. Held fixed at 42 for every result produced so far, so split "
        "variability is currently unmeasured."
    ),
}

FROZEN_COST = {
    "definition": "R = 'missing one fraud is as bad as R false alarms'",
    "C_FP": 1.0,
    "C_FN": "R",
    "R_default": float(C.R_DEFAULT),
    "R_default_derivation": "N_legit / N_fraud over the full dataset",
    "bayes_threshold": "lambda = 1 / (1 + R)",
    "R_train_vs_R_eval": (
        "R_train enters scale_pos_weight, the RUS target and the severity "
        "weights. R_eval enters threshold selection and the NER computation. "
        "They are equal at the default R in every experiment run so far, and "
        "MUST be recorded separately regardless -- the existing decision-cost "
        "sweep varies R_eval alone at fixed R_train = 773.70."
    ),
}

FROZEN_THRESHOLD_PROTOCOL = {
    "order": "TRAIN -> FIT -> VALIDATION -> SELECT -> LOCK -> TEST -> REPORT",
    "rules": ["fixed", "youden", "ner_optimal"],
    "selected_on": "val",
    "applied_to": "test",
    "implementation": "evaluate.select_threshold",
    "note": (
        "At R = N_neg/N_pos the Youden and NER-optimal thresholds are the same "
        "number for every model -- a theorem, not a coincidence. Configs B and C "
        "are one condition at the default R, giving six distinct conditions "
        "rather than seven."
    ),
}

FROZEN_METRICS = {
    "primary_ranking": "pr_auc",
    "primary_decision_risk": "ner",
    "also_reported": ["precision", "recall", "f1", "mcc", "fp", "fn",
                      "n_flagged", "alert_rate", "brier", "roc_auc"],
    "excluded": {
        "accuracy": (
            "Never computed. At 0.129% prevalence a constant-negative predictor "
            "scores 0.9987, so it cannot discriminate between models."
        ),
    },
    "roc_auc_caveat": (
        "Reported for comparability with the prior work only. Near-insensitive "
        "to false positives at this imbalance; not used to rank models."
    ),
    "uncertainty": {
        "procedure": "paired stratified bootstrap of the delta on shared resamples",
        "n_boot": 2000,
        "implementation": "evaluate.paired_bootstrap_pr_auc / paired_bootstrap_delta",
        "forbidden": (
            "CI overlap must not be used as a significance decision. Non-overlap "
            "implies a difference; overlap does not imply its absence."
        ),
        "language": (
            "Say 'the paired interval excludes zero'. Do not say 'statistically "
            "significant' unless a procedure supporting that phrase was run."
        ),
    },
}

#: Every experiment record must carry these. The freeze-phase check found seed
#: missing from most artifacts and R_train/R_eval distinguished in exactly one.
REQUIRED_EXPERIMENT_FIELDS = [
    "learner",
    "feature_set",
    "csl_variant",
    "R_train",
    "R_eval",
    "threshold_rule",
    "threshold",
    "model_seed",
    "split_seed",
    "split_strategy",
    "split",
    "val_prevalence",
    "test_prevalence",
]

#: Metrics every experiment record must carry alongside the fields above.
REQUIRED_METRIC_FIELDS = [
    "pr_auc", "ner", "precision", "recall", "f1", "mcc",
    "fp", "fn", "n_flagged", "alert_rate",
]


# --- What the evidence currently supports -------------------------------------

#: Deliberately phrased at the strength the evidence carries, no further. Each
#: entry names where it comes from so a reader can check it rather than trust it.
SUPPORTED_CONCLUSIONS = [
    {"claim": "PaySim contains highly recoverable simulator-specific fraud signatures.",
     "evidence": "results/02_rule_recoverability.csv -- three clauses reach "
                 "precision 0.99988, recall 0.97699"},
    {"claim": "errorBalanceOrig is unsuitable as an unrestricted feature: its "
              "predictive strength is tied to simulator bookkeeping.",
     "evidence": "results/02_single_feature_auc.csv (0.9020 alone); "
                 "results/02_drain_signature.csv (median gap exactly 0.0 for fraud)"},
    {"claim": "The main experiments therefore use no_origin_balance.",
     "evidence": "results/03_experimental_rung_v2.json -- validation-selected, "
                 "criterion unchanged"},
    {"claim": "CSL methods show consistently negative PR-AUC deltas against the "
              "unweighted baseline.",
     "evidence": "results/04_h1_paired_bootstrap.csv -- 12 of 12 negative, every "
                 "paired CI excluding zero, n_boot 2000"},
    {"claim": "The effect of CSL on NER is learner- and cost-dependent.",
     "evidence": "results/06_decision_cost_sweep.csv -- CSL-trained models reach "
                 "lower NER in 17 of 36 cells; note R_train is fixed at 773.70 "
                 "throughout, so this varies the decision cost only"},
    {"claim": "Threshold optimisation produces substantial risk reductions "
              "relative to a fixed 0.5 threshold.",
     "evidence": "results/06_threshold_markers.csv -- e.g. logistic regression "
                 "0.6591 -> 0.2499, XGBoost 0.2590 -> 0.1314"},
    {"claim": "Evaluation prevalence has a major effect on precision.",
     "evidence": "results/05_precision_collapse.csv; see "
                 "results/05_arm_interpretation.json for what the arms do and do "
                 "not isolate -- the A-to-B contrast does not vary prevalence alone"},
    {"claim": "Temporal generalisation is harder for 3 of 4 learners.",
     "evidence": "results/09_temporal_correction.csv -- on the repaired split "
                 "(69.68/15.30/15.02); the superseded split showed the opposite "
                 "through a base-rate artefact"},
    {"claim": "PaySim cannot empirically answer the authentication-vulnerability "
              "research question, because those variables do not exist in it.",
     "evidence": "constants.RAW_COLUMNS -- no session, device, SIM, PIN, channel "
                 "or location field"},
]

#: Claims the evidence does not reach. Listed because each is a plausible
#: overstatement of something above.
FORBIDDEN_CLAIMS = [
    "The feature set is simulator-independent or artefact-free. "
    f"The supportable phrase is '{FROZEN_RUNG_CLAIM}'.",
    CONFIG_F_FORBIDDEN_CLAIM,
    "CSL never helps. Its effect on end-to-end risk is learner- and "
    "cost-dependent; only the ranking-quality result is uniformly negative.",
    "The decision-cost sweep shows CSL winning at a given training cost ratio. "
    "Every CSL model in it was trained at R_train = 773.70.",
    "Evaluation prevalence alone accounts for the precision collapse. The "
    "notebook 05 arms also differ in syntheticness, ENN cleaning and "
    "train/test contamination.",
    "Any CSL delta is stable across seeds or splits. Every one rests on a "
    "single model seed and a single split seed.",
    AUTH_DESIGN_FORBIDDEN_CLAIM,
]


# --- Guards -------------------------------------------------------------------


class SupersededArtifactError(RuntimeError):
    """Raised when a superseded result is loaded as though it were current."""


@dataclass
class ConsistencyCheck:
    """One freeze-phase check and what it found."""

    name: str
    passed: bool
    detail: str
    evidence: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"check": self.name, "status": "PASS" if self.passed else "FAIL",
                "detail": self.detail, "evidence": self.evidence}


def artifact_status(name: str, *, results_dir: Path | None = None) -> str | None:
    """The experiment index's status for one artifact stem, or None if unlisted."""
    directory = Path(results_dir) if results_dir is not None else RESULTS_DIR
    index_path = directory / f"{INDEX_NAME}.json"
    if not index_path.exists():
        return None

    index = json.loads(index_path.read_text(encoding="utf-8"))
    for row in index.get("artifacts", []):
        if row["artifact"] == name:
            return row["status"]
    return None


def load_results(
    name: str,
    *,
    allow_superseded: bool = False,
    results_dir: Path | None = None,
) -> pd.DataFrame:
    """Load a results CSV, refusing superseded artifacts by default.

    The experiment index records which artifacts the repair phase replaced, but
    an index only informs -- it cannot stop ``pd.read_csv`` reading a stale file
    into a new analysis. This is the enforcement.

    Pass ``allow_superseded=True`` to read one deliberately, which the repair
    scripts do when producing the replacement or comparing against it.
    """
    directory = Path(results_dir) if results_dir is not None else RESULTS_DIR
    status = artifact_status(name, results_dir=directory)

    if status == "SUPERSEDED" and not allow_superseded:
        index = json.loads((directory / f"{INDEX_NAME}.json").read_text(encoding="utf-8"))
        note = next((r.get("note") for r in index["artifacts"]
                     if r["artifact"] == name), None)
        raise SupersededArtifactError(
            f"{name!r} is SUPERSEDED and must not be loaded as a current result.\n"
            f"  reason: {note}\n"
            f"  pass allow_superseded=True only to inspect it deliberately."
        )

    return pd.read_csv(directory / f"{name}.csv")


def validate_experiment_frame(
    frame: pd.DataFrame,
    *,
    require_metrics: bool = True,
) -> list[str]:
    """Missing required columns for a frame claiming to be an experiment record.

    Returns the missing field names rather than raising, so a caller can report
    them all at once. Empty list means the schema is satisfied.
    """
    required = list(REQUIRED_EXPERIMENT_FIELDS)
    if require_metrics:
        required += REQUIRED_METRIC_FIELDS
    return [f for f in required if f not in frame.columns]


def assert_frozen_rung(rung: str) -> None:
    """Fail loudly if an experiment is about to run at a non-frozen rung."""
    if rung in DIAGNOSTIC_RUNGS:
        raise ValueError(
            f"{rung!r} is a DIAGNOSTIC rung and must not be used as the main "
            f"experimental feature set. The frozen rung is {FROZEN_RUNG!r}."
        )
    if rung != FROZEN_RUNG:
        raise ValueError(
            f"main experiments must run at the frozen rung {FROZEN_RUNG!r}, "
            f"not {rung!r}. Changing it requires a recorded decision in "
            f"docs/methodology_change_log.md."
        )


def manifest(results_dir: Path | None = None) -> dict:
    """The frozen baseline as a plain dict, ready to serialise."""
    from . import provenance as P

    directory = Path(results_dir) if results_dir is not None else RESULTS_DIR
    dataset = P.dataset_record()
    git = P.git_state()

    return {
        "status": "FROZEN",
        "frozen_at_utc": None,          # filled by the writer
        "phase": "post-repair frozen baseline",
        "authority": "docs/repair_report.md",
        "dataset": {
            "identifier": dataset.get("dataset"),
            "sha256": dataset.get("csv_sha256"),
            "n_transactions": dataset.get("n_transactions"),
            "n_fraud": dataset.get("n_fraud"),
            "prevalence": dataset.get("prevalence"),
            "validated_on_load_by": "data.validate",
        },
        "feature_set": {
            "frozen_rung": FROZEN_RUNG,
            "claim": FROZEN_RUNG_CLAIM,
            "features_dropped": FEATURE_SETS[FROZEN_RUNG],
            "selected_on": "val",
            "selection_criterion": "best NER >= 0.05, least-ablated viable rung",
            "validation_ner_at_rung": 0.1146545932099288,
            "selectable_rungs": SELECTABLE_RUNGS,
            "diagnostic_rungs": sorted(DIAGNOSTIC_RUNGS),
            "rationale": [
                "origin-side simulator artefact evidence justified removal",
                "destination-side diagnostic did not establish an equivalent "
                "deterministic artefact",
                "validation-based selection still selects no_origin_balance",
                "therefore no evidence-based reason to change the main rung",
            ],
            "forbidden_claim": (
                "Do not describe this feature set as simulator-independent or "
                f"artefact-free. Say '{FROZEN_RUNG_CLAIM}'."
            ),
        },
        "split": FROZEN_SPLIT,
        "seeds": {"kinds": SEED_KINDS, "model_seed": C.RANDOM_SEED,
                  "split_seed": C.RANDOM_SEED},
        "cost": FROZEN_COST,
        "models": {"learners": FROZEN_LEARNERS,
                   "excluded": EXCLUDED_LEARNERS,
                   "hyperparameters": "models.make_learner defaults; fixed, not tuned"},
        "csl_taxonomy": CSL_TAXONOMY,
        "threshold_protocol": FROZEN_THRESHOLD_PROTOCOL,
        "metrics": FROZEN_METRICS,
        "required_experiment_fields": REQUIRED_EXPERIMENT_FIELDS,
        "required_metric_fields": REQUIRED_METRIC_FIELDS,
        "supported_conclusions": SUPPORTED_CONCLUSIONS,
        "forbidden_claims": FORBIDDEN_CLAIMS,
        "provenance": {
            "git_commit": git.get("commit"),
            "git_branch": git.get("branch"),
            "git_dirty": git.get("dirty"),
            "environment": "results/experiment_environment.json",
            "experiment_index": f"results/{INDEX_NAME}.json",
            "change_log": "results/methodology_change_log.json",
        },
        "human_readable": "docs/frozen_baseline_protocol.md",
    }


def load_manifest(results_dir: Path | None = None) -> dict:
    """The frozen manifest as written to disk."""
    directory = Path(results_dir) if results_dir is not None else RESULTS_DIR
    path = directory / f"{MANIFEST_NAME}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No frozen baseline at {path}. Run scripts/freeze/write_manifest.py."
        )
    return json.loads(path.read_text(encoding="utf-8"))
