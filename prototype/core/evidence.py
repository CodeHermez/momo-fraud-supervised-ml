"""Cached access to what the experiments already produced.

Nothing here computes a result. The figures, tables and held-out scores were all
written by the notebooks; this module only finds them, loads them once, and
groups them so a page can present them in the order the study made them.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"
DEMO_SAMPLE = PROJECT_ROOT / "prototype" / "data" / "demo_sample.parquet"

#: The held-out scores behind the shipped model -- the same file the model card
#: was computed from, so the threshold page can reproduce it rather than quote it.
TEST_PREDICTIONS = (
    PREDICTIONS_DIR / "xgboost__unweighted__test__seed42__no_origin_balance.parquet"
)

#: Notebook number -> what that notebook was asking. Drives the gallery's
#: section order and its lead-ins.
CHAPTERS: list[tuple[str, str, str]] = [
    ("01", "The data",
     "What PaySim contains, and why 0.129% prevalence makes accuracy a useless "
     "measure. Fraud appears in only two of the five transaction types."),
    ("02", "The leakage audit",
     "The measurement behind the section above: how much fraud is recoverable "
     "with no model at all, and which columns leak."),
    ("03", "Baselines and the ablation ladder",
     "Four learners across progressively harder feature sets, checked against "
     "two published PaySim results to confirm the pipeline is not broken."),
    ("04", "Does cost-sensitive training help?",
     "Class weighting and resampling, compared against simply moving the "
     "decision threshold. For the two strongest learners, thresholding wins."),
    ("05", "Replication and base-rate collapse",
     "Reproducing a published result on a 50/50 resampled test set, then "
     "watching its precision collapse when the natural 0.129% prevalence is "
     "restored. Precision falls by between 2.5x and 31x depending on learner."),
    ("06", "Risk, thresholds and review budgets",
     "Where to put the alert line once missing a fraud is priced against "
     "raising a false alarm, and how much fraud a fixed analyst headcount "
     "can actually catch."),
    ("07", "Interpretability",
     "Which features carry the decision, by SHAP, gain and permutation "
     "importance, set beside what the published literature reports."),
    ("09", "Velocity features",
     "Ten causal account-history features -- prior counts, fan-in, fan-out, "
     "time since last activity. They did not move the needle."),
    ("10", "Further optimisation",
     "Hyperparameter search, a second gradient-boosting implementation, "
     "stacking, and a seed-variance check."),
]


@st.cache_data(show_spinner=False)
def load_result(name: str) -> pd.DataFrame:
    """Read one CSV from ``results/``."""
    return pd.read_csv(RESULTS_DIR / name)


@st.cache_data(show_spinner=False)
def load_test_predictions() -> pd.DataFrame:
    """The held-out scores: ``y_true``, ``score``, ``severity`` for 954k rows."""
    return pd.read_parquet(TEST_PREDICTIONS)


@st.cache_data(show_spinner=False)
def load_demo_sample() -> pd.DataFrame:
    """The bundled labelled slice, with raw fields alongside recorded scores."""
    return pd.read_parquet(DEMO_SAMPLE)


@st.cache_data(show_spinner="Scoring the sample with the shipped model ...")
def load_demo_sample_scored() -> pd.DataFrame:
    """The bundled slice, re-scored by the model this prototype actually serves.

    The ``score`` column shipped in the sample came from the *train-only* model
    the experiments evaluated. ``artifacts/`` holds a refit on train+validation
    (85% of the data, per the model card), so the two disagree slightly. Scoring
    the sample here keeps the explorer's quadrants consistent with what happens
    when a row is opened -- 41k rows takes about a tenth of a second, so there is
    no reason to reuse a stale column and explain the mismatch away.
    """
    from . import schema, scorer

    sample = load_demo_sample()
    rows = sample[schema.REQUIRED_FIELDS].copy()
    rows["step"] = rows["step"].astype(int)
    rows["type"] = rows["type"].astype(str)

    scored = scorer.get_scorer().score_batch(rows)
    return sample.rename(columns={"score": "experiment_score"}).join(scored)


def demo_sample_available() -> bool:
    return DEMO_SAMPLE.exists()


def test_predictions_available() -> bool:
    return TEST_PREDICTIONS.exists()


def figures_for(prefix: str) -> list[Path]:
    """PNG figures belonging to one notebook, in filename order."""
    return sorted(FIGURES_DIR.glob(f"{prefix}_*.png"))


def results_for(prefix: str) -> list[Path]:
    """CSV tables belonging to one notebook, in filename order."""
    return sorted(RESULTS_DIR.glob(f"{prefix}_*.csv"))


def prettify(path: Path) -> str:
    """``06_recall_at_budget.png`` -> ``Recall at budget``."""
    stem = path.stem.split("_", 1)[-1]
    return stem.replace("_", " ").capitalize()
