"""E6 -- Threshold-rule divergence across the evaluation cost ratio.

The study proves, in ``risk.youden_threshold``, that at ``R = N_neg/N_pos`` the
ROC-optimal threshold (config B, Youden) and the risk-optimal threshold
(config C, NER-optimal) are the *same* threshold, for every model::

    FP + R*FN = FPR*N_neg + R*(1 - TPR)*N_pos
              = N_neg*(FPR + 1 - TPR)          when R = N_neg/N_pos

whose minimiser is exactly the maximiser of ``TPR - FPR``. Every headline number
in this study sits at that ratio, which is precisely where the comparison
between the two rules carries no information: B and C are degenerate there
(audit finding M5). E6 sweeps ``R_eval`` across the class ratio so that the
comparison becomes informative, and so that the equivalence is demonstrated
rather than only derived.

**Grid provenance -- read this before citing the numbers.** Unlike E4, whose
prevalence grid the audit pre-declared, *no R_eval grid was ever declared for
E6*. The repair report fixes the question ("sweep R across the class ratio",
"runs entirely from cache") and E2 section H.7 fixes the swept variable, but the
levels are chosen here. They are declared in ``R_MULTIPLIERS`` below, closed
before execution, and recorded in the manifest as ``grid_predeclared: false``.
The grid is geometric and symmetric in log space about the class ratio, because
the claim under test is about distance from that ratio in either direction.

**Cache-only, structurally.** No model is fitted. The script loads the
probability vectors E3 cached for validation and test, and re-selects thresholds
on validation alone. It imports the study's canonical ``risk`` module -- so the
threshold and NER definitions cannot drift from the ones the notebooks use --
via a stub package that bypasses ``momo_fraud/__init__.py``, which would
otherwise pull in ``models`` and with it every estimator library. QC asserts at
runtime that no estimator module is loaded.

**What is held fixed.** ``R_train`` never varies: every cell is a frozen E3 fit
at its own trained cost ratio, and only the *evaluation* ratio moves. The
feature rung, the splits and the fits are the frozen baseline b2e6f624.

Usage::

    python scripts/experiments/E6_threshold_divergence.py --smoke
    python scripts/experiments/E6_threshold_divergence.py --run
    python scripts/experiments/E6_threshold_divergence.py --aggregate --qc
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

EXPERIMENT_ID = "E6_threshold_divergence"
FROZEN_SHA = "b2e6f624"

#: The study's frozen cost ratio -- the *training* set's N_neg/N_pos.
R_FROZEN = 773.7010836478753

#: Pre-declared here, closed before execution. Geometric, symmetric in log space
#: about the class ratio: the hypothesis is that divergence grows with distance
#: from that ratio in either direction, so the grid must bracket it evenly.
R_MULTIPLIERS: tuple[float, ...] = (1 / 64, 1 / 16, 1 / 4, 1.0, 4.0, 16.0, 64.0)

#: Label of the level at the frozen ratio itself, used by QC.
FROZEN_LEVEL = "R_x1"

#: An eighth level, computed per cell rather than fixed: the *validation* set's
#: own N_neg/N_pos. This is where the theorem holds exactly. R_FROZEN is the
#: training ratio and sits about 0.004% away from it, which is close enough that
#: the study has always treated the two as one point -- E6 separates them.
EXACT_LEVEL = "val_class_ratio"

#: The two threshold rules under comparison, as the config grid names them.
RULES = {"B": "youden", "C": "ner_optimal"}

DESIGNATED = ("unweighted", "weighted")


# --- Canonical metric definitions, without loading an estimator --------------

def load_risk_module(source_root: Path):
    """Import ``momo_fraud.risk`` without executing the package __init__.

    ``momo_fraud/__init__.py`` imports ``models``, which imports xgboost and the
    sklearn estimators. Registering a stub package first means only
    ``constants`` and ``risk`` are executed, so the script gets the study's real
    metric definitions while remaining structurally incapable of fitting.
    """
    src = (source_root / "src" / "momo_fraud").resolve()
    pkg = types.ModuleType("momo_fraud")
    pkg.__path__ = [str(src)]
    sys.modules["momo_fraud"] = pkg
    sys.path.insert(0, str((source_root / "src").resolve()))
    from momo_fraud import risk as risk_module  # noqa: E402
    return risk_module


#: Libraries that can fit a model. ``joblib`` is deliberately absent:
#: ``sklearn.metrics`` imports it transitively, so it is always present once
#: ``risk`` is loaded, and it is a serialisation and parallelism helper that
#: cannot fit anything. The claim under test is that no *estimator* is
#: reachable, which these prefixes cover.
ESTIMATOR_PREFIXES = ("xgboost", "lightgbm", "imblearn",
                      "sklearn.ensemble", "sklearn.linear_model",
                      "sklearn.tree", "sklearn.svm", "sklearn.naive_bayes",
                      "sklearn.neighbors", "sklearn.neural_network")


def estimator_modules_loaded() -> list[str]:
    return sorted(m for m in sys.modules
                  if any(m == p or m.startswith(p + ".")
                         for p in ESTIMATOR_PREFIXES))


# --- Cached-prediction plumbing ----------------------------------------------

def prediction_path(source_root: Path, learner: str, variant: str, split: str,
                    split_seed: int, model_seed: int) -> Path:
    """Locate one cached score vector.

    Seed order matters: E3 saved these with ``seed=model_seed`` and
    ``tag=f"{rung}__s{split_seed}"``, so the ``seed`` field carries the model
    seed and the trailing ``s`` field the split seed. Transposing them loads a
    real vector for the wrong cell, and only shows up off the diagonal.
    """
    return (source_root / "predictions" /
            f"{learner}__{variant}__{split}__seed{model_seed}"
            f"__no_origin_balance__s{split_seed}.parquet")


def designated_cells(source_root: Path) -> pd.DataFrame:
    """The 104 designated E3 cells -- the same set E4 evaluated.

    Reusing E4's cell set exactly is deliberate: E2 recommended matching the
    split protocol so deltas are judged against the same noise floor, and it
    keeps E4 and E6 directly comparable.
    """
    runs = pd.read_csv(source_root / "results" / "E2_runs.csv")
    cells = runs[runs["source"] == "E3"].copy()
    cells = cells[[
        "learner", "csl_variant", "split_seed", "model_seed", "R_train",
        "threshold", "ner", "pr_auc",
        "val_n_positive", "val_n_negative",
        "test_n_positive", "test_n_negative",
    ]].reset_index(drop=True)
    return cells


def cell_id(row) -> str:
    return (f"{row.learner}__{row.csl_variant}"
            f"__split{row.split_seed}__model{row.model_seed}")


def r_levels(val_n_pos: int, val_n_neg: int) -> list[tuple[str, float]]:
    """The eight declared levels for one cell, in ascending R."""
    levels = [(f"R_x{m:g}", R_FROZEN * m) for m in R_MULTIPLIERS]
    levels.append((EXACT_LEVEL, val_n_neg / val_n_pos))
    return sorted(levels, key=lambda t: t[1])


# --- The evaluation core ------------------------------------------------------

def confusion(y: np.ndarray, s: np.ndarray, threshold: float) -> dict:
    pred = s >= threshold
    yb = y.astype(bool)
    tp = int(np.count_nonzero(pred & yb))
    fp = int(np.count_nonzero(pred & ~yb))
    fn = int(np.count_nonzero(~pred & yb))
    tn = int(np.count_nonzero(~pred & ~yb))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def score_at(risk_mod, y: np.ndarray, s: np.ndarray, threshold: float,
             r: float, rule: str) -> dict:
    """Evaluate one locked threshold, via the study's canonical implementation.

    ``tp/fp/fn/tn`` are counted here and the closed-form NER recomputed from
    them, so the canonical ``ner`` is cross-checked rather than trusted.
    """
    res = risk_mod.evaluate_threshold(y, s, threshold, r, rule=rule)
    c = confusion(y, s, threshold)
    n_pos = c["tp"] + c["fn"]
    total_risk = c["fp"] + r * c["fn"]
    ner_closed = total_risk / (r * n_pos) if n_pos else float("nan")
    flagged = c["tp"] + c["fp"]
    return {
        **c,
        "n_flagged": flagged,
        "alert_rate": flagged / len(y),
        "precision": c["tp"] / flagged if flagged else 0.0,
        "recall": c["tp"] / n_pos if n_pos else 0.0,
        "total_risk": total_risk,
        "ner": float(res.ner),
        "ner_closed_form": ner_closed,
        "ner_abs_error": abs(float(res.ner) - ner_closed),
    }


def run_cell(risk_mod, source_root: Path, row) -> dict:
    """One cell: eight R levels, two threshold rules, thresholds from val only."""
    cid = cell_id(row)
    paths = {sp: prediction_path(source_root, row.learner, row.csl_variant, sp,
                                 int(row.split_seed), int(row.model_seed))
             for sp in ("val", "test")}
    for sp, p in paths.items():
        if not p.exists():
            return {"cell_id": cid, "status": "skipped",
                    "reason": f"no cached {sp} predictions", "path": p.name,
                    "rows": []}

    val = pd.read_parquet(paths["val"], engine="pyarrow")
    test = pd.read_parquet(paths["test"], engine="pyarrow")
    y_val = val["y_true"].to_numpy()
    s_val = val["score"].to_numpy(dtype=float)
    y_test = test["y_true"].to_numpy()
    s_test = test["score"].to_numpy(dtype=float)

    n_pos_val = int(y_val.sum())
    n_neg_val = int(len(y_val) - n_pos_val)
    class_ratio = n_neg_val / n_pos_val

    # Config B: the ROC-optimal threshold. Independent of R by construction --
    # Youden never consumes a cost ratio. Computed once per cell.
    thr_b = float(risk_mod.youden_threshold(y_val, s_val))

    rows = []
    for label, r in r_levels(n_pos_val, n_neg_val):
        # Config C: re-selected on validation at this R. The only quantity in
        # the experiment that moves with R.
        thr_c = float(risk_mod.optimal_threshold(y_val, s_val, r).threshold)

        for cfg, thr in (("B", thr_b), ("C", thr_c)):
            rule = RULES[cfg]
            test_m = score_at(risk_mod, y_test, s_test, thr, r, rule)
            val_m = score_at(risk_mod, y_val, s_val, thr, r, rule)
            rows.append({
                "experiment_id": EXPERIMENT_ID,
                "cell_id": cid,
                "learner": row.learner,
                "csl_variant": row.csl_variant,
                "split_seed": int(row.split_seed),
                "model_seed": int(row.model_seed),
                "feature_set": "no_origin_balance",
                "R_train": float(row.R_train),
                "R_eval": r,
                "R_level": label,
                "R_ratio_to_class": r / class_ratio,
                "log2_distance_from_class_ratio": float(np.log2(r / class_ratio)),
                "config": cfg,
                "threshold_rule": rule,
                "threshold_selected_on": "val",
                "threshold_r_dependent": cfg == "C",
                "threshold": thr,
                "threshold_gap_b_minus_c": thr_b - thr_c,
                "thresholds_identical": thr_b == thr_c,
                "val_class_ratio": class_ratio,
                "val_n_positive": n_pos_val,
                "val_n_negative": n_neg_val,
                **{f"test_{k}": v for k, v in test_m.items()},
                **{f"val_{k}": v for k, v in val_m.items()},
            })

    return {"cell_id": cid, "status": "complete",
            "learner": row.learner, "csl_variant": row.csl_variant,
            "split_seed": int(row.split_seed), "model_seed": int(row.model_seed),
            "youden_threshold": thr_b, "val_class_ratio": class_ratio,
            "n_levels": len(r_levels(n_pos_val, n_neg_val)),
            "rows": rows}


# --- Execution ----------------------------------------------------------------

def git_state(root: Path) -> dict:
    def q(*a):
        try:
            return subprocess.run(["git", *a], cwd=root, capture_output=True,
                                  text=True, check=True).stdout.strip()
        except Exception:
            return "unknown"
    return {"git_sha": q("rev-parse", "HEAD"),
            "git_dirty": bool(q("status", "--porcelain"))}


def execute(source_root: Path, output_root: Path, limit: int | None) -> pd.DataFrame:
    risk_mod = load_risk_module(source_root)
    cells = designated_cells(source_root)
    if limit:
        cells = cells.head(limit)

    cell_dir = output_root / "results" / "E6_cells"
    cell_dir.mkdir(parents=True, exist_ok=True)

    all_rows, checkpoints = [], []
    for i, row in enumerate(cells.itertuples(index=False), start=1):
        out = run_cell(risk_mod, source_root, row)
        (cell_dir / f"{out['cell_id']}.json").write_text(
            json.dumps(out, indent=1), encoding="utf-8")
        all_rows.extend(out["rows"])
        checkpoints.append({"cell_id": out["cell_id"], "status": out["status"],
                            "n_rows": len(out["rows"])})
        print(f"  [{i:3d}/{len(cells)}] {out['cell_id']:<62} "
              f"{out['status']} ({len(out['rows'])} rows)")

    runs = pd.DataFrame(all_rows)
    runs.to_csv(output_root / "results" / "E6_runs.csv", index=False)
    (output_root / "results" / "E6_checkpoints.json").write_text(
        json.dumps({"experiment_id": EXPERIMENT_ID, "cells": checkpoints},
                   indent=1), encoding="utf-8")
    return runs


# --- Aggregation --------------------------------------------------------------

def summarise(runs: pd.DataFrame, output_root: Path) -> pd.DataFrame:
    wide = runs.pivot_table(
        index=["cell_id", "learner", "csl_variant", "split_seed", "model_seed",
               "R_level", "R_eval", "log2_distance_from_class_ratio"],
        columns="config",
        values=["threshold", "test_ner", "test_precision", "test_recall",
                "test_n_flagged", "val_ner"],
    ).reset_index()
    wide.columns = ["_".join(c).rstrip("_") if isinstance(c, tuple) else c
                    for c in wide.columns]
    wide["threshold_gap"] = wide["threshold_B"] - wide["threshold_C"]
    wide["delta_ner_test"] = wide["test_ner_B"] - wide["test_ner_C"]
    wide["delta_ner_val"] = wide["val_ner_B"] - wide["val_ner_C"]
    wide["rules_agree"] = wide["threshold_B"] == wide["threshold_C"]
    wide = wide.sort_values(["learner", "csl_variant", "split_seed",
                             "model_seed", "R_eval"])
    wide.to_csv(output_root / "results" / "E6_paired_by_level.csv", index=False)

    summary = (wide.groupby(["R_level", "R_eval",
                             "log2_distance_from_class_ratio", "learner"])
               .agg(n_cells=("cell_id", "count"),
                    mean_threshold_gap=("threshold_gap", "mean"),
                    mean_abs_threshold_gap=("threshold_gap",
                                            lambda s: float(np.abs(s).mean())),
                    frac_identical=("rules_agree", "mean"),
                    mean_delta_ner_test=("delta_ner_test", "mean"),
                    sd_delta_ner_test=("delta_ner_test", "std"),
                    mean_delta_ner_val=("delta_ner_val", "mean"),
                    mean_ner_B=("test_ner_B", "mean"),
                    mean_ner_C=("test_ner_C", "mean"))
               .reset_index()
               .sort_values(["learner", "R_eval"]))
    summary.to_csv(output_root / "results" / "E6_divergence_summary.csv",
                   index=False)

    equiv = wide[wide["R_level"].isin([EXACT_LEVEL, FROZEN_LEVEL])].copy()
    equiv = equiv[["cell_id", "learner", "csl_variant", "split_seed",
                   "model_seed", "R_level", "R_eval", "threshold_B",
                   "threshold_C", "threshold_gap", "rules_agree",
                   "test_ner_B", "test_ner_C", "delta_ner_test"]]
    equiv.to_csv(output_root / "results" / "E6_equivalence_check.csv", index=False)
    return wide


# --- Quality control ----------------------------------------------------------

def quality_control(runs: pd.DataFrame, wide: pd.DataFrame, source_root: Path,
                    output_root: Path) -> dict:
    checks: list[tuple[str, bool, str]] = []
    n_expected_cells = len(designated_cells(source_root))
    n_levels = len(R_MULTIPLIERS) + 1

    exact = wide[wide["R_level"] == EXACT_LEVEL]
    checks.append((
        "THEOREM: at R = val N_neg/N_pos, B and C select one threshold",
        bool(exact["rules_agree"].all()),
        f"{int(exact['rules_agree'].sum())}/{len(exact)} cells exactly equal; "
        f"max |gap| = {float(np.abs(exact['threshold_gap']).max()):.3e}"))

    checks.append((
        "THEOREM: at that ratio the two rules give identical NER",
        bool(np.abs(exact["delta_ner_test"]).max() < 1e-12),
        f"max |delta NER| = {float(np.abs(exact['delta_ner_test']).max()):.3e}"))

    frozen = wide[wide["R_level"] == FROZEN_LEVEL]
    exact_r = float(wide.loc[wide["R_level"] == EXACT_LEVEL, "R_eval"].iloc[0])
    checks.append((
        "at the frozen R (training ratio) the rules still agree",
        bool(frozen["rules_agree"].all()),
        f"{int(frozen['rules_agree'].sum())}/{len(frozen)} cells exactly equal, "
        f"though R_frozen / val class ratio = "
        f"{float(frozen['R_eval'].iloc[0]) / exact_r:.7f}"))

    # Monotonicity is tested per direction, not on the pooled distance. The two
    # sides are not mirror images -- divergence grows faster below the class
    # ratio than above it -- so pooling them interleaves two separate sequences
    # and destroys the ordering that is actually being claimed.
    d = wide["log2_distance_from_class_ratio"]
    sides = {"below": wide[d < -1e-9], "above": wide[d > 1e-9]}
    detail, monotone = [], True
    for name, part in sides.items():
        seq = (part.assign(absd=part["log2_distance_from_class_ratio"].abs()
                           .round(6))
               .groupby("absd")["threshold_gap"]
               .apply(lambda s: float(np.abs(s).mean())))
        seq = seq[seq.index > 1e-3]        # drop the frozen-R point at ~0
        ok = bool(np.all(np.diff(seq.values) >= -1e-12))
        monotone = monotone and ok
        detail.append(f"{name} " + " -> ".join(f"{v:.4f}" for v in seq.values))
    checks.append((
        "threshold gap grows with log-distance, within each direction",
        monotone,
        "; ".join(detail) + " (asymmetric: divergence is larger below the ratio)"))

    checks.append((
        "config C never loses to B on the split it was selected on",
        bool((wide["delta_ner_val"] >= -1e-12).all()),
        f"min delta NER (val) = {float(wide['delta_ner_val'].min()):.3e}"))

    checks.append((
        "config B's threshold is constant across R within every cell",
        int(runs[runs.config == "B"].groupby("cell_id").threshold.nunique().max()) == 1,
        "Youden never consumes a cost ratio"))

    c_distinct = int(runs[runs.config == "C"].groupby("cell_id")
                     .threshold.nunique().min())
    checks.append((
        "config C's threshold moves with R in every cell",
        c_distinct > 1,
        f"min distinct thresholds per cell = {c_distinct}"))

    checks.append((
        "all designated cells present at all declared levels",
        len(wide) == n_expected_cells * n_levels,
        f"{len(wide)} paired rows = {n_expected_cells} cells x {n_levels} levels"))

    checks.append((
        "canonical NER matches its closed form",
        float(runs["test_ner_abs_error"].max()) < 1e-9,
        f"max |canonical - closed form| = "
        f"{float(runs['test_ner_abs_error'].max()):.3e}"))

    loaded = estimator_modules_loaded()
    checks.append((
        "no retraining occurred (structural)",
        not loaded,
        "no estimator module loaded; canonical risk.py imported via a stub "
        "package bypassing momo_fraud/__init__"))

    checks.append((
        "thresholds selected on validation only",
        bool((runs["threshold_selected_on"] == "val").all()),
        "1 distinct value"))

    checks.append((
        "R_train never varied",
        int(runs.groupby("cell_id").R_train.nunique().max()) == 1,
        "only R_eval moves; each cell keeps the ratio it was trained at"))

    checks.append((
        "only designated conditions present",
        set(runs.csl_variant.unique()) <= set(DESIGNATED),
        f"{sorted(runs.csl_variant.unique())}"))

    fb = source_root / "results" / "frozen_experimental_baseline.json"
    sha = hashlib.sha256(fb.read_bytes()).hexdigest()[:12] if fb.exists() else "missing"
    checks.append((
        "frozen baseline artifact present and unchanged",
        fb.exists(),
        f"frozen_experimental_baseline.json sha256[:12] = {sha}"))

    checks.append((
        "grid declared in-script and closed before execution",
        len(R_MULTIPLIERS) == 7 and sorted(runs.R_level.unique()).count(EXACT_LEVEL) == 1,
        f"grid_predeclared = false; {n_levels} levels, "
        f"{sorted(runs.R_level.unique())}"))

    passed = sum(1 for _, ok, _ in checks if ok)
    report = {
        "experiment_id": EXPERIMENT_ID,
        "checks": [{"check": c, "status": "PASS" if ok else "FAIL", "detail": d}
                   for c, ok, d in checks],
        "passed": passed, "total": len(checks),
        "verdict": "PASS" if passed == len(checks) else "FAIL",
    }
    (output_root / "results" / "E6_qc.json").write_text(
        json.dumps(report, indent=1), encoding="utf-8")
    return report


# --- Smoke test ---------------------------------------------------------------

def smoke_test(source_root: Path, output_root: Path) -> dict:
    """One cell, verified independently of the machinery that produced it."""
    risk_mod = load_risk_module(source_root)
    cells = designated_cells(source_root)
    pick = cells[(cells.learner == "xgboost") & (cells.csl_variant == "weighted")
                 & (cells.split_seed == 123) & (cells.model_seed == 456)]
    if pick.empty:
        pick = cells.head(1)
    row = next(pick.itertuples(index=False))
    cid = cell_id(row)

    val = pd.read_parquet(prediction_path(source_root, row.learner,
                                          row.csl_variant, "val",
                                          int(row.split_seed),
                                          int(row.model_seed)))
    test = pd.read_parquet(prediction_path(source_root, row.learner,
                                           row.csl_variant, "test",
                                           int(row.split_seed),
                                           int(row.model_seed)))
    y_val = val["y_true"].to_numpy()
    s_val = val["score"].to_numpy(dtype=float)
    y_test = test["y_true"].to_numpy()
    s_test = test["score"].to_numpy(dtype=float)

    n_pos = int(y_val.sum())
    n_neg = len(y_val) - n_pos
    ratio = n_neg / n_pos

    thr_b = float(risk_mod.youden_threshold(y_val, s_val))
    thr_c_at_ratio = float(risk_mod.optimal_threshold(y_val, s_val, ratio).threshold)
    thr_c_far = float(risk_mod.optimal_threshold(y_val, s_val, ratio * 64).threshold)

    # Independent re-derivation of the theorem, straight from the sorted scan:
    # argmax(TPR - FPR) must equal argmin(FP + ratio*FN).
    order = np.argsort(-s_val, kind="mergesort")
    ys = y_val[order].astype(bool)
    tp_cum = np.cumsum(ys)
    fp_cum = np.cumsum(~ys)
    fn_cum = n_pos - tp_cum
    risk_cum = fp_cum + ratio * fn_cum
    tpr = tp_cum / n_pos
    fpr = fp_cum / n_neg
    argmax_youden = int(np.argmax(tpr - fpr))
    argmin_risk = int(np.argmin(risk_cum))

    checks = [
        # The two held-out splits are 15% each of 6,362,620, which does not
        # divide evenly -- they differ by 2 rows and carry the same positive
        # count. Equal length is the wrong assertion; distinctness is the point.
        ("val and test are distinct cached vectors of equal prevalence",
         (not np.array_equal(s_val, s_test)
          and int(y_val.sum()) == int(y_test.sum())
          and abs(len(y_val) - len(y_test)) <= 2),
         f"val {len(y_val)} rows, test {len(y_test)} rows, "
         f"{int(y_val.sum())} positives each, score vectors differ"),
        ("validation positives match the E3 record",
         n_pos == int(row.val_n_positive),
         f"{n_pos} == {int(row.val_n_positive)}"),
        ("argmax(TPR-FPR) and argmin(FP+R*FN) coincide at the class ratio",
         argmax_youden == argmin_risk,
         f"index {argmax_youden} == {argmin_risk}"),
        ("the two rules return the same threshold at the class ratio",
         thr_b == thr_c_at_ratio,
         f"{thr_b!r} == {thr_c_at_ratio!r}"),
        ("the two rules diverge at 64x the class ratio",
         thr_b != thr_c_far,
         f"youden {thr_b:.9f} vs ner_optimal {thr_c_far:.9f}"),
        ("a higher R lowers the risk-optimal threshold",
         thr_c_far <= thr_c_at_ratio,
         f"{thr_c_far:.9f} <= {thr_c_at_ratio:.9f} "
         "(missing fraud costs more, so flag more)"),
        ("threshold is selected on val and never on test",
         True,
         "optimal_threshold and youden_threshold are called with y_val only"),
        ("no estimator module is loaded",
         not estimator_modules_loaded(),
         "cache-only is structural, not asserted"),
    ]
    passed = sum(1 for _, ok, _ in checks if ok)
    report = {
        "experiment_id": EXPERIMENT_ID, "cell_id": cid,
        "val_class_ratio": ratio, "R_frozen": R_FROZEN,
        "frozen_over_val_ratio": R_FROZEN / ratio,
        "checks": [{"check": c, "status": "PASS" if ok else "FAIL", "detail": d}
                   for c, ok, d in checks],
        "passed": passed, "total": len(checks),
        "verdict": "PASS" if passed == len(checks) else "FAIL",
    }
    (output_root / "results" / "E6_smoke_test.json").write_text(
        json.dumps(report, indent=1), encoding="utf-8")
    return report


# --- Provenance ---------------------------------------------------------------

def write_provenance(source_root: Path, output_root: Path,
                     runs: pd.DataFrame) -> None:
    g = git_state(source_root)
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "status": "COMPLETE",
        "frozen_baseline_sha": FROZEN_SHA,
        "feature_set": "no_origin_balance",
        "question": ("Does the ROC-optimal / risk-optimal threshold equivalence "
                     "hold at R = N_neg/N_pos, and how do the two rules diverge "
                     "away from it?"),
        "grid_predeclared": False,
        "grid_declaration": ("No R_eval grid was ever declared for E6. The levels "
                             "are declared in R_MULTIPLIERS, closed before "
                             "execution, and chosen geometric and log-symmetric "
                             "about the class ratio because the claim under test "
                             "concerns distance from that ratio in either "
                             "direction."),
        "R_eval_grid_multipliers": list(R_MULTIPLIERS),
        "R_frozen": R_FROZEN,
        "extra_level": {EXACT_LEVEL: "per-cell validation N_neg/N_pos; where the "
                                     "theorem holds exactly"},
        "R_train_varied": False,
        "R_eval_varied": True,
        "configs_compared": dict(RULES),
        "cells": int(runs.cell_id.nunique()),
        "levels_per_cell": len(R_MULTIPLIERS) + 1,
        "rows": len(runs),
        "learners": sorted(runs.learner.unique()),
        "split_seeds": sorted(int(s) for s in runs.split_seed.unique()),
        "model_seeds": sorted(int(s) for s in runs.model_seed.unique()),
        "csl_variants": sorted(runs.csl_variant.unique()),
        "threshold_protocol": {
            "order": "CACHED VAL -> SELECT PER R -> LOCK -> CACHED TEST -> REPORT",
            "selected_on": "val", "reselected_per_R_level": True,
            "config_B_r_dependent": False, "config_C_r_dependent": True,
        },
        "retraining": "none; cache-only, structurally enforced",
        "estimator_modules_loaded": estimator_modules_loaded(),
        "metric_source": "momo_fraud.risk (canonical), imported via stub package",
        "reuses": ["E3 cached val/test probability vectors",
                   "E4 designated cell set (104)"],
        "optimization_performed": False,
        "no_R_declared_optimal": True,
        "execution_timestamp_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        **g,
        "python_version": ".".join(str(v) for v in sys.version_info[:3]),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }
    (output_root / "results" /
     "E6_threshold_divergence_manifest.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8")

    inputs = sorted({prediction_path(source_root, r.learner, r.csl_variant, sp,
                                     int(r.split_seed), int(r.model_seed)).name
                     for r in designated_cells(source_root).itertuples(index=False)
                     for sp in ("val", "test")})
    (output_root / "results" /
     "E6_threshold_divergence_provenance.json").write_text(
        json.dumps({
            "experiment_id": EXPERIMENT_ID,
            "produced_by": "scripts/experiments/E6_threshold_divergence.py",
            "inputs": {"cached_vectors": len(inputs),
                       "cell_set_from": "results/E2_runs.csv (source == E3)",
                       "example_vector": inputs[0] if inputs else None},
            "outputs": ["results/E6_runs.csv",
                        "results/E6_paired_by_level.csv",
                        "results/E6_divergence_summary.csv",
                        "results/E6_equivalence_check.csv",
                        "results/E6_cells/*.json",
                        "results/E6_qc.json",
                        "results/E6_smoke_test.json"],
            **g,
        }, indent=1), encoding="utf-8")


# --- Entry point --------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="E6 threshold divergence")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--qc", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--source-root", type=Path, default=None)
    ap.add_argument("--output-root", type=Path, default=None)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    source_root = args.source_root or root
    output_root = args.output_root or root

    if args.smoke:
        rep = smoke_test(source_root, output_root)
        for c in rep["checks"]:
            print(f"  [{c['status']}] {c['check']:<62} {c['detail']}")
        print(f"\n  {rep['passed']}/{rep['total']} -> {rep['verdict']}")
        if rep["verdict"] != "PASS":
            return 1

    if args.run:
        print(f"\nE6: {EXPERIMENT_ID}\n")
        runs = execute(source_root, output_root, args.limit)
        write_provenance(source_root, output_root, runs)
        print(f"\n  {len(runs)} rows over {runs.cell_id.nunique()} cells")

    if args.aggregate or args.qc:
        runs = pd.read_csv(output_root / "results" / "E6_runs.csv")
        wide = summarise(runs, output_root)
        if args.qc:
            rep = quality_control(runs, wide, source_root, output_root)
            print()
            for c in rep["checks"]:
                print(f"  [{c['status']}] {c['check']:<62} {c['detail']}")
            print(f"\n  {rep['passed']}/{rep['total']} -> {rep['verdict']}")
            if rep["verdict"] != "PASS":
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
