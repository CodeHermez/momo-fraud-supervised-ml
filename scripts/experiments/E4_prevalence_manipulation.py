"""E4 -- Arm D: prevalence-only manipulation.

Research question: how much of the precision collapse reported in the Lokanan
replication is attributable to *evaluated prevalence alone*, holding the model,
the operating point and the real positives fixed?

Notebook 05 measured that collapse at two points -- a balanced evaluation
(``arm_a``, precision 0.97-0.9995) and the natural 0.129% prevalence
(``arm_b``, precision 0.031-0.404). Two points cannot separate "the model is
worse" from "the arithmetic of precision changed". E4 converts that two-point
contrast into a measured curve.

**Design.** One model per cell, one threshold per cell, all real positives
retained, real negatives subsampled without replacement. No synthetic points,
no ENN, no resampling of positives, and -- structurally -- no retraining: this
script never imports a model library. It reads the probability vectors E3
already cached and re-evaluates them on subsampled negative sets.

**Why no refit.** Refitting per prevalence level would confound the evaluation
effect with a training effect, which is the exact error the arm exists to
isolate. The model and its threshold are frozen across every level.

**The grid.** The audit pre-declared "real negatives subsampled to
{50%, 10%, 1%, 0.129%}". That phrasing admits two readings -- target evaluated
prevalence, or retained fraction of negatives -- and they give different grids.
Rather than guess, the pre-declared grid below is the **union**, so both
readings are reported in full. Levels are closed and fixed before execution;
none is chosen after seeing results.

============  ==================  =======================  ==========
target        negative retention  realises                 reading
============  ==================  =======================  ==========
natural        1.0                0.1291% (control)        A
0.2581%        0.5                                         B
1%             ~0.128790                                   A
1.2765%        0.1                                         B
10%            ~0.011620                                   A
11.32%         0.01                                        B
50%            ~0.00129087        balanced                 A and B
============  ==================  =======================  ==========

Retention fractions are recomputed exactly per split from that split's own
positive and negative counts; the table above is illustrative.

**Cells.** The two designated reference conditions carried through E2 -- the
unweighted reference and class weighting at the frozen default R = 773.7011 --
for all four learners, across 4 split seeds and 4 model seeds (1 for logistic
regression, which is deterministic in the model seed). **104 cells**, every one
of them backed by a cached test vector. Nothing is refitted.

**Replicates.** 100 subsample replicates per cell per stochastic level, drawn
from a per-cell permutation of the negatives. Levels within a replicate are
**nested** -- the smaller subsample is a subset of the larger -- which pairs the
levels and removes sampling noise from between-level comparisons. Each level's
marginal is still a uniform random subset of the correct size. The natural level
involves no subsampling and is therefore deterministic: 1 replicate.

**What must not move.** Positives, the threshold, and therefore TP, FN and
recall are invariant across every level by construction. The QC pass asserts
this exactly rather than trusting it.

Usage::

    python scripts/experiments/E4_prevalence_manipulation.py --smoke
    python scripts/experiments/E4_prevalence_manipulation.py --run
    python scripts/experiments/E4_prevalence_manipulation.py --aggregate --qc
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

EXPERIMENT_ID = "E4_prevalence_manipulation"
FROZEN_SHA = "b2e6f624"

#: Frozen evaluation cost ratio. Never swept here -- that is E6's question.
R_EVAL = 773.7010836478753

#: Pre-declared, closed. Not added to, not removed from, not chosen after
#: seeing results. ``None`` means "natural prevalence", i.e. retain every
#: negative; it is the no-op control.
PREVALENCE_GRID: list[tuple[str, float | None, float | None]] = [
    # (label, target_prevalence, target_negative_retention)
    ("natural", None, 1.0),
    ("prev_0.2581pct", None, 0.5),
    ("prev_1pct", 0.01, None),
    ("prev_1.2765pct", None, 0.1),
    ("prev_10pct", 0.10, None),
    ("prev_11.32pct", None, 0.01),
    ("prev_50pct", 0.50, None),
]

N_REPLICATES = 100
RNG_BASE = 20260830

DESIGNATED = {
    "unweighted": 1.0,
    "weighted": R_EVAL,
}


# --- Cached-prediction plumbing ---------------------------------------------

def prediction_path(source_root: Path, learner: str, variant: str,
                    split_seed: int, model_seed: int) -> Path:
    """Locate one cached test vector.

    Note the seed order: E3 saved these with ``seed=model_seed`` and
    ``tag=f"{rung}__s{split_seed}"`` (``E3_multiseed_multisplit.py:190-194``),
    so the ``seed`` field carries the **model** seed and the trailing ``s``
    field the **split** seed. Transposing them silently loads a real cell that
    is the wrong one -- it only shows up off the split == model diagonal.
    """
    return (source_root / "predictions" /
            f"{learner}__{variant}__test__seed{model_seed}"
            f"__no_origin_balance__s{split_seed}.parquet")


def designated_cells(source_root: Path) -> pd.DataFrame:
    """The 104 designated cells, with each cell's locked threshold.

    Read from ``E2_runs.csv`` rather than recomputed: the threshold is the one
    E3 selected on validation, and re-deriving it here would risk selecting a
    different operating point than the one the study actually reports.
    """
    runs = pd.read_csv(source_root / "results" / "E2_runs.csv")
    cells = runs[runs["source"] == "E3"].copy()
    cells = cells[[
        "learner", "csl_variant", "split_seed", "model_seed", "R_train",
        "threshold", "pr_auc", "ner", "precision", "recall",
        "tp", "fp", "fn", "tn", "test_n_positive", "test_n_negative",
    ]]
    return cells.reset_index(drop=True)


# --- The evaluation core ------------------------------------------------------

class CellScores:
    """One cell's cached test vector, prepared once for many subsamples.

    Sorting and grouping are done a single time per cell. Each replicate then
    costs one pass over the negatives rather than a re-sort, which is what makes
    98 cells x 7 levels x 100 replicates tractable.
    """

    def __init__(self, y_true: np.ndarray, score: np.ndarray, threshold: float):
        order = np.argsort(-score.astype(np.float64), kind="stable")
        self.y = y_true[order].astype(np.int8)
        self.s = score[order].astype(np.float64)
        self.threshold = float(threshold)

        self.pos_pos = np.flatnonzero(self.y == 1)
        self.neg_pos = np.flatnonzero(self.y == 0)
        self.n_pos = int(self.pos_pos.size)
        self.n_neg = int(self.neg_pos.size)

        # Distinct-score groups. Average precision only changes at groups that
        # contain at least one positive, so only those are retained.
        new_group = np.empty(self.s.size, dtype=bool)
        new_group[0] = True
        np.not_equal(self.s[1:], self.s[:-1], out=new_group[1:])
        group_id = np.cumsum(new_group) - 1
        n_groups = int(group_id[-1]) + 1

        last_index = np.zeros(n_groups, dtype=np.int64)
        last_index[group_id] = np.arange(self.s.size)

        pos_groups = np.unique(group_id[self.pos_pos])
        cuts = last_index[pos_groups]

        # TP through the end of each positive-bearing group: fixed forever.
        self.tp_at_cut = np.searchsorted(self.pos_pos, cuts, side="right")
        # How many negatives sit at or before each cut: fixed forever.
        self.neg_before_cut = np.searchsorted(self.neg_pos, cuts, side="right")

        # Negatives at or above the locked threshold, in position order.
        self.n_neg_above = int(np.searchsorted(-self.s[self.neg_pos],
                                               -self.threshold, side="right"))
        self.tp_locked = int(np.count_nonzero(self.s[self.pos_pos] >= self.threshold))
        self.fn_locked = self.n_pos - self.tp_locked

    def _fp_cumulative(self, retained_mask: np.ndarray) -> np.ndarray:
        """Cumulative retained-negative count, indexed by negative position."""
        return np.concatenate(([0], np.cumsum(retained_mask)))

    def evaluate(self, retained_mask: np.ndarray) -> dict[str, float]:
        """Metrics for one subsample. ``retained_mask`` is over negatives."""
        csum = self._fp_cumulative(retained_mask)
        m = int(csum[-1])

        # --- threshold-free: average precision, tie-aware and exact ---
        fp_at_cut = csum[self.neg_before_cut]
        denom = self.tp_at_cut + fp_at_cut
        precision_k = np.divide(self.tp_at_cut, denom,
                                out=np.zeros_like(denom, dtype=float),
                                where=denom > 0)
        recall_k = self.tp_at_cut / self.n_pos
        delta_recall = np.diff(recall_k, prepend=0.0)
        pr_auc = float(np.sum(delta_recall * precision_k))

        # --- at the locked threshold ---
        fp = int(csum[self.n_neg_above])
        tp, fn = self.tp_locked, self.fn_locked
        tn = m - fp
        flagged = tp + fp
        total = self.n_pos + m
        precision = tp / flagged if flagged else 0.0
        recall = tp / self.n_pos
        total_risk = fp + R_EVAL * fn
        return {
            "n_eval": total,
            "n_negative_retained": m,
            "prevalence": self.n_pos / total,
            "pr_auc": pr_auc,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": (2 * precision * recall / (precision + recall)
                   if (precision + recall) else 0.0),
            "n_flagged": flagged,
            "alert_rate": flagged / total,
            "total_risk": total_risk,
            "ner": total_risk / (R_EVAL * self.n_pos),
        }


def retention_for(level: tuple[str, float | None, float | None],
                  n_pos: int, n_neg: int) -> tuple[int, float]:
    """Number of negatives to retain, and the retention fraction."""
    label, target_prev, target_ret = level
    if target_ret is not None:
        m = int(round(target_ret * n_neg))
    else:
        # prevalence p = n_pos / (n_pos + m)  ->  m = n_pos * (1 - p) / p
        m = int(round(n_pos * (1.0 - target_prev) / target_prev))
    m = max(1, min(m, n_neg))
    return m, m / n_neg


def run_cell(scores: CellScores, rng: np.random.Generator,
             n_replicates: int) -> list[dict]:
    """Every level for one cell, using nested subsamples per replicate."""
    sizes = [retention_for(lv, scores.n_pos, scores.n_neg)
             for lv in PREVALENCE_GRID]
    rows = []
    for rep in range(n_replicates):
        # One permutation per replicate; each level takes a prefix of it, so
        # smaller subsamples nest inside larger ones.
        rank = np.empty(scores.n_neg, dtype=np.int64)
        rank[rng.permutation(scores.n_neg)] = np.arange(scores.n_neg)
        for (label, _, _), (m, ret) in zip(PREVALENCE_GRID, sizes):
            deterministic = m == scores.n_neg
            if deterministic and rep > 0:
                continue
            mask = rank < m
            out = scores.evaluate(mask)
            out.update(level=label, replicate=rep,
                       negative_retention=ret,
                       n_negative_target=m,
                       deterministic_level=deterministic)
            rows.append(out)
    return rows


# --- Orchestration ------------------------------------------------------------

def git_state(root: Path) -> dict:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                      text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"],
                                             cwd=root, text=True).strip())
    except Exception:
        sha, dirty = None, None
    return {"git_sha": sha, "git_dirty": dirty}


def execute(source_root: Path, output_root: Path, n_replicates: int,
            limit: int | None = None) -> pd.DataFrame:
    cells = designated_cells(source_root)
    records, skipped = [], []

    for i, cell in cells.iterrows():
        if limit is not None and len(records) and i >= limit:
            break
        path = prediction_path(source_root, cell.learner, cell.csl_variant,
                               int(cell.split_seed), int(cell.model_seed))
        if not path.exists():
            skipped.append({
                "learner": cell.learner, "csl_variant": cell.csl_variant,
                "split_seed": int(cell.split_seed),
                "model_seed": int(cell.model_seed),
                "reason": "no cached test predictions", "path": str(path.name),
            })
            continue

        frame = pd.read_parquet(path, engine="pyarrow")
        scores = CellScores(frame["y_true"].to_numpy(),
                            frame["score"].to_numpy(),
                            float(cell.threshold))
        seed = (RNG_BASE + 1000 * int(cell.split_seed) + int(cell.model_seed)
                + hash(cell.learner + cell.csl_variant) % 9973)
        rng = np.random.default_rng(seed)

        for row in run_cell(scores, rng, n_replicates):
            row.update(
                experiment_id=EXPERIMENT_ID,
                learner=cell.learner,
                csl_variant=cell.csl_variant,
                R_train=float(cell.R_train),
                R_eval=R_EVAL,
                split_seed=int(cell.split_seed),
                model_seed=int(cell.model_seed),
                threshold=float(cell.threshold),
                threshold_source="E3 validation-selected, locked",
                rng_seed=seed,
                stored_pr_auc=float(cell.pr_auc),
                stored_ner=float(cell.ner),
                stored_precision=float(cell.precision),
                stored_recall=float(cell.recall),
                stored_fp=int(cell.fp),
            )
            records.append(row)
        print(f"  [{i + 1}/{len(cells)}] {cell.learner} {cell.csl_variant} "
              f"split{cell.split_seed} model{cell.model_seed}", flush=True)

    runs = pd.DataFrame(records)
    output_root.mkdir(parents=True, exist_ok=True)
    # gzip: the per-replicate record is ~28 MB raw, ~3.5 MB compressed.
    runs.to_csv(output_root / "E4_runs.csv.gz", index=False,
                compression="gzip")
    pd.DataFrame(skipped).to_csv(output_root / "E4_skipped_cells.csv", index=False)
    print(f"\nwrote {len(runs)} rows; {len(skipped)} cells skipped")
    return runs


def summarise(runs: pd.DataFrame, output_root: Path) -> pd.DataFrame:
    keys = ["learner", "csl_variant", "level"]
    agg = runs.groupby(keys).agg(
        n_cells=("split_seed", lambda s: s.astype(str).nunique()),
        n_rows=("pr_auc", "size"),
        prevalence_mean=("prevalence", "mean"),
        negative_retention=("negative_retention", "mean"),
        pr_auc_mean=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"),
        pr_auc_min=("pr_auc", "min"), pr_auc_max=("pr_auc", "max"),
        precision_mean=("precision", "mean"), precision_std=("precision", "std"),
        precision_min=("precision", "min"), precision_max=("precision", "max"),
        recall_mean=("recall", "mean"), recall_std=("recall", "std"),
        ner_mean=("ner", "mean"), ner_std=("ner", "std"),
        ner_min=("ner", "min"), ner_max=("ner", "max"),
        alert_rate_mean=("alert_rate", "mean"),
    ).reset_index()
    agg.to_csv(output_root / "E4_prevalence_summary.csv", index=False)
    return agg


def analytic_check(runs: pd.DataFrame, output_root: Path) -> pd.DataFrame:
    """Simulated precision against its closed form.

    Under uniform negative subsampling at rate ``r`` with the threshold fixed,
    E[FP] = r * FP_full and TP is unchanged, so precision should converge to
    TP / (TP + r * FP_full). Agreement is evidence the manipulation did what it
    claims; disagreement would mean the subsample is not uniform.
    """
    rows = []
    keys = ["learner", "csl_variant", "split_seed", "model_seed", "level"]
    for key, g in runs.groupby(keys):
        tp = float(g.tp.iloc[0])
        fp_full = float(g.stored_fp.iloc[0])
        r = float(g.negative_retention.iloc[0])
        expected = tp / (tp + r * fp_full) if (tp + r * fp_full) > 0 else 0.0
        observed = float(g.precision.mean())
        rows.append(dict(zip(keys, key)) | {
            "negative_retention": r,
            "precision_expected": expected,
            "precision_observed": observed,
            "abs_error": abs(expected - observed),
        })
    out = pd.DataFrame(rows)
    out.to_csv(output_root / "E4_analytic_check.csv", index=False)
    return out


def quality_control(runs: pd.DataFrame, source_root: Path,
                    output_root: Path) -> dict:
    checks: list[tuple[str, bool, str]] = []
    cell_keys = ["learner", "csl_variant", "split_seed", "model_seed"]

    levels = list(runs.level.unique())
    checks.append((
        "all pre-declared prevalence levels present",
        set(levels) == {lv[0] for lv in PREVALENCE_GRID},
        f"{len(levels)} levels: {sorted(levels)}"))

    # Recall, TP and FN cannot move: positives and threshold are both fixed.
    bad_recall = bad_tp = bad_fn = 0
    for _, g in runs.groupby(cell_keys):
        bad_recall += int(g.recall.nunique() > 1)
        bad_tp += int(g.tp.nunique() > 1)
        bad_fn += int(g.fn.nunique() > 1)
    checks.append(("recall invariant across prevalence levels", bad_recall == 0,
                   f"{bad_recall} cells with a moving recall"))
    checks.append(("TP and FN invariant across prevalence levels",
                   bad_tp == 0 and bad_fn == 0,
                   f"{bad_tp} TP / {bad_fn} FN violations"))

    # Every positive is kept, at every level.
    n_pos = runs.tp + runs.fn
    checks.append(("all real positives retained at every level",
                   int(n_pos.nunique()) == 1,
                   f"{sorted(n_pos.unique())} positives per evaluation"))

    # The natural level must reproduce the stored E3 numbers exactly.
    nat = runs[runs.level == "natural"]
    d_pr = (nat.pr_auc - nat.stored_pr_auc).abs().max()
    d_ner = (nat.ner - nat.stored_ner).abs().max()
    d_prec = (nat.precision - nat.stored_precision).abs().max()
    checks.append((
        "natural level reproduces the stored E3 cell",
        bool(d_pr < 1e-9 and d_ner < 1e-9 and d_prec < 1e-9),
        f"max |diff| PR-AUC {d_pr:.2e}, NER {d_ner:.2e}, precision {d_prec:.2e}"))

    checks.append(("natural level is the no-op control",
                   bool((nat.negative_retention == 1.0).all()),
                   "retention = 1.0, 1 deterministic replicate"))

    # Achieved prevalence must match the target it was constructed from.
    worst = 0.0
    for lv, target_prev, _ in PREVALENCE_GRID:
        if target_prev is None:
            continue
        got = runs[runs.level == lv].prevalence
        worst = max(worst, float((got - target_prev).abs().max()))
    checks.append(("achieved prevalence matches the target", worst < 1e-4,
                   f"max |achieved - target| = {worst:.2e}"))

    # Simulation against the closed form.
    ac = analytic_check(runs, output_root)
    checks.append(("observed precision matches its closed form",
                   float(ac.abs_error.max()) < 5e-3,
                   f"max |observed - expected| = {float(ac.abs_error.max()):.2e}"))

    checks.append(("no retraining occurred", True,
                   "cache-only; the script imports no model library"))
    checks.append(("threshold locked, never re-selected",
                   int(runs.groupby(cell_keys).threshold.nunique().max()) == 1,
                   "one threshold per cell across all levels"))
    checks.append(("R_eval fixed at the frozen default",
                   bool((runs.R_eval == R_EVAL).all()),
                   f"{R_EVAL:.4f}, 1 distinct value"))
    checks.append(("only designated conditions present",
                   set(runs.csl_variant.unique()) <= set(DESIGNATED),
                   str(sorted(runs.csl_variant.unique()))))

    # An empty frame writes a headerless file, which read_csv refuses to parse.
    try:
        skipped = pd.read_csv(output_root / "E4_skipped_cells.csv")
    except pd.errors.EmptyDataError:
        skipped = pd.DataFrame()
    n_cells = runs.groupby(cell_keys).ngroups
    checks.append(("every designated cell evaluated", len(skipped) == 0,
                   f"{n_cells} cells, {len(skipped)} skipped"))
    checks.append(("cell coverage recorded", True,
                   f"{n_cells} cells evaluated, {len(skipped)} skipped "
                   f"(no cached predictions)"))

    verdict = "PASS" if all(ok for _, ok, _ in checks) else "FAIL"
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "verdict": verdict,
        "n_cells": int(n_cells),
        "n_rows": int(len(runs)),
        "n_replicates": int(runs.replicate.max()) + 1,
        "checks": [{"check": c, "status": "PASS" if ok else "FAIL", "detail": d}
                   for c, ok, d in checks],
    }
    (output_root / "E4_qc.json").write_text(json.dumps(payload, indent=2))
    return payload


def smoke_test(source_root: Path, output_root: Path) -> dict:
    """One cell, checked against sklearn rather than against itself."""
    from sklearn.metrics import average_precision_score

    cells = designated_cells(source_root)
    cell = cells[(cells.learner == "xgboost") &
                 (cells.csl_variant == "weighted") &
                 (cells.split_seed == 42) &
                 (cells.model_seed == 42)].iloc[0]
    frame = pd.read_parquet(
        prediction_path(source_root, cell.learner, cell.csl_variant,
                        int(cell.split_seed), int(cell.model_seed)),
        engine="pyarrow")
    y = frame["y_true"].to_numpy()
    s = frame["score"].to_numpy()
    scores = CellScores(y, s, float(cell.threshold))

    checks = []
    full = scores.evaluate(np.ones(scores.n_neg, dtype=bool))
    sk_full = float(average_precision_score(y, s))
    checks.append(("fast AP equals sklearn on the full test set",
                   abs(full["pr_auc"] - sk_full) < 1e-12,
                   f"{full['pr_auc']:.12f} vs {sk_full:.12f}"))
    checks.append(("full-set PR-AUC equals the stored E3 value",
                   abs(full["pr_auc"] - float(cell.pr_auc)) < 1e-9,
                   f"{full['pr_auc']:.12f} vs {float(cell.pr_auc):.12f}"))

    rng = np.random.default_rng(0)
    rank = np.empty(scores.n_neg, dtype=np.int64)
    rank[rng.permutation(scores.n_neg)] = np.arange(scores.n_neg)
    m = scores.n_pos  # balanced
    mask = rank < m
    sub = scores.evaluate(mask)
    keep = np.concatenate([scores.pos_pos, scores.neg_pos[mask]])
    keep.sort()
    sk_sub = float(average_precision_score(scores.y[keep], scores.s[keep]))
    checks.append(("fast AP equals sklearn on a balanced subsample",
                   abs(sub["pr_auc"] - sk_sub) < 1e-12,
                   f"{sub['pr_auc']:.12f} vs {sk_sub:.12f}"))
    checks.append(("balanced subsample really is balanced",
                   abs(sub["prevalence"] - 0.5) < 1e-6,
                   f"prevalence {sub['prevalence']:.6f}"))
    checks.append(("recall unchanged by subsampling",
                   abs(sub["recall"] - full["recall"]) < 1e-12,
                   f"{full['recall']:.12f} -> {sub['recall']:.12f}"))
    checks.append(("every positive retained",
                   sub["tp"] + sub["fn"] == scores.n_pos,
                   f"{sub['tp'] + sub['fn']} of {scores.n_pos}"))
    checks.append(("precision rose while the model did not change",
                   sub["precision"] > full["precision"],
                   f"{full['precision']:.4f} -> {sub['precision']:.4f}"))

    verdict = "PASS" if all(ok for _, ok, _ in checks) else "FAIL"
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "cell": f"{cell.learner}__{cell.csl_variant}__split42__model42",
        "verdict": verdict,
        "checks": [{"check": c, "status": "PASS" if ok else "FAIL", "detail": d}
                   for c, ok, d in checks],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "E4_smoke_test.json").write_text(json.dumps(payload, indent=2))
    return payload


def write_provenance(source_root: Path, output_root: Path, runs: pd.DataFrame) -> None:
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "status": "COMPLETE",
        "frozen_baseline": FROZEN_SHA,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "design": {
            "manipulation": "test-set negatives subsampled without replacement",
            "positives": "all real positives retained; none synthesised",
            "model": "frozen per cell; no refitting at any level",
            "threshold": "E3 validation-selected, locked across all levels",
            "R_eval": R_EVAL,
            "prevalence_grid": [lv[0] for lv in PREVALENCE_GRID],
            "grid_rationale": (
                "union of both readings of the pre-declared "
                "{50%, 10%, 1%, 0.129%}: target prevalence and negative "
                "retention fraction"),
            "n_replicates": N_REPLICATES,
            "nested_subsamples": True,
        },
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "source_root": str(source_root),
        "n_rows": int(len(runs)),
    } | git_state(output_root)
    (output_root / "E4_prevalence_provenance.json").write_text(
        json.dumps(payload, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--qc", action="store_true")
    ap.add_argument("--replicates", type=int, default=N_REPLICATES)
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only the first N cells (development only)")
    ap.add_argument("--source-root", type=Path, default=None,
                    help="where cached predictions and E2_runs.csv live")
    ap.add_argument("--output-root", type=Path, default=None,
                    help="where E4_* artefacts are written")
    args = ap.parse_args()

    here = Path(__file__).resolve().parents[2]
    source_root = args.source_root or here
    output_root = args.output_root or (here / "results")

    if args.smoke:
        payload = smoke_test(source_root, output_root)
        print(json.dumps(payload, indent=2))
        if payload["verdict"] != "PASS":
            return 1

    runs = None
    if args.run:
        runs = execute(source_root, output_root, args.replicates, args.limit)

    if args.aggregate or args.qc:
        if runs is None:
            runs = pd.read_csv(output_root / "E4_runs.csv.gz")
        if args.aggregate:
            agg = summarise(runs, output_root)
            print(f"summary rows: {len(agg)}")
            write_provenance(source_root, output_root, runs)
        if args.qc:
            payload = quality_control(runs, source_root, output_root)
            print(json.dumps(payload, indent=2))
            if payload["verdict"] != "PASS":
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
