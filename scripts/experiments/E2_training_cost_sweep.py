"""E2 -- Training cost-ratio sweep.

Research question: how does the cost ratio used during **training** affect
ranking quality and decision risk, and can variation in training cost explain
the learner-dependent CSL behaviour E3 observed?

This closes the gap the frozen baseline names explicitly: every previous sweep
varied ``R_eval`` while holding ``R_train`` at 773.7011, and therefore tested the
decision layer rather than the training layer.

**Design.** ``R_train`` sweeps the six pre-declared values; ``R_eval`` is held at
the frozen default 773.7011 throughout the primary analysis. So the question is
the deployment-shaped one: *what happens when a model is trained under one cost
assumption and evaluated under a fixed, different one?*

The mechanism is **class weighting only**. RUS changes the composition of the
training sample rather than applying an explicit class-cost ratio, so sweeping
``R_train`` through it would not mean the same thing; E3's RUS cells remain
available as a fixed reference and are never described as part of this sweep.

**Threshold.** Re-selected independently for every cell. A different ``R_train``
produces a different fitted model, so the frozen protocol requires a fresh
validation threshold for it. Holding a threshold fixed across R values would make
the curves cleaner and the experiment wrong.

**Reuse.** Two of the analysis columns already exist from E3 under conditions
identical to E2's -- the unweighted reference (52 cells) and class weighting at
``R_train`` = 773.7011 (52 cells). They are read, not refitted. New fits cover the
remaining five ``R_train`` values: **260 fits.**

**Determinism.** Logistic regression's weighted arm measured
``training_seed_std`` = 0.000000 on full data in E3, and ``lbfgs`` never consumes
``random_state`` regardless of ``class_weight``, so its model-seed repetitions are
duplicates at every ``R_train``. 60 such cells are omitted. Decision tree is
**not** reduced: its 0.000052 is small but not zero.

Usage::

    python scripts/experiments/E2_training_cost_sweep.py --smoke
    python scripts/experiments/E2_training_cost_sweep.py --run
    python scripts/experiments/E2_training_cost_sweep.py --aggregate --qc
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from momo_fraud import baseline as B          # noqa: E402
from momo_fraud import constants as C         # noqa: E402
from momo_fraud import data as D              # noqa: E402
from momo_fraud import evaluate as E          # noqa: E402
from momo_fraud import experiment as X        # noqa: E402
from momo_fraud import models as M            # noqa: E402
from momo_fraud import provenance as P        # noqa: E402
from momo_fraud import splits as S            # noqa: E402
from momo_fraud.features import FeatureBuilder, columns_for   # noqa: E402

EXPERIMENT_ID = "E2_training_cost_sweep"
FROZEN_SHA = "b2e6f624"

RESULTS = PROJECT_ROOT / "results"
CELLS = RESULTS / "E2_cells"

#: Pre-declared. Not added to, not removed from, not chosen after seeing results.
R_TRAIN_GRID = [10.0, 50.0, 100.0, 773.7010836478753, 2000.0, 5000.0]
R_DEFAULT = float(C.R_DEFAULT)

#: Held fixed for the primary analysis. E2 varies training cost, not decision cost.
R_EVAL = R_DEFAULT

#: Already produced by E3 under identical conditions; read rather than refitted.
R_TRAIN_FROM_E3 = R_DEFAULT
NEW_R_TRAIN = [r for r in R_TRAIN_GRID if r != R_TRAIN_FROM_E3]

SPLIT_SEEDS = [42, 123, 456, 789]
MODEL_SEEDS = [42, 123, 456, 789]

VARIANT = "weighted"          # the only mechanism in the sweep
CONFIG = "E"                  # weighted + NER-optimal threshold

#: From E3 full-data evidence: LR weighted training_seed_std == 0.000000 exactly,
#: and lbfgs ignores random_state whatever class_weight holds. Decision tree is
#: deliberately NOT here -- 0.000052 is small, but it is not zero.
DETERMINISTIC_LEARNERS = {"logistic_regression"}


def r_tag(r: float) -> str:
    return f"R{r:g}"


def cell_id(learner: str, r_train: float, split_seed: int, model_seed: int) -> str:
    return f"{learner}__weighted__{r_tag(r_train)}__split{split_seed}__model{model_seed}"


def required_model_seeds(learner: str) -> list[int]:
    return [MODEL_SEEDS[0]] if learner in DETERMINISTIC_LEARNERS else MODEL_SEEDS


def planned_cells() -> list[tuple[str, float, int, int]]:
    return [(learner, r, split_seed, model_seed)
            for split_seed in SPLIT_SEEDS
            for r in NEW_R_TRAIN
            for learner in C.LEARNERS
            for model_seed in required_model_seeds(learner)]


# --- verifying that R_train actually reached the estimator ---------------------


def estimator_cost_parameter(estimator, learner: str) -> tuple[str, float | None]:
    """The weight the *fitted* estimator actually carries.

    Section 24 requires evidence that each requested ``R_train`` reached the
    training algorithm, not merely that the metadata says so. Different learners
    express the same cost ratio through different parameters, so this reads
    whichever one applies and returns the positive-class weight itself.
    """
    if learner == "logistic_regression":
        weights = estimator.named_steps["clf"].class_weight
        return "class_weight[1]", (None if weights is None else float(weights[1]))
    if learner == "xgboost":
        value = estimator.get_params().get("scale_pos_weight")
        return "scale_pos_weight", (None if value is None else float(value))
    weights = estimator.get_params().get("class_weight")
    return "class_weight[1]", (None if weights is None else float(weights[1]))


# --- split context -------------------------------------------------------------


class SplitContext:
    """Everything derived from one split seed, reused by every cell under it."""

    def __init__(self, df: pd.DataFrame, y: np.ndarray, split_seed: int, rung: str):
        self.split_seed = split_seed
        self.split = S.stratified_split(y, seed=split_seed)
        self.builder = FeatureBuilder().fit(df.iloc[self.split.train])

        X_all = self.builder.transform(df)
        self.cols = columns_for(rung, list(X_all.columns))
        self.parts = {k: X_all.iloc[getattr(self.split, k)][self.cols]
                      for k in ("train", "val", "test")}
        self.labels = {k: y[getattr(self.split, k)] for k in ("train", "val", "test")}

        self.prevalence = self.split.prevalence(y)
        self.class_ratio = self.split.class_ratio(y)
        self.sizes = self.split.sizes()

    def counts(self, part: str) -> tuple[int, int]:
        labels = self.labels[part]
        pos = int(labels.sum())
        return pos, int(len(labels) - pos)


# --- one cell ------------------------------------------------------------------


def run_cell(ctx: SplitContext, learner: str, r_train: float, model_seed: int,
             rung: str, git: dict, environment: dict) -> dict:
    """Fit at ``r_train``, threshold on validation at ``R_EVAL``, then score test."""
    started = time.perf_counter()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = X.fit_one(learner, VARIANT, ctx.parts["train"], ctx.labels["train"],
                        ctx.parts["val"], ctx.parts["test"],
                        r=r_train, seed=model_seed, keep_estimator=True)
        convergence_warnings = [str(w.message)[:200] for w in caught
                                if "converge" in str(w.message).lower()]

    # Read the cost parameter off the FITTED estimator, then release it.
    param_name, param_value = estimator_cost_parameter(fit.estimator, learner)
    r_train_reached = param_value is not None and abs(param_value - r_train) < 1e-9
    fit.estimator = None

    tag = f"{rung}__{r_tag(r_train)}__s{ctx.split_seed}"
    E.save_predictions(ctx.labels["val"], fit.val_scores, learner=learner,
                       config=VARIANT, split="val", seed=model_seed, tag=tag)
    E.save_predictions(ctx.labels["test"], fit.test_scores, learner=learner,
                       config=VARIANT, split="test", seed=model_seed, tag=tag)

    # SELECT on validation at R_EVAL -- never at r_train, never on test -- then LOCK.
    chosen = E.select_threshold(ctx.labels["val"], fit.val_scores, "ner_optimal", R_EVAL)
    threshold = float(chosen.threshold)

    train_pos, train_neg = ctx.counts("train")
    val_pos, val_neg = ctx.counts("val")
    test_pos, test_neg = ctx.counts("test")

    rows = []
    for part in ("val", "test"):
        rows.append(E.full_report(
            ctx.labels[part], fit.val_scores if part == "val" else fit.test_scores,
            threshold=threshold, r=R_EVAL,
            experiment_id=EXPERIMENT_ID,
            cell_id=cell_id(learner, r_train, ctx.split_seed, model_seed),
            learner=learner, csl_variant=VARIANT, csl_mechanism="class_weighting",
            config=CONFIG, level=M.CONFIG_SPECS[CONFIG].level,
            feature_set=rung, split=part, split_strategy="stratified",
            split_seed=ctx.split_seed, model_seed=model_seed,
            deterministic_in_model_seed=learner in DETERMINISTIC_LEARNERS,
            R_train=float(r_train), R_eval=float(R_EVAL),
            is_default_R_train=bool(r_train == R_DEFAULT),
            r_train_parameter=param_name,
            r_train_in_estimator=param_value,
            r_train_reached_estimator=bool(r_train_reached),
            convergence_warnings=len(convergence_warnings),
            threshold_rule="ner_optimal", threshold_selected_on="val",
            train_prevalence=ctx.prevalence["train"],
            val_prevalence=ctx.prevalence["val"],
            test_prevalence=ctx.prevalence["test"],
            train_n_positive=train_pos, train_n_negative=train_neg,
            val_n_positive=val_pos, val_n_negative=val_neg,
            test_n_positive=test_pos, test_n_negative=test_neg,
            n_train=fit.n_train, fit_seconds=fit.fit_seconds,
            git_sha=git.get("commit"), git_dirty=git.get("dirty"),
            python_version=environment["python_version"],
            sklearn_version=environment["packages"].get("scikit-learn"),
            xgboost_version=environment["packages"].get("xgboost"),
            timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ))

    return {
        "cell_id": cell_id(learner, r_train, ctx.split_seed, model_seed),
        "learner": learner, "R_train": float(r_train), "R_eval": float(R_EVAL),
        "split_seed": ctx.split_seed, "model_seed": model_seed,
        "r_train_parameter": param_name, "r_train_in_estimator": param_value,
        "r_train_reached_estimator": bool(r_train_reached),
        "convergence_warnings": convergence_warnings,
        "fit_seconds": fit.fit_seconds,
        "cell_seconds": time.perf_counter() - started,
        "status": "complete", "rows": rows,
    }


def validate_cell(payload: dict, rung: str) -> list[str]:
    problems = []
    frame = pd.DataFrame(payload["rows"])

    missing = B.validate_experiment_frame(frame)
    if missing:
        problems.append(f"missing required fields: {missing}")
    if not (frame["feature_set"] == rung).all():
        problems.append(f"feature_set is not {rung!r}")
    if not (frame["threshold_selected_on"] == "val").all():
        problems.append("threshold not selected on validation")
    if not (frame["R_eval"] == R_EVAL).all():
        problems.append("R_eval departs from the frozen default")
    if payload["R_train"] not in R_TRAIN_GRID:
        problems.append(f"R_train {payload['R_train']} not in the pre-declared grid")
    if not payload["r_train_reached_estimator"]:
        problems.append(
            f"R_train did not reach the estimator: requested {payload['R_train']}, "
            f"{payload['r_train_parameter']} = {payload['r_train_in_estimator']}")
    if frame["threshold"].nunique() != 1:
        problems.append("threshold differs between val and test")
    if not {"val", "test"} <= set(frame["split"]):
        problems.append("cell did not evaluate both val and test")
    return problems


def write_cell(payload: dict) -> Path:
    CELLS.mkdir(parents=True, exist_ok=True)
    path = CELLS / f"{payload['cell_id']}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, path)
    return path


def completed_cells() -> dict[str, dict]:
    out = {}
    if not CELLS.exists():
        return out
    for path in CELLS.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if payload.get("status") == "complete":
            out[payload["cell_id"]] = payload
    return out


def _frozen_fingerprint() -> dict[str, str]:
    import hashlib
    out = {}
    for name in ("frozen_experimental_baseline.json", "frozen_baseline_checks.json",
                 "03_experimental_rung_v2.json", "experiment_index.json",
                 "E3_runs.csv", "E3_delta_summary.csv", "E3_qc.json"):
        path = RESULTS / name
        if path.exists():
            out[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


# --- smoke test ----------------------------------------------------------------

#: A non-default R_train, a non-default split seed, a non-default model seed, and
#: a stochastic learner.
SMOKE_CELL = ("random_forest", 50.0, 456, 789)


def smoke_test(df, y, rung: str, git: dict, environment: dict) -> bool:
    learner, r_train, split_seed, model_seed = SMOKE_CELL
    print(f"Smoke test: {cell_id(learner, r_train, split_seed, model_seed)}\n")

    fingerprint_before = _frozen_fingerprint()

    ctx = SplitContext(df, y, split_seed, rung)
    payload = run_cell(ctx, learner, r_train, model_seed, rung, git, environment)
    frame = pd.DataFrame(payload["rows"])
    test = frame.query("split == 'test'").iloc[0]

    checks: list[tuple[str, bool, str]] = []

    checks.append(("R_train is non-default", r_train != R_DEFAULT,
                   f"{r_train:g} vs default {R_DEFAULT:.4f}"))
    checks.append(("R_eval fixed at the frozen default",
                   float(test["R_eval"]) == R_EVAL, f"{test['R_eval']:.4f}"))
    checks.append(("R_train and R_eval recorded independently",
                   float(test["R_train"]) != float(test["R_eval"]),
                   f"R_train {test['R_train']:g}, R_eval {test['R_eval']:.4f}"))

    # The mandatory check: the weight the fitted estimator actually carried.
    checks.append(("requested R_train reached the fitted estimator",
                   bool(payload["r_train_reached_estimator"]),
                   f"{payload['r_train_parameter']} = "
                   f"{payload['r_train_in_estimator']:g} (requested {r_train:g})"))

    # And that it changes the model, rather than merely being stored on it.
    other = X.fit_one(learner, VARIANT, ctx.parts["train"], ctx.labels["train"],
                      ctx.parts["val"], ctx.parts["test"],
                      r=R_DEFAULT, seed=model_seed, keep_estimator=True)
    other_name, other_value = estimator_cost_parameter(other.estimator, learner)
    scores_differ = not np.array_equal(
        np.asarray(payload["rows"][1]["n_flagged"]), None) and True
    default_scores = other.test_scores
    this_scores = E.load_predictions(learner, VARIANT, "test", model_seed,
                                     tag=f"{rung}__{r_tag(r_train)}__s{split_seed}")
    max_diff = float(np.max(np.abs(default_scores - this_scores["score"].to_numpy())))
    other.estimator = None
    checks.append(("different R_train produces a different fitted model",
                   max_diff > 0, f"max|score difference| vs R=773.70 = {max_diff:.4f}"))
    checks.append(("the comparison model carried the default weight",
                   other_value is not None and abs(other_value - R_DEFAULT) < 1e-9,
                   f"{other_name} = {other_value:.4f}"))

    checks.append(("correct frozen feature set", test["feature_set"] == rung,
                   str(test["feature_set"])))
    checks.append(("correct split seed", int(test["split_seed"]) == split_seed,
                   str(test["split_seed"])))
    checks.append(("correct model seed", int(test["model_seed"]) == model_seed,
                   str(test["model_seed"])))
    shares = [ctx.sizes[p] / len(df) for p in ("train", "val", "test")]
    checks.append(("split proportions 70/15/15",
                   all(abs(a - b) < 0.001 for a, b in
                       zip(shares, [C.SPLIT_TRAIN, C.SPLIT_VAL, C.SPLIT_TEST])),
                   " / ".join(f"{s:.4f}" for s in shares)))

    val_scores = E.load_predictions(learner, VARIANT, "val", model_seed,
                                    tag=f"{rung}__{r_tag(r_train)}__s{split_seed}")
    reselected = E.select_threshold(val_scores["y_true"], val_scores["score"],
                                    "ner_optimal", R_EVAL).threshold
    checks.append(("threshold selected on validation only, at R_eval",
                   abs(float(reselected) - float(test["threshold"])) < 1e-12,
                   f"{test['threshold']:.8g}"))

    test_selected = E.select_threshold(this_scores["y_true"], this_scores["score"],
                                       "ner_optimal", R_EVAL).threshold
    checks.append(("locked threshold is not the test-optimal one",
                   abs(float(test_selected) - float(test["threshold"])) > 0,
                   f"test-optimal would be {test_selected:.8g} -- not used"))

    from sklearn.metrics import average_precision_score
    from momo_fraud import risk as RK
    y_t = this_scores["y_true"].to_numpy()
    s_t = this_scores["score"].to_numpy()
    pred = s_t >= float(test["threshold"])
    checks.append(("metrics reproduce from cached predictions",
                   abs(average_precision_score(y_t, s_t) - test["pr_auc"]) < 1e-6
                   and abs(RK.normalized_expected_risk(y_t, pred, R_EVAL)
                           - test["ner"]) < 1e-9,
                   f"PR-AUC {test['pr_auc']:.6f}, NER {test['ner']:.6f}"))

    problems = validate_cell(payload, rung)
    checks.append(("provenance and schema complete", not problems,
                   "; ".join(problems) if problems else "all required fields present"))
    checks.append(("frozen baseline and E3 artifacts unchanged",
                   _frozen_fingerprint() == fingerprint_before, "sha256 match"))

    width = max(len(n) for n, _, _ in checks)
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<{width}}  {detail}")

    passed = all(ok for _, ok, _ in checks)
    print(f"\n  {sum(ok for _, ok, _ in checks)}/{len(checks)} -> "
          f"{'PASS' if passed else 'FAIL'}")

    E.save_json({
        "experiment_id": EXPERIMENT_ID,
        "smoke_cell": cell_id(learner, r_train, split_seed, model_seed),
        "verdict": "PASS" if passed else "FAIL",
        "r_train_requested": r_train,
        "r_train_in_estimator": payload["r_train_in_estimator"],
        "r_train_parameter": payload["r_train_parameter"],
        "max_score_difference_vs_default_R": max_diff,
        "checks": [{"check": n, "status": "PASS" if ok else "FAIL", "detail": d}
                   for n, ok, d in checks],
    }, "E2_smoke_test")

    if passed and not problems:
        write_cell(payload)
        print("  smoke cell retained as a completed cell")
    return passed


# --- the matrix ----------------------------------------------------------------


def run_matrix(df, y, rung: str, git: dict, environment: dict) -> None:
    plan = planned_cells()
    done = completed_cells()
    failures, retried = [], []

    print(f"\nE2 matrix: {len(plan)} planned, {len(done)} already complete, "
          f"{len(plan) - len(done)} to run\n")

    for split_seed in SPLIT_SEEDS:
        todo = [c for c in plan if c[2] == split_seed and cell_id(*c) not in done]
        if not todo:
            print(f"--- split seed {split_seed}: already complete ---", flush=True)
            continue

        print(f"--- split seed {split_seed}: {len(todo)} cells ---", flush=True)
        ctx = SplitContext(df, y, split_seed, rung)

        for learner, r_train, _, model_seed in todo:
            cid = cell_id(learner, r_train, split_seed, model_seed)
            for attempt in (1, 2):
                try:
                    payload = run_cell(ctx, learner, r_train, model_seed, rung,
                                       git, environment)
                    problems = validate_cell(payload, rung)
                    if problems:
                        raise ValueError(f"validation failed: {problems}")
                    write_cell(payload)
                    done[cid] = payload
                    row = next(x for x in payload["rows"] if x["split"] == "test")
                    warn = (f"  [{payload['convergence_warnings'] and 'CONVERGENCE' or ''}]"
                            if payload["convergence_warnings"] else "")
                    print(f"    [{len(done):>3}/{len(plan)}] {cid:<58} "
                          f"{payload['cell_seconds']:6.0f}s  "
                          f"PR-AUC {row['pr_auc']:.4f}  NER {row['ner']:.4f}{warn}",
                          flush=True)
                    break
                except Exception as exc:                       # noqa: BLE001
                    failures.append({"cell_id": cid, "attempt": attempt,
                                     "error": f"{type(exc).__name__}: {exc}",
                                     "traceback": traceback.format_exc()[-1500:]})
                    print(f"    [FAIL] {cid} attempt {attempt}: {exc}", flush=True)
                    if attempt == 1:
                        retried.append(cid)

        checkpoint(plan, done, failures, retried, split_seed)
        del ctx


def checkpoint(plan, done, failures, retried, after_split_seed: int) -> None:
    ids = [cell_id(*c) for c in plan]
    corrupted = []
    for path in CELLS.glob("*.json"):
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            corrupted.append(path.name)

    record = {
        "after_split_seed": after_split_seed,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "planned": len(plan), "completed": len(done),
        "failed": len(failures), "retried": len(retried),
        "missing": len(plan) - len(done),
        "duplicate_cell_ids": [i for i in set(ids) if ids.count(i) > 1],
        "corrupted_artifacts": corrupted,
        "frozen_fingerprint": _frozen_fingerprint(),
        "failures": failures,
    }
    path = RESULTS / "E2_checkpoints.json"
    history = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    history.append(record)
    path.write_text(json.dumps(history, indent=1), encoding="utf-8")
    print(f"    checkpoint: {record['completed']}/{record['planned']} complete, "
          f"{record['failed']} failed, {record['missing']} missing, "
          f"{len(corrupted)} corrupted", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--qc", action="store_true")
    args = parser.parse_args()
    stages = {"smoke": args.smoke, "run": args.run,
              "aggregate": args.aggregate, "qc": args.qc}
    if not any(stages.values()):
        stages = dict.fromkeys(stages, True)

    manifest = B.load_manifest()
    if manifest["status"] != "FROZEN":
        raise SystemExit(f"baseline status is {manifest['status']}, not FROZEN")
    rung = manifest["feature_set"]["frozen_rung"]
    B.assert_frozen_rung(rung)

    git, environment = P.git_state(), P.environment()
    print(f"{EXPERIMENT_ID} | rung {rung} | R_eval {R_EVAL:.4f} (fixed) | "
          f"R_train grid {[f'{r:g}' for r in R_TRAIN_GRID]}")
    print(f"  frozen SHA {FROZEN_SHA} | current {git['commit'][:8]}")

    df = D.load()
    y = df[C.TARGET].to_numpy()

    if stages["smoke"]:
        if not smoke_test(df, y, rung, git, environment):
            print("\nSMOKE TEST FAILED -- matrix not launched.")
            return 1
        print()

    if stages["run"]:
        run_matrix(df, y, rung, git, environment)

    if stages["aggregate"] or stages["qc"]:
        from E2_analysis import aggregate, quality_control       # noqa: PLC0415
        if stages["aggregate"]:
            aggregate(rung, git, environment)
        if stages["qc"]:
            return 0 if quality_control(rung) else 1
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
