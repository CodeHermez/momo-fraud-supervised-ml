"""E3 -- Multi-seed x multi-split CSL robustness.

Research question: does the observed effect of cost-sensitive learning on PR-AUC
and NER persist across independently varied training seeds and stratified
train/validation/test splits?

This is a robustness experiment. Nothing is tuned, no feature is selected, no
baseline is re-chosen. The frozen baseline
(``results/frozen_experimental_baseline.json``) supplies every setting except the
two variables under study.

**The matrix.** 4 learners x 4 fit variants x 4 split seeds x 4 model seeds =
256 cells in the full factorial. Logistic regression is deterministic in the
model seed for its three non-RUS variants -- ``lbfgs`` does not consume
``random_state``, and ``std_pr_auc`` is exactly 0.0 across four seeds on the full
training set -- so 36 of those cells would be bit-identical duplicates and are
not run. **220 fits.** The omitted cells are deterministic duplicates, not
missing observations, and LR's training-seed variance is reported as exactly 0
by construction rather than estimated from repeated identical fits.

RUS is stochastic for *every* learner including LR, because the undersampling
draw is seeded by the model seed. Those 16 LR cells are all run.

Seven configurations come from four fits: thresholding is post-hoc, so A/B/C
share one unweighted model and D/E share one weighted model.

**Resumability.** Each cell writes one atomically-replaced JSON under
``results/E3_cells/``. A cell counts as complete only after fit, validation
scoring, threshold selection, threshold lock, test scoring, metrics, provenance
and schema validation have all succeeded. A crash mid-cell leaves no file, so the
cell reruns. Completed cells are never overwritten.

Usage::

    python scripts/experiments/E3_multiseed_multisplit.py --smoke
    python scripts/experiments/E3_multiseed_multisplit.py --run
    python scripts/experiments/E3_multiseed_multisplit.py --aggregate --qc
    python scripts/experiments/E3_multiseed_multisplit.py            # all of it
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
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
from momo_fraud import provenance as P        # noqa: E402
from momo_fraud import splits as S            # noqa: E402
from momo_fraud.features import FeatureBuilder, columns_for   # noqa: E402

EXPERIMENT_ID = "E3_multiseed_multisplit"
FROZEN_SHA = "b2e6f624"

RESULTS = PROJECT_ROOT / "results"
CELLS = RESULTS / "E3_cells"

SPLIT_SEEDS = [42, 123, 456, 789]
MODEL_SEEDS = [42, 123, 456, 789]
VARIANTS = ["unweighted", "weighted", "instance_weighted", "rus"]

#: LR is deterministic in the model seed for these. Established two ways:
#: ``lbfgs`` never reads ``random_state`` (only sag/saga/liblinear do), and
#: ``results/10_multiseed_ner.csv`` records std_pr_auc == 0.0 exactly across four
#: seeds on the full 4.45M-row training set. RUS is excluded because the draw
#: itself consumes the model seed.
DETERMINISTIC_CELLS = {("logistic_regression", v) for v in
                       ("unweighted", "weighted", "instance_weighted")}

#: ``R_train`` for an unweighted fit. 1.0 is the literal value the learner
#: receives (scale_pos_weight=1.0 / class_weight=None), and means "no cost
#: weighting at training time". Recorded rather than left null so the column
#: stays numeric and R_train is never inferred from R_eval.
R_TRAIN_UNWEIGHTED = 1.0


# --- cell identity ------------------------------------------------------------


def cell_id(learner: str, variant: str, split_seed: int, model_seed: int) -> str:
    return f"{learner}__{variant}__split{split_seed}__model{model_seed}"


def required_model_seeds(learner: str, variant: str) -> list[int]:
    """Model seeds this cell actually needs. One if deterministic, else all four."""
    return [MODEL_SEEDS[0]] if (learner, variant) in DETERMINISTIC_CELLS else MODEL_SEEDS


def planned_cells() -> list[tuple[str, str, int, int]]:
    return [(learner, variant, split_seed, model_seed)
            for split_seed in SPLIT_SEEDS
            for learner in C.LEARNERS
            for variant in VARIANTS
            for model_seed in required_model_seeds(learner, variant)]


def omitted_cells() -> list[tuple[str, str, int, int]]:
    return [(learner, variant, split_seed, model_seed)
            for split_seed in SPLIT_SEEDS
            for learner in C.LEARNERS
            for variant in VARIANTS
            for model_seed in MODEL_SEEDS
            if model_seed not in required_model_seeds(learner, variant)]


# --- one split's data ---------------------------------------------------------


class SplitContext:
    """Everything derived from one split seed. Built once, reused by every cell.

    The FeatureBuilder is fitted on *this split's* training rows, so a new split
    seed means a new builder -- the high-risk cutoff and severity grid are fitted
    state and must not carry across partitions.
    """

    def __init__(self, df: pd.DataFrame, y: np.ndarray, split_seed: int, rung: str):
        self.split_seed = split_seed
        self.split = S.stratified_split(y, seed=split_seed)

        self.builder = FeatureBuilder().fit(df.iloc[self.split.train])
        X_all = self.builder.transform(df)
        self.cols = columns_for(rung, list(X_all.columns))
        severity_all = self.builder.severity_of(df)

        self.parts = {k: X_all.iloc[getattr(self.split, k)][self.cols]
                      for k in ("train", "val", "test")}
        self.labels = {k: y[getattr(self.split, k)] for k in ("train", "val", "test")}
        self.severity = {k: severity_all[getattr(self.split, k)]
                         for k in ("train", "val", "test")}

        self.prevalence = self.split.prevalence(y)
        self.class_ratio = self.split.class_ratio(y)
        self.sizes = self.split.sizes()

    def counts(self, part: str) -> tuple[int, int]:
        labels = self.labels[part]
        pos = int(labels.sum())
        return pos, int(len(labels) - pos)

    def prevalence_record(self) -> dict:
        record = {"split_seed": self.split_seed}
        for part in ("train", "val", "test"):
            pos, neg = self.counts(part)
            record.update({
                f"{part}_n": self.sizes[part],
                f"{part}_prevalence": self.prevalence[part],
                f"{part}_n_positive": pos,
                f"{part}_n_negative": neg,
                f"{part}_class_ratio": self.class_ratio[part],
            })
        return record


# --- one cell -----------------------------------------------------------------


def run_cell(ctx: SplitContext, learner: str, variant: str, model_seed: int,
             rung: str, r: float, git: dict, environment: dict) -> dict:
    """Fit one cell and evaluate every configuration it serves.

    Order is the frozen protocol and is not negotiable here: fit on train, score
    validation, select the threshold on validation, lock it, then score test.
    ``fit_one`` produces both score vectors together for caching, but the test
    vector is never passed to a selection call.
    """
    started = time.perf_counter()

    fit = X.fit_one(
        learner, variant,
        ctx.parts["train"], ctx.labels["train"], ctx.parts["val"], ctx.parts["test"],
        r=r, severity_train=ctx.severity["train"], seed=model_seed,
    )

    tag = f"{rung}__s{ctx.split_seed}"
    E.save_predictions(ctx.labels["val"], fit.val_scores, learner=learner,
                       config=variant, split="val", seed=model_seed, tag=tag)
    E.save_predictions(ctx.labels["test"], fit.test_scores, learner=learner,
                       config=variant, split="test", seed=model_seed, tag=tag)

    r_train = r if variant != "unweighted" else R_TRAIN_UNWEIGHTED
    train_pos, train_neg = ctx.counts("train")
    val_pos, val_neg = ctx.counts("val")
    test_pos, test_neg = ctx.counts("test")

    rows = []
    for config in X.FIT_VARIANTS[variant]:
        rule = X.CONFIG_RULES[config]

        # SELECT on validation, then LOCK. Nothing below reassigns `threshold`.
        chosen = E.select_threshold(ctx.labels["val"], fit.val_scores, rule, r)
        threshold = float(chosen.threshold)

        for part in ("val", "test"):
            rows.append(E.full_report(
                ctx.labels[part], fit.val_scores if part == "val" else fit.test_scores,
                threshold=threshold, r=r,
                # --- identity
                experiment_id=EXPERIMENT_ID,
                cell_id=cell_id(learner, variant, ctx.split_seed, model_seed),
                learner=learner, csl_variant=variant, config=config,
                level=X.M.CONFIG_SPECS[config].level,
                feature_set=rung, split=part, split_strategy="stratified",
                # --- the two experimental variables
                split_seed=ctx.split_seed, model_seed=model_seed,
                deterministic_in_model_seed=(learner, variant) in DETERMINISTIC_CELLS,
                # --- cost, recorded separately and never inferred
                R_train=r_train, R_eval=float(r),
                # --- threshold
                threshold_rule=rule, threshold_selected_on="val",
                # --- prevalence
                train_prevalence=ctx.prevalence["train"],
                val_prevalence=ctx.prevalence["val"],
                test_prevalence=ctx.prevalence["test"],
                train_n_positive=train_pos, train_n_negative=train_neg,
                val_n_positive=val_pos, val_n_negative=val_neg,
                test_n_positive=test_pos, test_n_negative=test_neg,
                # --- provenance
                n_train=fit.n_train, fit_seconds=fit.fit_seconds,
                git_sha=git.get("commit"), git_dirty=git.get("dirty"),
                python_version=environment["python_version"],
                sklearn_version=environment["packages"].get("scikit-learn"),
                xgboost_version=environment["packages"].get("xgboost"),
                timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ))

    return {
        "cell_id": cell_id(learner, variant, ctx.split_seed, model_seed),
        "learner": learner, "variant": variant,
        "split_seed": ctx.split_seed, "model_seed": model_seed,
        "configs": X.FIT_VARIANTS[variant],
        "fit_seconds": fit.fit_seconds,
        "cell_seconds": time.perf_counter() - started,
        "status": "complete",
        "rows": rows,
    }


def validate_cell(payload: dict, rung: str, r: float) -> list[str]:
    """Schema and protocol validation. A cell is complete only if this is empty."""
    problems = []
    frame = pd.DataFrame(payload["rows"])

    missing = B.validate_experiment_frame(frame)
    if missing:
        problems.append(f"missing required fields: {missing}")

    if not (frame["feature_set"] == rung).all():
        problems.append(f"feature_set is not {rung!r}")
    if not (frame["threshold_selected_on"] == "val").all():
        problems.append("threshold not selected on validation")
    if not (frame["R_eval"] == r).all():
        problems.append("R_eval departs from the frozen default")
    if frame["R_train"].isna().any():
        problems.append("R_train missing")
    if frame["split_seed"].isna().any() or frame["model_seed"].isna().any():
        problems.append("seed provenance missing")
    if payload["split_seed"] not in SPLIT_SEEDS:
        problems.append(f"split seed {payload['split_seed']} not pre-declared")
    if payload["model_seed"] not in MODEL_SEEDS:
        problems.append(f"model seed {payload['model_seed']} not pre-declared")

    # The locked threshold must be identical on val and test within a config.
    for config, group in frame.groupby("config"):
        if group["threshold"].nunique() != 1:
            problems.append(f"config {config}: threshold differs between val and test")

    if not {"val", "test"} <= set(frame["split"]):
        problems.append("cell did not evaluate both val and test")

    return problems


def write_cell(payload: dict) -> Path:
    """Atomic write: a crash mid-write leaves no file, so the cell reruns."""
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
            continue                      # truncated -> treat as incomplete
        if payload.get("status") == "complete":
            out[payload["cell_id"]] = payload
    return out


# --- smoke test ---------------------------------------------------------------

#: One representative cell: a stochastic learner, a non-default split seed, a
#: non-default model seed, and the RUS variant, which exercises training-only
#: resampling as well as the rest of the pipeline.
SMOKE_CELL = ("xgboost", "rus", 123, 456)


def smoke_test(df, y, rung: str, r: float, git: dict, environment: dict) -> bool:
    learner, variant, split_seed, model_seed = SMOKE_CELL
    print(f"Smoke test: {cell_id(learner, variant, split_seed, model_seed)}\n")

    baseline_before = _frozen_fingerprint()

    ctx = SplitContext(df, y, split_seed, rung)
    payload = run_cell(ctx, learner, variant, model_seed, rung, r, git, environment)
    frame = pd.DataFrame(payload["rows"])
    test = frame.query("split == 'test'").iloc[0]

    checks: list[tuple[str, bool, str]] = []

    checks.append(("feature rung is no_origin_balance",
                   test["feature_set"] == rung, str(test["feature_set"])))
    checks.append(("split seed correct",
                   int(test["split_seed"]) == split_seed, str(test["split_seed"])))
    checks.append(("model seed correct",
                   int(test["model_seed"]) == model_seed, str(test["model_seed"])))

    shares = [ctx.sizes[p] / len(df) for p in ("train", "val", "test")]
    checks.append(("split proportions 70/15/15",
                   all(abs(a - b) < 0.001 for a, b in
                       zip(shares, [C.SPLIT_TRAIN, C.SPLIT_VAL, C.SPLIT_TEST])),
                   " / ".join(f"{s:.4f}" for s in shares)))

    cutoff_train = float(np.percentile(df["amount"].iloc[ctx.split.train], 99.0))
    checks.append(("preprocessing fitted on training rows only",
                   abs(ctx.builder.high_risk_cutoff - cutoff_train) < 1e-6,
                   f"cutoff {ctx.builder.high_risk_cutoff:,.2f} == train p99"))

    from momo_fraud import models as M
    rus_rows = M.resample_indices(ctx.labels["train"], M.CONFIG_SPECS["G"], r,
                                  seed=model_seed)
    checks.append(("resampling applied to training rows only",
                   len(rus_rows) == payload["rows"][0]["n_train"]
                   and len(rus_rows) < ctx.sizes["train"],
                   f"n_train {payload['rows'][0]['n_train']:,} of "
                   f"{ctx.sizes['train']:,} training rows"))

    # Threshold must equal the value selected on validation, and nothing else.
    val_scores = E.load_predictions(learner, variant, "val", model_seed,
                                    tag=f"{rung}__s{split_seed}")
    reselected = E.select_threshold(val_scores["y_true"], val_scores["score"],
                                    test["threshold_rule"], r).threshold
    checks.append(("threshold selected on validation only",
                   abs(float(reselected) - float(test["threshold"])) < 1e-12,
                   f"{test['threshold']:.8g}"))

    test_scores = E.load_predictions(learner, variant, "test", model_seed,
                                     tag=f"{rung}__s{split_seed}")
    test_selected = E.select_threshold(test_scores["y_true"], test_scores["score"],
                                       test["threshold_rule"], r).threshold
    checks.append(("locked threshold differs from a test-selected one",
                   abs(float(test_selected) - float(test["threshold"])) > 0,
                   f"test-optimal would be {test_selected:.8g} -- not used"))

    checks.append(("R_train explicit", not pd.isna(test["R_train"]),
                   f"{test['R_train']:.4f}"))
    checks.append(("R_eval explicit", not pd.isna(test["R_eval"]),
                   f"{test['R_eval']:.4f}"))
    checks.append(("R_train and R_eval recorded independently",
                   "R_train" in test and "R_eval" in test,
                   "both columns present"))

    # Metrics recomputed from the cached vectors.
    from sklearn.metrics import average_precision_score
    from momo_fraud import risk as RK
    y_t = test_scores["y_true"].to_numpy()
    s_t = test_scores["score"].to_numpy()
    pred = s_t >= float(test["threshold"])
    checks.append(("PR-AUC reproduces from cached predictions",
                   abs(average_precision_score(y_t, s_t) - test["pr_auc"]) < 1e-6,
                   f"{test['pr_auc']:.6f}"))
    checks.append(("NER reproduces from cached predictions",
                   abs(RK.normalized_expected_risk(y_t, pred, r) - test["ner"]) < 1e-9,
                   f"{test['ner']:.6f}"))
    checks.append(("confusion counts reproduce",
                   int(((~y_t.astype(bool)) & pred).sum()) == int(test["fp"])
                   and int((y_t.astype(bool) & ~pred).sum()) == int(test["fn"]),
                   f"fp {int(test['fp'])}, fn {int(test['fn'])}"))

    problems = validate_cell(payload, rung, r)
    checks.append(("artifact provenance complete", not problems,
                   "; ".join(problems) if problems else "all required fields present"))

    checks.append(("frozen baseline artifacts unchanged",
                   _frozen_fingerprint() == baseline_before, "sha256 match"))

    width = max(len(n) for n, _, _ in checks)
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<{width}}  {detail}")

    passed = all(ok for _, ok, _ in checks)
    print(f"\n  {sum(ok for _, ok, _ in checks)}/{len(checks)} -> "
          f"{'PASS' if passed else 'FAIL'}")

    E.save_json({
        "experiment_id": EXPERIMENT_ID,
        "smoke_cell": cell_id(learner, variant, split_seed, model_seed),
        "verdict": "PASS" if passed else "FAIL",
        "checks": [{"check": n, "status": "PASS" if ok else "FAIL", "detail": d}
                   for n, ok, d in checks],
    }, "E3_smoke_test")

    if passed and not problems:
        write_cell(payload)          # a valid cell is a valid cell; keep it
        print(f"  smoke cell retained as a completed cell")
    return passed


def _frozen_fingerprint() -> dict[str, str]:
    """Hashes of the frozen baseline artifacts, to prove E3 did not touch them."""
    import hashlib
    out = {}
    for name in ("frozen_experimental_baseline.json", "frozen_baseline_checks.json",
                 "03_experimental_rung_v2.json", "experiment_index.json"):
        path = RESULTS / name
        if path.exists():
            out[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


# --- the matrix ---------------------------------------------------------------


def run_matrix(df, y, rung: str, r: float, git: dict, environment: dict) -> dict:
    plan = planned_cells()
    done = completed_cells()
    ledger = {"failures": [], "retried": []}

    print(f"\nE3 matrix: {len(plan)} planned, {len(done)} already complete, "
          f"{len(plan) - len(done)} to run\n")

    started = time.perf_counter()
    for split_seed in SPLIT_SEEDS:
        todo = [c for c in plan if c[2] == split_seed and cell_id(*c) not in done]
        if not todo:
            print(f"--- split seed {split_seed}: already complete ---")
            continue

        print(f"--- split seed {split_seed}: {len(todo)} cells ---", flush=True)
        ctx = SplitContext(df, y, split_seed, rung)
        print(f"    prevalence train {ctx.prevalence['train']:.5%} "
              f"val {ctx.prevalence['val']:.5%} test {ctx.prevalence['test']:.5%}",
              flush=True)

        for learner, variant, _, model_seed in todo:
            cid = cell_id(learner, variant, split_seed, model_seed)
            for attempt in (1, 2):
                try:
                    payload = run_cell(ctx, learner, variant, model_seed, rung, r,
                                       git, environment)
                    problems = validate_cell(payload, rung, r)
                    if problems:
                        raise ValueError(f"schema validation failed: {problems}")
                    write_cell(payload)
                    done[cid] = payload
                    test_row = next(x for x in payload["rows"] if x["split"] == "test")
                    print(f"    [{len(done):>3}/{len(plan)}] {cid:<52} "
                          f"{payload['cell_seconds']:6.0f}s  "
                          f"PR-AUC {test_row['pr_auc']:.4f}  NER {test_row['ner']:.4f}",
                          flush=True)
                    break
                except Exception as exc:                     # noqa: BLE001
                    failure = {"cell_id": cid, "attempt": attempt,
                               "error": f"{type(exc).__name__}: {exc}",
                               "traceback": traceback.format_exc()[-1500:]}
                    ledger["failures"].append(failure)
                    print(f"    [FAIL] {cid} attempt {attempt}: {exc}", flush=True)
                    if attempt == 1:
                        ledger["retried"].append(cid)
                    else:
                        break

        # Checkpoint integrity after each split seed. Integrity only -- no
        # parameter is read, changed, or decided on here.
        checkpoint(plan, done, ledger, split_seed)

        del ctx

    ledger["elapsed_seconds"] = time.perf_counter() - started
    return {"plan": plan, "done": done, "ledger": ledger}


def checkpoint(plan, done, ledger, after_split_seed: int) -> None:
    ids = [cell_id(*c) for c in plan]
    duplicates = [i for i in set(ids) if ids.count(i) > 1]
    corrupted = []
    for path in CELLS.glob("*.json"):
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            corrupted.append(path.name)

    record = {
        "after_split_seed": after_split_seed,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "planned": len(plan),
        "completed": len(done),
        "failed": len(ledger["failures"]),
        "missing": len(plan) - len(done),
        "duplicate_cell_ids": duplicates,
        "corrupted_artifacts": corrupted,
        "frozen_baseline_fingerprint": _frozen_fingerprint(),
    }
    path = RESULTS / "E3_checkpoints.json"
    history = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    history.append(record)
    path.write_text(json.dumps(history, indent=1), encoding="utf-8")

    print(f"    checkpoint: {record['completed']}/{record['planned']} complete, "
          f"{record['failed']} failed, {record['missing']} missing, "
          f"{len(duplicates)} duplicate ids, {len(corrupted)} corrupted", flush=True)


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
    r = float(manifest["cost"]["R_default"])
    B.assert_frozen_rung(rung)

    git, environment = P.git_state(), P.environment()
    print(f"{EXPERIMENT_ID} | rung {rung} | R {r:.4f} | "
          f"frozen SHA {FROZEN_SHA} | current {git['commit'][:8]}")

    df = D.load()
    y = df[C.TARGET].to_numpy()

    if stages["smoke"]:
        if not smoke_test(df, y, rung, r, git, environment):
            print("\nSMOKE TEST FAILED -- matrix not launched.")
            return 1
        print()

    if stages["run"]:
        run_matrix(df, y, rung, r, git, environment)

    if stages["aggregate"] or stages["qc"]:
        from E3_analysis import aggregate, quality_control     # noqa: PLC0415
        if stages["aggregate"]:
            aggregate(rung, r, git, environment)
        if stages["qc"]:
            return 0 if quality_control(rung, r) else 1

    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
