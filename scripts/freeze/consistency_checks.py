"""Freeze-phase consistency checks.

These are static and artifact-level checks, not experiments: nothing is fitted,
no model is scored, no data is resampled. They verify that the controls the
repair phase installed are actually in force before the baseline is declared
frozen.

Run with ``--json`` to write ``results/frozen_baseline_checks.json``.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from momo_fraud import baseline as B          # noqa: E402
from momo_fraud import constants as C         # noqa: E402
from momo_fraud import evaluate as E          # noqa: E402
from momo_fraud import experiment as X        # noqa: E402
from momo_fraud.baseline import ConsistencyCheck  # noqa: E402

NOTEBOOKS = PROJECT_ROOT / "notebooks"
RESULTS = PROJECT_ROOT / "results"
SRC = PROJECT_ROOT / "src" / "momo_fraud"

#: Notebooks that run a main experiment at the frozen rung. 00-02 are dataset
#: characterisation and 05 is the replication, which models Lokanan's feature set
#: rather than ours by design.
EXPERIMENT_NOTEBOOKS = ["04_cost_sensitive", "06_risk_analysis", "07_interpretability",
                        "08_model_export", "09_velocity_features",
                        "10_advanced_optimization"]

#: Notebook 03 produces the rung decision rather than consuming it.
RUNG_PRODUCER = "03_baselines"


def notebook_code(stem: str) -> str:
    nb = json.loads((NOTEBOOKS / f"{stem}.ipynb").read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


# --- 1. every experiment notebook loads the frozen rung ------------------------


def check_notebooks_load_frozen_rung() -> ConsistencyCheck:
    missing = [s for s in EXPERIMENT_NOTEBOOKS
               if "load_experimental_rung" not in notebook_code(s)]
    resolved = X.load_experimental_rung()
    wrong_rung = resolved["experimental_rung"] != B.FROZEN_RUNG

    return ConsistencyCheck(
        "notebooks_load_frozen_rung",
        not missing and not wrong_rung,
        (f"all {len(EXPERIMENT_NOTEBOOKS)} experiment notebooks resolve the rung "
         f"through experiment.load_experimental_rung, which returns "
         f"{resolved['experimental_rung']!r} (selected on {resolved['selected_on']})"
         if not missing and not wrong_rung else
         f"missing loader: {missing}; resolved rung {resolved['experimental_rung']!r}"),
        [{"notebook": s, "loads_rung": s not in missing} for s in EXPERIMENT_NOTEBOOKS]
        + [{"producer": RUNG_PRODUCER, "writes": "03_experimental_rung_v2.json"}],
    )


# --- 2. no notebook silently overrides the rung --------------------------------


def check_no_rung_override() -> ConsistencyCheck:
    offenders = []
    for stem in EXPERIMENT_NOTEBOOKS + [RUNG_PRODUCER]:
        code = notebook_code(stem)
        hardcoded = re.findall(r"""RUNG\s*=\s*["']([a-z_]+)["']""", code)
        literals = [m for m in re.findall(r"""feature_set\s*=\s*["']([a-zA-Z_0-9]+)["']""", code)
                    if m in B.SELECTABLE_RUNGS or m in B.DIAGNOSTIC_RUNGS]
        if hardcoded or literals:
            offenders.append({"notebook": stem, "hardcoded_RUNG": hardcoded,
                              "feature_set_literals": literals})

    return ConsistencyCheck(
        "no_rung_override",
        not offenders,
        ("no notebook hard-codes a rung name or passes a rung literal as feature_set"
         if not offenders else f"rung overridden in: {offenders}"),
        offenders,
    )


# --- 3. all main experiments use the same split --------------------------------


def check_single_split() -> ConsistencyCheck:
    calls, offenders = [], []
    for stem in EXPERIMENT_NOTEBOOKS + [RUNG_PRODUCER]:
        code = notebook_code(stem)
        for call in re.findall(r"stratified_split\(([^)]*)\)", code):
            calls.append({"notebook": stem, "call": f"stratified_split({call})"})
            # A seed argument here would mean a partition different from the frozen one.
            if "seed" in call or "random_state" in call:
                offenders.append({"notebook": stem, "call": call})

    temporal = [{"notebook": s, "uses_temporal": "temporal_split(" in notebook_code(s)}
                for s in EXPERIMENT_NOTEBOOKS if "temporal_split(" in notebook_code(s)]

    return ConsistencyCheck(
        "single_split_across_experiments",
        not offenders,
        (f"every main experiment calls stratified_split with the default "
         f"seed={C.RANDOM_SEED}, giving identical partitions"
         if not offenders else f"non-default split seed in: {offenders}"),
        calls + [{"note": "temporal split is a separate declared arm", **t} for t in temporal],
    )


# --- 4. no test-based selection remains ----------------------------------------


def check_no_test_selection() -> ConsistencyCheck:
    offenders = []

    # The library defaults that used to read test.
    import inspect
    headroom_default = inspect.signature(X.headroom_by_rung).parameters["split"].default
    best_default = inspect.signature(X.select_best_config).parameters["on"].default
    if headroom_default != "val":
        offenders.append({"function": "headroom_by_rung", "default": headroom_default})
    if best_default != "val":
        offenders.append({"function": "select_best_config", "default": best_default})

    # idxmin/idxmax over a frame filtered to test is the notebook-level pattern
    # that produced the original defect. Comments are stripped first: the
    # repaired cells *describe* the old pattern in prose right above the fixed
    # code, and a checker that cannot tell narration from instruction reports the
    # documentation as the defect.
    for stem in EXPERIMENT_NOTEBOOKS + [RUNG_PRODUCER]:
        for raw in notebook_code(stem).splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if not any(op in line for op in ("idxmin", "idxmax", "nsmallest")):
                continue
            if "test" not in line:
                continue
            # An explicit side-by-side against the superseded rule is recorded on
            # purpose, and names itself as such.
            if "if_selected_on_test" in line:
                continue
            offenders.append({"notebook": stem, "line": line})

    return ConsistencyCheck(
        "no_test_based_selection",
        not offenders,
        ("headroom_by_rung and select_best_config both default to split='val', and "
         "no notebook selects by idxmin/nsmallest over test rows"
         if not offenders else f"test-based selection found: {offenders}"),
        offenders,
    )


# --- 5. no threshold is selected from test labels ------------------------------


def check_threshold_never_from_test() -> ConsistencyCheck:
    """Every threshold-selection call must be handed validation labels."""
    selectors = ("select_threshold", "optimal_threshold", "youden_threshold")
    offenders, sites = [], []

    targets = [(f"notebooks/{s}.ipynb", notebook_code(s))
               for s in EXPERIMENT_NOTEBOOKS + [RUNG_PRODUCER]]
    for path in sorted((PROJECT_ROOT / "scripts").rglob("*.py")):
        targets.append((str(path.relative_to(PROJECT_ROOT)), path.read_text(encoding="utf-8")))
    for path in sorted(SRC.glob("*.py")):
        targets.append((str(path.relative_to(PROJECT_ROOT)), path.read_text(encoding="utf-8")))

    for name, code in targets:
        for line in code.splitlines():
            for selector in selectors:
                if f"{selector}(" not in line or f"def {selector}" in line:
                    continue
                arg = line.split(f"{selector}(", 1)[1]
                sites.append({"file": name, "call": line.strip()[:110]})
                # A selection call whose first argument names the test partition.
                if re.search(r"""^\s*(y_test|labels\[["']test["']\]|test\[)""", arg):
                    offenders.append({"file": name, "line": line.strip()})

    return ConsistencyCheck(
        "threshold_never_selected_on_test",
        not offenders,
        (f"all {len(sites)} threshold-selection call sites take validation labels"
         if not offenders else f"threshold selected from test labels: {offenders}"),
        sites,
    )


# --- 6. test predictions are produced but never consulted for selection --------


def check_test_scores_not_used_for_selection() -> ConsistencyCheck:
    """``fit_one`` scores val and test together; only val may inform a decision.

    Producing both vectors at fit time is deliberate -- it is what makes the
    cached risk sweep possible. What matters is that no selection function ever
    receives the test vector.
    """
    source = (SRC / "experiment.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "evaluate_configs")
    selection_calls = [
        n for n in ast.walk(fn)
        if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "select_threshold"
    ]
    bad = []
    for call in selection_calls:
        args = [ast.unparse(a) for a in call.args]
        if any("test" in a for a in args[:2]):
            bad.append(args)

    return ConsistencyCheck(
        "test_scores_never_feed_selection",
        bool(selection_calls) and not bad,
        (f"evaluate_configs calls select_threshold with "
         f"{[ast.unparse(a) for a in selection_calls[0].args[:2]]}; the test vector "
         f"is scored at fit time for caching but never reaches a selection call"
         if not bad else f"test vector reaches threshold selection: {bad}"),
        [{"call": ast.unparse(c)[:120]} for c in selection_calls],
    )


# --- 7. R_train and R_eval are explicit in the experiment schema ---------------


def check_r_train_r_eval_schema() -> ConsistencyCheck:
    required = {"R_train", "R_eval"}
    declared = required.issubset(set(B.REQUIRED_EXPERIMENT_FIELDS))

    # Which existing artifacts already satisfy it, for the record.
    existing = {}
    for path in sorted(RESULTS.glob("*.csv")):
        try:
            cols = set(pd.read_csv(path, nrows=1).columns)
        except Exception:
            continue
        if {"ner", "pr_auc"} & cols:
            existing[path.stem] = sorted(required & cols) or sorted({"R"} & cols)

    return ConsistencyCheck(
        "r_train_r_eval_explicit",
        declared,
        ("R_train and R_eval are both required by "
         "baseline.REQUIRED_EXPERIMENT_FIELDS, enforced by "
         "baseline.validate_experiment_frame for all future experiments"
         if declared else "R_train/R_eval missing from the required schema"),
        [{"artifact": k, "cost_columns_present": v} for k, v in existing.items()],
    )


# --- 8. experiment artifacts carry seed information ----------------------------


def check_seed_recorded() -> ConsistencyCheck:
    schema_ok = {"model_seed", "split_seed"}.issubset(set(B.REQUIRED_EXPERIMENT_FIELDS))

    without_seed = []
    for path in sorted(RESULTS.glob("*.csv")):
        try:
            cols = set(pd.read_csv(path, nrows=1).columns)
        except Exception:
            continue
        if {"ner", "pr_auc"} & cols and not ({"seed", "model_seed"} & cols):
            without_seed.append(path.stem)

    return ConsistencyCheck(
        "seed_recorded",
        schema_ok,
        (f"model_seed and split_seed are required for all future experiments; "
         f"{len(without_seed)} existing derived artifacts carry no seed column and "
         f"inherit it from results/experiment_environment.json"
         if schema_ok else "seed fields missing from the required schema"),
        [{"artifact": a, "seed_column": False} for a in without_seed],
    )


# --- 9. superseded results cannot be loaded as current -------------------------


def check_superseded_guard() -> ConsistencyCheck:
    index_path = RESULTS / f"{B.INDEX_NAME}.json"
    if not index_path.exists():
        return ConsistencyCheck("superseded_guard", False,
                                "experiment_index.json is absent", [])

    index = json.loads(index_path.read_text(encoding="utf-8"))
    superseded = [r["artifact"] for r in index["artifacts"] if r["status"] == "SUPERSEDED"]

    refused, leaked = [], []
    for name in superseded:
        if not (RESULTS / f"{name}.csv").exists():
            continue
        try:
            B.load_results(name)
            leaked.append(name)
        except B.SupersededArtifactError:
            refused.append(name)

    # The escape hatch must still work, or the repair scripts break.
    escape_ok = True
    if refused:
        try:
            B.load_results(refused[0], allow_superseded=True)
        except Exception:
            escape_ok = False

    return ConsistencyCheck(
        "superseded_cannot_load_as_current",
        not leaked and escape_ok,
        (f"baseline.load_results refuses all {len(refused)} superseded CSV "
         f"artifacts and still permits deliberate access via allow_superseded=True"
         if not leaked and escape_ok else
         f"loaded without refusal: {leaked}; escape hatch ok: {escape_ok}"),
        [{"artifact": a, "refused": True} for a in refused],
    )


CHECKS = [
    check_notebooks_load_frozen_rung,
    check_no_rung_override,
    check_single_split,
    check_no_test_selection,
    check_threshold_never_from_test,
    check_test_scores_not_used_for_selection,
    check_r_train_r_eval_schema,
    check_seed_recorded,
    check_superseded_guard,
]


def main() -> int:
    results = [check() for check in CHECKS]

    width = max(len(r.name) for r in results)
    print("Freeze-phase consistency checks\n")
    for r in results:
        print(f"  [{'PASS' if r.passed else 'FAIL'}] {r.name:<{width}}  {r.detail}")

    passed = sum(r.passed for r in results)
    verdict = "PASS" if passed == len(results) else "FAIL"
    print(f"\n  {passed}/{len(results)} checks passed -> {verdict}")

    if "--json" in sys.argv:
        E.save_json({
            "verdict": verdict,
            "n_passed": passed,
            "n_checks": len(results),
            "frozen_rung": B.FROZEN_RUNG,
            "checks": [r.as_dict() for r in results],
        }, "frozen_baseline_checks")
        print("\n  written: results/frozen_baseline_checks.json")

    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
