"""REPAIR 2 (code side) -- move selection off the test set in notebooks 03/04/07.

``repair_02_validation_selection.py`` regenerates the affected artifacts. This
edits the notebooks themselves, so re-running them reproduces the repaired
decision rather than the superseded one.

Idempotent: each edit asserts it finds exactly one match, so a second run fails
loudly instead of double-applying.
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = PROJECT_ROOT / "notebooks"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, nb: dict) -> None:
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")


def edit(nb: dict, idx: int, old: str, new: str) -> None:
    src = "".join(nb["cells"][idx]["source"])
    if src.count(old) != 1:
        raise AssertionError(
            f"cell {idx}: expected exactly 1 match, found {src.count(old)}\n"
            f"  looking for: {old[:80]!r}"
        )
    nb["cells"][idx]["source"] = src.replace(old, new).splitlines(keepends=True)


def repair_notebook_03() -> Path:
    path = NOTEBOOKS / "03_baselines.ipynb"
    nb = load(path)

    edit(nb, 1,
         "from momo_fraud.features import FEATURE_SETS, FeatureBuilder, columns_for",
         "from momo_fraud import provenance as P\n"
         "from momo_fraud.features import (\n"
         "    DIAGNOSTIC_FEATURE_SETS, FEATURE_SETS, FeatureBuilder, columns_for,\n"
         ")\n"
         "\n"
         "#: The ablation ladder proper. `no_balance_state` is a diagnostic rung\n"
         "#: added during the repair pass and is deliberately not a candidate here.\n"
         "RUNG_ORDER = [r for r in FEATURE_SETS if r not in DIAGNOSTIC_FEATURE_SETS]")

    edit(nb, 6, "for feature_set in FEATURE_SETS:", "for feature_set in RUNG_ORDER:")

    edit(nb, 7,
         'ladder = (results.query("split == \'test\' and config == \'C\'")\n'
         '          .pivot_table(index="feature_set", columns="learner", values=["pr_auc", "ner"])\n'
         '          .reindex(list(FEATURE_SETS)))\n'
         '\n'
         'headroom = (results.query("split == \'test\' and config == \'C\'")\n'
         '            .groupby("feature_set")["ner"].min()\n'
         '            .reindex(list(FEATURE_SETS))\n'
         '            .rename("best_ner"))\n'
         'E.save_results(headroom.reset_index().to_dict("records"), "03_ladder_headroom")',

         '# Reported on test, but SELECTED on validation -- two separate operations\n'
         '# below, and they must not be collapsed into one.\n'
         'ladder = (results.query("split == \'test\' and config == \'C\'")\n'
         '          .pivot_table(index="feature_set", columns="learner", values=["pr_auc", "ner"])\n'
         '          .reindex(RUNG_ORDER))\n'
         '\n'
         '# Headroom drives the rung choice, so it comes off **validation**. Reading\n'
         '# it from test makes every downstream cost-sensitive comparison\n'
         '# test-selected, which is what the earlier version of this cell did\n'
         '# (audit finding C3).\n'
         'headroom = (X.headroom_by_rung(results.query("config == \'C\'"), split="val")\n'
         '            .reindex(RUNG_ORDER).dropna())\n'
         'headroom_test = (X.headroom_by_rung(results.query("config == \'C\'"), split="test")\n'
         '                 .reindex(RUNG_ORDER).dropna())\n'
         '\n'
         'E.save_results(pd.DataFrame({\n'
         '    "feature_set": headroom.index,\n'
         '    "best_ner_val": headroom.values,\n'
         '    "best_ner_test": headroom_test.reindex(headroom.index).values,\n'
         '}).to_dict("records"), "03_ladder_headroom_v2")')

    edit(nb, 7,
         'print("Best achievable NER per rung (0 = solved, 1 = no better than doing nothing):")',
         'print("Best achievable VALIDATION NER per rung "\n'
         '      "(0 = solved, 1 = no better than doing nothing):")')

    edit(nb, 9,
         'viable = headroom[headroom >= HEADROOM_FLOOR]\n'
         'if viable.empty:\n'
         '    chosen = headroom.idxmax()\n'
         '    note = "no rung clears the floor; taking the one with the most headroom"\n'
         'else:\n'
         '    chosen = viable.index[0]   # the least-ablated rung that still has headroom\n'
         '    note = f"least-ablated rung with NER >= {HEADROOM_FLOOR}"\n'
         '\n'
         'decision = {\n'
         '    "experimental_rung": chosen,\n'
         '    "best_ner_at_rung": float(headroom[chosen]),\n'
         '    "headroom_floor": HEADROOM_FLOOR,\n'
         '    "rationale": note,\n'
         '    "features_dropped": FEATURE_SETS[chosen],\n'
         '    "rule_precision": float(best_rule["precision"]),\n'
         '    "rule_recall": float(best_rule["recall"]),\n'
         '}\n'
         'E.save_json(decision, "03_experimental_rung")',

         '# Selected on validation. The criterion itself is unchanged from the\n'
         '# original run -- it is not retuned because the numbers moved.\n'
         'chosen, note = X.choose_rung(headroom, RUNG_ORDER, floor=HEADROOM_FLOOR)\n'
         '\n'
         'decision = {\n'
         '    "experimental_rung": chosen,\n'
         '    "selected_on": "val",\n'
         '    "best_ner_at_rung_val": float(headroom[chosen]),\n'
         '    "best_ner_at_rung_test": float(headroom_test[chosen]),\n'
         '    "headroom_floor": HEADROOM_FLOOR,\n'
         '    "rationale": note,\n'
         '    "features_dropped": FEATURE_SETS[chosen],\n'
         '    "rule_precision": float(best_rule["precision"]),\n'
         '    "rule_recall": float(best_rule["recall"]),\n'
         '    "supersedes": "03_experimental_rung.json (selected on test NER)",\n'
         '}\n'
         'E.save_json(decision, "03_experimental_rung_v2")\n'
         'P.write(P.experiment_record(\n'
         '    name="03_experimental_rung_v2",\n'
         '    seed=C.RANDOM_SEED, cost_ratio=float(R),\n'
         '    split={"strategy": "stratified", "seed": C.RANDOM_SEED},\n'
         '    evaluation={"selection_split": "val", "criterion": note,\n'
         '                "floor": HEADROOM_FLOOR},\n'
         '), "03_experimental_rung_v2_provenance")')

    edit(nb, 9,
         'print(f"Experimental rung: {chosen}  (best NER {headroom[chosen]:.4f})")',
         'print(f"Experimental rung: {chosen}  '
         '(best VALIDATION NER {headroom[chosen]:.4f})")')

    edit(nb, 10, "rungs = list(FEATURE_SETS)", "rungs = RUNG_ORDER")
    edit(nb, 12, "for feature_set in FEATURE_SETS:", "for feature_set in RUNG_ORDER:")

    save(path, nb)
    return path


def repair_notebook_04() -> Path:
    path = NOTEBOOKS / "04_cost_sensitive.ipynb"
    nb = load(path)

    edit(nb, 1,
         'decision = json.loads((PROJECT_ROOT / "results" / "03_experimental_rung.json").read_text())\n'
         'RUNG = decision["experimental_rung"]',
         'decision = X.load_experimental_rung()\n'
         'RUNG = decision["experimental_rung"]')

    edit(nb, 1,
         'print(f"experimental rung : {RUNG}")',
         'print(f"experimental rung : {RUNG}  (selected on {decision[\'selected_on\']}, "\n'
         '      f"from {decision[\'source_file\']})")')

    edit(nb, 1,
         'print(f"  best NER there  : {decision[\'best_ner_at_rung\']:.4f}  ({decision[\'rationale\']})")',
         '_best = decision.get("best_ner_at_rung_val", decision.get("best_ner_at_rung"))\n'
         'print(f"  best NER there  : {_best:.4f}  ({decision[\'rationale\']})")')

    edit(nb, 14,
         'best = test.loc[test.groupby("learner")["ner"].idxmin(),\n'
         '                ["learner", "config", "level", "threshold_rule", "ner",\n'
         '                 "recall", "precision", "n_flagged"]]\n'
         'E.save_results(best.to_dict("records"), "04_best_config_per_learner")\n'
         '\n'
         'print("Lowest-risk configuration per learner:\\n")',

         '# Selected on VALIDATION, reported on test. Taking idxmin over the test\n'
         '# rows -- which this cell used to do -- chooses the configuration on the\n'
         '# same partition it then reports (audit finding C3).\n'
         'best_val = X.select_best_config(results, metric="ner", on="val")\n'
         'best = best_val.merge(\n'
         '    test[["learner", "config", "ner", "recall", "precision",\n'
         '          "n_flagged", "pr_auc"]].rename(columns={"ner": "ner_test"}),\n'
         '    on=["learner", "config"], how="left",\n'
         ')\n'
         'E.save_results(best.to_dict("records"), "04_best_config_per_learner_v2")\n'
         '\n'
         'print("Lowest-risk configuration per learner (selected on validation):\\n")')

    save(path, nb)
    return path


def repair_notebook_07() -> Path:
    path = NOTEBOOKS / "07_interpretability.ipynb"
    nb = load(path)

    edit(nb, 1,
         'decision = json.loads((PROJECT_ROOT / "results" / "03_experimental_rung.json").read_text())\n'
         'RUNG = decision["experimental_rung"]\n'
         'SEED = C.RANDOM_SEED\n'
         'LEARNER = "xgboost"          # lowest NER in notebook 04',

         'from momo_fraud import experiment as X\n'
         '\n'
         'decision = X.load_experimental_rung()\n'
         'RUNG = decision["experimental_rung"]\n'
         'SEED = C.RANDOM_SEED\n'
         '\n'
         '# Lowest VALIDATION NER in notebook 04. Reading it off the test rows --\n'
         '# which this constant used to do -- picks the model to interpret using the\n'
         '# partition the interpretation is then reported on (audit finding C3). It\n'
         '# resolves to xgboost either way; the point is that it is no longer\n'
         '# allowed not to.\n'
         'LEARNER = X.select_best_config(\n'
         '    pd.read_csv(PROJECT_ROOT / "results" / "04_cost_sensitive.csv")\n'
         '      .query("feature_set == @RUNG"),\n'
         '    metric="ner", on="val",\n'
         ').nsmallest(1, "ner_val")["learner"].iloc[0]')

    save(path, nb)
    return path


def main() -> None:
    for repair in (repair_notebook_03, repair_notebook_04, repair_notebook_07):
        path = repair()
        json.loads(path.read_text(encoding="utf-8"))  # must still parse
        print(f"  edited and validated: {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
