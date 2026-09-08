"""REPAIR 1 (part c) -- make notebook 09's temporal section report what it must.

The notebook printed ``split_temp.prevalence(y)`` to the cell output but saved
none of it, so ``09_velocity_vs_baseline.csv`` carried temporal NER values with
no record that the temporal test partition sat at a different base rate from the
population the cost ratio was derived from.

Two edits:

* the temporal split cell now records prevalence and class ratio per partition,
  and states plainly when ``R_DEFAULT`` is not anchored on that partition;
* the comparison cell points at the repaired artifacts rather than presenting
  its own temporal rows as the robustness check.

Idempotent: each edit asserts exactly one match.
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PATH = PROJECT_ROOT / "notebooks" / "09_velocity_features.ipynb"


def edit(nb: dict, idx: int, old: str, new: str) -> None:
    src = "".join(nb["cells"][idx]["source"])
    if src.count(old) != 1:
        raise AssertionError(
            f"cell {idx}: expected 1 match, found {src.count(old)}; "
            f"looking for {old[:70]!r}")
    nb["cells"][idx]["source"] = src.replace(old, new).splitlines(keepends=True)


def main() -> None:
    nb = json.loads(PATH.read_text(encoding="utf-8"))

    edit(nb, 9,
         'split_temp = S.temporal_split(df["step"])\n'
         'print("temporal:", split_temp.sizes(), split_temp.prevalence(y))',

         '# Boundaries are derived from the empirical cumulative row count and the\n'
         '# realised proportions are asserted inside temporal_split -- the previous\n'
         '# constants (520/632) assumed a uniform step distribution and realised\n'
         '# 95.6/3.0/1.4 on the real file, with nothing to catch it.\n'
         'split_temp = S.temporal_split(df["step"])\n'
         '\n'
         '# Hitting the row proportions does not equalise the class prior: PaySim\'s\n'
         '# fraud prevalence is non-stationary in simulated time. This is recorded\n'
         '# rather than printed, because a temporal NER is not interpretable without\n'
         '# it -- NER scores "flag nothing" at 1.0 for any R, but "flag everything"\n'
         '# reaches 1.0, and Youden coincides with the NER-optimal threshold, only\n'
         '# where R equals the evaluated partition\'s class ratio.\n'
         'temporal_sizes = split_temp.sizes()\n'
         'temporal_prevalence = split_temp.prevalence(y)\n'
         'temporal_ratios = split_temp.class_ratio(y)\n'
         '\n'
         'E.save_results([\n'
         '    {"partition": part, "n_rows": temporal_sizes[part],\n'
         '     "share": temporal_sizes[part] / len(df),\n'
         '     "n_fraud": int(y[getattr(split_temp, part)].sum()),\n'
         '     "prevalence": temporal_prevalence[part],\n'
         '     "class_ratio": temporal_ratios[part]}\n'
         '    for part in ("train", "val", "test")\n'
         '], "09_temporal_prevalence")\n'
         '\n'
         'print("temporal:", temporal_sizes)\n'
         'for part in ("train", "val", "test"):\n'
         '    print(f"  {part:6s} prevalence {temporal_prevalence[part]:.5%}  "\n'
         '          f"N_neg/N_pos {temporal_ratios[part]:8.1f}")\n'
         'print(f"\\nR_DEFAULT = {R:.1f}; temporal test class ratio = "\n'
         '      f"{temporal_ratios[\'test\']:.1f} -- NER at R_DEFAULT is comparable with\\n"\n'
         '      "the stratified arm but is NOT anchored on this partition. "\n'
         '      "scripts/repairs/repair_01_temporal_split.py\\nreports all three cost "\n'
         '      "ratios; see results/09_temporal_repaired.csv.")')

    edit(nb, 12,
         'E.save_results(comparison.to_dict("records"), "09_velocity_vs_baseline")',

         '# NOTE: the temporal rows produced by this cell were invalidated by the\n'
         '# split defect (audit C1) and are superseded by\n'
         '# results/09_split_strategy_comparison.csv, built by\n'
         '# scripts/repairs/repair_01b_split_comparison.py, which reports the\n'
         '# temporal arm at three cost ratios with prevalence recorded. The\n'
         '# stratified rows here remain valid.\n'
         'E.save_results(comparison.to_dict("records"), "09_velocity_vs_baseline")')

    PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    json.loads(PATH.read_text(encoding="utf-8"))
    print(f"  edited and validated: {PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
